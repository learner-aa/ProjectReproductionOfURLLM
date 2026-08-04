"""
Instruction 数据集构建模块

将用户画像 + 交互序列转化为 LLM 微调所需的 Instruction 格式数据。
支持 Alpaca 风格和 ChatML 格式。
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_utils import PROCESSED_DIR, load_json, save_json
from prompt_templates import (
    PROMPT_II_RECOMMEND_BASE,
    PROMPT_II_RECOMMEND_WITH_PROFILE,
    PROMPT_II_RECOMMEND_COT,
    format_interaction_sequence,
    format_attributes,
)

logger = logging.getLogger(__name__)


# ============================================================
# Instruction 数据生成
# ============================================================

def build_single_instruction(
    user_id: str,
    user_seq: List[Dict],
    target_item: Dict,
    user_profile: Optional[Dict],
    item_metadata: Dict[str, Dict],
    item_attributes: Dict[str, Dict],
    template_type: str = "profile",
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
) -> Dict[str, str]:
    """
    为单个用户构建一条 Instruction 数据。

    Args:
        user_id: 用户 ID
        user_seq: 用户交互序列 [{"item_id": str, "domain": str}, ...]
        target_item: 目标推荐物品 {"item_id": str, "domain": str}
        user_profile: 用户画像字典
        item_metadata: 物品元数据
        item_attributes: 物品属性
        template_type: "base" | "profile" | "cot"
        domain_x_name: X 域名称
        domain_y_name: Y 域名称

    Returns:
        {"instruction": str, "input": str, "output": str}
    """
    target_domain = target_item.get("domain", domain_x_name)
    target_title = item_metadata.get(
        target_item["item_id"], {}
    ).get("title", target_item["item_id"])

    # 构建交互序列文本
    seq_text = format_interaction_sequence(
        user_seq, item_metadata, max_display=15
    )

    # 构建画像文本
    if user_profile and template_type != "base":
        behavior = user_profile.get("behavior", {})
        semantic = user_profile.get("semantic", {})
        embedding = user_profile.get("embedding", {})

        # 近邻物品文本
        similar_x = embedding.get("top_similar_items_x", [])
        similar_y = embedding.get("top_similar_items_y", [])
        similar_items = ", ".join(
            [i["title"] for i in similar_x[:3]] +
            [i["title"] for i in similar_y[:3]]
        ) or "N/A"

        profile_text = (
            f"Total interactions: {behavior.get('total_interactions', 0)}\n"
            f"Domain distribution: {domain_x_name}({behavior.get('domain_x_count', 0)}), "
            f"{domain_y_name}({behavior.get('domain_y_count', 0)})\n"
            f"Top attributes: {format_attributes(semantic.get('preferred_attributes', []))}\n"
            f"Top categories: {format_attributes(semantic.get('preferred_categories', []))}"
        )

        preferred_attrs = format_attributes(semantic.get("preferred_attributes", []))
        preferred_cats = format_attributes(semantic.get("preferred_categories", []))

    else:
        profile_text = "N/A"
        similar_items = "N/A"
        preferred_attrs = "N/A"
        preferred_cats = "N/A"

    # 选择模板
    if template_type == "base":
        instruction = PROMPT_II_RECOMMEND_BASE.format(
            target_domain=target_domain,
            user_profile_text=profile_text,
            interaction_sequence=seq_text,
        )
    elif template_type == "cot":
        instruction = PROMPT_II_RECOMMEND_COT.format(
            target_domain=target_domain,
            user_profile_text=profile_text,
            interaction_sequence=seq_text,
        )
    else:  # profile
        instruction = PROMPT_II_RECOMMEND_WITH_PROFILE.format(
            target_domain=target_domain,
            user_profile_text=profile_text,
            interaction_sequence=seq_text,
            preferred_attributes=preferred_attrs,
            preferred_categories=preferred_cats,
            similar_items=similar_items,
        )

    return {
        "instruction": f"Based on the user's interaction history and profile, recommend a new {target_domain} item.",
        "input": instruction.replace("Instruction:", "").replace("Input:", "").strip(),
        "output": target_title,
        "user_id": user_id,
        "target_item_id": target_item["item_id"],
        "target_domain": target_domain,
    }


def build_instruction_dataset(
    split_data: Dict[str, Dict],
    user_profiles: Dict[str, Dict],
    item_metadata: Dict[str, Dict],
    item_attributes: Dict[str, Dict],
    template_type: str = "profile",
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
) -> List[Dict[str, str]]:
    """
    批量构建 Instruction 数据集。

    Args:
        split_data: {user_id: {"seq": [...], "target": {...}}} (valid 或 test 格式)
                    或 {user_id: {"seq": [...]}} (train 格式，需自行构造target)
        user_profiles: 用户画像
        item_metadata: 物品元数据
        item_attributes: 物品属性
        template_type: "base" | "profile" | "cot"
        domain_x_name: X 域名称
        domain_y_name: Y 域名称

    Returns:
        list of instruction dicts
    """
    instructions = []
    skipped = 0

    for user_id, data in split_data.items():
        seq = data.get("seq", [])
        target = data.get("target")

        # 训练集没有 target，用序列最后一个作为 target
        if target is None and len(seq) >= 2:
            target = seq[-1]
            seq = seq[:-1]
        elif target is None:
            skipped += 1
            continue

        profile = user_profiles.get(user_id, {})

        try:
            inst = build_single_instruction(
                user_id=user_id,
                user_seq=seq,
                target_item=target,
                user_profile=profile,
                item_metadata=item_metadata,
                item_attributes=item_attributes,
                template_type=template_type,
                domain_x_name=domain_x_name,
                domain_y_name=domain_y_name,
            )
            instructions.append(inst)
        except Exception as e:
            logger.warning(f"构建 {user_id} 的 instruction 失败: {e}")
            skipped += 1

    logger.info(
        f"Instruction 数据构建: {len(instructions)} 条, 跳过 {skipped} 条"
    )
    return instructions


# ============================================================
# ChatML 格式转换 (用于某些模型的微调)
# ============================================================

def convert_to_chatml(instructions: List[Dict]) -> List[Dict]:
    """
    将 Alpaca 格式转为 ChatML 格式。

    Returns:
        [{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}]
    """
    chatml_data = []
    for inst in instructions:
        messages = [
            {
                "role": "system",
                "content": "You are a cross-domain recommendation assistant. "
                           "Based on user profiles and interaction history, "
                           "recommend items that match user preferences."
            },
            {
                "role": "user",
                "content": f"{inst['instruction']}\n\n{inst['input']}"
            },
            {
                "role": "assistant",
                "content": inst["output"]
            },
        ]
        chatml_data.append({
            "messages": messages,
            "user_id": inst.get("user_id"),
            "target_item_id": inst.get("target_item_id"),
            "target_domain": inst.get("target_domain"),
        })
    return chatml_data


# ============================================================
# 主流程
# ============================================================

def build_all_instruction_data(config: Dict):
    """
    构建全部 Instruction 数据集 (train/valid/test)。

    Args:
        config: pipeline 配置
    """
    logger.info("=" * 60)
    logger.info("开始构建 Instruction 数据集")
    logger.info("=" * 60)

    # 加载数据
    train_data = load_json(PROCESSED_DIR / "train.json")
    valid_data = load_json(PROCESSED_DIR / "valid.json")
    test_data = load_json(PROCESSED_DIR / "test.json")
    item_metadata = load_json(PROCESSED_DIR / "item_metadata.json")
    id_mapping = load_json(PROCESSED_DIR / "id_mapping.json")

    # 用户画像 (可选)
    profile_path = PROCESSED_DIR / "user_profiles.json"
    user_profiles = load_json(profile_path) if profile_path.exists() else {}

    # 物品属性 (可选)
    attr_path = PROCESSED_DIR / "item_attributes.json"
    item_attributes = load_json(attr_path) if attr_path.exists() else {}

    # 配置
    inst_cfg = config.get("instruction", {})
    template_type = inst_cfg.get("template_type", "profile")
    output_format = inst_cfg.get("output_format", "alpaca")
    domain_cfg = config.get("domains", {})
    domain_x = domain_cfg.get("x", "Entertainment")
    domain_y = domain_cfg.get("y", "Education")

    # 构建各 split 的 instruction 数据
    logger.info("构建训练集 instructions...")
    train_inst = build_instruction_dataset(
        train_data, user_profiles, item_metadata, item_attributes,
        template_type, domain_x, domain_y,
    )

    logger.info("构建验证集 instructions...")
    valid_inst = build_instruction_dataset(
        valid_data, user_profiles, item_metadata, item_attributes,
        template_type, domain_x, domain_y,
    )

    logger.info("构建测试集 instructions...")
    test_inst = build_instruction_dataset(
        test_data, user_profiles, item_metadata, item_attributes,
        template_type, domain_x, domain_y,
    )

    # 格式转换
    if output_format == "chatml":
        logger.info("转换为 ChatML 格式...")
        train_inst = convert_to_chatml(train_inst)
        valid_inst = convert_to_chatml(valid_inst)
        test_inst = convert_to_chatml(test_inst)

    # 保存
    save_json(train_inst, PROCESSED_DIR / "train_instructions.json")
    save_json(valid_inst, PROCESSED_DIR / "valid_instructions.json")
    save_json(test_inst, PROCESSED_DIR / "test_instructions.json")

    # 统计
    logger.info("=" * 60)
    logger.info("Instruction 数据集构建完成:")
    logger.info(f"  训练集: {len(train_inst)} 条")
    logger.info(f"  验证集: {len(valid_inst)} 条")
    logger.info(f"  测试集: {len(test_inst)} 条")
    logger.info(f"  模板类型: {template_type}")
    logger.info(f"  输出格式: {output_format}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        build_all_instruction_data(cfg)
    else:
        build_all_instruction_data({})
