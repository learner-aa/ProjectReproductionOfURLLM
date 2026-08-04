"""
原始数据预处理模块

将 Amazon Entertainment-Education 跨域数据集解析为统一格式:
- 用户交互序列
- 物品元数据
- ID 映射表
- train/valid/test 划分 (leave-one-out)
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_utils import (
    DATA_DIR, PROCESSED_DIR, RAW_DIR,
    load_json, save_json, ensure_dirs,
)

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

DEFAULT_CONFIG = {
    "min_interactions": 3,       # 最少交互次数过滤
    "max_seq_len": 15,           # 最大序列长度 (与 DG 模型一致)
    "random_seed": 2040,         # 随机种子 (与 DG 模型一致)
    "domain_x_name": "Entertainment",
    "domain_y_name": "Education",
}


# ============================================================
# 原始数据解析
# ============================================================

def parse_amazon_reviews(filepath: str, domain_label: str) -> List[Dict]:
    """
    解析 Amazon 评论/交互数据文件 (JSON Lines 格式)。

    Args:
        filepath: 原始数据文件路径
        domain_label: 域标签 ("Entertainment" 或 "Education")

    Returns:
        list of {"user_id": str, "item_id": str, "timestamp": int, "domain": str}
    """
    interactions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                interactions.append({
                    "user_id": str(record.get("user_id", record.get("reviewerID", ""))),
                    "item_id": str(record.get("item_id", record.get("asin", ""))),
                    "timestamp": int(record.get("timestamp", record.get("unixReviewTime", 0))),
                    "domain": domain_label,
                })
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"跳过无效记录: {e}")
                continue
    logger.info(f"解析 {filepath}: {len(interactions)} 条交互, 域={domain_label}")
    return interactions


def parse_amazon_metadata(filepath: str) -> Dict[str, Dict]:
    """
    解析 Amazon 物品元数据文件 (JSON Lines 格式)。

    Returns:
        dict: {item_id: {"title": str, "description": str, "category": list, "brand": str}}
    """
    metadata = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                item_id = str(record.get("asin", record.get("item_id", "")))
                desc = record.get("description", "")
                if isinstance(desc, list):
                    desc = " ".join(desc)
                metadata[item_id] = {
                    "title": record.get("title", ""),
                    "description": desc,
                    "category": record.get("categories", record.get("category", [])),
                    "brand": record.get("brand", ""),
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"跳过无效元数据: {e}")
                continue
    logger.info(f"解析 {filepath}: {len(metadata)} 个物品元数据")
    return metadata


# ============================================================
# 数据清洗与序列构建
# ============================================================

def build_user_sequences(
    interactions_x: List[Dict],
    interactions_y: List[Dict],
    min_interactions: int = 3,
    max_seq_len: int = 15,
) -> Dict[str, List[Dict]]:
    """
    合并双域交互并按时间排序构建用户序列。

    Args:
        interactions_x: X 域交互列表
        interactions_y: Y 域交互列表
        min_interactions: 最少交互次数 (过滤冷用户)
        max_seq_len: 最大序列长度

    Returns:
        dict: {user_id: [{"item_id": str, "domain": str, "timestamp": int}, ...]}
    """
    user_interactions = defaultdict(list)

    for inter in interactions_x + interactions_y:
        user_interactions[inter["user_id"]].append({
            "item_id": inter["item_id"],
            "domain": inter["domain"],
            "timestamp": inter["timestamp"],
        })

    # 按时间排序 + 过滤
    filtered = {}
    for user_id, seq in user_interactions.items():
        seq.sort(key=lambda x: x["timestamp"])
        if len(seq) >= min_interactions:
            # 截断到最大长度
            filtered[user_id] = seq[-max_seq_len:]

    logger.info(
        f"序列构建完成: {len(user_interactions)} 用户 → "
        f"{len(filtered)} 用户 (过滤<{min_interactions}次交互)"
    )
    return filtered


def extract_item_set(user_sequences: Dict[str, List[Dict]]) -> Dict[str, List[str]]:
    """
    从用户序列中提取物品集合，按域分类。

    Returns:
        {"all": [...], "X": [...], "Y": [...]}
    """
    items = {"all": set(), "X": set(), "Y": set()}
    for seq in user_sequences.values():
        for item in seq:
            items["all"].add(item["item_id"])
            if item["domain"] in ("Entertainment", "X"):
                items["X"].add(item["item_id"])
            else:
                items["Y"].add(item["item_id"])

    items = {k: sorted(list(v)) for k, v in items.items()}
    logger.info(
        f"物品集合: 全部={len(items['all'])}, "
        f"X域={len(items['X'])}, Y域={len(items['Y'])}"
    )
    return items


# ============================================================
# 数据划分 (leave-one-out)
# ============================================================

def split_leave_one_out(
    user_sequences: Dict[str, List[Dict]],
    seed: int = 2040,
) -> Tuple[Dict, Dict, Dict]:
    """
    Leave-one-out 划分:
    - 最后一个交互 → test
    - 倒数第二个 → valid
    - 其余 → train

    Returns:
        (train_data, valid_data, test_data)
        每个为 {user_id: {"seq": [...], "target": {...}}}
    """
    random.seed(seed)

    train_data = {}
    valid_data = {}
    test_data = {}

    for user_id, seq in user_sequences.items():
        if len(seq) < 3:
            continue

        test_target = seq[-1]
        valid_target = seq[-2]
        train_seq = seq[:-2]

        train_data[user_id] = {
            "seq": train_seq,
        }
        valid_data[user_id] = {
            "seq": train_seq,  # 验证时使用训练序列
            "target": valid_target,
        }
        test_data[user_id] = {
            "seq": seq[:-1],  # 测试时使用含验证的序列
            "target": test_target,
        }

    logger.info(
        f"数据划分: train={len(train_data)}, "
        f"valid={len(valid_data)}, test={len(test_data)}"
    )
    return train_data, valid_data, test_data


# ============================================================
# ID 映射表构建
# ============================================================

def build_id_mapping(
    item_set: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    构建物品 ID 与索引的映射表。

    Returns:
        {
            "item_id_to_index": {item_id: int},
            "index_to_item_id": {int: item_id},
            "domain_x_items": [item_id, ...],
            "domain_y_items": [item_id, ...],
            "num_items": int,
            "num_x_items": int,
            "num_y_items": int,
        }
    """
    all_items = item_set["all"]
    id_to_idx = {item_id: idx for idx, item_id in enumerate(all_items)}
    idx_to_id = {idx: item_id for item_id, idx in id_to_idx.items()}

    mapping = {
        "item_id_to_index": id_to_idx,
        "index_to_item_id": {str(k): v for k, v in idx_to_id.items()},
        "domain_x_items": item_set["X"],
        "domain_y_items": item_set["Y"],
        "num_items": len(all_items),
        "num_x_items": len(item_set["X"]),
        "num_y_items": len(item_set["Y"]),
    }
    return mapping


# ============================================================
# 主流程
# ============================================================

def preprocess(
    review_x_path: Optional[str] = None,
    review_y_path: Optional[str] = None,
    meta_x_path: Optional[str] = None,
    meta_y_path: Optional[str] = None,
    config: Optional[Dict] = None,
):
    """
    执行完整的预处理流程。

    Args:
        review_x_path: X域交互数据路径
        review_y_path: Y域交互数据路径
        meta_x_path: X域物品元数据路径
        meta_y_path: Y域物品元数据路径
        config: 预处理配置
    """
    ensure_dirs()
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # 1. 解析原始数据
    logger.info("=" * 60)
    logger.info("Step 1: 解析原始交互数据")
    logger.info("=" * 60)

    if review_x_path and review_y_path:
        interactions_x = parse_amazon_reviews(review_x_path, cfg["domain_x_name"])
        interactions_y = parse_amazon_reviews(review_y_path, cfg["domain_y_name"])
    else:
        logger.warning("未提供原始交互数据路径，跳过解析。请将数据放入 data/raw/ 目录")
        return

    # 2. 解析元数据
    logger.info("=" * 60)
    logger.info("Step 2: 解析物品元数据")
    logger.info("=" * 60)

    item_meta = {}
    if meta_x_path:
        meta_x = parse_amazon_metadata(meta_x_path)
        for item_id, meta in meta_x.items():
            meta["domain"] = cfg["domain_x_name"]
            item_meta[item_id] = meta
    if meta_y_path:
        meta_y = parse_amazon_metadata(meta_y_path)
        for item_id, meta in meta_y.items():
            meta["domain"] = cfg["domain_y_name"]
            item_meta[item_id] = meta

    save_json(item_meta, PROCESSED_DIR / "item_metadata.json")

    # 3. 构建用户序列
    logger.info("=" * 60)
    logger.info("Step 3: 构建用户交互序列")
    logger.info("=" * 60)

    user_sequences = build_user_sequences(
        interactions_x, interactions_y,
        min_interactions=cfg["min_interactions"],
        max_seq_len=cfg["max_seq_len"],
    )

    # 保存交互序列 (简化格式)
    interactions_simple = {
        uid: [item["item_id"] for item in seq]
        for uid, seq in user_sequences.items()
    }
    save_json(interactions_simple, PROCESSED_DIR / "interactions.json")

    # 4. 提取物品集合 + 构建 ID 映射
    logger.info("=" * 60)
    logger.info("Step 4: 构建 ID 映射表")
    logger.info("=" * 60)

    item_set = extract_item_set(user_sequences)
    id_mapping = build_id_mapping(item_set)
    save_json(id_mapping, PROCESSED_DIR / "id_mapping.json")

    # 5. 数据划分
    logger.info("=" * 60)
    logger.info("Step 5: Leave-one-out 数据划分")
    logger.info("=" * 60)

    train_data, valid_data, test_data = split_leave_one_out(
        user_sequences, seed=cfg["random_seed"]
    )
    save_json(train_data, PROCESSED_DIR / "train.json")
    save_json(valid_data, PROCESSED_DIR / "valid.json")
    save_json(test_data, PROCESSED_DIR / "test.json")

    # 6. 输出统计
    logger.info("=" * 60)
    logger.info("预处理完成！统计信息:")
    logger.info(f"  用户数: {len(user_sequences)}")
    logger.info(f"  物品总数: {id_mapping['num_items']}")
    logger.info(f"  X域物品: {id_mapping['num_x_items']}")
    logger.info(f"  Y域物品: {id_mapping['num_y_items']}")
    logger.info(f"  训练集: {len(train_data)} 用户")
    logger.info(f"  验证集: {len(valid_data)} 用户")
    logger.info(f"  测试集: {len(test_data)} 用户")
    logger.info(f"  产出目录: {PROCESSED_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess()
