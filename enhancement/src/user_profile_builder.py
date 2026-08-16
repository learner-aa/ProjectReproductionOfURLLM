"""
用户画像构建模块 (核心)

从三个维度构建 LLM 可理解的用户画像:
1. 行为画像: 交互频次、域偏好、序列模式
2. 语义画像: 聚合物品属性标签，提取高频偏好主题
3. 特征画像: DG 656维向量 → 近邻物品描述文本化

画像构建复用已有 DG 模型产出的特征向量。
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    PROCESSED_DIR,
    load_dg_features,
    load_dg_scores,
    load_json,
    save_json,
    cosine_similarity,
    find_topk_similar,
)
from prompt_templates import (
    PROFILE_TEMPLATE_COMPACT,
    PROFILE_TEMPLATE_DETAILED,
    format_interaction_sequence,
    format_attributes,
)

logger = logging.getLogger(__name__)


# ============================================================
# 行为画像
# ============================================================

def build_behavior_profile(
    user_seq: List[Dict],
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
) -> Dict[str, Any]:
    """
    从交互序列构建行为画像。

    Args:
        user_seq: [{"item_id": str, "domain": str}, ...]
        domain_x_name: X 域名称
        domain_y_name: Y 域名称

    Returns:
        行为画像字典
    """
    if not user_seq:
        return {
            "total_interactions": 0,
            "domain_x_count": 0,
            "domain_y_count": 0,
            "cross_domain_ratio": 0.0,
            "recent_items": [],
            "sequence_text": "",
        }

    x_items = [i for i in user_seq if i.get("domain") in (domain_x_name, "X")]
    y_items = [i for i in user_seq if i.get("domain") in (domain_y_name, "Y")]
    total = len(user_seq)

    # 跨域比率 (衡量用户在两个域的活跃均衡度)
    if total > 0:
        cross_ratio = 1.0 - abs(len(x_items) - len(y_items)) / total
    else:
        cross_ratio = 0.0

    return {
        "total_interactions": total,
        "domain_x_count": len(x_items),
        "domain_y_count": len(y_items),
        "cross_domain_ratio": cross_ratio,
        "recent_items": [i["item_id"] for i in user_seq[-5:]],
        "all_item_ids": [i["item_id"] for i in user_seq],
    }


# ============================================================
# 语义画像
# ============================================================

def build_semantic_profile(
    user_seq: List[Dict],
    item_attributes: Dict[str, Dict],
    item_metadata: Dict[str, Dict],
    top_k_attributes: int = 10,
    top_k_categories: int = 5,
) -> Dict[str, Any]:
    """
    从交互物品的属性标签构建语义画像。

    Args:
        user_seq: 用户交互序列
        item_attributes: {item_id: {"attributes": [...]}}
        item_metadata: {item_id: {"title": str, "category": list}}
        top_k_attributes: 保留的 top 属性数
        top_k_categories: 保留的 top 类别数

    Returns:
        语义画像字典
    """
    all_attrs = []
    all_categories = []

    for item in user_seq:
        item_id = item["item_id"] if isinstance(item, dict) else item

        # 聚合属性
        attrs = item_attributes.get(item_id, {}).get("attributes", [])
        all_attrs.extend(attrs)

        # 聚合类别
        cats = item_metadata.get(item_id, {}).get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        all_categories.extend(cats)

    # 统计频次
    attr_counter = Counter(all_attrs)
    cat_counter = Counter(all_categories)

    return {
        "preferred_attributes": attr_counter.most_common(top_k_attributes),
        "preferred_categories": cat_counter.most_common(top_k_categories),
        "attribute_diversity": len(attr_counter),
        "category_diversity": len(cat_counter),
    }


# ============================================================
# 特征画像 (DG 向量文本化)
# ============================================================

def build_embedding_profile(
    user_fea_x: np.ndarray,
    user_fea_y: np.ndarray,
    item_fea_x: np.ndarray,
    item_fea_y: np.ndarray,
    id_mapping: Dict[str, Any],
    item_metadata: Dict[str, Dict],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    将 DG 656维用户特征向量转化为文本描述。

    策略: 用用户向量与物品向量做余弦相似度，取 top-k 近邻物品的标题。

    Args:
        user_fea_x: (656,) X域用户特征
        user_fea_y: (656,) Y域用户特征
        item_fea_x: (num_items, 656) X域物品特征
        item_fea_y: (num_items, 656) Y域物品特征
        id_mapping: ID 映射表
        item_metadata: 物品元数据
        top_k: 近邻物品数

    Returns:
        特征画像字典
    """
    idx_to_id = id_mapping.get("index_to_item_id", {})

    # X 域近邻物品
    top_x_idx, top_x_scores = find_topk_similar(user_fea_x, item_fea_x, k=top_k)
    top_x_items = []
    for idx, score in zip(top_x_idx, top_x_scores):
        item_id = idx_to_id.get(str(idx), str(idx))
        title = item_metadata.get(item_id, {}).get("title", item_id)
        top_x_items.append({"item_id": item_id, "title": title, "score": float(score)})

    # Y 域近邻物品
    top_y_idx, top_y_scores = find_topk_similar(user_fea_y, item_fea_y, k=top_k)
    top_y_items = []
    for idx, score in zip(top_y_idx, top_y_scores):
        item_id = idx_to_id.get(str(idx), str(idx))
        title = item_metadata.get(item_id, {}).get("title", item_id)
        top_y_items.append({"item_id": item_id, "title": title, "score": float(score)})

    # 向量统计摘要
    def vector_stats(fea: np.ndarray) -> Dict:
        return {
            "mean": float(fea.mean()),
            "std": float(fea.std()),
            "max": float(fea.max()),
            "min": float(fea.min()),
            "l2_norm": float(np.linalg.norm(fea)),
        }

    return {
        "top_similar_items_x": top_x_items,
        "top_similar_items_y": top_y_items,
        "embedding_stats_x": vector_stats(user_fea_x),
        "embedding_stats_y": vector_stats(user_fea_y),
    }


# ============================================================
# 画像文本化 (供 LLM 输入)
# ============================================================

def profile_to_text(
    profile: Dict[str, Any],
    item_metadata: Dict[str, Dict],
    template: str = "detailed",
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
) -> str:
    """
    将用户画像字典转化为 LLM 可读的文本。

    Args:
        profile: 完整画像字典 (含 behavior, semantic, embedding)
        item_metadata: 物品元数据
        template: "compact" 或 "detailed"
        domain_x_name: X 域名称
        domain_y_name: Y 域名称

    Returns:
        画像文本
    """
    behavior = profile.get("behavior", {})
    semantic = profile.get("semantic", {})
    embedding = profile.get("embedding", {})

    # 近期物品标题化
    recent_ids = behavior.get("recent_items", [])
    recent_titles = [
        item_metadata.get(iid, {}).get("title", iid) for iid in recent_ids
    ]

    if template == "compact":
        return PROFILE_TEMPLATE_COMPACT.format(
            total_count=behavior.get("total_interactions", 0),
            domain_x_name=domain_x_name,
            domain_y_name=domain_y_name,
            x_count=behavior.get("domain_x_count", 0),
            y_count=behavior.get("domain_y_count", 0),
            recent_items=", ".join(recent_titles[-5:]),
        )

    # detailed 模板
    top_similar_x = embedding.get("top_similar_items_x", [])
    top_similar_y = embedding.get("top_similar_items_y", [])
    similar_items_text = ", ".join(
        [i["title"] for i in top_similar_x[:3]] +
        [i["title"] for i in top_similar_y[:3]]
    )

    # 交互序列文本化
    all_ids = behavior.get("all_item_ids", [])
    seq_lines = []
    for iid in all_ids[-10:]:
        title = item_metadata.get(iid, {}).get("title", iid)
        domain = item_metadata.get(iid, {}).get("domain", "?")
        seq_lines.append(f"  [{domain}] {title}")
    recent_seq = "\n".join(seq_lines) if seq_lines else "  (empty)"

    return PROFILE_TEMPLATE_DETAILED.format(
        total_count=behavior.get("total_interactions", 0),
        domain_x_name=domain_x_name,
        domain_y_name=domain_y_name,
        x_count=behavior.get("domain_x_count", 0),
        y_count=behavior.get("domain_y_count", 0),
        cross_ratio=behavior.get("cross_domain_ratio", 0.0),
        top_attributes=format_attributes(semantic.get("preferred_attributes", [])),
        top_categories=format_attributes(semantic.get("preferred_categories", [])),
        similar_items=similar_items_text or "N/A",
        recent_sequence=recent_seq,
    )


# ============================================================
# 主流程: 批量构建用户画像
# ============================================================

def build_all_user_profiles(
    interactions: Dict[str, List[str]],
    item_metadata: Dict[str, Dict],
    item_attributes: Dict[str, Dict],
    id_mapping: Dict[str, Any],
    dg_features: Optional[Dict[str, np.ndarray]] = None,
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
    top_k_similar: int = 5,
) -> Dict[str, Dict]:
    """
    为所有用户构建完整画像。

    Args:
        interactions: {user_id: [item_id, ...]}
        item_metadata: {item_id: {"title": str, "domain": str, ...}}
        item_attributes: {item_id: {"attributes": [...]}}
        id_mapping: ID 映射表
        dg_features: DG 模型特征 (可选)
        domain_x_name: X 域名称
        domain_y_name: Y 域名称
        top_k_similar: 近邻物品数

    Returns:
        {user_id: {
            "behavior": {...},
            "semantic": {...},
            "embedding": {...},
            "profile_text": str,
        }}
    """
    profiles = {}
    total = len(interactions)

    for idx, (user_id, item_ids) in enumerate(interactions.items()):
        # 构建结构化序列
        user_seq = []
        for iid in item_ids:
            meta = item_metadata.get(iid, {})
            user_seq.append({
                "item_id": iid,
                "domain": meta.get("domain", "Unknown"),
                "title": meta.get("title", iid),
            })

        # 1. 行为画像
        behavior = build_behavior_profile(user_seq, domain_x_name, domain_y_name)

        # 2. 语义画像
        semantic = build_semantic_profile(
            user_seq, item_attributes, item_metadata
        )

        # 3. 特征画像 (如果有 DG 特征)
        embedding = {}
        if dg_features is not None:
            # 注意: DG 特征按索引对应，需要确认 user_id 与索引的映射
            # 这里假设 test 用户按顺序对应
            pass  # 在 run_pipeline 中处理

        profile = {
            "behavior": behavior,
            "semantic": semantic,
            "embedding": embedding,
            "user_id": user_id,
        }

        # 4. 文本化
        profile["profile_text"] = profile_to_text(
            profile, item_metadata,
            template="detailed",
            domain_x_name=domain_x_name,
            domain_y_name=domain_y_name,
        )

        profiles[user_id] = profile

        if (idx + 1) % 500 == 0:
            logger.info(f"画像构建进度: {idx + 1}/{total}")

    logger.info(f"画像构建完成: {len(profiles)} 个用户")
    return profiles


def build_embedding_profiles_for_test_users(
    dg_features: Dict[str, np.ndarray],
    id_mapping: Dict[str, Any],
    item_metadata: Dict[str, Dict],
    top_k: int = 5,
) -> Dict[int, Dict]:
    """
    为测试集用户构建特征画像 (基于 DG 向量)。

    测试用户的 DG 特征存储在 test_x_fea / test_y_fea 中，
    按行索引对应用户。

    Args:
        dg_features: load_dg_features() 的返回值
        id_mapping: ID 映射表
        item_metadata: 物品元数据
        top_k: 近邻数

    Returns:
        {test_user_index: embedding_profile_dict}
    """
    test_x = dg_features["test_x_fea"]  # (num_test_users, 656)
    test_y = dg_features["test_y_fea"]
    item_x = dg_features["train_x_fea"]  # (num_items, 656)
    item_y = dg_features["train_y_fea"]

    num_test_users = test_x.shape[0]
    logger.info(f"为 {num_test_users} 个测试用户构建特征画像...")

    embedding_profiles = {}
    for i in range(num_test_users):
        ep = build_embedding_profile(
            user_fea_x=test_x[i],
            user_fea_y=test_y[i],
            item_fea_x=item_x,
            item_fea_y=item_y,
            id_mapping=id_mapping,
            item_metadata=item_metadata,
            top_k=top_k,
        )
        embedding_profiles[i] = ep

    logger.info(f"特征画像构建完成: {len(embedding_profiles)} 个测试用户")
    return embedding_profiles


# ============================================================
# 入口
# ============================================================

def run_profile_building(config: Dict):
    """
    运行完整的画像构建流程。

    Args:
        config: pipeline 配置字典
    """
    logger.info("=" * 60)
    logger.info("开始构建用户画像")
    logger.info("=" * 60)

    # 加载数据
    interactions = load_json(PROCESSED_DIR / "interactions.json")
    item_metadata = load_json(PROCESSED_DIR / "item_metadata.json")
    id_mapping = load_json(PROCESSED_DIR / "id_mapping.json")

    # 物品属性 (可选，如未提取则用空字典)
    attr_path = PROCESSED_DIR / "item_attributes.json"
    if attr_path.exists():
        item_attributes = load_json(attr_path)
    else:
        logger.warning("物品属性文件不存在，语义画像将为空")
        item_attributes = {}

    # DG 特征 (可选)
    dg_features = None
    try:
        dg_features = load_dg_features()
    except FileNotFoundError:
        logger.warning("DG 特征文件不存在，特征画像将为空")

    # 构建全量用户画像
    domain_cfg = config.get("domains", {})
    profiles = build_all_user_profiles(
        interactions=interactions,
        item_metadata=item_metadata,
        item_attributes=item_attributes,
        id_mapping=id_mapping,
        dg_features=dg_features,
        domain_x_name=domain_cfg.get("x", "Entertainment"),
        domain_y_name=domain_cfg.get("y", "Education"),
    )

    # 保存用户画像
    save_json(profiles, PROCESSED_DIR / "user_profiles.json")

    # 为测试用户构建特征画像
    if dg_features is not None:
        test_embedding_profiles = build_embedding_profiles_for_test_users(
            dg_features=dg_features,
            id_mapping=id_mapping,
            item_metadata=item_metadata,
        )
        save_json(test_embedding_profiles, PROCESSED_DIR / "test_embedding_profiles.json")

    logger.info("=" * 60)
    logger.info("用户画像构建完成！")
    logger.info(f"  画像文件: {PROCESSED_DIR / 'user_profiles.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        run_profile_building(cfg)
    else:
        logger.info(f"使用默认配置运行")
        run_profile_building({})
