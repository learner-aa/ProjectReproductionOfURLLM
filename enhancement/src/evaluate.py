"""
评估模块

计算推荐系统的标准评估指标:
- Hit Rate (HR@K)
- Normalized Discounted Cumulative Gain (NDCG@K)
- Mean Reciprocal Rank (MRR)

支持:
- LLM 预测结果评估
- 与 DG 基线对比
- 冷/热启动分场景分析
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    OUTPUT_DIR, PROCESSED_DIR,
    load_json, save_json,
    load_dg_scores, load_dg_candidates, load_dg_config,
    get_processed_path, get_dataset_suffix,
)

logger = logging.getLogger(__name__)


# ============================================================
# 指标计算
# ============================================================

def hit_rate_at_k(predictions: List[str], targets: List[str], k: int = 10) -> float:
    """
    计算 HR@K (Hit Rate at K)。

    对于 LLM 推荐，K 通常设为 1 (因为 LLM 只生成一个推荐)。
    如果考虑 BM25 扩展的 top-K，可以调大 K。

    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
        k: 截断位置

    Returns:
        HR@K 值
    """
    hits = 0
    for pred, target in zip(predictions, targets):
        if isinstance(pred, list):
            # top-K 候选
            pred_k = pred[:k]
        else:
            pred_k = [pred]
        if any(p.lower().strip() == target.lower().strip() for p in pred_k):
            hits += 1
    return hits / len(targets) if targets else 0.0


def ndcg_at_k(predictions: List[str], targets: List[str], k: int = 10) -> float:
    """计算 NDCG@K"""
    ndcg_sum = 0.0
    for pred, target in zip(predictions, targets):
        if isinstance(pred, list):
            pred_k = pred[:k]
        else:
            pred_k = [pred]
        for rank, p in enumerate(pred_k, 1):
            if p.lower().strip() == target.lower().strip():
                ndcg_sum += 1.0 / np.log2(rank + 1)
                break
    return ndcg_sum / len(targets) if targets else 0.0


def mean_reciprocal_rank(predictions: List[str], targets: List[str]) -> float:
    """计算 MRR"""
    rr_sum = 0.0
    for pred, target in zip(predictions, targets):
        if isinstance(pred, list):
            for rank, p in enumerate(pred, 1):
                if p.lower().strip() == target.lower().strip():
                    rr_sum += 1.0 / rank
                    break
        else:
            if pred.lower().strip() == target.lower().strip():
                rr_sum += 1.0
    return rr_sum / len(targets) if targets else 0.0


def compute_all_metrics(
    predictions: List[str],
    targets: List[str],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    计算所有评估指标。

    Returns:
        {"HR@1": float, "HR@5": float, ..., "NDCG@1": float, ..., "MRR": float}
    """
    metrics = {}
    for k in k_values:
        metrics[f"HR@{k}"] = hit_rate_at_k(predictions, targets, k)
        metrics[f"NDCG@{k}"] = ndcg_at_k(predictions, targets, k)
    metrics["MRR"] = mean_reciprocal_rank(predictions, targets)
    return metrics


# ============================================================
# 物品属性 Jaccard 相似度扩展 (URLLM 论文核心方法)
# ============================================================

def build_attribute_index(item_attributes: Dict) -> Tuple[Dict, Dict]:
    """
    构建属性倒排索引和物品属性字典。

    Returns:
        item_id_to_attrs: {item_id: set(attributes)}
        attr_to_items: {attribute: set(item_ids)}
    """
    item_id_to_attrs = {}
    attr_to_items = defaultdict(set)
    for item_id, info in item_attributes.items():
        attrs = set(info.get("attributes", []))
        item_id_to_attrs[item_id] = attrs
        for attr in attrs:
            attr_to_items[attr].add(item_id)
    return item_id_to_attrs, attr_to_items


def jaccard_similarity(set1: set, set2: set) -> float:
    """计算 Jaccard 相似度"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def find_item_id_by_title(title: str, title_to_id: Dict) -> Optional[str]:
    """通过标题查找物品 ID（精确匹配 + 大小写不敏感匹配）"""
    if not title:
        return None
    title_clean = title.strip()
    # 精确匹配
    if title_clean in title_to_id:
        return title_to_id[title_clean]
    # 大小写不敏感匹配
    title_lower = title_clean.lower()
    for t, iid in title_to_id.items():
        if t.lower() == title_lower:
            return iid
    return None


def expand_to_topk(
    seed_item_id: str,
    item_id_to_attrs: Dict[str, set],
    attr_to_items: Dict[str, set],
    k: int = 20,
) -> List[str]:
    """
    使用 Jaccard 相似度将种子物品扩展为 top-K 候选列表。

    使用倒排索引加速：只计算与种子物品有共同属性的物品的相似度。

    Args:
        seed_item_id: 种子物品 ID
        item_id_to_attrs: 物品 ID 到属性集合的映射
        attr_to_items: 属性到物品集合的倒排索引
        k: 返回的候选数量

    Returns:
        top-K 相似物品 ID 列表（按相似度降序排列）
    """
    if seed_item_id not in item_id_to_attrs:
        return [seed_item_id] if seed_item_id else []

    seed_attrs = item_id_to_attrs[seed_item_id]
    if not seed_attrs:
        return [seed_item_id]

    # 使用倒排索引找到候选物品（有共同属性的物品）
    candidate_items = set()
    for attr in seed_attrs:
        candidate_items.update(attr_to_items.get(attr, set()))
    candidate_items.discard(seed_item_id)

    # 限制候选数量，避免 OOM（最多 2000 个候选）
    if len(candidate_items) > 2000:
        candidate_items = set(list(candidate_items)[:2000])

    # 计算 Jaccard 相似度并排序
    similarities = []
    for cand_id in candidate_items:
        cand_attrs = item_id_to_attrs.get(cand_id, set())
        sim = jaccard_similarity(seed_attrs, cand_attrs)
        similarities.append((cand_id, sim))

    # 按相似度降序排列，取 top-K
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_ids = [item_id for item_id, _ in similarities[:k]]

    # 种子物品放在第一位
    return [seed_item_id] + top_k_ids


def compute_expanded_metrics(
    predictions: List[str],
    targets: List[str],
    target_item_ids: List[str],
    title_to_id: Dict[str, str],
    item_id_to_attrs: Dict[str, set],
    attr_to_items: Dict[str, set],
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict[str, float]:
    """
    使用物品相似度扩展计算 HR@K, NDCG@K, MRR。

    对于每个预测：
    1. 将预测文本匹配到物品 ID
    2. 使用 Jaccard 相似度扩展为 top-K 候选列表
    3. 检查 target_item_id 是否在 top-K 中
    """
    metrics = {}

    # 为每个预测生成 top-K 候选列表
    expanded_lists = []
    match_count = 0
    import gc
    for i, (pred, target_id) in enumerate(zip(predictions, target_item_ids)):
        seed_id = find_item_id_by_title(pred, title_to_id)
        if seed_id:
            match_count += 1
            topk = expand_to_topk(seed_id, item_id_to_attrs, attr_to_items, k=max(k_values))
        else:
            # 如果无法匹配到物品 ID，只使用预测本身
            topk = [pred]
        expanded_lists.append((topk, target_id))
        if (i + 1) % 500 == 0:
            logger.info(f"  扩展进度: {i+1}/{len(predictions)}")
            gc.collect()

    logger.info(f"预测匹配到物品 ID 的数量: {match_count}/{len(predictions)} ({match_count*100/len(predictions):.1f}%)")

    for k in k_values:
        hits = 0
        ndcg_sum = 0.0
        mrr_sum = 0.0

        for topk, target_id in expanded_lists:
            # 检查 target_id 是否在 top-K 中
            found_rank = None
            for rank, item_id in enumerate(topk[:k], 1):
                if str(item_id) == str(target_id):
                    found_rank = rank
                    break

            if found_rank is not None:
                hits += 1
                ndcg_sum += 1.0 / np.log2(found_rank + 1)
                mrr_sum += 1.0 / found_rank

        n = len(predictions)
        metrics[f"HR@{k}"] = hits / n if n > 0 else 0.0
        metrics[f"NDCG@{k}"] = ndcg_sum / n if n > 0 else 0.0

    metrics["MRR"] = mrr_sum / len(predictions) if predictions else 0.0
    return metrics


# ============================================================
# 模糊匹配评估 (LLM 生成的文本可能不完全匹配)
# ============================================================

def fuzzy_match_score(prediction: str, target: str) -> float:
    """
    模糊匹配评分。

    评分规则:
    - 完全匹配: 1.0
    - 包含关系: 0.5
    - 编辑距离较近: 0.3
    - 不匹配: 0.0
    """
    pred = prediction.lower().strip()
    tgt = target.lower().strip()

    if pred == tgt:
        return 1.0
    if tgt in pred or pred in tgt:
        return 0.5

    # 简单的 token overlap
    pred_tokens = set(pred.split())
    tgt_tokens = set(tgt.split())
    if pred_tokens and tgt_tokens:
        overlap = len(pred_tokens & tgt_tokens) / max(len(pred_tokens), len(tgt_tokens))
        if overlap > 0.5:
            return 0.3

    return 0.0


def compute_fuzzy_metrics(
    predictions: List[str],
    targets: List[str],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    使用模糊匹配计算评估指标。

    Args:
        predictions: LLM 预测文本列表
        targets: 真实物品标题列表
        threshold: 模糊匹配阈值

    Returns:
        指标字典
    """
    fuzzy_hits = sum(
        1 for p, t in zip(predictions, targets)
        if fuzzy_match_score(p, t) >= threshold
    )
    partial_hits = sum(
        fuzzy_match_score(p, t) for p, t in zip(predictions, targets)
    )

    n = len(targets)
    return {
        "fuzzy_HR@1": fuzzy_hits / n if n else 0.0,
        "partial_HR@1": partial_hits / n if n else 0.0,
        "exact_HR@1": sum(1 for p, t in zip(predictions, targets)
                         if p.lower().strip() == t.lower().strip()) / n if n else 0.0,
    }


# ============================================================
# 冷/热启动分析
# ============================================================

def analyze_cold_warm_start(
    results: List[Dict],
    interactions: Dict[str, List],
    cold_threshold: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    按冷/热启动分别评估。

    Args:
        results: 推理结果列表 [{"user_id": str, "prediction": str, "ground_truth": str}]
        interactions: 用户交互序列
        cold_threshold: 冷启动阈值 (交互数 < 此值视为冷启动)

    Returns:
        {"cold": metrics_dict, "warm": metrics_dict, "all": metrics_dict}
    """
    cold_preds, cold_targets = [], []
    warm_preds, warm_targets = [], []

    for r in results:
        user_id = r.get("user_id")
        pred = r.get("prediction", "")
        target = r.get("ground_truth", "")

        seq_len = len(interactions.get(user_id, []))
        if seq_len < cold_threshold:
            cold_preds.append(pred)
            cold_targets.append(target)
        else:
            warm_preds.append(pred)
            warm_targets.append(target)

    return {
        "cold": compute_all_metrics(cold_preds, cold_targets),
        "warm": compute_all_metrics(warm_preds, warm_targets),
        "all": compute_all_metrics(
            [r["prediction"] for r in results],
            [r["ground_truth"] for r in results],
        ),
        "stats": {
            "cold_users": len(cold_preds),
            "warm_users": len(warm_preds),
            "total": len(results),
        },
    }


# ============================================================
# DG 基线评估 (从评分矩阵计算)
# ============================================================

def evaluate_dg_baseline(
    test_data: Dict[str, Dict],
    id_mapping: Dict[str, Any],
    k_values: List[int] = [1, 5, 10, 20],
    dataset: str = "GM",
) -> Dict[str, float]:
    """
    从 DG 模型的评分矩阵计算基线指标。

    Args:
        test_data: 测试集数据
        id_mapping: ID 映射表
        k_values: K 值列表
        dataset: "GM" 或 "AO"

    Returns:
        指标字典
    """
    try:
        scores = load_dg_scores(dataset=dataset)
    except FileNotFoundError:
        logger.warning("DG 评分矩阵不存在，跳过基线评估")
        return {}

    item_to_idx = id_mapping.get("item_id_to_index", {})
    user_ids = list(test_data.keys())

    metrics = {f"HR@{k}": 0.0 for k in k_values}
    metrics.update({f"NDCG@{k}": 0.0 for k in k_values})
    metrics["MRR"] = 0.0
    valid_count = 0

    for ui, user_id in enumerate(user_ids):
        if ui >= scores.shape[0]:
            break
        target = test_data[user_id].get("target", {})
        target_id = target.get("item_id", "")
        target_idx = item_to_idx.get(target_id)

        if target_idx is None:
            continue

        user_scores = scores[ui]
        sorted_items = np.argsort(user_scores)[::-1]

        valid_count += 1
        target_rank = np.where(sorted_items == target_idx)[0]
        if len(target_rank) > 0:
            rank = target_rank[0] + 1
            metrics["MRR"] += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    metrics[f"HR@{k}"] += 1.0
                    metrics[f"NDCG@{k}"] += 1.0 / np.log2(rank + 1)

    if valid_count > 0:
        for key in metrics:
            metrics[key] /= valid_count

    metrics["evaluated_users"] = valid_count
    return metrics


# ============================================================
# 域外率统计
# ============================================================

def compute_out_of_domain_rate(
    predictions: List[str],
    target_domains: List[str],
    title_to_domain: Dict[str, str],
) -> Dict[str, float]:
    """
    统计 LLM 生成的域外推荐比例。

    Args:
        predictions: LLM 预测的物品标题
        target_domains: 每个样本的目标域
        title_to_domain: 预构建的 标题(小写) → 域 映射

    Returns:
        {"ood_rate": float, "ood_count": int, "total": int}
    """
    ood_count = 0
    total = len(predictions)

    for pred, target_domain in zip(predictions, target_domains):
        pred_lower = pred.lower().strip()
        matched_domain = title_to_domain.get(pred_lower)

        if matched_domain and matched_domain != target_domain:
            ood_count += 1
        elif matched_domain is None:
            # 无法匹配到真实物品，视为潜在域外
            ood_count += 0.5  # 半计入

    return {
        "ood_rate": ood_count / total if total else 0.0,
        "ood_count": ood_count,
        "total": total,
    }


# ============================================================
# 主评估流程
# ============================================================

def run_evaluation(config: Optional[Dict] = None, dataset: str = "GM"):
    """
    执行完整评估流程。

    Args:
        config: 配置字典
        dataset: "GM" 或 "AO"
    """
    logger.info("=" * 60)
    logger.info(f"开始评估 (dataset={dataset})")
    logger.info("=" * 60)

    # 加载推理结果
    suffix = get_dataset_suffix(dataset)
    pred_file = OUTPUT_DIR / "predictions" / f"test_predictions{suffix}.json"
    if not pred_file.exists():
        logger.error(f"推理结果不存在: {pred_file}")
        return

    results = load_json(pred_file)

    # 清理预测结果中的 tokenizer 特殊标记 (</s>, <s>, etc.)
    for r in results:
        if r.get("prediction"):
            r["prediction"] = r["prediction"].replace("</s>", "").replace("<s>", "").strip()

    import gc

    # 1. 精确匹配评估 (无需大文件，先算以降低峰值内存)
    predictions = [r["prediction"] for r in results]
    targets = [r["ground_truth"] for r in results]
    exact_metrics = compute_all_metrics(predictions, targets)
    logger.info("精确匹配指标:")
    for k, v in exact_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 2. 模糊匹配评估 (无需大文件)
    fuzzy_metrics = compute_fuzzy_metrics(predictions, targets)
    logger.info("模糊匹配指标:")
    for k, v in fuzzy_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 3. 加载 item_metadata，提取 title_to_id / title_to_domain 后立即释放
    title_to_id = {}
    title_to_domain = {}
    meta_path = get_processed_path("item_metadata.json", dataset)
    if meta_path.exists():
        item_metadata = load_json(meta_path)
        for iid, info in item_metadata.items():
            if iid == "idBefore":
                continue
            title = info.get("title", "")
            if title:
                title_to_id[title] = iid
                title_to_domain[title.lower().strip()] = info.get("domain", "Unknown")
        del item_metadata
        gc.collect()
        logger.info(f"item_metadata 已提取映射 (title_to_id={len(title_to_id)})")

    # 4. 加载 item_attributes，构建属性索引后立即释放
    item_id_to_attrs, attr_to_items = {}, defaultdict(set)
    attr_path = get_processed_path("item_attributes.json", dataset)
    if attr_path.exists():
        item_attributes = load_json(attr_path)
        item_id_to_attrs, attr_to_items = build_attribute_index(item_attributes)
        del item_attributes
        gc.collect()
        logger.info(f"item_attributes 已构建索引 (items={len(item_id_to_attrs)})")

    # 5. 物品相似度扩展评估 (URLLM 论文核心方法)
    expanded_metrics = {}
    if item_id_to_attrs and title_to_id:
        target_item_ids = []
        test_instructions_path = get_processed_path("test_instructions.json", dataset)
        if test_instructions_path.exists():
            test_items = load_json(test_instructions_path)
            user_to_target = {}
            for item in test_items:
                uid = item.get("user_id")
                tid = item.get("target_item_id")
                if uid and tid:
                    user_to_target[uid] = str(tid)
            del test_items
            gc.collect()
            for r in results:
                uid = r.get("user_id")
                target_item_ids.append(user_to_target.get(uid, ""))
        else:
            target_item_ids = [""] * len(results)

        expanded_metrics = compute_expanded_metrics(
            predictions, targets, target_item_ids,
            title_to_id, item_id_to_attrs, attr_to_items,
        )
        logger.info("物品相似度扩展指标:")
        for k, v in expanded_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    # 6. 冷/热启动分析 (加载 interactions，用完即释放)
    cold_warm = None
    inter_path = get_processed_path("interactions.json", dataset)
    if inter_path.exists():
        interactions = load_json(inter_path)
        cold_warm = analyze_cold_warm_start(results, interactions)
        del interactions
        gc.collect()
        logger.info("冷启动指标:")
        for k, v in cold_warm["cold"].items():
            logger.info(f"  cold_{k}: {v:.4f}")
        logger.info("热启动指标:")
        for k, v in cold_warm["warm"].items():
            logger.info(f"  warm_{k}: {v:.4f}")
        logger.info(f"  冷启动用户: {cold_warm['stats']['cold_users']}")
        logger.info(f"  热启动用户: {cold_warm['stats']['warm_users']}")

    # 7. 域外率 (用预构建的 title_to_domain，无需原始 item_metadata)
    ood = None
    if title_to_domain:
        target_domains = [r.get("target_domain", "") for r in results]
        ood = compute_out_of_domain_rate(predictions, target_domains, title_to_domain)
        logger.info(f"域外率: {ood['ood_rate']:.2%} ({ood['ood_count']:.0f}/{ood['total']})")

    # 8. DG 基线对比 (加载 test_data + id_mapping，用完即释放)
    dg_metrics = {}
    test_path = get_processed_path("test.json", dataset)
    map_path = get_processed_path("id_mapping.json", dataset)
    if test_path.exists() and map_path.exists():
        test_data = load_json(test_path)
        id_mapping = load_json(map_path)
        if test_data and id_mapping:
            dg_metrics = evaluate_dg_baseline(test_data, id_mapping, dataset=dataset)
            del test_data, id_mapping
            gc.collect()
            if dg_metrics:
                logger.info("DG 基线指标:")
                for k, v in dg_metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  DG_{k}: {v:.4f}")

    # 保存评估结果
    eval_result = {
        "exact_metrics": exact_metrics,
        "fuzzy_metrics": fuzzy_metrics,
        "expanded_metrics": expanded_metrics,
    }
    if cold_warm:
        eval_result["cold_warm"] = cold_warm
    if ood:
        eval_result["out_of_domain"] = ood
    if dg_metrics:
        eval_result["dg_baseline"] = dg_metrics

    eval_file = str(OUTPUT_DIR / "eval_results" / f"evaluation{suffix}.json")
    save_json(eval_result, eval_file)
    logger.info(f"评估结果已保存: {eval_file}")

    # 6. 对比表
    logger.info("=" * 60)
    logger.info("对比表:")
    logger.info(f"{'方法':<20} {'HR@1':>8} {'HR@5':>8} {'HR@10':>8} {'MRR':>8}")
    logger.info(f"{'LLM+画像':<20} {exact_metrics.get('HR@1', 0):>8.4f} {exact_metrics.get('HR@5', 0):>8.4f} {exact_metrics.get('HR@10', 0):>8.4f} {exact_metrics.get('MRR', 0):>8.4f}")
    if dg_metrics:
        logger.info(f"{'DG基线':<20} {dg_metrics.get('HR@1', 0):>8.4f} {dg_metrics.get('HR@5', 0):>8.4f} {dg_metrics.get('HR@10', 0):>8.4f} {dg_metrics.get('MRR', 0):>8.4f}")
    logger.info("=" * 60)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="评估推荐结果")
    ap.add_argument("--dataset", choices=["GM", "AO"], default="GM", help="数据集 (默认 GM)")
    args = ap.parse_args()
    run_evaluation(dataset=args.dataset)
