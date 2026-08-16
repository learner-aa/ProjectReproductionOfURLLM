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
    OUTPUT_DIR, PROCESSED_DIR, DATASET_SUFFIX,
    load_json, save_json,
    load_dg_scores, load_dg_candidates, load_dg_config,
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
) -> Dict[str, float]:
    """
    从 DG 模型的评分矩阵计算基线指标。

    Args:
        test_data: 测试集数据
        id_mapping: ID 映射表
        k_values: K 值列表

    Returns:
        指标字典
    """
    try:
        scores = load_dg_scores()
    except FileNotFoundError:
        logger.warning("DG 评分矩阵不存在，跳过基线评估")
        return {}

    item_to_idx = id_mapping.get("item_id_to_index", {})
    user_ids = list(test_data.keys())

    metrics = {}
    for k in k_values:
        metrics[f"HR@{k}"] = 0.0
        metrics[f"NDCG@{k}"] = 0.0
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
    item_metadata: Dict[str, Dict],
) -> Dict[str, float]:
    """
    统计 LLM 生成的域外推荐比例。

    Args:
        predictions: LLM 预测的物品标题
        target_domains: 每个样本的目标域
        item_metadata: 物品元数据 (含 domain 信息)

    Returns:
        {"ood_rate": float, "ood_count": int, "total": int}
    """
    # 构建标题 → 域的映射
    title_to_domain = {}
    for item_id, meta in item_metadata.items():
        title = meta.get("title", "").lower().strip()
        if title:
            title_to_domain[title] = meta.get("domain", "Unknown")

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
# User Hit Rate (UHR) — 论文 Table 6
# ============================================================

def compute_uhr(
    retrieval_results: Dict[str, Dict[str, List[str]]],
    test_data: Dict[str, Dict],
    interactions: Dict[str, List[str]],
    top_k_list: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """
    计算 User Hit Rate (UHR): 检索到的相似用户中有多少
    在训练集中与目标物品有过交互。

    论文 Table 6 中用 UHR 衡量检索质量:
    UHR@k = (命中目标物品的检索用户数) / (总检索用户数)

    Args:
        retrieval_results: {"test": {user_id: [retrieved_user_ids]}}
        test_data: {user_id: {"target": {"item_id": str}}}
        interactions: {user_id: [item_id, ...]}
        top_k_list: 计算的 k 值列表

    Returns:
        {"UHR@1": float, "UHR@3": float, "UHR@5": float}
    """
    test_retrieval = retrieval_results.get("test", {})
    metrics = {}

    for k in top_k_list:
        hits = 0
        total = 0

        for user_id, data in test_data.items():
            retrieved = test_retrieval.get(user_id, [])[:k]
            if not retrieved:
                continue

            target_item = data.get("target", {}).get("item_id", "")
            if not target_item:
                continue

            total += 1

            # 检查检索用户是否交互过目标物品
            for r_uid in retrieved:
                r_items = interactions.get(r_uid, [])
                if target_item in r_items:
                    hits += 1
                    break

        metrics[f"UHR@{k}"] = hits / total if total > 0 else 0.0

    metrics["evaluated_users"] = total if top_k_list else 0
    return metrics


# ============================================================
# 主评估流程
# ============================================================

def run_evaluation(config: Optional[Dict] = None):
    """
    执行完整评估流程。

    Args:
        config: 配置字典
    """
    logger.info("=" * 60)
    logger.info("开始评估")
    logger.info("=" * 60)

    # 加载推理结果
    pred_file = OUTPUT_DIR / "predictions" / f"test_predictions{DATASET_SUFFIX}.json"
    if not pred_file.exists():
        logger.error(f"推理结果不存在: {pred_file}")
        return

    results = load_json(pred_file)

    # 加载辅助数据
    interactions = {}
    inter_path = PROCESSED_DIR / "interactions.json"
    if inter_path.exists():
        interactions = load_json(inter_path)

    item_metadata = {}
    meta_path = PROCESSED_DIR / "item_metadata.json"
    if meta_path.exists():
        item_metadata = load_json(meta_path)

    id_mapping = {}
    map_path = PROCESSED_DIR / "id_mapping.json"
    if map_path.exists():
        id_mapping = load_json(map_path)

    test_data = {}
    test_path = PROCESSED_DIR / "test.json"
    if test_path.exists():
        test_data = load_json(test_path)

    # 1. 精确匹配评估
    predictions = [r["prediction"] for r in results]
    targets = [r["ground_truth"] for r in results]
    exact_metrics = compute_all_metrics(predictions, targets)
    logger.info("精确匹配指标:")
    for k, v in exact_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 2. 模糊匹配评估
    fuzzy_metrics = compute_fuzzy_metrics(predictions, targets)
    logger.info("模糊匹配指标:")
    for k, v in fuzzy_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 3. 冷/热启动分析
    if interactions:
        cold_warm = analyze_cold_warm_start(results, interactions)
        logger.info("冷启动指标:")
        for k, v in cold_warm["cold"].items():
            logger.info(f"  cold_{k}: {v:.4f}")
        logger.info("热启动指标:")
        for k, v in cold_warm["warm"].items():
            logger.info(f"  warm_{k}: {v:.4f}")
        logger.info(f"  冷启动用户: {cold_warm['stats']['cold_users']}")
        logger.info(f"  热启动用户: {cold_warm['stats']['warm_users']}")

    # 4. 域外率
    if item_metadata:
        target_domains = [r.get("target_domain", "") for r in results]
        ood = compute_out_of_domain_rate(predictions, target_domains, item_metadata)
        logger.info(f"域外率: {ood['ood_rate']:.2%} ({ood['ood_count']:.0f}/{ood['total']})")

    # 5. DG 基线对比
    dg_metrics = {}
    if test_data and id_mapping:
        dg_metrics = evaluate_dg_baseline(test_data, id_mapping)
        if dg_metrics:
            logger.info("DG 基线指标:")
            for k, v in dg_metrics.items():
                if isinstance(v, float):
                    logger.info(f"  DG_{k}: {v:.4f}")

    # 6. Refined predictions 评估 (Answer Refinement 后)
    refined_metrics = {}
    refined_fuzzy = {}
    expanded_metrics = {}
    refined_file = OUTPUT_DIR / "refined_predictions" / f"refined_predictions{DATASET_SUFFIX}.json"
    if refined_file.exists():
        refined_results = load_json(refined_file)
        refined_preds = [r["prediction"] for r in refined_results]
        refined_targets = [r["ground_truth"] for r in refined_results]
        refined_metrics = compute_all_metrics(refined_preds, refined_targets)
        refined_fuzzy = compute_fuzzy_metrics(refined_preds, refined_targets)
        logger.info("精炼后 (Refined) 精确匹配指标:")
        for k, v in refined_metrics.items():
            logger.info(f"  Refined_{k}: {v:.4f}")
        logger.info("精炼后 (Refined) 模糊匹配指标:")
        for k, v in refined_fuzzy.items():
            logger.info(f"  Refined_{k}: {v:.4f}")

        # 6.5 Top-K 候选评估 (使用 refine_answers 的 BM25 top-20 候选)
        if any("top_k_candidates" in r for r in refined_results):
            topk_preds = []
            for r in refined_results:
                candidates = r.get("top_k_candidates", [])
                if candidates:
                    topk_preds.append(candidates)
                else:
                    topk_preds.append([r["prediction"]])
            expanded_metrics = compute_all_metrics(topk_preds, refined_targets)
            logger.info("Top-K 候选评估指标 (BM25 top-20):")
            for k, v in expanded_metrics.items():
                logger.info(f"  TopK_{k}: {v:.4f}")
        else:
            logger.warning("refined_predictions 中无 top_k_candidates, 跳过 top-K 评估")

        # 精炼后域外率
        if item_metadata:
            refined_domains = [r.get("target_domain", "") for r in refined_results]
            refined_ood = compute_out_of_domain_rate(
                refined_preds, refined_domains, item_metadata
            )
            logger.info(
                f"精炼后域外率: {refined_ood['ood_rate']:.2%} "
                f"({refined_ood['ood_count']:.0f}/{refined_ood['total']})"
            )
        else:
            refined_ood = {}
    else:
        refined_ood = {}

    # 7. UHR (User Hit Rate)
    uhr_metrics = {}
    retrieval_path = PROCESSED_DIR / "retrieval_results.json"
    if retrieval_path.exists() and test_data and interactions:
        retrieval_results = load_json(retrieval_path)
        uhr_metrics = compute_uhr(retrieval_results, test_data, interactions)
        logger.info("User Hit Rate (检索质量):")
        for k, v in uhr_metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

    # 保存评估结果
    eval_result = {
        "exact_metrics": exact_metrics,
        "fuzzy_metrics": fuzzy_metrics,
    }
    if interactions:
        eval_result["cold_warm"] = cold_warm
    if item_metadata:
        eval_result["out_of_domain"] = ood
    if dg_metrics:
        eval_result["dg_baseline"] = dg_metrics
    if refined_metrics:
        eval_result["refined_metrics"] = refined_metrics
        eval_result["refined_fuzzy"] = refined_fuzzy
        if refined_ood:
            eval_result["refined_ood"] = refined_ood
    if expanded_metrics:
        eval_result["expanded_metrics"] = expanded_metrics
    if uhr_metrics:
        eval_result["uhr"] = uhr_metrics

    eval_file = str(OUTPUT_DIR / "eval_results" / f"evaluation{DATASET_SUFFIX}.json")
    save_json(eval_result, eval_file)
    logger.info(f"评估结果已保存: {eval_file}")

    # 8. 对比表
    logger.info("=" * 60)
    logger.info("对比表:")
    logger.info(f"{'方法':<25} {'HR@1':>8} {'HR@5':>8} {'HR@10':>8} {'MRR':>8}")
    logger.info(
        f"{'LLM (原始)':<25} "
        f"{exact_metrics.get('HR@1', 0):>8.4f} "
        f"{exact_metrics.get('HR@5', 0):>8.4f} "
        f"{exact_metrics.get('HR@10', 0):>8.4f} "
        f"{exact_metrics.get('MRR', 0):>8.4f}"
    )
    if refined_metrics:
        logger.info(
            f"{'LLM+Refinement':<25} "
            f"{refined_metrics.get('HR@1', 0):>8.4f} "
            f"{refined_metrics.get('HR@5', 0):>8.4f} "
            f"{refined_metrics.get('HR@10', 0):>8.4f} "
            f"{refined_metrics.get('MRR', 0):>8.4f}"
        )
    if dg_metrics:
        logger.info(
            f"{'DG基线':<25} "
            f"{dg_metrics.get('HR@1', 0):>8.4f} "
            f"{dg_metrics.get('HR@5', 0):>8.4f} "
            f"{dg_metrics.get('HR@10', 0):>8.4f} "
            f"{dg_metrics.get('MRR', 0):>8.4f}"
        )
    logger.info("=" * 60)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_evaluation()
