"""
Answer Refinement 模块 (论文 §4.2.3, Algorithm 1)

LLM 生成的推荐文本 → BM25 grounding 映射到真实物品空间
→ 域检查 → 域外则回退到 DG 模型推荐结果 I₁

Algorithm 1 (论文):
    I2 = LLM(Ur, Su)
    max_I = max{I2[0:m]}, min_I = min{I2[0:m]}
    I = I2
    if max_I > X_id in recommend domain A: I = I1
    if min_I > Y_id in recommend domain B: I = I1
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    DATASET_SUFFIX,
    load_dg_scores,
    load_dg_candidates,
    load_id_mapping,
    load_item_metadata,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)


# ============================================================
# BM25 Grounding
# ============================================================

class BM25Grounding:
    """
    基于 BM25 (或 token overlap 降级) 将 LLM 输出映射到真实物品。

    优先使用 rank_bm25 库；若未安装，自动降级为 token overlap。
    """

    def __init__(
        self,
        item_metadata: Dict[str, Dict],
        use_bm25: bool = True,
    ):
        """
        Args:
            item_metadata: {item_id: {"title": str, ...}}
            use_bm25: 是否尝试使用 rank_bm25 库
        """
        self.item_metadata = item_metadata

        # 构建标题列表和 item_id 列表
        self.item_ids = []
        self.titles = []
        self.title_lower = []
        self._title_to_item_id = {}

        for item_id, meta in item_metadata.items():
            title = meta.get("title", "").strip()
            if title:
                self.item_ids.append(item_id)
                self.titles.append(title)
                self.title_lower.append(title.lower())
                self._title_to_item_id[title.lower()] = item_id

        # 尝试构建 BM25 索引
        self._bm25 = None
        if use_bm25:
            try:
                from rank_bm25 import BM25Okapi
                corpus = [self._tokenize(t) for t in self.titles]
                self._bm25 = BM25Okapi(corpus)
                logger.info(
                    f"BM25 索引构建完成: {len(self.titles)} 物品 "
                    f"(rank_bm25 库)"
                )
            except ImportError:
                logger.warning(
                    "rank_bm25 未安装，降级为 token overlap 匹配。"
                    "安装: pip install rank_bm25"
                )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词: 小写 + 按非字母数字切分"""
        return re.findall(r'[a-z0-9]+', text.lower())

    def ground(
        self,
        llm_output: str,
        top_m: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        将 LLM 输出文本映射到最匹配的真实物品。

        Args:
            llm_output: LLM 生成的推荐文本
            top_m: 返回 top-m 候选

        Returns:
            [{"item_id": str, "title": str, "score": float}, ...]
            按分数降序排列
        """
        if not llm_output.strip():
            return []

        # BM25 或 token overlap 检索 top-m 候选
        query_tokens = self._tokenize(llm_output)
        if not query_tokens:
            return []

        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
        else:
            scores = self._token_overlap_scores(query_tokens)

        # Top-m
        top_indices = np.argsort(scores)[::-1][:top_m]
        results = []
        for idx in top_indices:
            results.append({
                "item_id": self.item_ids[idx],
                "title": self.titles[idx],
                "score": float(scores[idx]),
            })

        # 精确匹配加分: 若 LLM 输出精确匹配某标题,将其置顶并给最高分
        output_lower = llm_output.strip().lower()
        if output_lower in self._title_to_item_id:
            exact_id = self._title_to_item_id[output_lower]
            exact_title = self.item_metadata[exact_id].get("title", "")
            # 检查精确匹配项是否已在结果中
            found = False
            for r in results:
                if r["item_id"] == exact_id:
                    r["score"] = max(r["score"], 1.0)
                    found = True
                    break
            if not found:
                results.insert(0, {
                    "item_id": exact_id,
                    "title": exact_title,
                    "score": 1.0,
                })
                results = results[:top_m]  # 保持 top_m 数量

        return results

    def _token_overlap_scores(self, query_tokens: List[str]) -> np.ndarray:
        """Token overlap 降级方案"""
        query_set = set(query_tokens)
        scores = np.zeros(len(self.title_lower))
        for i, title_tokens in enumerate(
            [self._tokenize(t) for t in self.titles]
        ):
            title_set = set(title_tokens)
            if title_set:
                overlap = len(query_set & title_set)
                scores[i] = overlap / max(len(query_set), 1)
        return scores


# ============================================================
# 域检查
# ============================================================

def check_domain(
    grounded_items: List[Dict[str, Any]],
    target_domain: str,
    item_metadata: Dict[str, Dict],
    domain_x_name: str = "Entertainment",
    domain_y_name: str = "Education",
) -> bool:
    """
    检查 grounding 结果是否属于目标域。

    论文逻辑: 如果 top-m 中任一物品属于错误域，视为域外。

    Args:
        grounded_items: BM25 grounding 结果列表
        target_domain: 期望的目标域
        item_metadata: 物品元数据
        domain_x_name: X 域名称
        domain_y_name: Y 域名称

    Returns:
        True = 域内, False = 域外
    """
    if not grounded_items:
        return False

    # 统一域名比较
    # 注意: AO 数据集标签混用, Art=Entertainment, Office=Education
    # 与 knn_retriever.py 的 _resolve_dg_domain 保持一致
    def normalize_domain(d: str) -> str:
        d = d.lower().strip()
        if d in (domain_x_name.lower(), "x", "art", "entertainment"):
            return "X"
        if d in (domain_y_name.lower(), "y", "office", "education"):
            return "Y"
        return d

    target_norm = normalize_domain(target_domain)

    for item in grounded_items:
        item_id = item.get("item_id", "")
        meta = item_metadata.get(item_id, {})
        item_domain = normalize_domain(meta.get("domain", ""))

        if item_domain and item_domain != target_norm:
            return False

    return True


# ============================================================
# DG 模型回退
# ============================================================

def dg_fallback(
    test_user_index: int,
    dg_scores: Optional[np.ndarray] = None,
    id_mapping: Optional[Dict[str, Any]] = None,
    item_metadata: Optional[Dict[str, Dict]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    从 DG 模型评分矩阵获取推荐结果作为回退 (论文中的 I₁)。

    返回 top-K 候选 (用于评估 HR@K),top-1 作为最终推荐。

    Args:
        test_user_index: 测试用户索引
        dg_scores: (num_test_users, num_items) 评分矩阵
        id_mapping: ID 映射表
        item_metadata: 物品元数据
        top_k: 返回的候选数量

    Returns:
        {"item_id": str, "title": str, "score": float,
         "top_k_titles": [str, ...], "top_k_ids": [str, ...]}
    """
    if dg_scores is None:
        dg_scores = load_dg_scores()
    if id_mapping is None:
        id_mapping = load_id_mapping()
    if item_metadata is None:
        item_metadata = load_item_metadata()

    if test_user_index >= dg_scores.shape[0]:
        logger.warning(
            f"test_user_index {test_user_index} 超出评分矩阵范围 "
            f"({dg_scores.shape[0]})"
        )
        return {"item_id": "", "title": "", "score": 0.0,
                "top_k_titles": [], "top_k_ids": []}

    user_scores = dg_scores[test_user_index]
    # Top-K 候选
    top_k_indices = np.argsort(user_scores)[::-1][:top_k]

    idx_to_id = id_mapping.get("index_to_item_id", {})
    top_k_titles = []
    top_k_ids = []
    for idx in top_k_indices:
        iid = idx_to_id.get(str(int(idx)), str(int(idx)))
        title = item_metadata.get(iid, {}).get("title", iid)
        top_k_titles.append(title)
        top_k_ids.append(iid)

    # top-1 作为最终推荐
    top_idx = int(top_k_indices[0])
    item_id = idx_to_id.get(str(top_idx), str(top_idx))
    title = item_metadata.get(item_id, {}).get("title", item_id)

    return {
        "item_id": item_id,
        "title": title,
        "score": float(user_scores[top_idx]),
        "top_k_titles": top_k_titles,
        "top_k_ids": top_k_ids,
    }


# ============================================================
# 主编排函数
# ============================================================

def refine_predictions(
    config: Optional[Dict] = None,
) -> List[Dict]:
    """
    对 LLM 推理结果执行 Answer Refinement:
    1. BM25 grounding → 真实物品
    2. 域检查
    3. 域外则 DG 回退

    Args:
        config: 配置字典，可包含:
            - refinement.top_m: BM25 候选数 (默认 5)
            - refinement.use_bm25: 是否使用 rank_bm25 (默认 True)
            - refinement.fallback_enabled: 是否启用 DG 回退 (默认 True)

    Returns:
        精炼后的预测结果列表
    """
    logger.info("=" * 60)
    logger.info("开始 Answer Refinement")
    logger.info("=" * 60)

    cfg = config or {}
    ref_cfg = cfg.get("refinement", {})
    domain_cfg = cfg.get("domains", {})
    domain_x = domain_cfg.get("x", "Entertainment")
    domain_y = domain_cfg.get("y", "Education")
    top_m = ref_cfg.get("top_m", 5)
    use_bm25 = ref_cfg.get("use_bm25", True)
    fallback_enabled = ref_cfg.get("fallback_enabled", True)

    # 加载数据
    pred_file = OUTPUT_DIR / "predictions" / f"test_predictions{DATASET_SUFFIX}.json"
    if not pred_file.exists():
        logger.error(f"推理结果不存在: {pred_file}")
        return []

    results = load_json(pred_file)
    item_metadata = load_item_metadata()
    id_mapping = load_id_mapping()

    # 构建 BM25 grounding
    grounder = BM25Grounding(item_metadata, use_bm25=use_bm25)

    # 加载 DG 评分矩阵 (回退用)
    dg_scores = None
    if fallback_enabled:
        try:
            dg_scores = load_dg_scores()
            logger.info(f"DG 评分矩阵已加载: {dg_scores.shape}")
        except FileNotFoundError:
            logger.warning("DG 评分矩阵不存在，域外回退将跳过")
            fallback_enabled = False

    # 加载 AO DG 候选矩阵 (用于 OOD 样本的 top-K)
    dg_candidates = None
    if DATASET_SUFFIX == "_AO":
        cand_path = OUTPUT_DIR.parent / "t4_G2_final_DGresult_test_candidate_AO.npy"
        if cand_path.exists():
            dg_candidates = np.load(str(cand_path))
            # 去重: 原论文每用户2行,取偶数行
            if dg_candidates.shape[0] % 2 == 0:
                dg_candidates = dg_candidates[0::2]
            logger.info(f"AO DG 候选矩阵已加载: {dg_candidates.shape} (去重后)")
        else:
            logger.warning(f"AO DG 候选矩阵不存在: {cand_path}")

    # 逐条精炼
    refined_results = []
    stats = {"total": 0, "grounded": 0, "in_domain": 0, "fallback": 0}

    for idx, result in enumerate(results):
        llm_output = result.get("prediction", "")
        target_domain = result.get("target_domain", domain_x)
        stats["total"] += 1

        new_result = dict(result)  # 复制原始结果

        # Step 1: BM25 grounding (top_m=20 以保存 top-K 候选)
        grounded = grounder.ground(llm_output, top_m=max(top_m, 20))

        if grounded:
            best = grounded[0]
            stats["grounded"] += 1

            # 保存 top-20 候选标题列表 (用于评估 HR@K)
            new_result["top_k_candidates"] = [g["title"] for g in grounded[:20]]

            # Step 2: 域检查
            in_domain = check_domain(
                grounded, target_domain, item_metadata,
                domain_x, domain_y,
            )

            if in_domain:
                # 域内: 使用 BM25 结果
                new_result["prediction_original"] = llm_output
                new_result["prediction"] = best["title"]
                new_result["grounded_item_id"] = best["item_id"]
                new_result["grounded_score"] = best["score"]
                new_result["refined"] = True
                stats["in_domain"] += 1
            elif fallback_enabled:
                # 域外: DG 回退
                dg_result = dg_fallback(
                    idx, dg_scores, id_mapping, item_metadata, top_k=20
                )
                new_result["prediction_original"] = llm_output
                new_result["prediction"] = dg_result["title"]
                new_result["grounded_item_id"] = dg_result["item_id"]
                new_result["fallback_to_dg"] = True
                new_result["refined"] = True
                # 对 AO: 用 DG 候选矩阵的 top-K 替换 BM25 候选
                # (BM25 候选全跨域, DG 候选包含正确域物品)
                if dg_candidates is not None and idx < dg_candidates.shape[0]:
                    # 从前 1000 个候选中过滤出目标域物品,取 top-20
                    dg_cand_ids = dg_candidates[idx][:1000]
                    target_norm_local = target_domain.lower().strip()
                    dg_cand_titles = []
                    for cid in dg_cand_ids:
                        cid_str = str(int(cid))
                        meta = item_metadata.get(cid_str, {})
                        d = meta.get("domain", "").lower()
                        # Art=Entertainment, Office=Education
                        if target_norm_local in ("art", "entertainment") and d == "entertainment":
                            dg_cand_titles.append(meta.get("title", cid_str))
                        elif target_norm_local in ("office", "education") and d == "education":
                            dg_cand_titles.append(meta.get("title", cid_str))
                        if len(dg_cand_titles) >= 20:
                            break
                    if dg_cand_titles:
                        new_result["top_k_candidates"] = dg_cand_titles
                stats["fallback"] += 1
            else:
                new_result["refined"] = False
        else:
            # grounding 失败: 保持原始 LLM 输出
            new_result["refined"] = False

        refined_results.append(new_result)

    # 保存
    output_file = OUTPUT_DIR / "refined_predictions" / f"refined_predictions{DATASET_SUFFIX}.json"
    save_json(refined_results, output_file)

    # 统计
    logger.info("=" * 60)
    logger.info("Answer Refinement 完成:")
    logger.info(f"  总样本: {stats['total']}")
    logger.info(f"  BM25 匹配成功: {stats['grounded']}")
    logger.info(f"  域内保留: {stats['in_domain']}")
    logger.info(f"  DG 回退: {stats['fallback']}")
    logger.info(f"  未精炼: {stats['total'] - stats['in_domain'] - stats['fallback']}")
    logger.info(f"  结果保存: {output_file}")
    logger.info("=" * 60)

    return refined_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import yaml
    import os

    # 根据 DATASET_SUFFIX 加载对应配置文件
    suffix = os.environ.get("DATASET_SUFFIX", "")
    config_name = f"pipeline_config{suffix}.yaml"
    config_path = Path(__file__).parent.parent / "config" / config_name
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        logger.info(f"已加载配置: {config_path}")
    else:
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
    refine_predictions(cfg)
