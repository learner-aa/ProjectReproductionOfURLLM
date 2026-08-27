"""
Full-item-set top-K 评估 (论文口径)

对已有 predictions JSON 计算全物品集 ranking 指标:
- BM25 将 LLM 输出 ranking 到全物品集 → HR@1/5/10/20 + NDCG@K + MRR
  (target 恒在候选集, 与论文"全物品集作候选避免采样偏差"一致)
- DG 候选矩阵 (load_dg_candidates, 2000×10000) baseline → HR@K + MRR

用法:
  python eval_fullset.py --pred-file outputs/AO/predictions/test_predictions.json --tag raw_8_25
  python eval_fullset.py --pred-file outputs/AO/refined_predictions/refined_predictions.json --tag refined_8_25 --no-dg
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import (
    get_output_dir,
    load_dg_candidates,
    load_id_mapping,
    load_item_metadata,
    load_json,
    save_json,
    set_dataset,
)

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class TitleRanker:
    """BM25 对全物品集标题做 ranking (向量化稀疏打分, 避免 rank_bm25 的纯 Python 慢循环)"""

    def __init__(self, item_metadata: Dict[str, Dict]):
        self.item_ids: List[str] = []
        self.titles: List[str] = []
        for iid, meta in item_metadata.items():
            t = meta.get("title", "").strip()
            if t:
                self.item_ids.append(iid)
                self.titles.append(t)
        self.n = len(self.item_ids)

        # 预计算: token -> (doc_indices, weight) 稀疏结构 (BM25)
        k1, b = 1.5, 0.75
        doc_lens = np.array([len(tokenize(t)) for t in self.titles], dtype=np.float64)
        avgdl = doc_lens.mean() + 1e-8
        df: Dict[str, int] = {}
        token_docs: Dict[str, List[Tuple[int, float]]] = {}
        for i, t in enumerate(self.titles):
            seen = {}
            for tok in tokenize(t):
                seen[tok] = seen.get(tok, 0) + 1
            for tok, tf in seen.items():
                df[tok] = df.get(tok, 0) + 1
                token_docs.setdefault(tok, []).append((i, tf))
        self._tok_index: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for tok, entries in token_docs.items():
            idx = np.array([e[0] for e in entries], dtype=np.int64)
            tf = np.array([e[1] for e in entries], dtype=np.float64)
            idf = np.log((self.n - df[tok] + 0.5) / (df[tok] + 0.5) + 1.0)
            dl = doc_lens[idx]
            w = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            self._tok_index[tok] = (idx, w)

    def scores(self, pred: str) -> np.ndarray:
        q = [tok for tok in tokenize(pred) if tok in self._tok_index]
        if not q:
            return np.zeros(self.n)
        out = np.zeros(self.n, dtype=np.float64)
        for tok in q:
            idx, w = self._tok_index[tok]
            np.add.at(out, idx, w)
        return out

    def rank_of(self, pred: str, target_idx: int) -> int:
        """返回 target 的 1-based rank (rank>=1; 无匹配/空 pred 给 n+1)"""
        if target_idx < 0 or target_idx >= self.n:
            return self.n + 1
        s = self.scores(pred)
        if not s.any():
            return self.n + 1
        order = np.argsort(-s, kind="stable")
        pos = np.where(order == target_idx)[0]
        return int(pos[0]) + 1 if len(pos) else self.n + 1


def _metrics_from_ranks(ranks: List[int], n: int, k_values=(1, 5, 10, 20)) -> Dict:
    metrics = {}
    for k in k_values:
        hits = sum(1 for r in ranks if r <= k)
        ndcg = sum(1.0 / np.log2(r + 1) for r in ranks if r <= k)
        metrics[f"HR@{k}"] = hits / n if n else 0.0
        metrics[f"NDCG@{k}"] = ndcg / n if n else 0.0
    metrics["MRR"] = (sum(1.0 / r for r in ranks) / n) if n else 0.0
    return metrics


def eval_fullset(
    pred_file: str,
    tag: str = "fullset",
    with_dg: bool = True,
    dg_root: str = None,
) -> Dict:
    results = load_json(pred_file)
    item_metadata = load_item_metadata()
    id_mapping = load_id_mapping()
    item_to_idx = id_mapping.get("item_id_to_index", {})

    ranker = TitleRanker(item_metadata)
    title_to_idx: Dict[str, int] = {}
    for iid, meta in item_metadata.items():
        t = meta.get("title", "").strip().lower()
        if t and t not in title_to_idx:
            title_to_idx[t] = int(item_to_idx.get(iid, -1))

    dg_candidates = None
    if with_dg:
        try:
            dg_candidates = load_dg_candidates()
        except Exception as e:
            logger.warning(f"DG 候选矩阵不可用: {e}")

    ranks, dg_ranks = [], []
    ranks_ids = []
    miss_rank, dg_miss_rank = [], []
    n_target_resolved = 0

    for r in results:
        target_id = str(r.get("target_item_id", ""))
        target_idx = int(item_to_idx.get(target_id, -1)) if target_id else -1
        if target_idx < 0:
            gt = str(r.get("ground_truth", "")).strip().lower()
            target_idx = title_to_idx.get(gt, -1)
        if target_idx < 0:
            ranks_ids.append(ranker.n + 1)
            miss_rank.append(ranker.n + 1)
            if dg_candidates is not None:
                dg_miss_rank.append(dg_candidates.shape[1] + 1)
            continue
        n_target_resolved += 1

        pred = r.get("prediction", "")

        if isinstance(pred, list):
            # 输出已是排序列表 (论文 I) → target 的 rank = 在列表中的位置
            # 注意: 列表长度可能 < K (如精确匹配短路只返回 1 项)。
            # miss 必须给大 rank (ranker.n+1), 否则 len+1<=K 会被误计为命中。
            rank = ranker.n + 1
            for i, p in enumerate(pred):
                if title_to_idx.get(str(p).strip().lower(), -1) == target_idx:
                    rank = i + 1
                    break
            ranks.append(rank)

            # ID 口径 (论文: I₂ 是 item ID 列表, 按 ID 匹配, 无标题歧义/大小写问题)
            pred_ids = r.get("prediction_ids", pred)
            rank_id = ranker.n + 1
            for i, p in enumerate(pred_ids):
                if item_to_idx.get(str(p)) == target_idx:
                    rank_id = i + 1
                    break
            ranks_ids.append(rank_id)
        else:
            ranks.append(ranker.rank_of(str(pred), target_idx))
            ranks_ids.append(ranker.n + 1)

        if dg_candidates is not None:
            dgi = r.get("dg_index")
            row = dg_candidates[dgi] if (dgi is not None and dgi < dg_candidates.shape[0]) else None
            if row is None:
                dg_miss_rank.append(dg_candidates.shape[1] + 1)
            else:
                pos = np.where(row == target_idx)[0]
                dg_ranks.append(int(pos[0]) + 1 if len(pos) else dg_candidates.shape[1] + 1)

    # 未解析 target 的样本算 miss (rank = n+1)
    all_ranks = ranks + miss_rank
    llm_metrics = _metrics_from_ranks(all_ranks, len(results))
    all_ranks_ids = ranks_ids + miss_rank
    llm_metrics_ids = _metrics_from_ranks(all_ranks_ids, len(results))

    out = {
        "tag": tag,
        "pred_file": str(pred_file),
        "n_samples": len(results),
        "n_target_resolved": n_target_resolved,
        "n_unresolved_target": len(miss_rank),
        "llm_bm25_fullset": llm_metrics,
        "llm_bm25_fullset_ids": llm_metrics_ids,
    }
    if dg_candidates is not None:
        dg_all = dg_ranks + dg_miss_rank
        out["dg_candidate_baseline"] = _metrics_from_ranks(dg_all, len(results))

    # 打印
    def fmt(m):
        return "  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items())

    print("=" * 70)
    print(f"[{tag}] pred_file={pred_file}")
    print(f"  n={len(results)} (target 可解析 {n_target_resolved}, 缺失 {len(miss_rank)})")
    print(f"  LLM(BM25 全物品集, title 口径):")
    print(fmt(out["llm_bm25_fullset"]))
    print(f"  LLM(BM25 全物品集, ID 口径):")
    print(fmt(out["llm_bm25_fullset_ids"]))
    if "dg_candidate_baseline" in out:
        print(f"  DG(候选矩阵 top-10000):")
        print(fmt(out["dg_candidate_baseline"]))
    print("=" * 70)

    out_path = get_output_dir() / "eval_results" / f"fullset_{tag}.json"
    save_json(out, out_path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", required=True)
    ap.add_argument("--tag", default="fullset")
    ap.add_argument("--no-dg", action="store_true")
    ap.add_argument("--dg-root", default="/root/autodl-tmp/URLLM-project/DG_Final")
    args = ap.parse_args()
    set_dataset("AO", args.dg_root)
    eval_fullset(args.pred_file, args.tag, with_dg=not args.no_dg, dg_root=args.dg_root)
