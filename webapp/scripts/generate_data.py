"""
前端数据生成脚本
从 enhancement pipeline 产物生成 webapp 所需的数据文件
"""
import json
import re
import os
from pathlib import Path
from collections import Counter

ENH = Path("/root/autodl-tmp/URLLM-project/enhancement")
WEBAPP = Path("/root/autodl-tmp/URLLM-project/webapp")
DATA_DIR = WEBAPP / "public" / "data"
SRC_DATA_DIR = WEBAPP / "src" / "data"  # 供前端直接 import

DOMAIN_KIND = {"Entertainment": "movie", "Education": "game", "Arts": "art", "Office": "office"}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, p):
    os.makedirs(Path(p).parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {p}")


def parse_history(input_text):
    """从 instruction input 解析用户交互历史"""
    history = []
    in_history = False
    for line in input_text.split("\n"):
        if "Interaction History" in line:
            in_history = True
            continue
        if in_history:
            if line.strip().startswith("=== ") or "Preference Summary" in line:
                break
            m = re.match(r"\s*\[(\w+)\]\s*(.+)", line)
            if m:
                domain, title = m.group(1), m.group(2).strip()
                kind = DOMAIN_KIND.get(domain, "movie")
                history.append({"title": title, "kind": kind, "domain": domain})
    return history


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


# ============================================================
# 1. training_logs.json
# ============================================================
def gen_training_logs():
    print("[1] training_logs.json")
    ts = load_json(ENH / "outputs/lora_weights/checkpoint-19735/trainer_state.json")
    steps = []
    for entry in ts["log_history"]:
        if "loss" in entry:
            step = {
                "step": entry["step"],
                "loss": round(entry["loss"], 4),
                "learning_rate": round(entry.get("learning_rate", 0), 8),
                "epoch": round(entry["epoch"], 4),
            }
            if "eval_loss" in entry:
                step["eval_loss"] = round(entry["eval_loss"], 4)
            steps.append(step)
    # best_metric 可能为 None (未启用 load_best_model_at_end), 从 log_history 中获取最后的 eval_loss
    best_metric = ts.get("best_metric")
    best_step = ts.get("best_global_step")
    if best_metric is None:
        # 从 log_history 中找最后一条 eval 记录
        for entry in reversed(ts["log_history"]):
            if "eval_loss" in entry:
                best_metric = entry["eval_loss"]
                best_step = entry.get("step", ts.get("global_step", 19730))
                break
    if best_metric is None:
        best_metric = 0.0
    if best_step is None:
        best_step = ts.get("global_step", 19730)

    out = {
        "steps": steps,
        "best_step": best_step,
        "best_eval_loss": round(best_metric, 4),
        "total_steps": 19730,
        "description": "LoRA 微调训练日志 (19730 steps / 5 epochs, 7 modules, final_loss=0.3914)",
    }
    save_json(out, DATA_DIR / "training_logs.json")
    save_json(out, SRC_DATA_DIR / "training_logs.json")
    return out


# ============================================================
# 2. dataset_stats.json
# ============================================================
def gen_dataset_stats():
    print("[2] dataset_stats.json")
    interactions = load_json(ENH / "data/processed/interactions.json")
    item_meta = load_json(ENH / "data/processed/item_metadata.json")
    item_domain = {str(iid): meta.get("domain", "") for iid, meta in item_meta.items()}

    num_users = len(interactions)
    ent_items, edu_items = set(), set()
    ent_inter, edu_inter = 0, 0
    seq_lens = []

    for uid, items in interactions.items():
        seq_lens.append(len(items))
        for iid in items:
            domain = item_domain.get(str(iid), "")
            if domain == "Entertainment":
                ent_items.add(str(iid)); ent_inter += 1
            elif domain == "Education":
                edu_items.add(str(iid)); edu_inter += 1

    avg_seq = sum(seq_lens) / len(seq_lens) if seq_lens else 0

    out = {
        "movie_game": {
            "name": "Entertainment-Education (Movie-Game)",
            "source_domain": {"name": "Entertainment", "num_items": len(ent_items), "num_interactions": ent_inter},
            "target_domain": {"name": "Education", "num_items": len(edu_items), "num_interactions": edu_inter},
            "num_users": num_users,
            "avg_sequence_length": round(avg_seq, 1),
        },
    }
    save_json(out, DATA_DIR / "dataset_stats.json")
    save_json(out, SRC_DATA_DIR / "dataset_stats.json")


# ============================================================
# 3. mockData: datasets + metricsSnapshots + extraMetrics
# ============================================================
def gen_mockdata():
    print("[3] mockData (datasets + metricsSnapshots)")
    eval_result = load_json(ENH / "outputs/eval_results/evaluation.json")
    test_insts = load_json(ENH / "data/processed/test_instructions.json")
    preds = load_json(ENH / "outputs/predictions/test_predictions.json")

    # --- 加载用户池 (120 个,用于计算相似度) ---
    # 清理预测结果中的 tokenizer 特殊标记 (</s>, <s>, etc.)
    for p in preds:
        if p.get("prediction"):
            p["prediction"] = p["prediction"].replace("</s>", "").replace("<s>", "").strip()
    pred_map = {p["user_id"]: p for p in preds}
    user_pool = []  # [{id, history, item_set, domain, pred, gt}]
    for inst in test_insts[:160]:
        uid = inst.get("user_id")
        pred = pred_map.get(uid)
        if not pred:
            continue
        history = parse_history(inst.get("input", ""))
        if len(history) < 2:
            continue
        item_set = set(h["title"].lower().strip() for h in history)
        user_pool.append({
            "id": uid,
            "history": history,
            "item_set": item_set,
            "domain": inst.get("target_domain", ""),
            "pred": pred.get("prediction", ""),
            "gt": pred.get("ground_truth", ""),
        })
        if len(user_pool) >= 120:
            break

    # --- 选择 24 个演示用户 (12 Entertainment + 12 Education) ---
    ent_users = [u for u in user_pool if u["domain"] == "Entertainment"][:12]
    edu_users = [u for u in user_pool if u["domain"] == "Education"][:12]
    demo_users = ent_users + edu_users

    # --- 为每个演示用户计算相似用户 + 检索池 ---
    demo_data = []
    for du in demo_users:
        sims = []
        for other in user_pool:
            if other["id"] == du["id"]:
                continue
            sim = jaccard(du["item_set"], other["item_set"])
            if sim > 0:
                sims.append((other, sim))
        sims.sort(key=lambda x: x[1], reverse=True)

        # 检索池 Top-15 (前5个 selected=true),不足则用其他用户补充
        pool = []
        for i, (other, sim) in enumerate(sims[:15]):
            pool.append({
                "id": f"User#{other['id']}",
                "similarity": round(sim, 4),
                "recentTitles": [h["title"] for h in other["history"][:3]],
                "selected": i < 5,
            })
        # 不足时用其他用户填充(即使相似度为 0)
        if len(pool) < 15:
            for other in user_pool:
                if other["id"] != du["id"] and not any(p["id"] == f"User#{other['id']}" for p in pool):
                    pool.append({
                        "id": f"User#{other['id']}",
                        "similarity": 0.0,
                        "recentTitles": [h["title"] for h in other["history"][:3]],
                        "selected": len(pool) < 5,
                    })
                    if len(pool) >= 15:
                        break

        # 相似用户 Top-5 (从 sims 取,不足则从检索池补充)
        similar_users = []
        for other, sim in sims[:5]:
            similar_users.append({
                "id": f"User#{other['id']}",
                "similarity": round(sim, 4),
                "recentTitles": [h["title"] for h in other["history"][:3]],
            })
        # 不足 5 个时,从检索池中补充
        if len(similar_users) < 5:
            for rp_cand in pool:
                if not any(su["id"] == rp_cand["id"] for su in similar_users):
                    similar_users.append({
                        "id": rp_cand["id"],
                        "similarity": rp_cand["similarity"],
                        "recentTitles": rp_cand["recentTitles"],
                    })
                    if len(similar_users) >= 5:
                        break

        reason = f"模型基于用户画像(交互{len(du['history'])}次)生成推荐。真实标签: {du['gt'][:60]}"
        demo_data.append({
            "id": f"User#{du['id']}",
            "history": [{"title": h["title"], "kind": h["kind"]} for h in du["history"][:8]],
            "result": {
                "title": du["pred"][:80] if du["pred"] else "N/A",
                "kind": DOMAIN_KIND.get(du["domain"], "movie"),
                "reason": reason,
                "similarUsers": similar_users,
            },
            "retrievalPool": pool,
        })

    datasets = [{"key": "movie-game", "label": "影视 · 游戏", "users": demo_data}]

    # --- metricsSnapshots ---
    exact = eval_result["exact_metrics"]
    expanded = eval_result.get("expanded_metrics", {})
    fuzzy = eval_result.get("fuzzy_metrics", {})
    ood = eval_result.get("out_of_domain", {})
    dg = eval_result.get("dg_baseline", {})
    # 优先使用 Jaccard 物品相似度扩展指标 (URLLM 论文核心方法)
    # expanded 把单条预测扩展为 top-K 候选, HR@K 有区分度; exact 单候选时所有 K 相同
    src = expanded if expanded else exact
    metrics = {
        "movie-game": {
            "hr": round(src.get("HR@1", 0), 4),
            "recallAtK": {1: round(src.get("HR@1", 0), 4), 5: round(src.get("HR@5", 0), 4),
                          10: round(src.get("HR@10", 0), 4), 20: round(src.get("HR@20", 0), 4)},
            "ndcgAtK": {1: round(src.get("NDCG@1", 0), 4), 5: round(src.get("NDCG@5", 0), 4),
                        10: round(src.get("NDCG@10", 0), 4), 20: round(src.get("NDCG@20", 0), 4)},
            "totalUsers": 3601,
            "meanSimilarUserCount": round(sum(len(d["result"]["similarUsers"]) for d in demo_data) / max(len(demo_data), 1), 1),
        },
    }

    # 额外指标 (Dashboard 展示用)
    extra_metrics = {
        "fuzzy_HR1": round(fuzzy.get("fuzzy_HR@1", 0), 4),
        "partial_HR1": round(fuzzy.get("partial_HR@1", 0), 4),
        "exact_HR1": round(fuzzy.get("exact_HR@1", 0), 4),
        "mrr": round(src.get("MRR", 0), 4),
        "ood_rate": round(ood.get("ood_rate", 0), 4),
        "ood_count": int(ood.get("ood_count", 0)),
        "total": int(ood.get("total", 3601)),
        "dg_baseline": {
            "HR@1": round(dg.get("HR@1", 0), 4),
            "HR@5": round(dg.get("HR@5", 0), 4),
            "HR@10": round(dg.get("HR@10", 0), 4),
            "HR@20": round(dg.get("HR@20", 0), 4),
            "MRR": round(dg.get("MRR", 0), 4),
        },
        "model_info": "LoRA 微调 (19730 steps / 5 epochs)",
    }

    # 加载 training_logs (供 Dashboard 展示训练曲线)
    training_logs = load_json(DATA_DIR / "training_logs.json")

    out = {"datasets": datasets, "metricsSnapshots": metrics, "extraMetrics": extra_metrics, "trainingLogs": training_logs}
    save_json(out, DATA_DIR / "real_data.json")
    save_json(out, SRC_DATA_DIR / "real_data.json")  # 供前端直接 import
    # datasets 单独输出到 src/data,供前端直接 import
    save_json(datasets, SRC_DATA_DIR / "datasets.json")
    print(f"  demo users: {len(demo_data)}")
    print(f"  HR@1={metrics['movie-game']['hr']}, fuzzy_HR@1={extra_metrics['fuzzy_HR1']}")
    return out


if __name__ == "__main__":
    print("=== 生成前端数据(从 enhancement 产物) ===")
    gen_training_logs()
    gen_dataset_stats()
    result = gen_mockdata()
    print("\n=== 生成完成 ===")
