"""
前端数据生成脚本
从 enhancement pipeline 产物生成 webapp 所需的数据文件

支持 GM / AO 双数据集，从 outputs/{GM,AO}/ 和 data/processed/{GM,AO}/ 读取真实产物。
生成:
  - src/data/eval_data.json      Dashboard 页面评估指标
  - src/data/datasets.json       Workbench 页面演示用户
  - src/data/training_logs.json  训练曲线
  - src/data/dataset_stats.json  数据集统计 (legacy)
  - src/data/real_data.json      legacy
"""
import json
import re
import os
from pathlib import Path
from collections import Counter
import numpy as np

ENH = Path("/root/autodl-tmp/URLLM-project/enhancement")
WEBAPP = Path("/root/autodl-tmp/URLLM-project/webapp")
DATA_DIR = WEBAPP / "public" / "data"
SRC_DATA_DIR = WEBAPP / "src" / "data"  # 供前端直接 import

# DG 用户相似度矩阵路径 (原始论文产物)
DG_BASE = Path("/root/autodl-tmp/ours/跨域推荐搭建")

# 域 -> 前端展示类型
DOMAIN_KIND = {
    "Entertainment": "movie",
    "Education": "game",
    "Game": "game",
    "Movie": "movie",
    "Arts": "art",
    "Office": "office",
    "Art": "art",
}

# 数据集配置: GM (Movie->Game) / AO (Office->Art)
DATASETS = {
    "GM": {
        "label": "影视 · 游戏",
        "source_domain": "Movie",
        "target_domain": "Game",
        "processed_dir": ENH / "data/processed/GM",
        "outputs_dir": ENH / "outputs/GM",
        "dg_matrix": DG_BASE / "Movie-Game" / "best_trte_XORY_DG_.npy",
    },
    "AO": {
        "label": "办公 · 艺术",
        "source_domain": "Office",
        "target_domain": "Art",
        "processed_dir": ENH / "data/processed/AO",
        "outputs_dir": ENH / "outputs/AO",
        "dg_matrix": DG_BASE / "Art-Office" / "best_trte_XORY_DG_390_.npy",
    },
}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, p):
    os.makedirs(Path(p).parent, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {p}")


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def cosine_attrs(dict_a, dict_b):
    """基于偏好属性权重的余弦相似度, 区分相同属性名但不同权重的用户"""
    if not dict_a or not dict_b:
        return 0.0
    common = dict_a.keys() & dict_b.keys()
    if not common:
        return 0.0
    dot = sum(dict_a[k] * dict_b[k] for k in common)
    norm_a = sum(v * v for v in dict_a.values()) ** 0.5
    norm_b = sum(v * v for v in dict_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# 解析用户交互历史 (支持 GM / AO 两种格式)
# ============================================================
def parse_history(input_text, dataset_key):
    """从 instruction input 解析用户自身的交互历史"""
    history = []
    if dataset_key == "GM":
        # GM 格式: "The user has played the following video games before: Movie:title | Game:title | ... | Please recommend"
        m = re.search(
            r"The user has played the following [^:]*before:\s*(.+?)(?:Please recommend|$)",
            input_text,
            re.DOTALL,
        )
        if not m:
            return history
        body = m.group(1)
        for item in body.split(" | "):
            item = item.strip()
            if not item:
                continue
            pm = re.match(r"(Movie|Game):\s*(.+)", item)
            if pm:
                domain, title = pm.group(1), pm.group(2).strip()
                kind = "movie" if domain == "Movie" else "game"
                history.append({"title": title, "kind": kind, "domain": domain})
    else:
        # AO 格式: "=== Target User Interaction History ===\n  [Art] title\n  [Office] title"
        in_history = False
        for line in input_text.split("\n"):
            if "Target User Interaction History" in line:
                in_history = True
                continue
            if in_history:
                if line.strip().startswith("=== ") or "Please recommend" in line:
                    break
                m = re.match(r"\s*\[(\w+)\]\s*(.+)", line)
                if m:
                    domain, title = m.group(1), m.group(2).strip()
                    kind = DOMAIN_KIND.get(domain, "office")
                    history.append({"title": title, "kind": kind, "domain": domain})
    return history


# ============================================================
# 1. training_logs.json
# ============================================================
def _build_log_from_trainer_state(ts_path, default_steps, desc):
    """从 trainer_state.json 构建训练日志 (单数据集)。"""
    if not ts_path.exists():
        return None
    ts = load_json(ts_path)
    steps = []
    for entry in ts["log_history"]:
        if "loss" in entry:
            # 训练损失条目
            step = {
                "step": entry["step"],
                "loss": round(entry["loss"], 4),
                "learning_rate": round(entry.get("learning_rate", 0), 8),
                "epoch": round(entry["epoch"], 4),
            }
            if "eval_loss" in entry:
                step["eval_loss"] = round(entry["eval_loss"], 4)
            steps.append(step)
        elif "eval_loss" in entry:
            # 验证损失条目 (HuggingFace Trainer 单独记录, 不含 train loss)
            steps.append({
                "step": entry["step"],
                "epoch": round(entry.get("epoch", 0), 4),
                "eval_loss": round(entry["eval_loss"], 4),
            })
    best_metric = ts.get("best_metric")
    best_step = ts.get("best_global_step")
    if best_metric is None:
        for entry in reversed(ts["log_history"]):
            if "eval_loss" in entry:
                best_metric = entry["eval_loss"]
                best_step = entry.get("step", ts.get("global_step", default_steps))
                break
    if best_metric is None:
        best_metric = 0.0
    if best_step is None:
        best_step = ts.get("global_step", default_steps)
    total_steps = ts.get("global_step", default_steps)
    final_loss = steps[-1]["loss"] if steps else 0.0
    return {
        "steps": steps,
        "best_step": best_step,
        "best_eval_loss": round(best_metric, 4),
        "total_steps": total_steps,
        "final_loss": final_loss,
        "description": desc,
    }


def gen_training_logs():
    print("[1] training_logs.json")
    # 保留现有 GM 训练日志 (checkpoint 已归档不在仓库)
    existing = {}
    if (SRC_DATA_DIR / "training_logs.json").exists():
        existing = load_json(SRC_DATA_DIR / "training_logs.json")

    out = {}
    # GM: checkpoint-19735 trainer_state 已不在新目录结构内，沿用现有真实日志
    if "GM" in existing and existing["GM"].get("steps"):
        out["GM"] = existing["GM"]
        print(f"  GM: 沿用现有训练日志 (steps={existing['GM'].get('total_steps')})")
    else:
        # 兜底: 尝试从已知路径加载
        gm_path = ENH / "outputs/lora_weights/checkpoint-19735/trainer_state.json"
        gm_log = _build_log_from_trainer_state(
            gm_path, 19735,
            "LoRA 微调训练日志 (19735 steps / 5 epochs, 7 modules, final_loss=0.3914)",
        )
        if gm_log:
            out["GM"] = gm_log

    # AO: 从 checkpoint-1500 读取最新训练日志 (3 epochs)
    ao_path = ENH / "outputs/AO/lora_weights/checkpoint-1500/trainer_state.json"
    ao_final = 0.0
    if ao_path.exists():
        ao_ts = load_json(ao_path)
        for entry in reversed(ao_ts.get("log_history", [])):
            if "loss" in entry:
                ao_final = round(entry["loss"], 4)
                break
    ao_log = _build_log_from_trainer_state(
        ao_path, 1500,
        f"LoRA 微调训练日志 (1500 steps / 3 epochs, 7 modules, final_loss={ao_final})",
    )
    if ao_log:
        out["AO"] = ao_log
        print(f"  AO: 加载最新训练日志 (steps={ao_log['total_steps']}, final_loss={ao_log['final_loss']})")
    elif "AO" in existing:
        out["AO"] = existing["AO"]
        print("  AO: 沿用现有训练日志 (新 checkpoint 未找到)")

    save_json(out, DATA_DIR / "training_logs.json")
    save_json(out, SRC_DATA_DIR / "training_logs.json")
    return out


# ============================================================
# 2. dataset_stats.json (legacy)
# ============================================================
def gen_dataset_stats():
    print("[2] dataset_stats.json")
    interactions_gm = load_json(ENH / "data/processed/GM/interactions.json")
    item_meta_gm = load_json(ENH / "data/processed/GM/item_metadata.json")
    item_domain_gm = {str(iid): meta.get("domain", "") for iid, meta in item_meta_gm.items()}

    ent_items, edu_items = set(), set()
    ent_inter, edu_inter = 0, 0
    seq_lens = []
    for uid, items in interactions_gm.items():
        seq_lens.append(len(items))
        for iid in items:
            domain = item_domain_gm.get(str(iid), "")
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
            "num_users": len(interactions_gm),
            "avg_sequence_length": round(avg_seq, 1),
        },
    }
    save_json(out, DATA_DIR / "dataset_stats.json")
    save_json(out, SRC_DATA_DIR / "dataset_stats.json")


# ============================================================
# 3. eval_data.json (Dashboard / Overview 真实评估指标)
# ============================================================
def _build_metrics(eval_result):
    """从 evaluation.json 构建 metrics (使用 refined 后指标, K 值有区分度)"""
    refined = eval_result.get("refined_metrics", {}) or eval_result.get("exact_metrics", {})
    refined_fuzzy = eval_result.get("refined_fuzzy", {}) or eval_result.get("fuzzy_metrics", {})
    refined_ood = eval_result.get("refined_ood", {}) or eval_result.get("out_of_domain", {})
    cw_stats = eval_result.get("cold_warm", {}).get("stats", {})

    return {
        "hr1": round(refined.get("HR@1", 0), 4),
        "hr5": round(refined.get("HR@5", 0), 4),
        "hr10": round(refined.get("HR@10", 0), 4),
        "hr20": round(refined.get("HR@20", 0), 4),
        "ndcg1": round(refined.get("NDCG@1", 0), 4),
        "ndcg5": round(refined.get("NDCG@5", 0), 4),
        "ndcg10": round(refined.get("NDCG@10", 0), 4),
        "ndcg20": round(refined.get("NDCG@20", 0), 4),
        "mrr": round(refined.get("MRR", 0), 4),
        "fuzzyHr1": round(refined_fuzzy.get("fuzzy_HR@1", 0), 4),
        "partialHr1": round(refined_fuzzy.get("partial_HR@1", 0), 4),
        "exactHr1": round(refined_fuzzy.get("exact_HR@1", 0), 4),
        "oodRate": round(refined_ood.get("ood_rate", 0), 4),
        "oodCount": int(round(refined_ood.get("ood_count", 0))),
        "totalUsers": int(refined_ood.get("total", cw_stats.get("total", 0))),
        "coldUsers": int(cw_stats.get("cold_users", 0)),
        "warmUsers": int(cw_stats.get("warm_users", 0)),
    }


def _build_dg_baseline(eval_result):
    dg = eval_result.get("dg_baseline", {})
    return {
        "hr1": round(dg.get("HR@1", 0), 4),
        "hr5": round(dg.get("HR@5", 0), 4),
        "hr10": round(dg.get("HR@10", 0), 4),
        "hr20": round(dg.get("HR@20", 0), 4),
        "mrr": round(dg.get("MRR", 0), 4),
    }


def _build_stats(cfg, processed_dir, test_insts):
    """从 processed 数据构建数据集统计"""
    interactions_path = processed_dir / "interactions.json"
    item_meta_path = processed_dir / "item_metadata.json"
    train_insts_path = processed_dir / "train_instructions.json"

    num_users = 0
    source_items, target_items = set(), set()
    source_inter, target_inter = 0, 0
    avg_seq = 0.0

    if interactions_path.exists() and item_meta_path.exists():
        interactions = load_json(interactions_path)
        item_meta = load_json(item_meta_path)
        item_domain = {str(iid): meta.get("domain", "") for iid, meta in item_meta.items()}
        # 根据数据集确定源域/目标域 (item_metadata 中的 domain 字段值)
        if cfg["source_domain"] == "Movie":
            src_key, tgt_key = "Entertainment", "Education"  # GM: Movie=Entertainment, Game=Education
        else:  # AO: Office->Art (item_metadata 中域名为 "Art" 非 "Arts")
            src_key, tgt_key = "Office", "Art"
        seq_lens = []
        for uid, items in interactions.items():
            seq_lens.append(len(items))
            for iid in items:
                domain = item_domain.get(str(iid), "")
                if domain == src_key:
                    source_items.add(str(iid)); source_inter += 1
                elif domain == tgt_key:
                    target_items.add(str(iid)); target_inter += 1
        num_users = len(interactions)
        avg_seq = round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else 0

    train_instructions = 0
    if train_insts_path.exists():
        train_instructions = len(load_json(train_insts_path))

    return {
        "numUsers": num_users,
        "testUsers": len(test_insts),
        "sourceItems": len(source_items),
        "targetItems": len(target_items),
        "sourceInteractions": source_inter,
        "targetInteractions": target_inter,
        "avgSeqLength": avg_seq,
        "trainInstructions": train_instructions,
    }


def _build_training_info(dataset_key, training_logs):
    """从 training_logs 提取训练摘要"""
    log = training_logs.get(dataset_key, {})
    total_steps = log.get("total_steps", 0)
    final_loss = log.get("final_loss", 0.0)
    best_eval = log.get("best_eval_loss", 0.0)
    # 推断 epochs
    if dataset_key == "GM":
        epochs = 5
    else:
        epochs = 3
    return {
        "totalSteps": total_steps,
        "epochs": epochs,
        "finalLoss": round(final_loss, 4),
        "bestEvalLoss": round(best_eval, 4),
    }


def gen_eval_data(training_logs):
    print("[3] eval_data.json")
    out = {}
    for key, cfg in DATASETS.items():
        eval_path = cfg["outputs_dir"] / "eval_results/evaluation.json"
        test_insts_path = cfg["processed_dir"] / "test_instructions.json"
        if not eval_path.exists():
            print(f"  {key}: 评估结果不存在, 跳过")
            continue
        eval_result = load_json(eval_path)
        test_insts = load_json(test_insts_path) if test_insts_path.exists() else []

        metrics = _build_metrics(eval_result)
        dg = _build_dg_baseline(eval_result)
        stats = _build_stats(cfg, cfg["processed_dir"], test_insts)
        training = _build_training_info(key, training_logs)

        out[key] = {
            "label": cfg["label"],
            "sourceDomain": cfg["source_domain"],
            "targetDomain": cfg["target_domain"],
            "stats": stats,
            "metrics": metrics,
            "dgBaseline": dg,
            "training": training,
        }
        print(f"  {key}: HR@1={metrics['hr1']}, HR@10={metrics['hr10']}, MRR={metrics['mrr']}, "
              f"DG_HR@1={dg['hr1']}, testUsers={stats['testUsers']}, trainInsts={stats['trainInstructions']}")

    save_json(out, SRC_DATA_DIR / "eval_data.json")
    save_json(out, DATA_DIR / "eval_data.json")
    return out


# ============================================================
# 4. datasets.json (Workbench 演示用户, GM + AO)
# ============================================================
def _build_demo_users_for_dataset(key, cfg, max_users=24):
    """为单个数据集挑选演示用户 + 检索池 (含真实用户画像)"""
    test_insts_path = cfg["processed_dir"] / "test_instructions.json"
    preds_path = cfg["outputs_dir"] / "predictions/test_predictions.json"
    profiles_path = cfg["processed_dir"] / "user_profiles.json"
    if not (test_insts_path.exists() and preds_path.exists()):
        print(f"  {key}: 缺少测试指令或预测结果, 跳过")
        return []

    test_insts = load_json(test_insts_path)
    preds = load_json(preds_path)
    # 真实用户画像 (含偏好属性 / 交互总数 / 域分布)
    user_profiles = load_json(profiles_path) if profiles_path.exists() else {}
    # 清理 tokenizer 特殊标记
    for p in preds:
        if p.get("prediction"):
            p["prediction"] = p["prediction"].replace("</s>", "").replace("<s>", "").strip()
    pred_map = {p["user_id"]: p for p in preds}

    def _get_profile(uid):
        """从 user_profiles.json 读取真实画像, 提取前端展示字段。
        AO 测试集 user_id 形如 '8000#X' (KNN 变体), 需取 '#' 前的原始 uid。"""
        raw_uid = str(uid).split("#")[0] if isinstance(uid, str) and "#" in str(uid) else str(uid)
        p = user_profiles.get(raw_uid) or user_profiles.get(uid) or user_profiles.get(str(uid))
        if not p:
            return None
        beh = p.get("behavior", {})
        sem = p.get("semantic", {})
        attrs = sem.get("preferred_attributes", [])
        return {
            "totalInteractions": beh.get("total_interactions", 0),
            "domainXCount": beh.get("domain_x_count", 0),
            "domainYCount": beh.get("domain_y_count", 0),
            "preferredAttributes": [
                {"name": a[0], "weight": a[1]} for a in attrs[:6] if isinstance(a, list) and len(a) >= 2
            ],
        }

    # 构建用户池 (用于挑选演示用户, 取前 300 条有效用户)
    # AO 测试集 user_id 形如 "8000#X" (KNN 变体), 取 '#' 前为原始 uid, 同一原始用户只保留首个变体
    seen_raw_uids = set()
    user_pool = []
    for test_idx, inst in enumerate(test_insts[:300]):
        uid = inst.get("user_id")
        raw_uid = str(uid).split("#")[0] if isinstance(uid, str) and "#" in str(uid) else str(uid)
        if raw_uid in seen_raw_uids:
            continue  # 跳过同一原始用户的其他变体
        pred = pred_map.get(uid)
        if not pred:
            continue
        history = parse_history(inst.get("input", ""), key)
        if len(history) < 2:
            continue
        seen_raw_uids.add(raw_uid)
        item_set = set(h["title"].lower().strip() for h in history)
        profile = _get_profile(uid)
        user_pool.append({
            "id": uid,
            "raw_uid": raw_uid,
            "test_idx": test_idx,  # DG 矩阵行索引
            "history": history,
            "item_set": item_set,
            "domain": inst.get("target_domain", ""),
            "pred": pred.get("prediction", ""),
            "gt": pred.get("ground_truth", ""),
            "profile": profile,
        })
        if len(user_pool) >= 120:
            break

    # 按 target_domain 分组, 均匀挑选演示用户
    domains_present = {}
    for u in user_pool:
        domains_present.setdefault(u["domain"], []).append(u)
    demo_users = []
    half = max_users // 2
    # 取出现次数最多的两个域各取一半
    sorted_domains = sorted(domains_present.keys(), key=lambda d: len(domains_present[d]), reverse=True)
    for d in sorted_domains[:2]:
        demo_users.extend(domains_present[d][:half])
    # 不足则用其他用户补足
    if len(demo_users) < max_users:
        for u in user_pool:
            if u not in demo_users:
                demo_users.append(u)
                if len(demo_users) >= max_users:
                    break

    # 加载 DG 用户相似度矩阵 (原始论文产物)
    # 矩阵行 = test_instructions 顺序, 列 = train_instructions 顺序
    dg_mat = np.load(str(cfg["dg_matrix"])) if cfg.get("dg_matrix") and Path(cfg["dg_matrix"]).exists() else None
    train_insts = load_json(cfg["processed_dir"] / "train_instructions.json") if dg_mat is not None else []
    interactions = load_json(cfg["processed_dir"] / "interactions.json") if dg_mat is not None else {}
    item_metadata = load_json(cfg["processed_dir"] / "item_metadata.json") if dg_mat is not None else {}
    train_col_count = len(train_insts)
    print(f"  {key}: DG 矩阵 {dg_mat.shape if dg_mat is not None else 'N/A'}, "
          f"train_instructions={train_col_count}")

    def _get_train_user_titles(col_idx):
        """矩阵列 -> train_instructions user_id -> interactions 物品标题"""
        if col_idx >= train_col_count:
            return None, None
        uid = train_insts[col_idx].get("user_id", "")
        raw_uid = str(uid).split("#")[0]
        items = interactions.get(str(raw_uid), [])
        titles = []
        for item_id in items[-3:]:  # 最近 3 个交互
            m = item_metadata.get(str(item_id), {})
            titles.append(m.get("title", "?")[:40])
        return raw_uid, titles

    # 为每个演示用户用 DG 矩阵计算相似用户 + 检索池
    demo_data = []
    for du in demo_users:
        if dg_mat is not None and du["test_idx"] < dg_mat.shape[0]:
            row = dg_mat[du["test_idx"]]
            # 取 top-50 列, 过滤超出 train_instructions 范围的列 (GM 矩阵有 35941 列但 train 只有 31570)
            sorted_cols = np.argsort(row)[::-1]
            valid_entries = []
            for c in sorted_cols:
                if c < train_col_count:
                    valid_entries.append((int(c), float(row[c])))
                if len(valid_entries) >= 20:
                    break
            # softmax 转换: 在 top-K 相似用户中计算相对相似度份额
            # top-1 不会是 1.0, 且有真实区分度 (DG 矩阵原始值是负数, 值越大越相似)
            if valid_entries:
                vals = np.array([v for _, v in valid_entries[:15]])
                shifted = vals - vals.max()  # 数值稳定
                exp_vals = np.exp(shifted)
                norm_vals = exp_vals / exp_vals.sum()
            else:
                norm_vals = []
            # 构建检索池 Top-15 + 相似用户 Top-5
            pool = []
            similar_users = []
            for i, ((col, _), nv) in enumerate(zip(valid_entries, norm_vals)):
                raw_uid, titles = _get_train_user_titles(col)
                if not raw_uid or not titles:
                    continue
                sim_val = round(float(nv), 4)
                entry = {
                    "id": f"User#{raw_uid}",
                    "similarity": sim_val,
                    "recentTitles": titles,
                    "selected": i < 5,
                }
                pool.append(entry)
                if i < 5:
                    similar_users.append({
                        "id": entry["id"],
                        "similarity": sim_val,
                        "recentTitles": titles,
                    })
                if len(pool) >= 15:
                    break
        else:
            # Fallback: 用物品 jaccard
            pool = []
            similar_users = []
            for i, other in enumerate(u for u in user_pool if u["raw_uid"] != du["raw_uid"]):
                sim = jaccard(du["item_set"], other["item_set"])
                entry = {
                    "id": f"User#{other['raw_uid']}",
                    "similarity": round(sim, 4),
                    "recentTitles": [h["title"] for h in other["history"][:3]],
                    "selected": i < 5,
                }
                pool.append(entry)
                if i < 5:
                    similar_users.append({"id": entry["id"], "similarity": sim,
                                          "recentTitles": entry["recentTitles"]})
                if len(pool) >= 15:
                    break

        # 真实画像 + 推荐理由 (使用 user_profiles.json 的真实交互总数)
        profile = du.get("profile")
        real_total = profile["totalInteractions"] if profile else len(du["history"])
        reason = f"模型基于用户画像(真实交互{real_total}次)生成推荐。真实标签: {du['gt'][:60]}"
        result_kind = DOMAIN_KIND.get(du["domain"], "movie")
        user_entry = {
            "id": f"User#{du['raw_uid']}",
            "history": [{"title": h["title"], "kind": h["kind"]} for h in du["history"][:8]],
            "result": {
                "title": du["pred"][:80] if du["pred"] else "N/A",
                "kind": result_kind,
                "reason": reason,
                "similarUsers": similar_users,
            },
            "retrievalPool": pool,
        }
        if profile:
            user_entry["profile"] = profile
        demo_data.append(user_entry)
    return demo_data


def gen_datasets():
    print("[4] datasets.json")
    datasets_out = []
    for key, cfg in DATASETS.items():
        users = _build_demo_users_for_dataset(key, cfg)
        datasets_out.append({
            "key": key,  # 'GM' / 'AO' 与 App.tsx 切换逻辑匹配
            "label": cfg["label"],
            "users": users,
        })
        print(f"  {key}: 演示用户 {len(users)} 个")
    save_json(datasets_out, SRC_DATA_DIR / "datasets.json")
    save_json(datasets_out, DATA_DIR / "datasets.json")
    return datasets_out


# ============================================================
# 5. real_data.json (legacy, 保持兼容)
# ============================================================
def gen_real_data(training_logs, eval_data):
    print("[5] real_data.json (legacy)")
    datasets_out = load_json(SRC_DATA_DIR / "datasets.json")
    # 用 GM 数据构造 legacy metricsSnapshots
    gm = eval_data.get("GM", {})
    metrics = gm.get("metrics", {})
    out = {
        "datasets": datasets_out,
        "metricsSnapshots": {
            "movie-game": {
                "hr": metrics.get("hr1", 0),
                "recallAtK": {1: metrics.get("hr1", 0), 5: metrics.get("hr5", 0),
                               10: metrics.get("hr10", 0), 20: metrics.get("hr20", 0)},
                "ndcgAtK": {1: metrics.get("ndcg1", 0), 5: metrics.get("ndcg5", 0),
                             10: metrics.get("ndcg10", 0), 20: metrics.get("ndcg20", 0)},
                "totalUsers": metrics.get("totalUsers", 3601),
                "meanSimilarUserCount": 5.0,
            },
        },
        "extraMetrics": {
            "fuzzy_HR1": metrics.get("fuzzyHr1", 0),
            "partial_HR1": metrics.get("partialHr1", 0),
            "exact_HR1": metrics.get("exactHr1", 0),
            "mrr": metrics.get("mrr", 0),
            "ood_rate": metrics.get("oodRate", 0),
            "ood_count": metrics.get("oodCount", 0),
            "total": metrics.get("totalUsers", 3601),
            "dg_baseline": gm.get("dgBaseline", {}),
            "model_info": "LoRA 微调 (19735 steps / 5 epochs)",
        },
        "trainingLogs": training_logs,
    }
    save_json(out, DATA_DIR / "real_data.json")
    save_json(out, SRC_DATA_DIR / "real_data.json")


if __name__ == "__main__":
    print("=== 生成前端数据 (从 enhancement 真实产物) ===")
    training_logs = gen_training_logs()
    gen_dataset_stats()
    eval_data = gen_eval_data(training_logs)
    gen_datasets()
    gen_real_data(training_logs, eval_data)
    print("\n=== 生成完成 ===")
    print(f"输出目录: {SRC_DATA_DIR}")
