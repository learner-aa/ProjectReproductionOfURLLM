"""
DG 模型数据预处理模块 (通用版)

解析 DG_Final 产物, 使 enhance 管线的 ID 空间与 DG 索引完全对齐:

- 物品清单:   {dataset}/item_list*.csv
  * 列: idBefore,item_title,idAfter[,item_attribute]
  * idAfter 即 DG 统一索引空间 (AO: 源 0..18638, 目标 18639..38395; GM: 源 0..71066, 目标 71067..183299)
- 交互序列:   {dataset}/{train,valid,test}_F*.txt, 物品 id 直接使用 DG 索引
  * AO 格式: uid\\thistoryCount\\titem|ts|pos...,  目标 = 最后一个物品
  * GM 格式: uid\\tuser_ts\\titem|ts|ts...,       目标 = ts == user_ts 的物品
- 物品属性:   DG_src/dataset/item_prompt_{AO|GM}/*_exat_*.json (qqid = DG 索引)

产出 (data/processed/):
- item_metadata.json   {dg_index: {title, domain, attribute}}
- item_attributes.json {dg_index: {intro, attributes}}   (从 DG 已产出的 exat 文件读取)
- id_mapping.json      对齐 DG 索引空间
- interactions.json    {user_id: [item_id, ...]}  (不含目标物品)
- train.json / valid.json / test.json
  键为 sample_id, 值含 {user_id, dg_index, seq, target, domain}
"""

import csv
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_utils import (
    get_processed_dir,
    save_json, ensure_dirs,
    get_dataset_dir,
    set_dataset, CURRENT_DATASET, CURRENT_DG_ROOT,
)

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

DEFAULT_CONFIG = {
    "min_interactions": 3,       # 最少交互次数过滤 (用户级别)
    "max_seq_len": 15,           # 最大序列长度 (与 DG 模型一致)
    "random_seed": 2040,         # 随机种子
    "train_file": "train_F2.txt",   # AO 默认; GM 用 train_F.txt
    "valid_file": "valid_F2.txt",   # AO 默认; GM 用 valid_F.txt
    "test_file": "test_F2.txt",     # AO 默认; GM 用 test_F.txt
}

# 各数据集的源域/目标域名称
DOMAIN_NAMES = {
    "AO": ("Art", "Office"),
    "GM": ("Movie", "Game"),
}


# ============================================================
# 物品清单解析 (idAfter = DG 索引)
# ============================================================

def parse_item_lists(
    dataset_name: str,
    domain_x_name: str = "Art",
    domain_y_name: str = "Office",
) -> Tuple[Dict[int, Dict], int, int]:
    """
    解析源/目标域物品 CSV。

    Returns:
        (items, num_source, num_target)
        items: {dg_index: {"title", "domain", "attribute", "id_before"}}
        num_source / num_target: 源域/目标域物品数量 (决定 DG 索引分界)
    """
    if dataset_name == "GM":
        source_csv = get_dataset_dir() / "item_listM_F.csv"
        target_csv = get_dataset_dir() / "item_listG_AM_F.csv"
    else:
        source_csv = get_dataset_dir() / "item_listA_F.csv"
        target_csv = get_dataset_dir() / "item_listO_AA_F.csv"

    items: Dict[int, Dict] = {}
    num_source = 0

    for csv_path, domain in [(source_csv, domain_x_name), (target_csv, domain_y_name)]:
        if not csv_path.exists():
            raise FileNotFoundError(f"物品清单不存在: {csv_path}")
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row["idAfter"])
                except (KeyError, ValueError):
                    continue
                items[idx] = {
                    "title": (row.get("item_title") or "").strip(),
                    "domain": domain,
                    "attribute": (row.get("item_attribute") or "").strip(),
                    "id_before": int(row.get("idBefore", 0) or 0),
                }
                if domain == domain_x_name:
                    num_source += 1

    num_target = len(items) - num_source
    logger.info(
        f"物品清单解析完成: 源域={num_source}, 目标域={num_target}, "
        f"总计={len(items)}, 索引范围=[{min(items)}..{max(items)}]"
    )
    return items, num_source, num_target


# ============================================================
# 交互序列解析 (DG txt)
# ============================================================

def _target_of_a_line(
    parts: List[str],
    target_mode: str,
    n_source: int,
    domain_x_name: str,
    domain_y_name: str,
) -> Tuple[List[int], int, str]:
    """
    从一行交互记录中解析历史序列与目标物品。

    Args:
        parts: 一行按 \\t 切分的字段
        target_mode: "last" (AO, 目标=最后一个物品) 或 "ts" (GM, 目标=ts==user_ts 的物品)
        n_source: 源域物品数 (用于索引分界判断目标域)

    Returns:
        (seq_item_ids, target_item_id, target_domain)
    """
    items = [p.split("|")[0].strip() for p in parts[2:]]
    items = [int(i) for i in items if i]

    if target_mode == "last":
        if len(items) < 2:
            return [], 0, domain_y_name
        target = items[-1]
        seq = items[:-1]
    else:  # ts 模式 (GM): 目标 = ts == 字段[1] 的物品
        user_ts = parts[1]
        target = None
        seq = []
        for p in parts[2:]:
            f = p.split("|")
            if len(f) < 2:
                continue
            if f[1] == user_ts:
                target = int(f[0])
            else:
                seq.append(int(f[0]))
        if target is None:
            # 降级: 取最后一个物品
            target = int(parts[-1].split("|")[0])
            seq = [int(p.split("|")[0]) for p in parts[2:-1]]

    domain = domain_y_name if target >= n_source else domain_x_name
    return seq, target, domain


def parse_split(
    txt_path: Path,
    target_mode: str,
    n_source: int,
    domain_x_name: str,
    domain_y_name: str,
    max_seq_len: int = 15,
) -> Dict[str, Dict]:
    """
    解析单个划分文件 (train/valid/test)。

    键: sample_id
      - AO: 同一用户有 X/Y 两个目标行 → "{uid}#X" / "{uid}#Y"
      - GM: 每用户 1 行 → str(uid)
    值: {user_id, dg_index, seq, target, domain}
      dg_index = 行号, 对应 DG 特征文件 (train/test) 的行索引。

    Returns:
        {sample_id: {...}}
    """
    data: Dict[str, Dict] = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            uid = str(parts[0])
            seq, target_id, target_domain = _target_of_a_line(
                parts, target_mode, n_source, domain_x_name, domain_y_name
            )
            if not seq:
                continue

            # 截断历史 (保留最近 max_seq_len), 物品 id 统一转字符串 (元数据/映射均用 str 键)
            seq = [str(x) for x in seq[-max_seq_len:]]
            target_id = str(target_id)

            # 构建 sample_id
            #   AO: 同一用户有 X/Y 两个目标行 → "{uid}#X" / "{uid}#Y"
            #   GM: 每用户 1 行 → str(uid)
            if target_mode == "last":
                sample_id = f"{uid}#{'Y' if target_domain == domain_y_name else 'X'}"
            else:
                sample_id = uid

            # 防止同一用户同行同域导致键冲突 (AO 理论上不会, 防御性处理)
            if sample_id in data:
                sample_id = f"{sample_id}-{line_idx}"

            data[sample_id] = {
                "user_id": uid,
                "dg_index": line_idx,
                "seq": seq,
                "target": {"item_id": target_id, "domain": target_domain},
                "domain": target_domain,
            }

    logger.info(
        f"解析 {txt_path.name}: {len(data)} 条样本, 目标模式={target_mode}, "
        f"seq长度<= {max_seq_len}"
    )
    return data


# ============================================================
# 物品属性 (DG 已产出的 LLM 提取结果)
# ============================================================

def load_precomputed_attributes() -> Dict[str, Dict]:
    """
    从 DG_src/dataset/item_prompt_{AO|GM}/*_exat_*.json 读取物品属性。

    每个 entry: {"qqid": int, "choices": [{"message": {"content": "[...]"}}]}
    qqid 即 DG 索引。

    Returns:
        {item_id_str: {"intro": str, "attributes": [str]}}
    """
    dir_name = f"item_prompt_{CURRENT_DATASET}"
    search_dirs = [
        Path(CURRENT_DG_ROOT) / "DG_src" / "dataset" / dir_name,
        get_dataset_dir() / dir_name,
    ]

    files = []
    for d in search_dirs:
        files = sorted(glob.glob(str(d / "*_exat_*.json")))
        if files:
            break

    attributes: Dict[str, Dict] = {}
    if not files:
        logger.warning(f"未找到 DG 预提取属性文件 (搜索 {search_dirs[0]}), item_attributes.json 将为空")
        return attributes

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"跳过属性文件 {fp}: {e}")
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            qqid = entry.get("qqid")
            if qqid is None:
                continue
            content = ""
            choices = entry.get("choices") or []
            if choices:
                content = (choices[0].get("message") or {}).get("content") or ""
            attrs = _parse_attr_list(content)
            if attrs or content:
                attributes[str(qqid)] = {
                    "intro": "",
                    "attributes": attrs,
                    "item_id": str(qqid),
                }

    logger.info(f"从 DG 预提取属性加载: {len(attributes)} 个物品 (from {len(files)} 文件)")
    return attributes


def _parse_attr_list(content: str) -> List[str]:
    """解析 LLM 输出的属性 JSON 列表字符串"""
    content = content.strip()
    if not content:
        return []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(a).strip() for a in data if str(a).strip()]
    except json.JSONDecodeError:
        pass
    # 降级: 尝试提取 [...] 子串
    start, end = content.find("["), content.rfind("]")
    if 0 <= start < end:
        try:
            data = json.loads(content[start:end + 1])
            if isinstance(data, list):
                return [str(a).strip() for a in data if str(a).strip()]
        except json.JSONDecodeError:
            pass
    return []


def merge_csv_attributes(
    item_attributes: Dict[str, Dict],
    items: Dict[int, Dict],
):
    """对缺失属性的物品, 用 CSV 的 item_attribute 列回填 (AO)"""
    filled = 0
    for idx, meta in items.items():
        if str(idx) not in item_attributes:
            attr_text = meta.get("attribute", "").strip()
            if attr_text:
                item_attributes[str(idx)] = {
                    "intro": "",
                    "attributes": attr_text.split(),
                    "item_id": str(idx),
                }
                filled += 1
    if filled:
        logger.info(f"CSV 属性列回填: {filled} 个物品")


# ============================================================
# 主流程
# ============================================================

def preprocess(config: Optional[Dict] = None):
    """
    执行完整的 DG 数据预处理流程。

    Args:
        config: 预处理配置 (pipeline 的 preprocess 段),
                可选包含 dataset.name / dataset.dg_root / domains.x / domains.y
    """
    ensure_dirs()

    # 从配置同步数据集上下文 (standalone 运行时保证一致)
    ds_cfg = (config or {}).get("dataset", {})
    ds_name = (ds_cfg.get("name") or CURRENT_DATASET).upper()
    if ds_name != CURRENT_DATASET or ds_cfg.get("dg_root"):
        set_dataset(ds_name, ds_cfg.get("dg_root"))

    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # 域名称
    domain_x = (config or {}).get("domains", {}).get("x") or DOMAIN_NAMES[CURRENT_DATASET][0]
    domain_y = (config or {}).get("domains", {}).get("y") or DOMAIN_NAMES[CURRENT_DATASET][1]

    # 目标模式: AO 用 last, GM 用 ts
    target_mode = "last" if CURRENT_DATASET == "AO" else "ts"

    logger.info("=" * 60)
    logger.info(f"Step 1: 解析物品清单 (dataset={CURRENT_DATASET})")
    logger.info("=" * 60)
    items, n_source, n_target = parse_item_lists(CURRENT_DATASET, domain_x, domain_y)

    # 物品元数据
    item_metadata = {}
    for idx, meta in items.items():
        item_metadata[str(idx)] = {
            "title": meta["title"],
            "domain": meta["domain"],
            "description": meta.get("attribute", ""),
        }
    save_json(item_metadata, get_processed_dir() / "item_metadata.json")

    # 物品属性
    logger.info("=" * 60)
    logger.info("Step 2: 加载 DG 预提取物品属性")
    logger.info("=" * 60)
    item_attributes = load_precomputed_attributes()
    merge_csv_attributes(item_attributes, items)
    save_json(item_attributes, get_processed_dir() / "item_attributes.json")
    logger.info(f"  共 {len(item_attributes)} 个物品具备属性")

    # ID 映射表 (对齐 DG 索引空间)
    logger.info("=" * 60)
    logger.info("Step 3: 构建 ID 映射表 (DG 索引空间)")
    logger.info("=" * 60)
    domain_x_items = sorted(str(idx) for idx, m in items.items() if m["domain"] == domain_x)
    domain_y_items = sorted(str(idx) for idx, m in items.items() if m["domain"] == domain_y)
    id_mapping = {
        "item_id_to_index": {str(idx): idx for idx in items},
        "index_to_item_id": {str(idx): idx for idx in items},
        "domain_x_items": domain_x_items,
        "domain_y_items": domain_y_items,
        "num_items": len(items),
        "num_x_items": len(domain_x_items),
        "num_y_items": len(domain_y_items),
        "num_source": n_source,
        "num_target": n_target,
        "dataset": CURRENT_DATASET,
    }
    save_json(id_mapping, get_processed_dir() / "id_mapping.json")

    # 交互序列 (用户级别, 供行为画像/检索文本/UHR 使用)
    logger.info("=" * 60)
    logger.info("Step 4: 解析交互序列 (train/valid/test)")
    logger.info("=" * 60)

    dataset_dir = get_dataset_dir()
    splits = {}
    for split_name, file_key in [("train", "train_file"), ("valid", "valid_file"), ("test", "test_file")]:
        txt_path = dataset_dir / cfg[file_key]
        if not txt_path.exists():
            logger.warning(f"{split_name} 文件不存在: {txt_path}, 跳过")
            splits[split_name] = {}
            continue
        splits[split_name] = parse_split(
            txt_path, target_mode, n_source, domain_x, domain_y,
            max_seq_len=cfg["max_seq_len"],
        )

    # interactions.json: {user_id: [item_id, ...]} (历史序列, 不含目标)
    interactions: Dict[str, List[str]] = defaultdict(list)
    for split_name, split_data in splits.items():
        for sample in split_data.values():
            uid = sample["user_id"]
            seq = sample["seq"]
            if len(interactions[uid]) < len(seq):
                interactions[uid] = list(seq)
    # 过滤冷用户 (少于 min_interactions)
    filtered = {
        uid: seq for uid, seq in interactions.items()
        if len(seq) >= cfg["min_interactions"]
    }
    logger.info(f"用户交互序列: {len(interactions)} → {len(filtered)} (过滤 <{cfg['min_interactions']} 次)")
    save_json(filtered, get_processed_dir() / "interactions.json")

    # 保存各 split
    for split_name in ("train", "valid", "test"):
        split_data = splits[split_name]
        # 过滤历史过短 (无法构成目标) 的样本
        keep = {
            sid: s for sid, s in split_data.items()
            if len(s["seq"]) >= 1 and s["target"]["item_id"] is not None
        }
        save_json(keep, get_processed_dir() / f"{split_name}.json")
        logger.info(f"  {split_name}: {len(keep)} 条样本")

    # 统计
    logger.info("=" * 60)
    logger.info("预处理完成！统计信息:")
    logger.info(f"  数据集: {CURRENT_DATASET} ({domain_x} → {domain_y})")
    logger.info(f"  物品总数: {id_mapping['num_items']} (源 {n_source}, 目标 {n_target})")
    logger.info(f"  用户数: {len(filtered)}")
    logger.info(f"  训练集: {len(splits['train'])} 条, 验证集: {len(splits['valid'])} 条, "
                f"测试集: {len(splits['test'])} 条")
    logger.info(f"  产出目录: {get_processed_dir()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    preprocess()
