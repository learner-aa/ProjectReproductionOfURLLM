"""
数据加载与保存工具模块

提供统一的数据 I/O 接口，支持:
- numpy 特征文件加载 (DG 模型产物)
- JSON 数据加载/保存
- 物品-用户 ID 映射管理
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # enhancement/

# DG 模型产物 (跨域推荐搭建结果) - 多数据集配置
_DG_FINAL_DIR = PROJECT_ROOT.parent / "DG_Final"
DG_FILE_CONFIG: Dict[str, Dict] = {
    "GM": {
        "results_dir": _DG_FINAL_DIR / "GM" / "DG",
        "saver_dir": _DG_FINAL_DIR / "GM" / "saver",
        "scores_file": "best_trte_XORY_DG_.npy",
        "candidates_file": "t4_G2_final_DGresult_test_candidate.npy",
        "feature_prefix": "DGGM",
    },
    "AO": {
        "results_dir": _DG_FINAL_DIR / "AO" / "DG",
        "saver_dir": _DG_FINAL_DIR / "AO" / "saver",
        "scores_file": "best_trte_XORY_DG_390_.npy",
        "candidates_file": "t4_G2_final_DGresult_test_candidate_AO.npy",
        "feature_prefix": "DGAO",
    },
}

# 兼容旧引用 (默认 GM)
DG_RESULTS_DIR = DG_FILE_CONFIG["GM"]["results_dir"]
DG_SAVER_DIR = DG_FILE_CONFIG["GM"]["saver_dir"]
SAVED_MODELS_DIR = DG_RESULTS_DIR

# item_prompt 数据 (DeepSeek V4 结果)
ITEM_PROMPT_GM_DIR = (_DG_FINAL_DIR / "DG_src"
                      / "dataset" / "Entertainment-Education_Amazon" / "item_prompt_GM")
ITEM_PROMPT_AO_DIR = (_DG_FINAL_DIR / "DG_src"
                      / "dataset" / "Entertainment-Education_AO" / "item_prompt_AO")

# 物品列表 CSV + 训练数据
DG_GM_DIR = _DG_FINAL_DIR / "GM"
DG_AO_DIR = _DG_FINAL_DIR / "AO"

# 增强 Pipeline 自身数据路径
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def get_dataset_suffix(dataset: str = "GM") -> str:
    """返回数据集文件名后缀: GM -> '', AO -> '_AO'"""
    ds = dataset.upper()
    return "" if ds == "GM" else f"_{ds}"


def get_processed_path(filename: str, dataset: str = "GM") -> Path:
    """根据数据集返回 processed 目录下的文件路径。
    GM: interactions.json; AO: interactions_AO.json
    """
    suffix = get_dataset_suffix(dataset)
    stem = filename.rsplit(".", 1)[0]
    ext = filename.rsplit(".", 1)[1] if "." in filename else "json"
    return PROCESSED_DIR / f"{stem}{suffix}.{ext}"


def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [PROCESSED_DIR, RAW_DIR, OUTPUT_DIR,
              OUTPUT_DIR / "lora_weights",
              OUTPUT_DIR / "predictions",
              OUTPUT_DIR / "eval_results"]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# DG 模型特征加载
# ============================================================

def load_dg_features(dataset: str = "GM") -> Dict[str, np.ndarray]:
    """
    加载 DG 双图模型产出的特征文件。

    Args:
        dataset: "GM" 或 "AO"

    Returns:
        dict with keys:
            - train_x_fea: (num_items, 656) X域物品特征
            - train_y_fea: (num_items, 656) Y域物品特征
            - test_x_fea:  (num_test_users, 656) X域用户特征
            - test_y_fea:  (num_test_users, 656) Y域用户特征
    """
    cfg = DG_FILE_CONFIG[dataset.upper()]
    prefix = cfg["feature_prefix"]
    results_dir = cfg["results_dir"]
    files = {
        "train_x_fea": results_dir / f"{prefix}_final_train_x_fea.npy",
        "train_y_fea": results_dir / f"{prefix}_final_train_y_fea.npy",
        "test_x_fea": results_dir / f"{prefix}_final_test_x_fea.npy",
        "test_y_fea": results_dir / f"{prefix}_final_test_y_fea.npy",
    }
    features = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"DG 特征文件不存在: {path}")
        features[key] = np.load(str(path))
        logger.info(f"已加载 {key}: shape={features[key].shape}, dtype={features[key].dtype}")
    return features


def load_dg_scores(dataset: str = "GM") -> np.ndarray:
    """加载 DG 模型测试评分矩阵 (num_test_users, num_items)"""
    cfg = DG_FILE_CONFIG[dataset.upper()]
    path = cfg["saver_dir"] / cfg["scores_file"]
    if not path.exists():
        raise FileNotFoundError(f"评分矩阵不存在: {path}")
    scores = np.load(str(path))
    logger.info(f"已加载评分矩阵({dataset}): shape={scores.shape}")
    return scores


def load_dg_candidates(dataset: str = "GM") -> np.ndarray:
    """加载 DG 模型候选物品矩阵 (num_test_users, 10000)"""
    cfg = DG_FILE_CONFIG[dataset.upper()]
    path = cfg["results_dir"] / cfg["candidates_file"]
    if not path.exists():
        raise FileNotFoundError(f"候选矩阵不存在: {path}")
    candidates = np.load(str(path))
    logger.info(f"已加载候选矩阵({dataset}): shape={candidates.shape}")
    return candidates


def load_dg_config(dataset: str = "GM") -> Dict[str, Any]:
    """加载 DG 模型训练配置"""
    cfg = DG_FILE_CONFIG[dataset.upper()]
    path = cfg["results_dir"] / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"已加载 DG 配置({dataset}): model={config.get('model')}, hidden={config.get('hidden_units')}")
    return config


# ============================================================
# JSON 数据加载/保存
# ============================================================

def load_json(path: Union[str, Path]) -> Any:
    """加载 JSON 文件"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"已加载 JSON: {path} ({len(data)} entries)")
    return data


def save_json(data: Any, path: Union[str, Path], indent: int = 2):
    """保存 JSON 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    logger.info(f"已保存 JSON: {path}")


# ============================================================
# 交互序列处理
# ============================================================

def load_interactions(path: Optional[Union[str, Path]] = None, dataset: str = "GM") -> Dict[str, List[str]]:
    """
    加载用户交互序列。

    Args:
        path: JSON 文件路径, 格式 {"user_id": ["item_id_1", "item_id_2", ...]}
              若为 None 则根据 dataset 使用默认路径
        dataset: "GM" 或 "AO"

    Returns:
        dict: {user_id: [item_id, ...]}
    """
    if path is None:
        path = get_processed_path("interactions.json", dataset)
    return load_json(path)


def load_item_metadata(path: Optional[Union[str, Path]] = None, dataset: str = "GM") -> Dict[str, Dict]:
    """
    加载物品元数据。

    Args:
        path: JSON 文件路径
        dataset: "GM" 或 "AO"

    Returns:
        dict: {item_id: metadata_dict}
    """
    if path is None:
        path = get_processed_path("item_metadata.json", dataset)
    return load_json(path)


def load_item_attributes(path: Optional[Union[str, Path]] = None, dataset: str = "GM") -> Dict[str, Dict]:
    """
    加载 LLM 提取的物品属性。

    Args:
        path: JSON 文件路径
        dataset: "GM" 或 "AO"

    Returns:
        dict: {item_id: {"intro": str, "attributes": [str, ...]}}
    """
    if path is None:
        path = get_processed_path("item_attributes.json", dataset)
    return load_json(path)


def load_id_mapping(path: Optional[Union[str, Path]] = None, dataset: str = "GM") -> Dict[str, Any]:
    """
    加载物品 ID 映射表。

    Args:
        path: JSON 文件路径
        dataset: "GM" 或 "AO"

    Returns:
        dict with keys:
            - item_id_to_index: {item_id: dg_index}
            - index_to_item_id: {dg_index: item_id}
            - domain_x_items: [item_ids in domain X]
            - domain_y_items: [item_ids in domain Y]
    """
    if path is None:
        path = get_processed_path("id_mapping.json", dataset)
    return load_json(path)


# ============================================================
# 特征相似度计算
# ============================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算余弦相似度。

    Args:
        a: (m, d) 或 (d,)
        b: (n, d)

    Returns:
        (m, n) 或 (n,) 相似度矩阵
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


def find_topk_similar(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    查找 top-k 最相似的候选项。

    Args:
        query: (d,) 查询向量
        candidates: (n, d) 候选向量矩阵
        k: 返回数量

    Returns:
        (top_k_indices, top_k_scores)
    """
    sims = cosine_similarity(query, candidates).flatten()
    top_k_idx = np.argsort(sims)[::-1][:k]
    top_k_scores = sims[top_k_idx]
    return top_k_idx, top_k_scores
