"""
数据加载与保存工具模块

提供统一的数据 I/O 接口，支持:
- numpy 特征文件加载 (DG 模型产物)
- JSON 数据加载/保存
- 物品-用户 ID 映射管理

通用数据集支持:
- 通过 set_dataset(name, dg_root) 切换 AO / GM
- DG 特征路径: {dg_root}/{name}/DG/DG{NAME}_final_{key}_fea.npy
- DG 特征内容为「用户向量」(非物品向量): 每行对应 train/test 的一条交互样本
"""

import json
import os
import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 当前数据集上下文 (默认 AO, 运行时由 set_dataset 覆盖)
# 注意: dg_root 必须在 pipeline_config.yaml 中显式配置, 这里不再提供 Windows 默认值
CURRENT_DATASET = "AO"
CURRENT_DG_ROOT: Optional[str] = None

# 当前数据集对应的处理目录/输出目录 (默认 AO 子目录, 由 set_dataset 切换)
# 路径隔离: data/processed/{AO|GM}/  和  outputs/{AO|GM}/
CURRENT_PROCESSED_DIR = PROCESSED_DIR / "AO"
CURRENT_OUTPUT_DIR = OUTPUT_DIR / "AO"


def set_dataset(name: str, dg_root: Optional[str] = None):
    """
    切换 DG 数据集上下文 (AO / GM)。

    切换后, processed_dir 指向 data/processed/{name}/, output_dir 指向 outputs/{name}/,
    实现 AO/GM 数据与产物的物理隔离。

    Args:
        name: 数据集名 ("AO" / "GM")
        dg_root: DG 模型根目录, None 时保持当前值 (默认 None, 必须由 yaml 配置)
    """
    global CURRENT_DATASET, CURRENT_DG_ROOT, CURRENT_PROCESSED_DIR, CURRENT_OUTPUT_DIR
    name = name.upper()
    assert name in ("AO", "GM"), f"未知数据集: {name}, 仅支持 AO/GM"
    CURRENT_DATASET = name
    if dg_root:
        candidate = str(dg_root)
        if not Path(candidate).exists():
            local_default = str(PROJECT_ROOT.parent / "DG_Final")
            logger.warning(
                f"dg_root 不存在: {candidate}, 回退到本地默认: {local_default}"
            )
            candidate = local_default
        CURRENT_DG_ROOT = candidate
    elif CURRENT_DG_ROOT is None:
        local_default = str(PROJECT_ROOT.parent / "DG_Final")
        if Path(local_default).exists():
            logger.info(f"dg_root 未配置, 使用本地默认: {local_default}")
            CURRENT_DG_ROOT = local_default
    # 切换 processed/outputs 目录到对应数据集子目录
    CURRENT_PROCESSED_DIR = PROCESSED_DIR / name
    CURRENT_OUTPUT_DIR = OUTPUT_DIR / name
    logger.info(
        f"数据集上下文: {name}, dg_root={CURRENT_DG_ROOT}, "
        f"processed_dir={CURRENT_PROCESSED_DIR}, output_dir={CURRENT_OUTPUT_DIR}"
    )


def get_dg_root() -> str:
    """DG 模型根目录"""
    assert CURRENT_DG_ROOT, (
        "DG root 未配置, 请在 pipeline_config.yaml 设置 dataset.dg_root"
    )
    return CURRENT_DG_ROOT


def get_processed_dir() -> Path:
    """当前数据集的处理数据目录 (data/processed/{AO|GM}/)"""
    return CURRENT_PROCESSED_DIR


def get_output_dir() -> Path:
    """当前数据集的输出目录 (outputs/{AO|GM}/)"""
    return CURRENT_OUTPUT_DIR


def get_dataset_dir() -> Path:
    """DG 数据集的根目录 (含 txt / csv / DG 子目录)"""
    assert CURRENT_DG_ROOT, (
        "DG root 未配置, 请在 pipeline_config.yaml 设置 dataset.dg_root"
    )
    return Path(CURRENT_DG_ROOT) / CURRENT_DATASET


def get_dg_dir() -> Path:
    """DG 特征目录 (含 *_final_*_fea.npy)"""
    return get_dataset_dir() / "DG"


def get_dg_prefix() -> str:
    """DG 特征文件前缀, 如 AO -> 'DGAO', GM -> 'DGGM'"""
    return f"DG{CURRENT_DATASET}"


def ensure_dirs():
    """确保所有必要目录存在 (含当前数据集子目录)"""
    for d in [PROCESSED_DIR, RAW_DIR, OUTPUT_DIR,
              CURRENT_PROCESSED_DIR,
              CURRENT_OUTPUT_DIR,
              CURRENT_OUTPUT_DIR / "lora_weights",
              CURRENT_OUTPUT_DIR / "predictions",
              CURRENT_OUTPUT_DIR / "refined_predictions",
              CURRENT_OUTPUT_DIR / "eval_results"]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# DG 模型特征加载
# ============================================================

def load_dg_features() -> Dict[str, np.ndarray]:
    """
    加载 DG 双图模型产出的特征文件。

    注意: DG 保存的每一行是「用户向量」(每条 train/test 样本一行),
    而非物品特征。AO: train 16000 行 (8000 用户 × 2 目标行), test 2000 行;
    GM: train 35941 行 (每用户 1 行), test 3601 行。

    Returns:
        dict with keys:
            - train_x_fea: (num_train_rows, 656) X域用户特征
            - train_y_fea: (num_train_rows, 656) Y域用户特征
            - test_x_fea:  (num_test_rows, 656)  X域用户特征
            - test_y_fea:  (num_test_rows, 656)  Y域用户特征
    """
    prefix = get_dg_prefix()
    files = {
        "train_x_fea": get_dg_dir() / f"{prefix}_final_train_x_fea.npy",
        "train_y_fea": get_dg_dir() / f"{prefix}_final_train_y_fea.npy",
        "test_x_fea": get_dg_dir() / f"{prefix}_final_test_x_fea.npy",
        "test_y_fea": get_dg_dir() / f"{prefix}_final_test_y_fea.npy",
    }
    features = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"DG 特征文件不存在: {path}")
        features[key] = np.load(str(path))
        logger.info(f"已加载 {key}: shape={features[key].shape}, dtype={features[key].dtype}")
    return features


def load_dg_scores() -> np.ndarray:
    """
    加载 DG 对比学习 MLP 产出的测试评分矩阵 (num_test_users, num_items)。

    路径搜索: {dg_root}/{dataset}/saver/best_trte_XORY_DG*.npy
    (由 Final_train_contrasive_searcher.py 生成, 文件名含 hidden 维度后缀)
    """
    patterns = [
        get_dataset_dir() / "saver" / "best_trte_XORY_DG*.npy",
        get_dataset_dir() / "best_trte_XORY_DG*.npy",
        Path(CURRENT_DG_ROOT) / "best_trte_XORY_DG*.npy",
    ]
    for pattern in patterns:
        matches = glob.glob(str(pattern))
        if matches:
            path = matches[0]
            scores = np.load(path)
            logger.info(f"已加载评分矩阵: {path} shape={scores.shape}")
            return scores
    raise FileNotFoundError(
        f"DG 评分矩阵不存在 (搜索 {patterns[0]} 附近)。"
        f"请先运行 {get_dataset_dir()/'Final_train_contrasive_searcher.py'} 生成。"
    )


def load_dg_candidates() -> np.ndarray:
    """加载 DG 模型候选物品矩阵 (num_test_users, 10000)"""
    candidates = None
    for pattern in [
        get_dataset_dir() / "DG" / "*DGresult*test_candidate*.npy",
        Path(CURRENT_DG_ROOT) / "*DGresult*test_candidate*.npy",
        get_dataset_dir() / "*candidate*.npy",
    ]:
        matches = glob.glob(str(pattern))
        if matches:
            candidates = np.load(matches[0])
            logger.info(f"已加载候选矩阵: {matches[0]} shape={candidates.shape}")
            break
    if candidates is None:
        raise FileNotFoundError(f"DG 候选矩阵不存在: {CURRENT_DG_ROOT}/*DGresult*test_candidate*.npy")
    return candidates


def load_dg_config() -> Dict[str, Any]:
    """加载 DG 模型训练配置"""
    path = get_dataset_dir() / "config.json"
    if not path.exists():
        path = Path(CURRENT_DG_ROOT) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"DG 配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"已加载 DG 配置: model={config.get('model')}, hidden={config.get('hidden_units')}")
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

def load_interactions(path: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
    """
    加载用户交互序列。

    Args:
        path: JSON 文件路径, 格式 {"user_id": ["item_id_1", "item_id_2", ...]}
              若为 None 则使用默认路径

    Returns:
        dict: {user_id: [item_id, ...]}
    """
    if path is None:
        path = get_processed_dir() / "interactions.json"
    return load_json(path)


def load_item_metadata(path: Optional[Union[str, Path]] = None) -> Dict[str, Dict]:
    """
    加载物品元数据。

    Args:
        path: JSON 文件路径
              格式 {"item_id": {"title": str, "description": str, "category": str, "domain": str}}

    Returns:
        dict: {item_id: metadata_dict}
    """
    if path is None:
        path = get_processed_dir() / "item_metadata.json"
    return load_json(path)


def load_item_attributes(path: Optional[Union[str, Path]] = None) -> Dict[str, Dict]:
    """
    加载 LLM 提取的物品属性。

    Returns:
        dict: {item_id: {"intro": str, "attributes": [str, ...]}}
    """
    if path is None:
        path = get_processed_dir() / "item_attributes.json"
    return load_json(path)


def load_id_mapping(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    加载物品 ID 映射表。

    Returns:
        dict with keys:
            - item_id_to_index: {item_id: dg_index}
            - index_to_item_id: {dg_index: item_id}
            - domain_x_items: [item_ids in domain X]
            - domain_y_items: [item_ids in domain Y]
    """
    if path is None:
        path = get_processed_dir() / "id_mapping.json"
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


# ============================================================
# 检索结果存取
# ============================================================

def get_retrieval_path() -> Path:
    """当前数据集的检索结果路径 (动态, 跟随 set_dataset 切换)"""
    return get_processed_dir() / "retrieval_results.json"


def save_retrieval_results(
    retrieval_results: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
):
    """
    保存 KNN 用户检索结果。

    Args:
        retrieval_results: {
            "train": {user_id: [retrieved_user_id, ...]},
            "valid": {user_id: [retrieved_user_id, ...]},
            "test":  {user_id: [retrieved_user_id, ...]},
            "metadata": {"k": int, "using_real_vectors": bool, ...}
        }
        path: 输出路径
    """
    if path is None:
        path = get_retrieval_path()
    save_json(retrieval_results, path)
    logger.info(f"检索结果已保存: {path}")


def load_retrieval_results(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    加载 KNN 用户检索结果。

    Returns:
        检索结果字典
    """
    if path is None:
        path = get_retrieval_path()
    return load_json(path)
