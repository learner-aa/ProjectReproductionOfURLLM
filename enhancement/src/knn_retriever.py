"""
KNN 用户检索模块 (论文 §4.2.1)

从 DG 模型产出的特征向量中检索与目标用户最相似的训练用户，
为 LLM 提供 few-shot 示例。

当前使用近似方案：训练用户向量 = 其交互物品的特征均值。
预留 load_real_vectors() 接口，可替换为 DG 模型真实输出的训练用户向量。

论文公式 (14):
    E'_X = H_S + H_X   (全域 + X 域偏好)
    E'_Y = H_S + H_Y   (全域 + Y 域偏好)
    Ur = KNN(u, U_train, E)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    PROCESSED_DIR,
    DATASET_SUFFIX,
    cosine_similarity,
    load_dg_features,
    load_interactions,
    load_id_mapping,
    load_item_metadata,
    load_retrieval_results,
    save_retrieval_results,
)

logger = logging.getLogger(__name__)


# ============================================================
# 近似训练用户向量构建
# ============================================================

def build_approximate_user_vectors(
    interactions: Dict[str, List[str]],
    item_features_x: np.ndarray,
    item_features_y: np.ndarray,
    id_mapping: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    用用户交互物品的特征均值近似用户向量。

    Args:
        interactions: {user_id: [item_id, ...]}
        item_features_x: (num_items, dim) X 域物品特征
        item_features_y: (num_items, dim) Y 域物品特征
        id_mapping: ID 映射表 (含 item_id_to_index)

    Returns:
        (user_vecs_x, user_vecs_y, user_ids)
        user_vecs_x[i] 对应 user_ids[i] 的 X 域近似向量
    """
    item_to_idx = id_mapping.get("item_id_to_index", {})
    dim = item_features_x.shape[1]

    user_vecs_x = []
    user_vecs_y = []
    user_ids = []

    for user_id, item_ids in interactions.items():
        # 收集该用户交互物品在 DG 特征中的索引
        indices = []
        for iid in item_ids:
            idx = item_to_idx.get(iid)
            if idx is not None and idx < item_features_x.shape[0]:
                indices.append(idx)

        if not indices:
            # 无有效物品，用零向量
            user_vecs_x.append(np.zeros(dim, dtype=np.float32))
            user_vecs_y.append(np.zeros(dim, dtype=np.float32))
        else:
            indices = np.array(indices)
            user_vecs_x.append(item_features_x[indices].mean(axis=0))
            user_vecs_y.append(item_features_y[indices].mean(axis=0))

        user_ids.append(user_id)

    user_vecs_x = np.array(user_vecs_x, dtype=np.float32)
    user_vecs_y = np.array(user_vecs_y, dtype=np.float32)

    logger.info(
        f"近似训练用户向量构建完成: {len(user_ids)} 用户, "
        f"维度={dim}, X 域均值范数={np.mean(np.linalg.norm(user_vecs_x, axis=1)):.3f}"
    )
    return user_vecs_x, user_vecs_y, user_ids


# ============================================================
# UserRetriever 核心类
# ============================================================

class UserRetriever:
    """
    KNN 用户检索器。

    支持:
    - 域区分检索 (X 域推荐用 X 向量, Y 域用 Y 向量)
    - 训练时排除自身 (取 top-2)
    - 推理时取 top-k
    - 替换真实训练用户向量 (load_real_vectors)
    """

    def __init__(
        self,
        dg_features: Optional[Dict[str, np.ndarray]] = None,
        interactions: Optional[Dict[str, List[str]]] = None,
        id_mapping: Optional[Dict[str, Any]] = None,
        item_metadata: Optional[Dict[str, Dict]] = None,
        domain_x_name: str = "Entertainment",
        domain_y_name: str = "Education",
    ):
        """
        Args:
            dg_features: load_dg_features() 返回值 (可选，None 时自动加载)
            interactions: {user_id: [item_id, ...]} (可选)
            id_mapping: ID 映射表 (可选)
            item_metadata: 物品元数据 (可选)
            domain_x_name: X 域名称
            domain_y_name: Y 域名称
        """
        self.domain_x_name = domain_x_name
        self.domain_y_name = domain_y_name
        self._using_real_vectors = False

        # 加载数据
        if dg_features is None:
            dg_features = load_dg_features()
        if interactions is None:
            interactions = load_interactions()
        if id_mapping is None:
            id_mapping = load_id_mapping()
        if item_metadata is None:
            meta_path = PROCESSED_DIR / f"item_metadata{DATASET_SUFFIX}.json"
            if meta_path.exists():
                from data_utils import load_json
                item_metadata = load_json(meta_path)
            else:
                item_metadata = {}

        self.interactions = interactions
        self.id_mapping = id_mapping
        self.item_metadata = item_metadata

        # DG 特征 (物品特征,非用户向量)
        self.item_fea_x = dg_features["train_x_fea"]
        self.item_fea_y = dg_features["train_y_fea"]

        # 测试用户 DG 向量: 自动去重 (原论文每用户2行,取偶数行)
        raw_test_x = dg_features["test_x_fea"]
        raw_test_y = dg_features["test_y_fea"]
        self.test_fea_x = raw_test_x[0::2] if raw_test_x.shape[0] % 2 == 0 else raw_test_x
        self.test_fea_y = raw_test_y[0::2] if raw_test_y.shape[0] % 2 == 0 else raw_test_y
        if raw_test_x.shape[0] != self.test_fea_x.shape[0]:
            logger.info(f"测试用户向量去重: {raw_test_x.shape[0]} → {self.test_fea_x.shape[0]}")

        # 构建近似训练用户向量
        self.train_user_x, self.train_user_y, self.train_user_ids = \
            build_approximate_user_vectors(
                interactions, self.item_fea_x, self.item_fea_y, id_mapping
            )

        # 训练用户 user_id -> 在 train_user_ids 中的索引
        self._train_user_to_idx = {
            uid: i for i, uid in enumerate(self.train_user_ids)
        }

        logger.info(
            f"UserRetriever 初始化完成: "
            f"{len(self.train_user_ids)} 训练用户, "
            f"测试用户向量 shape={self.test_fea_x.shape}, "
            f"近似向量 (后续可通过 load_real_vectors 替换)"
        )

    def load_real_vectors(
        self,
        train_user_x_path: str,
        train_user_y_path: str,
    ):
        """
        替换近似向量为 DG 模型真实输出的训练用户向量。

        自动处理:
        1. 去重: 原论文每用户2行,取偶数行去重
        2. 数量适配: 真实向量只覆盖训练用户(8000),interactions 中的
           valid/test 用户无真实向量,对应位置清零(检索时跳过)

        Args:
            train_user_x_path: X 域训练用户向量 .npy 路径
            train_user_y_path: Y 域训练用户向量 .npy 路径
        """
        real_x = np.load(train_user_x_path)
        real_y = np.load(train_user_y_path)

        # 自动去重: 如果行数是偶数,取偶数行(原论文每用户2行)
        if real_x.shape[0] % 2 == 0:
            real_x_dedup = real_x[0::2]
            real_y_dedup = real_y[0::2]
            logger.info(
                f"DG 向量去重: {real_x.shape[0]} → {real_x_dedup.shape[0]} "
                f"(每用户2行,取偶数行)"
            )
            real_x = real_x_dedup
            real_y = real_y_dedup

        num_train = len(self.train_user_ids)
        num_real = real_x.shape[0]

        if num_real == num_train:
            # 完全匹配,全部替换
            self.train_user_x = real_x.astype(np.float32)
            self.train_user_y = real_y.astype(np.float32)
        elif num_real < num_train:
            # 真实向量只覆盖训练用户(前 num_real 个),其余(valid/test)清零
            # 注意: train_user_ids 由 interactions 构建,顺序为
            # [train_users (8000), valid_users (577), test_users (1000)]
            self.train_user_x = np.zeros_like(self.train_user_x)
            self.train_user_y = np.zeros_like(self.train_user_y)
            self.train_user_x[:num_real] = real_x.astype(np.float32)
            self.train_user_y[:num_real] = real_y.astype(np.float32)
            logger.info(
                f"真实向量 {num_real} 替换前 {num_real} 个训练用户向量, "
                f"其余 {num_train - num_real} 个(valid/test)清零,检索时跳过"
            )
        else:
            # 真实向量多于训练用户,只取前 num_train 个
            self.train_user_x = real_x[:num_train].astype(np.float32)
            self.train_user_y = real_y[:num_train].astype(np.float32)
            logger.warning(
                f"真实向量 {num_real} > 训练用户 {num_train}, "
                f"只取前 {num_train} 个"
            )

        self._using_real_vectors = True
        logger.info(
            f"已替换为真实训练用户向量: shape={self.train_user_x.shape}, "
            f"维度={self.train_user_x.shape[1]}"
        )

    def _resolve_dg_domain(self, target_domain: str) -> bool:
        """
        根据 target_domain 判断应使用 DG X 域还是 Y 域向量。

        DG 模型 AO 数据集: X 域=Art(csv_paths[0]=item_listA), Y 域=Office(csv_paths[1]=item_listO)
        但数据文件中 domain 标签可能为 "Entertainment"(=Art) / "Education"(=Office) (GM 标签混用)

        映射规则 (固定,不依赖 domain_x_name/domain_y_name 配置):
        - Art / Entertainment → DG X 域 (True)
        - Office / Education → DG Y 域 (False)
        - 默认 → DG X 域 (True)
        """
        td = (target_domain or "").lower()
        # Y 域 (Office) 标签 - 先判断 Y 域,避免被 X 域的 domain_x_name 误匹配
        if td in ("office", "education", "y"):
            return False
        # X 域 (Art) 标签
        if td in ("art", "entertainment", "x"):
            return True
        # 默认用 X 域
        return True

    def _get_query_vector(
        self,
        user_id: Optional[str],
        test_user_index: Optional[int],
        target_domain: str,
    ) -> np.ndarray:
        """获取查询向量"""
        is_x = self._resolve_dg_domain(target_domain)

        if test_user_index is not None:
            # 测试用户: 直接用 DG 模型输出的向量
            fea = self.test_fea_x if is_x else self.test_fea_y
            if test_user_index >= fea.shape[0]:
                raise IndexError(
                    f"test_user_index {test_user_index} 超出范围 "
                    f"(共 {fea.shape[0]} 个测试用户)"
                )
            return fea[test_user_index]

        elif user_id is not None:
            # 训练用户: 用近似向量
            idx = self._train_user_to_idx.get(user_id)
            if idx is None:
                raise KeyError(f"用户 {user_id} 不在训练集中")
            vecs = self.train_user_x if is_x else self.train_user_y
            return vecs[idx]

        raise ValueError("必须提供 user_id 或 test_user_index")

    def _get_library_vectors(self, target_domain: str) -> np.ndarray:
        """获取检索库向量 (训练用户向量)"""
        is_x = self._resolve_dg_domain(target_domain)
        return self.train_user_x if is_x else self.train_user_y

    def retrieve(
        self,
        user_id: Optional[str] = None,
        test_user_index: Optional[int] = None,
        target_domain: str = "X",
        k: int = 3,
        exclude_self: bool = False,
    ) -> List[str]:
        """
        KNN 检索最相似的训练用户。

        Args:
            user_id: 训练用户 ID (与 test_user_index 二选一)
            test_user_index: 测试用户索引 (与 user_id 二选一)
            target_domain: 目标域 ("X" / "Y" / 域名)
            k: 返回的相似用户数
            exclude_self: 是否排除自身 (训练集检索时设为 True)

        Returns:
            相似训练用户的 user_id 列表
        """
        query = self._get_query_vector(user_id, test_user_index, target_domain)
        library = self._get_library_vectors(target_domain)

        # 余弦相似度
        sims = cosine_similarity(query, library).flatten()

        # 排除自身
        if exclude_self and user_id is not None:
            self_idx = self._train_user_to_idx.get(user_id)
            if self_idx is not None:
                sims[self_idx] = -999.0

        # Top-k
        top_k_idx = np.argsort(sims)[::-1][:k]

        return [self.train_user_ids[i] for i in top_k_idx]

    def get_retrieved_user_text(
        self,
        retrieved_user_ids: List[str],
        max_items_per_user: int = 5,
    ) -> str:
        """
        将检索到的用户的交互序列格式化为 few-shot 文本。

        论文 Prompt II 格式:
          "There is another similar user who has played
           movies and games before: Movie: X | Game: Y | ..."

        域标签映射 (修正 AO 数据集中 GM 标签混用):
        - Entertainment → Art (DG X 域)
        - Education → Office (DG Y 域)

        Args:
            retrieved_user_ids: 检索到的训练用户 ID 列表
            max_items_per_user: 每个用户最多显示的物品数

        Returns:
            格式化文本
        """
        # 域标签映射: 数据文件中的错误标签 → 正确标签
        domain_label_map = {
            "Entertainment": "Art",
            "Education": "Office",
            "Art": "Art",
            "Office": "Office",
        }

        lines = []
        for uid in retrieved_user_ids:
            items = self.interactions.get(uid, [])
            if not items:
                continue

            item_texts = []
            for iid in items[-max_items_per_user:]:
                meta = self.item_metadata.get(iid, {})
                title = meta.get("title", iid)
                domain = meta.get("domain", "")
                # 使用域标签映射
                label = domain_label_map.get(domain, domain or "Item")
                item_texts.append(f"{label}: {title}")

            if item_texts:
                lines.append(
                    f"There is a similar user who has interacted with: "
                    + " | ".join(item_texts)
                )

        return "\n".join(lines) if lines else "(no similar users found)"

    def retrieve_for_split(
        self,
        split_data: Dict[str, Dict],
        split_name: str,
        k: int = 3,
        test_user_offset: int = 0,
    ) -> Dict[str, List[str]]:
        """
        为某个数据划分 (train/valid/test) 批量检索相似用户。

        使用 numpy 矩阵乘法一次性计算所有相似度,避免逐个用户循环。

        Args:
            split_data: {user_id: {"seq": [...], "target": {...}}}
            split_name: "train" / "valid" / "test"
            k: 检索数量 (训练集建议 2, 测试集建议 3-5)
            test_user_offset: 测试用户索引起始偏移量 (用于 test split)

        Returns:
            {user_id: [retrieved_user_id, ...]}

        注意: valid 集没有真实 DG 向量(原论文只保存 train/test 用户向量),
              所以 valid split 直接返回空字典,build_instruction 时不会添加检索文本。
              这会导致 valid loss 与 train 略有差异,但不影响早停和监控。
        """
        # valid 集没有真实 DG 向量,跳过检索
        if split_name == "valid":
            logger.info(
                f"检索跳过 [{split_name}]: valid 集无真实 DG 向量, "
                f"返回空字典 ({len(split_data)} 用户)"
            )
            return {uid: [] for uid in split_data.keys()}

        results = {}
        is_test = split_name == "test"
        exclude_self = split_name == "train"
        total = len(split_data)
        user_ids = list(split_data.keys())

        # 按 target_domain 分组 (X 域 / Y 域),分别批量检索
        domain_groups = {"X": [], "Y": []}
        for idx, (uid, data) in enumerate(split_data.items()):
            target = data.get("target", {})
            target_domain = target.get("domain", self.domain_x_name) if isinstance(target, dict) else self.domain_x_name
            if not target_domain:
                target_domain = self.domain_x_name
            is_x = self._resolve_dg_domain(target_domain)
            domain_groups["X" if is_x else "Y"].append((idx, uid))

        logger.info(
            f"检索分组 [{split_name}]: X域={len(domain_groups['X'])}用户, "
            f"Y域={len(domain_groups['Y'])}用户"
        )

        for domain_key, group in domain_groups.items():
            if not group:
                continue

            is_x = (domain_key == "X")
            library = self.train_user_x if is_x else self.train_user_y  # (num_train, dim)
            library_norm = library / (np.linalg.norm(library, axis=1, keepdims=True) + 1e-8)

            if is_test:
                # 测试用户: 用 test_fea_x/y 查询
                fea = self.test_fea_x if is_x else self.test_fea_y
                query_indices = [test_user_offset + idx for idx, _ in group]
                query_vecs = fea[query_indices]  # (n, dim)
            else:
                # 训练用户: 用 train_user_x/y 查询
                query_indices = [self._train_user_to_idx[uid] for _, uid in group]
                query_vecs = library[query_indices]  # (n, dim)

            # 批量计算相似度
            query_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
            sims = query_norm @ library_norm.T  # (n, num_train)

            # 排除自身 (训练集)
            if exclude_self:
                for i, qidx in enumerate(query_indices):
                    sims[i, qidx] = -999.0

            # Top-k
            for i, (idx, uid) in enumerate(group):
                top_k_idx = np.argsort(sims[i])[::-1][:k]
                results[uid] = [self.train_user_ids[j] for j in top_k_idx]

            logger.info(
                f"检索完成 [{split_name}/{domain_key}域]: {len(group)}用户, "
                f"k={k}, exclude_self={exclude_self}"
            )

        logger.info(
            f"检索完成 [{split_name}]: 共 {len(results)} 用户"
        )
        return results


# ============================================================
# 主流程: 运行检索并保存
# ============================================================

def run_user_retrieval(config: Dict):
    """
    运行 KNN 用户检索流程。

    Args:
        config: pipeline 配置字典，可包含:
            - retrieval.k_train: 训练集检索数 (默认 2)
            - retrieval.k_test: 测试集检索数 (默认 3)
            - retrieval.real_vectors_x: 真实 X 向量路径 (可选)
            - retrieval.real_vectors_y: 真实 Y 向量路径 (可选)
            - domains.x: X 域名称
            - domains.y: Y 域名称
    """
    logger.info("=" * 60)
    logger.info("开始 KNN 用户检索")
    logger.info("=" * 60)

    # 配置
    ret_cfg = config.get("retrieval", {})
    domain_cfg = config.get("domains", {})
    domain_x = domain_cfg.get("x", "Entertainment")
    domain_y = domain_cfg.get("y", "Education")
    k_train = ret_cfg.get("k_train", 2)
    k_test = ret_cfg.get("k_test", 3)

    # 初始化检索器
    retriever = UserRetriever(
        domain_x_name=domain_x,
        domain_y_name=domain_y,
    )

    # 替换真实向量 (如果配置了路径)
    real_x_path = ret_cfg.get("real_vectors_x")
    real_y_path = ret_cfg.get("real_vectors_y")
    if real_x_path and real_y_path:
        if Path(real_x_path).exists() and Path(real_y_path).exists():
            retriever.load_real_vectors(real_x_path, real_y_path)
            logger.info("已加载真实训练用户向量")
        else:
            logger.warning(
                f"真实向量文件不存在: {real_x_path} / {real_y_path}, "
                f"继续使用近似向量"
            )

    # 加载数据划分
    from data_utils import load_json
    train_data = load_json(PROCESSED_DIR / f"train{DATASET_SUFFIX}.json")
    valid_data = load_json(PROCESSED_DIR / f"valid{DATASET_SUFFIX}.json")
    test_data = load_json(PROCESSED_DIR / f"test{DATASET_SUFFIX}.json")

    # 批量检索
    logger.info(f"训练集检索 (k={k_train}, 排除自身, 取 top-2)...")
    train_retrieval = retriever.retrieve_for_split(
        train_data, "train", k=k_train
    )

    logger.info(f"验证集检索 (k={k_test})...")
    valid_retrieval = retriever.retrieve_for_split(
        valid_data, "valid", k=k_test
    )

    logger.info(f"测试集检索 (k={k_test})...")
    test_retrieval = retriever.retrieve_for_split(
        test_data, "test", k=k_test
    )

    # 保存结果
    retrieval_results = {
        "train": train_retrieval,
        "valid": valid_retrieval,
        "test": test_retrieval,
        "metadata": {
            "k_train": k_train,
            "k_test": k_test,
            "using_real_vectors": retriever._using_real_vectors,
            "num_train_users": len(retriever.train_user_ids),
            "vector_dim": retriever.train_user_x.shape[1],
            "domain_x": domain_x,
            "domain_y": domain_y,
        },
    }
    save_retrieval_results(retrieval_results)

    logger.info("=" * 60)
    logger.info("KNN 用户检索完成!")
    logger.info(f"  训练集: {len(train_retrieval)} 用户")
    logger.info(f"  验证集: {len(valid_retrieval)} 用户")
    logger.info(f"  测试集: {len(test_retrieval)} 用户")
    logger.info(f"  向量来源: {'真实 DG 输出' if retriever._using_real_vectors else '物品均值近似'}")
    logger.info("=" * 60)

    return retriever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import yaml

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
        logger.warning(f"配置文件不存在: {config_path}")
    run_user_retrieval(cfg)
