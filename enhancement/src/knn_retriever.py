"""
KNN 用户检索模块 (论文 §4.2.1, 公式 14)

从 DG 模型产出的用户向量中检索与目标用户最相似的训练用户,
为 LLM 提供 few-shot 示例。

关键修正 (通用版):
- DG train_x/y_fea 的每一行就是「用户向量」(每条训练样本一行), 直接作为检索库,
  不再用「交互物品特征均值」近似。
- 测试用户向量来自 test_x/y_fea, 行号 = split 数据中的 dg_index。
- 检索时按目标域选择视图: X 域推荐用 X-view 向量, Y 域推荐用 Y-view 向量。
- 检索结果按 user_id 去重 (AO 同一用户有 X/Y 两行)。

论文公式 (14):
    E'_X = H_S + H_X   (全域 + X 域偏好)
    E'_Y = H_S + H_Y   (全域 + Y 域偏好)
    Ur = KNN(u, U_train, E)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_utils import (
    get_processed_dir,
    cosine_similarity,
    load_dg_features,
    load_interactions,
    load_item_metadata,
    load_json,
    save_retrieval_results,
)

logger = logging.getLogger(__name__)


class UserRetriever:
    """
    KNN 用户检索器。

    检索库 = DG 训练用户向量 (train_x_fea / train_y_fea)。
    - 域区分检索 (X 域推荐用 X 向量, Y 域用 Y 向量)
    - 训练时按 user_id 排除自身
    - 推理时取 top-k
    """

    def __init__(
        self,
        dg_features: Optional[Dict[str, np.ndarray]] = None,
        interactions: Optional[Dict[str, List[str]]] = None,
        train_data: Optional[Dict[str, Dict]] = None,
        item_metadata: Optional[Dict[str, Dict]] = None,
        domain_x_name: str = "Art",
        domain_y_name: str = "Office",
    ):
        """
        Args:
            dg_features: load_dg_features() 返回值 (None 时自动加载)
            interactions: {user_id: [item_id, ...]} (None 时自动加载)
            train_data: train.json, {sample_id: {user_id, dg_index, ...}}
                        用于构建 train_user_ids (每行对应 user_id)
            item_metadata: 物品元数据 (None 时自动加载)
            domain_x_name: X 域名称
            domain_y_name: Y 域名称
        """
        self.domain_x_name = domain_x_name
        self.domain_y_name = domain_y_name
        self._using_real_vectors = True

        # 加载数据
        if dg_features is None:
            dg_features = load_dg_features()
        if interactions is None:
            interactions = load_interactions()
        if train_data is None:
            train_path = get_processed_dir() / "train.json"
            train_data = load_json(train_path) if train_path.exists() else {}
        if item_metadata is None:
            item_metadata = load_item_metadata()

        self.interactions = interactions
        self.item_metadata = item_metadata
        self.train_data = train_data

        # DG 特征 = 用户向量 (每行一条训练/测试样本)
        self.train_user_x = dg_features["train_x_fea"]   # (num_train_rows, 656)
        self.train_user_y = dg_features["train_y_fea"]
        self.test_fea_x = dg_features["test_x_fea"]      # (num_test_rows, 656)
        self.test_fea_y = dg_features["test_y_fea"]

        # 每行训练向量对应的 user_id (按 train.json 顺序与特征行对齐)
        self.train_user_ids: List[str] = []
        for sample in train_data.values():
            self.train_user_ids.append(str(sample.get("user_id", "")))

        n_train = len(self.train_user_ids)
        if n_train != self.train_user_x.shape[0]:
            logger.warning(
                f"train.json 样本数 {n_train} 与 DG 训练特征行数 "
                f"{self.train_user_x.shape[0]} 不一致。将按行号对齐, 可能错位。"
            )

        logger.info(
            f"UserRetriever 初始化完成: {n_train} 条训练样本向量 "
            f"(真实 DG 用户向量), 维度={self.train_user_x.shape[1]}, "
            f"测试样本 {self.test_fea_x.shape[0]} 条"
        )

    def load_real_vectors(
        self,
        train_user_x_path: str,
        train_user_y_path: str,
    ):
        """
        覆盖检索库为用户自定义的训练用户向量。

        要求: 向量行数与 train_user_ids 一致。
        """
        real_x = np.load(train_user_x_path)
        real_y = np.load(train_user_y_path)
        if real_x.shape[0] != len(self.train_user_ids):
            raise ValueError(
                f"真实向量行数 {real_x.shape[0]} 与训练样本数 "
                f"{len(self.train_user_ids)} 不匹配"
            )
        self.train_user_x = real_x.astype(np.float32)
        self.train_user_y = real_y.astype(np.float32)
        self._using_real_vectors = True
        logger.info(f"已覆盖训练用户向量: shape={real_x.shape}")

    def _get_library_vectors(self, target_domain: str) -> np.ndarray:
        """获取检索库向量 (训练用户向量), 按目标域选视图"""
        is_x = target_domain in (self.domain_x_name, "X")
        return self.train_user_x if is_x else self.train_user_y

    def retrieve(
        self,
        query_fea: np.ndarray,
        dg_index: int,
        target_domain: str = "X",
        k: int = 3,
        exclude_user_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        KNN 检索最相似的训练用户 (按 user_id 去重)。

        Args:
            query_fea: 查询向量矩阵 (test_x/y_fea 或 train_x/y_fea)
            dg_index: 查询向量所在行号
            target_domain: 目标域 ("X" / "Y" / 域名)
            k: 返回的相似用户数
            exclude_user_ids: 需排除的用户 (训练时排除自身)

        Returns:
            相似训练用户的 user_id 列表 (去重)
        """
        if dg_index >= query_fea.shape[0]:
            raise IndexError(
                f"dg_index {dg_index} 超出特征行数 {query_fea.shape[0]}"
            )
        query = query_fea[dg_index]
        library = self._get_library_vectors(target_domain)

        # 余弦相似度
        sims = cosine_similarity(query, library).flatten()

        # 排除指定用户 (自身)
        if exclude_user_ids:
            excl = set(exclude_user_ids)
            for i, uid in enumerate(self.train_user_ids):
                if uid in excl:
                    sims[i] = -999.0

        # Top-k, 按 user_id 去重 (AO 同一用户有 X/Y 两行)
        order = np.argsort(sims)[::-1]
        seen: set = set()
        result: List[str] = []
        for i in order:
            uid = self.train_user_ids[i]
            if uid in seen:
                continue
            seen.add(uid)
            result.append(uid)
            if len(result) >= k:
                break
        return result

    def get_retrieved_user_text(
        self,
        retrieved_user_ids: List[str],
        max_items_per_user: int = 5,
    ) -> str:
        """
        将检索到的用户的交互序列格式化为 few-shot 文本。

        论文 Prompt II 格式:
          "There is a similar user who has interacted with: Movie: X | Game: Y | ..."

        Args:
            retrieved_user_ids: 检索到的训练用户 ID 列表
            max_items_per_user: 每个用户最多显示的物品数

        Returns:
            格式化文本
        """
        lines = []
        for uid in retrieved_user_ids:
            items = self.interactions.get(uid, [])
            if not items:
                continue

            item_texts = []
            for iid in items[-max_items_per_user:]:
                iid = str(iid)
                meta = self.item_metadata.get(iid, {})
                title = meta.get("title", iid)
                domain = meta.get("domain", "")
                if domain in (self.domain_x_name, "X"):
                    label = self.domain_x_name
                elif domain in (self.domain_y_name, "Y"):
                    label = self.domain_y_name
                else:
                    label = domain or "Item"
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
    ) -> Dict[str, List[str]]:
        """
        为某个数据划分批量检索相似用户。

        Args:
            split_data: {sample_id: {"user_id", "dg_index", "target": {...}}}
            split_name: "train" / "test" (valid 无 DG 特征, 跳过)
            k: 检索数量 (训练集建议 2, 测试集建议 3-5)

        Returns:
            {sample_id: [retrieved_user_id, ...]}
        """
        results = {}
        is_test = split_name == "test"
        exclude_self = split_name == "train"

        total = len(split_data)
        for idx, (sample_id, data) in enumerate(split_data.items()):
            dg_index = data.get("dg_index")
            if dg_index is None:
                logger.warning(f"{sample_id} 无 dg_index, 跳过检索")
                continue

            user_id = str(data.get("user_id", ""))
            target = data.get("target", {})
            target_domain = target.get("domain", self.domain_x_name)
            if not target_domain:
                target_domain = self.domain_x_name

            # 选择查询向量矩阵
            is_x = target_domain in (self.domain_x_name, "X")
            if is_test:
                query_fea = self.test_fea_x if is_x else self.test_fea_y
            else:
                query_fea = self.train_user_x if is_x else self.train_user_y

            exclude = {user_id} if exclude_self else None
            try:
                retrieved = self.retrieve(
                    query_fea, dg_index, target_domain, k=k,
                    exclude_user_ids=exclude,
                )
            except IndexError as e:
                logger.warning(f"{sample_id} 检索失败: {e}")
                retrieved = []

            results[sample_id] = retrieved

            if (idx + 1) % 500 == 0:
                logger.info(f"检索进度 [{split_name}]: {idx + 1}/{total}")

        logger.info(
            f"检索完成 [{split_name}]: {len(results)} 条, k={k}, "
            f"exclude_self={exclude_self}"
        )
        return results


# ============================================================
# 主流程: 运行检索并保存
# ============================================================

def run_user_retrieval(config: Dict):
    """
    运行 KNN 用户检索流程。

    Args:
        config: pipeline 配置字典, 可包含:
            - retrieval.k_train: 训练集检索数 (默认 2)
            - retrieval.k_test: 测试集检索数 (默认 3)
            - domains.x / domains.y: 域名称
    """
    logger.info("=" * 60)
    logger.info("开始 KNN 用户检索")
    logger.info("=" * 60)

    # 配置
    ret_cfg = config.get("retrieval", {})
    domain_cfg = config.get("domains", {})
    domain_x = domain_cfg.get("x", "Art")
    domain_y = domain_cfg.get("y", "Office")
    k_train = ret_cfg.get("k_train", 2)
    k_test = ret_cfg.get("k_test", 3)

    # 初始化检索器 (自动加载 DG 真实用户向量)
    retriever = UserRetriever(
        domain_x_name=domain_x,
        domain_y_name=domain_y,
    )

    # 加载数据划分
    train_data = load_json(get_processed_dir() / "train.json")
    test_data = load_json(get_processed_dir() / "test.json")

    # 批量检索 (valid 无 DG 特征向量, 跳过)
    logger.info(f"训练集检索 (k={k_train}, 排除自身)...")
    train_retrieval = retriever.retrieve_for_split(
        train_data, "train", k=k_train
    )

    logger.info(f"测试集检索 (k={k_test})...")
    test_retrieval = retriever.retrieve_for_split(
        test_data, "test", k=k_test
    )

    # 保存结果
    retrieval_results = {
        "train": train_retrieval,
        "valid": {},
        "test": test_retrieval,
        "metadata": {
            "k_train": k_train,
            "k_test": k_test,
            "using_real_vectors": retriever._using_real_vectors,
            "num_train_samples": len(train_data),
            "vector_dim": retriever.train_user_x.shape[1],
            "domain_x": domain_x,
            "domain_y": domain_y,
            "note": "valid 无 DG 特征向量, 未检索",
        },
    }
    save_retrieval_results(retrieval_results)

    logger.info("=" * 60)
    logger.info("KNN 用户检索完成!")
    logger.info(f"  训练集: {len(train_retrieval)} 条")
    logger.info(f"  测试集: {len(test_retrieval)} 条")
    logger.info(f"  向量来源: DG 模型真实用户输出")
    logger.info("=" * 60)

    return retriever


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    from data_utils import set_dataset
    ds = cfg.get("dataset", {})
    set_dataset(ds.get("name", "AO"), ds.get("dg_root"))
    run_user_retrieval(cfg)
