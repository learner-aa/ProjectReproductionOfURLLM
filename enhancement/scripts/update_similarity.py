"""
更新前端 datasets.json 中的用户相似度分数。

根因: 24个测试用户之间没有任何共同物品,
基于物品ID的Jaccard/余弦相似度全为0。
原论文使用 DG 模型产出的语义向量计算相似度,
即使没有共同物品也能得到非零的语义相似度。

本脚本:
1. 加载 GM 数据集的 DG 测试用户向量 (test_x_fea, test_y_fea)
2. 用 X 域和 Y 域向量的平均值作为综合用户向量
3. 计算显示用户与相似用户之间的余弦相似度
4. 更新 datasets.json 中的 similarUsers 和 retrievalPool 的 similarity 字段
"""

import json
import numpy as np
from pathlib import Path

# 路径配置
PROJECT_ROOT = Path("/root/autodl-tmp/URLLM-project")
DATASETS_JSON = PROJECT_ROOT / "webapp/src/data/datasets.json"
TEST_JSON = PROJECT_ROOT / "enhancement/data/processed/test.json"
TEST_X_FEA = PROJECT_ROOT / "enhancement/saved_models/DGGM_final_test_x_fea.npy"
TEST_Y_FEA = PROJECT_ROOT / "enhancement/saved_models/DGGM_final_test_y_fea.npy"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def main():
    # 1. 加载 DG 测试用户向量
    test_x = np.load(str(TEST_X_FEA))  # (3601, 656)
    test_y = np.load(str(TEST_Y_FEA))  # (3601, 656)
    print(f"加载 DG 向量: test_x shape={test_x.shape}, test_y shape={test_y.shape}")

    # 综合用户向量 = X域和Y域向量的平均值
    test_combined = (test_x + test_y) / 2.0

    # 2. 加载 test.json 建立用户ID到索引的映射
    with open(TEST_JSON) as f:
        test_data = json.load(f)
    test_keys = list(test_data.keys())
    user_id_to_idx = {uid: idx for idx, uid in enumerate(test_keys)}
    print(f"测试用户数: {len(test_keys)}")

    # 3. 加载 datasets.json
    with open(DATASETS_JSON) as f:
        datasets = json.load(f)

    # 4. 更新每个用户的相似度
    total_updated = 0
    for ds in datasets:
        for user in ds["users"]:
            user_id = user["id"].replace("User#", "")
            user_idx = user_id_to_idx.get(user_id)
            if user_idx is None:
                print(f"警告: 用户 {user_id} 不在 test.json 中,跳过")
                continue

            user_vec = test_combined[user_idx]

            # 更新 similarUsers
            similar_users = user.get("result", {}).get("similarUsers", [])
            for sim_user in similar_users:
                sim_uid = sim_user["id"].replace("User#", "")
                sim_idx = user_id_to_idx.get(sim_uid)
                if sim_idx is None:
                    print(f"  警告: 相似用户 {sim_uid} 不在 test.json 中,保持原值")
                    continue
                sim_vec = test_combined[sim_idx]
                sim_score = cosine_similarity(user_vec, sim_vec)
                # 保留4位小数
                sim_user["similarity"] = round(sim_score, 4)
                total_updated += 1

            # 更新 retrievalPool
            for pool_user in user.get("retrievalPool", []):
                sim_uid = pool_user["id"].replace("User#", "")
                sim_idx = user_id_to_idx.get(sim_uid)
                if sim_idx is None:
                    print(f"  警告: 检索池用户 {sim_uid} 不在 test.json 中,保持原值")
                    continue
                sim_vec = test_combined[sim_idx]
                sim_score = cosine_similarity(user_vec, sim_vec)
                pool_user["similarity"] = round(sim_score, 4)
                total_updated += 1

    # 5. 保存更新后的 datasets.json
    with open(DATASETS_JSON, "w", encoding="utf-8") as f:
        json.dump(datasets, f, ensure_ascii=False, indent=2)

    print(f"\n更新完成! 共更新 {total_updated} 个相似度分数")

    # 6. 打印一些样本验证
    print("\n=== 样本验证 ===")
    for ds in datasets:
        for user in ds["users"][:2]:
            print(f"\n用户 {user['id']}:")
            similar = user.get("result", {}).get("similarUsers", [])
            for su in similar[:3]:
                print(f"  相似用户 {su['id']}: similarity={su['similarity']}")
            pool = user.get("retrievalPool", [])
            for pu in pool[:3]:
                print(f"  检索池 {pu['id']}: similarity={pu['similarity']}")


if __name__ == "__main__":
    main()
