# 项目结构说明

## 目录总览

```
enhancement/
├── README.md                              # 使用指南
├── struct.md                              # 本文件 - 结构说明
├── requirements.txt                       # Python 依赖清单
├── 2406.03085v1.pdf                       # 参考论文
│
├── config/                                # 配置文件 (AO/GM 共用一份)
│   ├── pipeline_config.yaml                # Pipeline 全局配置 (切换 dataset.name 选数据集)
│   ├── lora_config.yaml                    # LoRA 微调配置 (通用)
│   └── server_env.yaml                     # 服务器环境说明 (通用)
│
├── src/                                   # 源代码
│   ├── __init__.py                        # 包初始化
│   ├── data_utils.py                      # 数据 I/O 工具
│   ├── preprocess.py                      # DG 产物预处理
│   ├── attribute_extraction.py            # 物品属性提取 (含预计算复用)
│   ├── prompt_templates.py                # Prompt 模板管理
│   ├── user_profile_builder.py            # 用户画像构建 (核心)
│   ├── knn_retriever.py                   # 用户检索 (DG 向量)
│   ├── build_instruction_data.py          # Instruction 数据集构建
│   ├── llm_finetune.py                    # LoRA 微调
│   ├── llm_inference.py                   # LLM 推理
│   ├── answer_refinement.py               # Answer Refinement (BM25 落地)
│   ├── evaluate.py                        # 评估指标计算
│   └── run_pipeline.py                    # 主流程编排
│
├── scripts/                               # Shell 运行脚本
│   ├── setup_env.sh                       # 环境安装
│   ├── run_preprocess.sh                  # 数据预处理
│   ├── run_finetune.sh                    # LoRA 微调
│   ├── run_inference.sh                   # LLM 推理
│   └── run_eval.sh                        # 评估
│
├── data/                                  # 数据目录 (运行时生成)
│   └── processed/                         # 处理后数据 (按数据集隔离: AO/ GM/, 各含下方完整文件清单)
│       ├── interactions.json              # 用户交互序列
│       ├── item_metadata.json             # 物品元数据
│       ├── item_attributes.json           # 物品属性 (复用 DG 预提取)
│       ├── id_mapping.json                # 物品 ID 映射表 (DG 索引空间)
│       ├── train.json                     # 训练集
│       ├── valid.json                     # 验证集
│       ├── test.json                      # 测试集
│       ├── user_profiles.json             # 用户画像
│       ├── test_embedding_profiles.json   # 测试用户特征画像
│       ├── retrieval_results.json         # 用户检索结果
│       ├── train_instructions.json        # 训练 Instruction 数据
│       ├── valid_instructions.json        # 验证 Instruction 数据
│       └── test_instructions.json         # 测试 Instruction 数据
│
└── outputs/                               # 输出目录 (按数据集隔离: AO/ GM/, 各含下方完整子结构)
    ├── lora_weights/                      # LoRA adapter 权重
    │   └── final/                         # 最终模型
    ├── predictions/                       # LLM 推理结果
    │   └── test_predictions.json
    ├── refined_predictions/               # Answer Refinement 结果
    │   └── refined_predictions.json
    ├── eval_results/                      # 评估结果
    │   └── evaluation.json
    ├── logs/                              # 训练日志 (TensorBoard)
    └── pipeline.log                       # Pipeline 运行日志
```

## 模块详解

### 1. `src/data_utils.py` — 数据 I/O 工具

**职责**: 提供统一的数据加载/保存接口

**主要函数**:
| 函数 | 说明 |
|------|------|
| `load_dg_features()` | 加载 DG 模型的 4 个特征文件 (train/test × X/Y) |
| `load_dg_scores()` | 加载 DG 测试评分矩阵 |
| `load_dg_candidates()` | 加载 DG 候选物品矩阵 |
| `load_dg_config()` | 加载 DG 训练配置 |
| `load_json()` / `save_json()` | 通用 JSON I/O |
| `load_interactions()` | 加载用户交互序列 |
| `load_item_metadata()` | 加载物品元数据 |
| `load_item_attributes()` | 加载 LLM 提取的物品属性 |
| `load_id_mapping()` | 加载物品 ID 映射表 |
| `cosine_similarity()` | 余弦相似度计算 |
| `find_topk_similar()` | 查找 top-k 最相似项 |

**依赖**: numpy, json, pathlib

---

### 2. `src/preprocess.py` — DG 产物预处理

**职责**: 解析 DG 已对齐产物 (txt/csv/exat json)，构建训练/验证/测试集，ID 空间与 DG 索引对齐

**主要函数**:
| 函数 | 说明 |
|------|------|
| `parse_item_lists()` | 解析物品清单 CSV (idAfter = DG 索引) |
| `_target_of_a_line()` | 提取目标物品 (AO: last / GM: ts 模式) |
| `parse_split()` | 解析单个划分 txt, 产出 {sample_id: {dg_index, seq, target, domain}} |
| `load_precomputed_attributes()` | 从 DG 预提取 exat json 读取物品属性 |
| `merge_csv_attributes()` | CSV 属性列回填缺失物品 |
| `preprocess()` | 主流程入口 (含 set_dataset 上下文) |

**输入**: `{dg_root}/{dataset}/item_list*.csv`, `{train,valid,test}_F*.txt`, `*_exat_*.json`
**输出**: `data/processed/interactions.json`, `item_metadata.json`, `item_attributes.json`, `id_mapping.json`, `train/valid/test.json`

---

### 3. `src/attribute_extraction.py` — LLM 物品属性提取

**职责**: 利用 LLM 的 Chain-of-Thought 能力提取物品结构化属性

**主要类/函数**:
| 类/函数 | 说明 |
|---------|------|
| `APIClient` | OpenAI 兼容 API 客户端 (GPT/通义千问) |
| `LocalModelClient` | 本地 transformers 模型客户端 |
| `parse_attribute_response()` | 解析 LLM 返回的属性文本 |
| `extract_attributes_single()` | 提取单个物品的属性 |
| `extract_all_attributes()` | 批量提取 (支持断点续传) |
| `run_attribute_extraction()` | 主流程入口 |

**输入**: `data/processed/item_metadata.json`
**输出**: `data/processed/item_attributes.json`
**依赖**: openai (API模式) 或 transformers (本地模式)

---

### 4. `src/prompt_templates.py` — Prompt 模板管理

**职责**: 集中管理所有 LLM Prompt 模板

**模板列表**:
| 模板名 | 用途 |
|--------|------|
| `PROMPT_I_ATTRIBUTE_EXTRACTION_EN/ZH` | 物品属性提取 (COT) |
| `PROMPT_I_BATCH` | 批量属性提取 |
| `PROMPT_II_RECOMMEND_BASE` | 基础推荐 prompt |
| `PROMPT_II_RECOMMEND_WITH_PROFILE` | 画像增强推荐 prompt |
| `PROMPT_II_RECOMMEND_COT` | CoT 推荐 prompt |
| `PROFILE_TEMPLATE_COMPACT` | 紧凑画像模板 |
| `PROFILE_TEMPLATE_DETAILED` | 详细画像模板 |

**辅助函数**: `format_interaction_sequence()`, `format_attributes()`, `get_prompt_template()`

---

### 5. `src/user_profile_builder.py` — 用户画像构建 (核心)

**职责**: 从三个维度构建 LLM 可理解的用户画像

**三维度画像**:
| 维度 | 构建函数 | 数据来源 |
|------|----------|----------|
| 行为画像 | `build_behavior_profile()` | 交互序列统计 |
| 语义画像 | `build_semantic_profile()` | 物品属性聚合 |
| 特征画像 | `build_embedding_profile()` | DG 656维向量→近邻物品 |

**主要函数**:
| 函数 | 说明 |
|------|------|
| `build_behavior_profile()` | 统计交互频次、域偏好、跨域比率 |
| `build_semantic_profile()` | 聚合属性标签频次，提取 top 偏好 |
| `build_embedding_profile()` | 用户向量与训练用户向量相似, 取其交互物品作线索 |
| `profile_to_text()` | 将画像字典转化为 LLM 可读文本 |
| `build_all_user_profiles()` | 批量构建全量用户画像 |
| `build_embedding_profiles_for_test_users()` | 为测试用户构建特征画像 |
| `run_profile_building()` | 主流程入口 |

**输入**: `interactions.json`, `item_metadata.json`, `item_attributes.json`, DG 特征
**输出**: `user_profiles.json`, `test_embedding_profiles.json`

---

### 6. `src/build_instruction_data.py` — Instruction 数据构建

**职责**: 将用户画像 + 交互序列转化为 LLM 微调格式

**主要函数**:
| 函数 | 说明 |
|------|------|
| `build_single_instruction()` | 为单个用户构建一条 Instruction |
| `build_instruction_dataset()` | 批量构建 Instruction 数据集 |
| `convert_to_chatml()` | Alpaca → ChatML 格式转换 |
| `build_all_instruction_data()` | 主流程入口 |

**支持格式**:
- **Alpaca**: `{"instruction": str, "input": str, "output": str}`
- **ChatML**: `{"messages": [{"role": "system/user/assistant", "content": str}]}`

**输入**: `train/valid/test.json`, `user_profiles.json`
**输出**: `train/valid/test_instructions.json`

---

### 7. `src/llm_finetune.py` — LoRA 微调

**职责**: 对基座 LLM 进行参数高效微调

**主要函数**:
| 函数 | 说明 |
|------|------|
| `load_instruction_dataset()` | 加载并 tokenize Instruction 数据 |
| `load_model_and_tokenizer()` | 加载基座模型 + 应用 LoRA |
| `train()` | 主训练流程 |

**训练流程**: 加载模型 → 准备 k-bit 训练 → 应用 LoRA → Tokenize 数据 → Trainer 训练 → 保存

**输入**: `train_instructions.json`, `valid_instructions.json`
**输出**: `outputs/lora_weights/final/`

---

### 8. `src/llm_inference.py` — LLM 推理

**职责**: 加载微调模型，批量生成推荐

**主要类/函数**:
| 类/函数 | 说明 |
|---------|------|
| `LLMRecommender` | LLM 推理器 (支持单条和批量) |
| `LLMRecommender.generate()` | 单条推理 |
| `LLMRecommender.generate_batch()` | 批量推理 |
| `build_test_prompts()` | 从 Instruction 构建 prompt |
| `run_inference()` | 主流程入口 |

**输入**: `test_instructions.json`, `outputs/lora_weights/final/`
**输出**: `outputs/predictions/test_predictions.json`

---

### 9. `src/evaluate.py` — 评估

**职责**: 计算推荐指标，对比基线，分析冷/热启动

**评估指标**:
| 指标 | 函数 |
|------|------|
| HR@K (Hit Rate) | `hit_rate_at_k()` |
| NDCG@K | `ndcg_at_k()` |
| MRR | `mean_reciprocal_rank()` |
| 模糊匹配 | `fuzzy_match_score()`, `compute_fuzzy_metrics()` |
| 冷/热启动分析 | `analyze_cold_warm_start()` |
| DG 基线对比 | `evaluate_dg_baseline()` |
| 域外率统计 | `compute_out_of_domain_rate()` |
| 完整评估 | `run_evaluation()` |

**输入**: `outputs/predictions/test_predictions.json`
**输出**: `outputs/eval_results/evaluation.json`

---

### 10. `src/run_pipeline.py` — 主流程编排

**职责**: 提供一键运行和分阶段执行能力

**9 个执行阶段**:
| # | 阶段 | 说明 |
|---|------|------|
| 1 | `preprocess` | DG 产物预处理 |
| 2 | `extract_attributes` | 物品属性提取 (复用预计算) |
| 3 | `build_profiles` | 用户画像构建 |
| 4 | `user_retrieval` | 用户检索 (DG 向量 KNN) |
| 5 | `build_instructions` | Instruction 数据构建 |
| 6 | `finetune` | LoRA 微调 |
| 7 | `inference` | LLM 推理 |
| 8 | `refinement` | Answer Refinement (BM25 落地) |
| 9 | `evaluate` | 评估 |

**命令行参数**:
```
--stage <name>     只运行指定阶段
--until <name>     运行到指定阶段为止
--config <path>    自定义配置文件
--dry-run          预览不执行
```

## 数据流图

```
{dg_root}/{dataset} (DG 产物: csv/txt/exat_npy)
    │
    ▼
preprocess.py ──→ data/processed/
    │                  ├── interactions.json
    │                  ├── item_metadata.json
    │                  ├── item_attributes.json
    │                  ├── id_mapping.json
    │                  └── train/valid/test.json
    │
    ▼
attribute_extraction.py ──→ item_attributes.json (复用预计算)
    │
    ▼
user_profile_builder.py ──→ user_profiles.json, test_embedding_profiles.json
    │
    ▼
knn_retriever.py ──→ retrieval_results.json
    │
    ▼
build_instruction_data.py ──→ train/valid/test_instructions.json
    │
    ▼
llm_finetune.py ──→ outputs/lora_weights/final/
    │
    ▼
llm_inference.py ──→ outputs/predictions/test_predictions.json
    │
    ▼
answer_refinement.py ──→ outputs/refined_predictions/refined_predictions.json
    │
    ▼
evaluate.py ──→ outputs/eval_results/evaluation.json
```
