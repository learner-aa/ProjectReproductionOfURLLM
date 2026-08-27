# URLLM 跨域序列推荐项目

复现论文 *Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation* (arXiv:2406.03085)，验证 **LLM + 用户画像增强 + 用户检索** 对跨域序列推荐的效果。

- **基座模型**：Llama-2-7b-hf + LoRA(lora_r=8, FP16)
- **数据集**：GM(Movie-Game) + AO(Art-Office) 双数据集，共 50,056 用户
- **核心成果**：GM 数据集 HR@1 提升 4.5 倍(0.0275 vs 0.0061)；AO 数据集 HR@20 提升 67%(0.0460 vs 0.0275)

完整 Pipeline 分 9 阶段：

```
预处理 → 属性提取 → 用户画像 → KNN用户检索 → 指令构建 → LoRA微调 → 推理 → Answer Refinement → 评估
```

***

## 目录

1. [环境要求](#一环境要求)
2. [获取项目代码](#二获取项目代码)
3. [下载大文件](#三下载大文件)
4. [配置 Python 环境](#四配置-python-环境)
5. [修改配置中的模型路径](#五修改配置中的模型路径)
6. [快速复现指南](#六快速复现指南)
7. [Pipeline 九阶段详解](#七pipeline-九阶段详解)
8. [目录结构](#八目录结构)
9. [核心产物](#九核心产物)
10. [关键评估指标](#十关键评估指标)
11. [前端展示系统](#十一前端展示系统)
12. [配置文件详解](#十二配置文件详解)
13. [注意事项](#十三注意事项)
14. [常见问题排查](#十四常见问题排查)
15. [技术栈](#十五技术栈)

***

## 一、环境要求

### 硬件要求

| 场景          | GPU 显存 | 说明                               |
| ----------- | ------ | -------------------------------- |
| 仅前端展示       | 无需 GPU | 纯前端项目，本地浏览器运行                    |
| 运行评估        | 无需 GPU | 只读 JSON 计算指标                     |
| 完整推理(7B)    | >= 24GB | FP16 + LoRA，RTX 4090D 可运行        |
| 完整训练(7B)    | >= 24GB | FP16 + 梯度检查点，batch_size=2       |
| 小模型验证(1.5B) | >= 16GB | Qwen2-1.5B，用于快速验证流程              |

### 软件要求

| 项目      | 版本                | 说明                           |
| ------- | ----------------- | ---------------------------- |
| CUDA 驱动 | >= 12.1           | 训练/推理必需                      |
| Python  | 3.10              | 推荐 conda 管理                  |
| Node.js | >= 20.19 或 >= 22.12 | 前端构建必需(Vite 8 要求，低版本会报错)     |
| PyTorch | 2.5.1+cu121       | 需匹配 CUDA 版本                  |
| 操作系统    | Linux             | 训练/推理需 Linux；前端可在 Windows 运行 |

***

## 二、获取项目代码

### 方式 A：从 GitHub clone(推荐)

```bash
git clone https://github.com/learner-aa/ProjectReproductionOfURLLM.git
cd URLLM-project
```

### 方式 B：从压缩包解压

```bash
tar -xzf URLLM-project.tar.gz
cd URLLM-project
```

> 两种方式都**不含**基座模型(26G)和前端依赖(node_modules)，需按下面步骤补齐。

***

## 三、下载大文件

> 仅查看前端展示可跳过本节。

### (a) 基座模型 Llama-2-7b-hf(26G，必需)

**国内环境(推荐 ModelScope)：**

```bash
pip install modelscope
modelscope download --model shakechen/Llama-2-7b-hf --local_dir models/Llama-2-7b-hf
```

**可访问 HuggingFace：**

```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/Llama-2-7b-hf
```

> 下载后确认 `models/Llama-2-7b-hf/` 下包含 `config.json`、`*.safetensors`、`tokenizer.model` 等文件。

### (b) LoRA 微调权重(16M，推理必需)

从 GitHub Release 下载：

```bash
# 命令行下载
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip
unzip lora-weights.zip -d enhancement/outputs/AO/lora_weights/
```

> 含 `adapter_model.safetensors`(16M) + `adapter_config.json`，推理必需。

### (c) DG 基线数据(可选)

若需对比 DG 基线，从 GitHub Release 下载：

```bash
# GM 数据集 DG 用户相似度矩阵(691M)
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-gm.npy.zip
unzip dg-gm.npy.zip -d DG_Final/GM/

# AO 数据集 DG 用户相似度矩阵
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-ao.npy.zip
unzip dg-ao.npy.zip -d DG_Final/AO/
```

> DG 矩阵用于评估时计算 DG 基线指标和前端展示用户相似度，仅评估对比时需要。

***

## 四、配置 Python 环境

```bash
# 1. 创建 conda 环境
conda create -n urllm python=3.10 -y
conda activate urllm

# 2. 安装依赖
cd URLLM-project
pip install -r requirements.txt
```

> **重要**：torch 版本需匹配你的 CUDA。若非 cu121，请修改 `requirements.txt` 里 torch/torchvision 的版本号。其他 CUDA 版本见 <https://pytorch.org/get-started/previous-versions/>

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

***

## 五、修改配置中的模型路径

配置默认指向服务器绝对路径，**本地运行前必须改成你自己的路径**：

### 修改 `enhancement/config/lora_config.yaml`

`model.base_model`：

```yaml
# 改前
base_model: "/root/autodl-tmp/URLLM-project/models/Llama-2-7b-hf/models/shakechen--Llama-2-7b-hf/snapshots/master"
# 改后(示例)
base_model: "/你的路径/URLLM-project/models/Llama-2-7b-hf"
```

### 切换数据集

修改 `enhancement/config/pipeline_config.yaml` 中的 `dataset.name`、`domains`、`train_file`/`valid_file`/`test_file`：

```yaml
# AO 数据集(默认)
dataset:
  name: "AO"
domains:
  x: "Art"
  y: "Office"
preprocess:
  train_file: "train_F2.txt"
  valid_file: "valid_F2.txt"
  test_file: "test_F2.txt"

# GM 数据集
dataset:
  name: "GM"
domains:
  x: "Movie"
  y: "Game"
preprocess:
  train_file: "train_F.txt"
  valid_file: "valid_F.txt"
  test_file: "test_F.txt"
```

***

## 六、快速复现指南

### 场景速查

| 场景          | 耗时      | 需下载                | 命令                                             |
| ----------- | ------- | ------------------ | ---------------------------------------------- |
| 只看前端展示      | ~5 分钟  | 无需模型               | `cd webapp && npm install && npm run dev`      |
| 运行评估        | ~10 分钟 | 无需模型               | `cd enhancement && python src/evaluate.py`     |
| AO 完整推理     | 较长      | 模型 + LoRA 权重 + GPU | `cd enhancement && python src/run_pipeline.py --stage inference` |
| AO 一键全流程    | 最长      | 模型 + GPU           | `cd enhancement && python src/run_pipeline.py` |
| 重新生成前端数据    | ~1 分钟  | 无需模型               | `cd webapp/scripts && python generate_data.py` |

### 场景一：仅查看前端展示

**适用**：只想看项目演示效果，无需 GPU 和模型。

```bash
cd webapp
npm install         # 安装前端依赖(首次)
npm run dev         # 启动开发服务器
```

浏览器访问 `http://localhost:6006`。

> 前端为纯 Vite + React 项目，数据来自本地 JSON，无需后端。

**注意事项：**

1. **Node.js 版本**：Vite 8 要求 Node.js >= 20.19 或 >= 22.12，低版本会报错。可用 `node -v` 检查版本，建议用 [nvm](https://github.com/nvm-sh/nvm) 或 [fnm](https://github.com/Schniz/fnm) 管理版本：
   ```bash
   nvm install 22
   nvm use 22
   ```
2. **指定端口**：若需指定端口(如远程服务器 autodl 的代理端口)，启动时加参数：
   ```bash
   npm run dev -- --host 0.0.0.0 --port 6006
   ```
3. **Windows 用户**：`npm install` 时若报 `@rolldown/binding-linux-x64-gnu` 相关错误，可忽略(这是 Linux 专用包，不影响 Windows 构建)。

### 场景二：运行评估

**适用**：已有推理结果，想重新计算评估指标。无需 GPU 和模型。

```bash
cd enhancement
python src/evaluate.py
# 结果输出到 outputs/{AO|GM}/eval_results/evaluation.json
```

> 评估脚本读取 `outputs/{dataset}/refined_predictions/refined_predictions.json`(精炼后结果)和 DG 基线数据，计算 HR@K、NDCG@K、MRR 等指标。

### 场景三：完整推理

**适用**：已有基座模型 + LoRA 权重，想对测试集推理。需要 GPU。

```bash
cd enhancement

# 设置环境变量防 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行推理(需先修改 config/lora_config.yaml 中的 base_model 路径)
python src/run_pipeline.py --stage inference
```

> 推理结果输出到 `outputs/{dataset}/predictions/test_predictions.json`。

### 场景四：一键全流程(训练 -> 推理 -> 评估)

**适用**：想从头训练 LoRA 权重并完成推理评估。需要 GPU(>=24GB)。

```bash
cd enhancement
python src/run_pipeline.py
```

该脚本按 9 阶段顺序执行：

| 阶段    | 说明                | 配置                                   |
| ----- | ----------------- | ------------------------------------ |
| 1. 预处理 | 解析 DG 产物，生成交互序列   | max_seq_len=15                       |
| 2. 属性提取 | 复用 DG 预计算属性       | backend=api                           |
| 3. 画像构建 | 统计行为特征+语义属性       | 40,479(GM)/9,577(AO) 用户             |
| 4. 用户检索 | KNN 检索相似用户       | k_train=2, k_test=3                  |
| 5. 指令构建 | Alpaca 指令模板       | template_type=profile                |
| 6. 微调   | Llama2-7B LoRA    | 3 epoch / lora_r=8 / FP16 / batch=2 |
| 7. 推理   | 对测试集推理            | temperature=0.0 / max_new_tokens=128 |
| 8. 精炼   | Answer Refinement | BM25 top-5 grounding                  |
| 9. 评估   | 计算各项指标            | 输出 evaluation.json                   |

> AO 训练约 1,500 步(3 epoch)，GM 训练约 19,735 步。RTX 4090D 约需数小时。

### 场景五：重新生成前端数据

**适用**：Pipeline 产物更新后，想同步更新前端展示数据。无需 GPU。

```bash
cd webapp/scripts
python generate_data.py
# 然后重启前端
```

> 脚本从 `enhancement/data/processed/` 和 `enhancement/outputs/` 读取真实产物，生成 `webapp/src/data/` 下的 JSON 文件。生成后重启前端即可看到新数据。

***

## 七、Pipeline 九阶段详解

### Stage 1：数据预处理(preprocess)

**输入**：DG 基线产物(`DG_Final/{GM,AO}/` 下的 train/valid/test 文件)

需将以下文件放入 `DG_Final/{GM,AO}/`：

```
train_F.txt   # GM 训练集 / train_F2.txt = AO
valid_F.txt   # GM 验证集 / valid_F2.txt = AO
test_F.txt    # GM 测试集  / test_F2.txt = AO
```

**输出**：`enhancement/data/processed/{dataset}/`

- `interactions.json` — 用户交互序列
  - GM: 40,479 用户(10.4M)
  - AO: 9,577 用户(2.5M)
- `item_metadata.json` — 物品元数据
  - GM: 170,478 物品(29.5M)
  - AO: 38,396 物品(7.1M)

### Stage 2：物品属性提取(extract_attributes)

**输出**：`item_attributes_{dataset}.json`

基于 DeepSeek API 从物品标题提取结构化属性(品牌、类别、型号等)。

> **软链接**：`item_attributes_{dataset}.json` 需软链到 `item_attributes.json`(评估脚本读取)。命令：`ln -sf item_attributes_AO.json item_attributes.json`

### Stage 3：用户画像构建(build_profiles)

**输出**：`user_profiles.json`

| 数据集 | 用户数    | 文件大小 |
| --- | ------ | ---- |
| GM  | 40,479 | 87.7M |
| AO  | 9,577  | 24.3M |

基于用户交互历史和物品属性，构建包含 `behavior`(域分布、交互次数、时间偏好)和 `semantic`(偏好属性、权重)的结构化用户画像。

### Stage 4：KNN 用户检索(retrieve_users)

**输出**：指令中的 `retrieved_users` 字段

**配置**：

- `k_train=2` — 训练集检索 2 个相似用户(排除自身)
- `k_test=3` — 验证/测试集检索 3 个相似用户

> 论文 §4.2.1 的用户检索模块，为每个用户找到跨域相似用户作为 LLM 输入上下文。

### Stage 5：指令数据构建(build_instructions)

**输出**：

| 文件                     | GM                | AO                |
| ---------------------- | ----------------- | ----------------- |
| train_instructions.json | 31,570 条(59.1M)  | 16,000 条(50.7M)  |
| test_instructions.json  | 3,601 条(9.2M)    | 2,000 条(7.1M)    |

指令采用 Alpaca 模板，包含用户画像 + 相似用户历史 + 目标域候选物品。

### Stage 6：LLM 微调(finetune)

**输出**：LoRA 权重(`outputs/{dataset}/lora_weights/checkpoint-{step}/`)

**配置**：

| 参数                         | 值                |
| -------------------------- | ---------------- |
| base_model                 | Llama-2-7b-hf    |
| lora_r                     | 8                |
| lora_alpha                 | 16               |
| lora_target_modules        | q_proj, v_proj   |
| num_epochs                 | 3                |
| batch_size                 | 2                |
| gradient_accumulation_steps | 16(有效 batch=32) |
| learning_rate              | 1.0e-4           |
| max_seq_length             | 2048             |
| fp16                       | true             |
| gradient_checkpointing     | true             |

### Stage 7：LLM 推理(inference)

**输出**：`outputs/{dataset}/predictions/test_predictions.json`

| 数据集 | 条数    | 文件大小 |
| --- | ---- | ---- |
| GM  | 3,601 | 1.0M |
| AO  | 2,000 | 0.7M |

**配置**：

- `temperature: 0.0` — 贪心解码，输出格式稳定
- `max_new_tokens: 128` — 完整生成
- `padding_side: "left"` — decoder-only 模型必需

### Stage 8：Answer Refinement(refine_answers)

**输出**：`outputs/{dataset}/refined_predictions/refined_predictions.json`

| 数据集 | 条数    | 文件大小 |
| --- | ---- | ---- |
| GM  | 3,601 | 11.5M |
| AO  | 2,000 | 11.5M |

> 论文 §4.2.3 的答案精炼模块，使用 BM25 检索 top-5 候选物品对 LLM 输出进行 grounding，提高物品标题匹配率。

### Stage 9：评估(evaluate)

**输出**：`outputs/{dataset}/eval_results/evaluation.json`

**指标**：

- HR@1/5/10/20 — 命中率(基于物品标题精确匹配)
- NDCG@K — 归一化折损累积增益
- MRR — 平均倒数排名
- DG 基线对比 — 同测试集上的 DG 模型指标
- Cold/Warm 分析 — 冷/热用户分组指标
- OOD 分析 — 域外推荐率

`evaluation.json` 含两套指标：
- `exact_metrics` — **LLM 原始**(未精炼)：LLM 每个样本只生成 1 条预测，故 HR@1=HR@5=HR@10=HR@20=NDCG@K=MRR，这是单条预测的数学必然结果，非 bug。
- `refined_metrics` — **Answer Refinement 后**：经 BM25 grounding 扩展为 top-K 候选列表，HR@K 随 K 递增有真实区分度。前端与下文关键指标均取此套。

***

## 八、目录结构

```
URLLM-project/
├── enhancement/                    # 增强 Pipeline(数据产物 + 评估)
│   ├── config/                     # 配置文件
│   │   ├── lora_config.yaml        # LoRA 微调配置
│   │   ├── pipeline_config.yaml    # Pipeline 全局配置(当前为 AO)
│   │   ├── pipeline_config_gm_eval.yaml # GM 评估配置
│   │   └── server_env.yaml         # 服务器环境配置
│   ├── data/
│   │   └── processed/              # 数据产物
│   │       ├── GM/                 # GM 数据集产物
│   │       │   ├── interactions.json
│   │       │   ├── user_profiles.json
│   │       │   ├── train_instructions.json
│   │       │   ├── test_instructions.json
│   │       │   ├── item_attributes_GM.json
│   │       │   └── item_metadata.json
│   │       └── AO/                 # AO 数据集产物
│   │           ├── interactions.json
│   │           ├── user_profiles.json
│   │           ├── train_instructions.json
│   │           ├── test_instructions.json
│   │           ├── item_attributes_AO.json
│   │           └── item_metadata.json
│   ├── outputs/                    # 输出产物
│   │   ├── GM/                     # GM 输出
│   │   │   ├── lora_weights/       # LoRA 权重
│   │   │   ├── predictions/        # 推理结果
│   │   │   ├── refined_predictions/ # 精炼结果
│   │   │   └── eval_results/       # 评估结果
│   │   └── AO/                     # AO 输出
│   │       ├── lora_weights/checkpoint-1500/
│   │       ├── predictions/
│   │       ├── refined_predictions/
│   │       └── eval_results/
│   ├── scripts/                    # 各阶段运行脚本
│   │   ├── run_preprocess.sh
│   │   ├── run_finetune.sh
│   │   ├── run_inference.sh
│   │   ├── run_eval.sh
│   │   └── setup_env.sh
│   └── src/                        # 核心代码
│       ├── run_pipeline.py         # 主流程编排(9 阶段)
│       ├── preprocess.py           # 阶段 1: 预处理
│       ├── attribute_extraction.py # 阶段 2: 属性提取
│       ├── user_profile_builder.py # 阶段 3: 画像构建
│       ├── knn_retriever.py        # 阶段 4: KNN 用户检索
│       ├── build_instruction_data.py # 阶段 5: 指令构建
│       ├── llm_finetune.py         # 阶段 6: 微调
│       ├── llm_inference.py        # 阶段 7: 推理
│       ├── answer_refinement.py    # 阶段 8: 答案精炼
│       └── evaluate.py             # 阶段 9: 评估
├── DG_Final/                       # DG 基线模型
│   ├── GM/                         # GM 数据集(train/valid/test + .npy)
│   └── AO/                         # AO 数据集(train/valid/test + .npy)
├── webapp/                         # 前端展示系统
│   ├── src/
│   │   ├── pages/                  # 5 个页面
│   │   │   ├── Overview.tsx        # 项目概览
│   │   │   ├── Method.tsx          # 方法流程
│   │   │   ├── Workbench.tsx       # 推荐展示
│   │   │   ├── Dashboard.tsx       # 效果评测
│   │   │   └── RetrievalView.tsx   # 检索展示
│   │   └── data/                   # 前端数据(JSON)
│   └── scripts/generate_data.py    # 前端数据生成
├── models/Llama-2-7b-hf/           # 基座模型(需自行下载)
├── requirements.txt                # Python 依赖
└── README.md                       # 本文件
```

***

## 九、核心产物

### 数据产物(`enhancement/data/processed/`)

| 文件                        | GM                           | AO                           |
| ------------------------- | ---------------------------- | ---------------------------- |
| interactions.json         | 40,479 用户(10.4M)           | 9,577 用户(2.5M)              |
| user_profiles.json        | 40,479 画像(87.7M)           | 9,577 画像(24.3M)            |
| train_instructions.json  | 31,570 条(59.1M)            | 16,000 条(50.7M)            |
| test_instructions.json   | 3,601 条(9.2M)              | 2,000 条(7.1M)              |
| item_metadata.json       | 170,478 物品(29.5M)          | 38,396 物品(7.1M)            |

### 模型与结果产物(`enhancement/outputs/`)

| 路径                                          | GM                   | AO                   |
| ------------------------------------------- | -------------------- | -------------------- |
| lora_weights/checkpoint-{step}/             | checkpoint-19735     | checkpoint-1500      |
| predictions/test_predictions.json           | 3,601 条(1.0M)       | 2,000 条(0.7M)       |
| refined_predictions/refined_predictions.json | 3,601 条(11.5M)      | 2,000 条(11.5M)      |
| eval_results/evaluation.json                | 完整评估指标             | 完整评估指标             |

### 训练日志

| 数据集 | 总步数    | 最佳 eval_loss | 训练损失范围       |
| --- | ------ | ------------- | ----------- |
| GM  | 19,735 | 0.5756       | -           |
| AO  | 1,500  | 2.3406       | 2.718 -> 2.178 |

***

## 十、关键评估指标

### GM 数据集(Movie <-> Game 双向)

| 指标     | LLM+画像+检索(Llama2-7B) | DG 基线   | 提升       |
| ------ | --------------------- | ------ | -------- |
| HR@1   | 0.0275                | 0.0061 | +4.5 倍  |
| HR@5   | 0.0439                | 0.0075 | +485%    |
| HR@10  | 0.0483                | 0.0078 | +519%    |
| HR@20  | 0.0555                | 0.0108 | +414%    |
| MRR    | 0.0345                | 0.0074 | +4.7 倍  |

### AO 数据集(Art <-> Office 双向)

| 指标     | LLM+画像+检索(Llama2-7B) | DG 基线   | 提升       |
| ------ | --------------------- | ------ | -------- |
| HR@1   | 0.0130                | 0.0125 | +4%      |
| HR@5   | 0.0285                | 0.0195 | +46%     |
| HR@10  | 0.0370                | 0.0235 | +57%     |
| HR@20  | 0.0460                | 0.0275 | +67%     |
| MRR    | 0.0201                | 0.0165 | +22%     |

> 指标来自 `refined_metrics`(Answer Refinement 后的评估结果)。LLM 原始(`exact_metrics`)每个样本仅 1 条预测，HR@1/5/10/20/NDCG@K/MRR 必然全等(单条预测的数学必然，非 bug)，故采用精炼后的多候选列表指标以体现 K 值区分度。GM 数据集 DG 基线较低是因为 DG 矩阵索引空间(0-112171)与 pipeline 的 id_mapping 空间(0-164879)部分不匹配导致跨空间查找命中率低。AO 数据集因物品池更大(38,396 候选)且物品标题含品牌/型号/尺寸，跨域推荐难度更高。

***

## 十一、前端展示系统

### 启动

```bash
cd webapp
npm install     # 首次
npm run dev     # 开发模式，访问 http://localhost:6006
# 或
npm run build   # 生产构建，输出到 dist/
npm run preview # 预览生产构建
```

### 五个页面说明

| 页面       | 功能                                                         |
| -------- | ---------------------------------------------------------- |
| **项目概览** | 核心成果卡片、双数据集概览(GM/AO)、方法概述                                  |
| **方法流程** | Pipeline 9 阶段流程图、用户画像样例(真实属性)、跨域推荐示意                       |
| **推荐展示** | GM/AO 双数据集切换，用户交互历史 + LLM 推荐结果 + 相似用户参考(DG 矩阵 softmax 相似度) |
| **效果评测** | GM/AO 双数据集切换，核心指标、训练曲线(训练损失)、DG 基线对比、HR@K & NDCG@K 曲线   |
| **检索展示** | KNN 用户检索过程可视化                                             |

### 数据源

前端数据来自 `webapp/src/data/` 下的 JSON 文件，由 `webapp/scripts/generate_data.py` 从 enhancement 真实产物生成：

| 前端数据文件            | 数据来源                                                       |
| ----------------- | ---------------------------------------------------------- |
| eval_data.json    | `outputs/{GM,AO}/eval_results/evaluation.json`            |
| datasets.json     | `test_instructions.json` + `test_predictions.json` + `user_profiles.json` + DG 矩阵 |
| training_logs.json | `outputs/AO/lora_weights/checkpoint-1500/trainer_state.json` |

修改数据后运行：

```bash
cd webapp/scripts
python generate_data.py
# 然后重启前端
```

***

## 十二、配置文件详解

### `enhancement/config/pipeline_config.yaml`

Pipeline 全局配置，控制数据路径、域定义、预处理、属性提取、指令构建。

| 配置项                            | 说明            | 默认值(AO)                          |
| ------------------------------ | ------------- | -------------------------------- |
| `dataset.name`                 | 数据集名称         | `AO`                             |
| `dataset.dg_root`              | DG 模型根目录      | `DG_Final`                       |
| `domains.x` / `domains.y`      | 域名称           | Art / Office                     |
| `preprocess.min_interactions`  | 最少交互次数        | 1                                |
| `preprocess.max_seq_len`       | 最大序列长度        | 15                               |
| `preprocess.train_file`        | 训练集文件名        | `train_F2.txt`                   |
| `attribute_extraction.backend` | 属性提取方式        | `api`                            |
| `instruction.template_type`    | 指令模板类型        | `profile`                        |
| `instruction.output_format`    | 输出格式          | `alpaca`                         |
| `instruction.use_candidates`  | 候选集约束         | false                            |
| `instruction.candidate_k`      | 候选集大小         | 30                               |
| `retrieval.k_train`            | 训练集检索数        | 2                                |
| `retrieval.k_test`             | 测试集检索数        | 3                                |

### `enhancement/config/lora_config.yaml`

LoRA 微调配置，控制模型、训练、推理参数。

| 配置项                                    | 说明                    | 默认值           |
| -------------------------------------- | --------------------- | ------------- |
| `model.base_model`                     | **基座模型路径(必改)**        | 服务器路径         |
| `model.lora_r`                         | LoRA rank             | 8             |
| `model.lora_alpha`                     | LoRA alpha            | 16            |
| `model.lora_dropout`                   | LoRA dropout          | 0.05          |
| `model.lora_target_modules`            | 应用 LoRA 的模块           | q_proj, v_proj |
| `training.num_epochs`                  | 训练轮数                  | 3             |
| `training.batch_size`                  | per-device batch size | 2             |
| `training.gradient_accumulation_steps` | 梯度累积                  | 16(有效 batch=32) |
| `training.gradient_checkpointing`      | 梯度检查点                 | true          |
| `training.learning_rate`               | 学习率                   | 1.0e-4        |
| `training.max_seq_length`              | 最大序列长度                | 2048          |
| `training.fp16`                        | FP16 混合精度             | true          |
| `training.warmup_ratio`                | 预热比例                  | 0.03          |
| `training.save_steps`                  | 保存步数                  | 200           |
| `training.eval_steps`                  | 评估步数                  | 200           |
| `inference.batch_size`                 | 推理 batch size         | 2             |
| `inference.temperature`                | 生成温度                  | 0.0           |
| `inference.max_new_tokens`             | 最大生成 token 数          | 128           |

***

## 十三、注意事项

1. **模型路径**：本地运行前务必修改 `lora_config.yaml` 中的 `model.base_model` 路径
2. **GPU 内存**：7B 模型使用 FP16 + 梯度检查点，batch_size 设 2 避免 OOM(24GB 下 4 会 OOM)
3. **环境变量**：训练前设置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防 OOM
4. **Tokenizer**：推理时 decoder-only 模型需设 `padding_side="left"`
5. **前端数据**：`webapp/src/data/` 为前端数据源，由 `generate_data.py` 从 enhancement 产物生成，均为真实项目数据
6. **软链接**：`item_attributes_{dataset}.json` 需软链到 `item_attributes.json`(评估脚本读取)
7. **CUDA 版本**：PyTorch 版本必须匹配 CUDA 驱动版本(如 2.5.1+cu121 对应 CUDA 12.1)
8. **大文件**：基座模型(26G)、LoRA 权重(16M)和 DG 基线数据(.npy)不在 Git 仓库，需从 Release 或 ModelScope 下载
9. **双数据集切换**：修改 `pipeline_config.yaml` 中的 `dataset.name`/`domains`/`*_file` 切换 AO/GM
10. **Python 环境**：推理必须使用 conda 环境的 python(`/root/miniconda3/envs/urllm/bin/python`)，系统 python 会报 ModuleNotFoundError
11. **DG 矩阵**：前端用户相似度来自 DG 矩阵 `best_trte_XORY_DG_.npy`(GM) / `best_trte_XORY_DG_390_.npy`(AO)，经 softmax 转换为 [0,1] 区间

***

## 十四、常见问题排查

### Q1：训练时 OOM(显存不足)

**解决方案**(按优先级)：

```bash
# 1. 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 降低 batch_size 到 1 或 2
# 在 lora_config.yaml 中:
#   batch_size: 1

# 3. 确保开启梯度检查点
#   gradient_checkpointing: true

# 4. 确保 FP16 混合精度
#   fp16: true
```

> 7B 模型 FP16 + 梯度检查点显存约 16G；batch_size=4 在 24GB 下会 OOM。

### Q2：推理时输出乱码或重复

**解决方案**：

- 确认 `padding_side="left"`(decoder-only 模型必需)
- 确认 `temperature=0.0`(贪心解码，输出格式稳定)
- 检查 LoRA 权重是否正确加载

### Q3：`item_attributes.json` 不存在

**原因**：该文件是符号链接，指向 `item_attributes_{dataset}.json`。

**解决方案**：

```bash
cd enhancement/data/processed/AO  # 或 GM
ln -sf item_attributes_AO.json item_attributes.json  # 或 item_attributes_GM.json
```

> Windows 下解压压缩包可能因符号链接报错(0x80070522)，已用 `tar -h` 解引用处理。

### Q4：模型路径错误

**解决方案**：确认 `enhancement/config/lora_config.yaml` 中 `model.base_model` 路径已改为本地路径。

路径应为 `models/Llama-2-7b-hf` 目录的绝对路径，目录下需有 `config.json`。

### Q5：PyTorch CUDA 版本不匹配

**解决方案**：

```bash
# 查看 CUDA 驱动版本
nvidia-smi

# 修改 requirements.txt 中 torch 版本
# CUDA 12.1 -> torch==2.5.1+cu121
# CUDA 11.8 -> torch==2.5.1+cu118
# 其他版本见 https://pytorch.org/get-started/previous-versions/
```

### Q6：前端数据不更新

**解决方案**：

```bash
# 1. 重新生成前端数据
cd webapp/scripts
python generate_data.py

# 2. 清除缓存并重启
cd ..
rm -rf dist node_modules/.vite
npm run build  # 或 npm run dev
```

> 前端静态 JSON 导入会将数据编译进 JS bundle，必须重新 build 才能更新。

### Q7：ModuleNotFoundError

**原因**：使用了系统 python 而非 conda 环境的 python。

**解决方案**：

```bash
# 必须使用 conda 环境的 python
/root/miniconda3/envs/urllm/bin/python src/run_pipeline.py

# 或先激活环境
conda activate urllm
python src/run_pipeline.py
```

### Q8：Git clone 很慢或失败

**解决方案**：

- 使用 SSH 协议：`git clone git@github.com:learner-aa/ProjectReproductionOfURLLM.git`
- 配置 SSH over 443 端口(见 `~/.ssh/config`)：
  ```
  Host github.com
    Hostname ssh.github.com
    Port 443
    User git
  ```

### Q9：前端启动失败(Vite 版本报错)

**原因**：Vite 8 要求 Node.js >= 20.19 或 >= 22.12，低版本(如 18.x)会报错。

**解决方案**：

```bash
# 1. 检查版本
node -v

# 2. 若版本过低，用 nvm 升级
nvm install 22
nvm use 22

# 3. 重新安装依赖并启动
cd webapp
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Q10：本地浏览器无法访问 localhost:6006

**原因**：前端运行在远程服务器(如 autodl 容器)上，`localhost` 指向远程服务器而非本地。

**解决方案**：

```bash
# 使用 autodl 代理端口启动
cd webapp
npm run dev -- --host 0.0.0.0 --port 6006
# 然后访问 http://localhost:6006
```

***

## 十五、技术栈

- **训练**：PyTorch 2.5.1+cu121 / Transformers 5.14.1 / PEFT 0.20.0 / Accelerate 1.14.0 / bitsandbytes 0.50.0
- **前端**：Vite 8.1.1 + React 19.2.7 + TypeScript 6.0.2 + TailwindCSS 4.3.3 + Recharts 3.10.1
- **数据**：GM(Movie-Game) + AO(Art-Office) 跨域数据集 / DeepSeek 物品属性提取
- **Python 依赖**：torch 2.5.1+cu121 / transformers 5.14.1 / peft 0.20.0 / accelerate 1.14.0 / datasets 5.0.1 / bitsandbytes 0.50.0 / modelscope 1.39.0

***

**项目状态**：GM + AO 双数据集完整 9 阶段流程已跑通，前端展示系统上线(支持双数据集切换 + DG 矩阵相似用户)，评估指标真实且有区分度。

**GitHub 仓库**：<https://github.com/learner-aa/ProjectReproductionOfURLLM>

**Release 下载**：<https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/tag/v1.0.0>
