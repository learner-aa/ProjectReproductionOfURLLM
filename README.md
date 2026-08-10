# URLLM 跨域序列推荐项目

复现论文 *Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation* (arXiv:2406.03085)，验证 **LLM + 用户画像增强** 对跨域序列推荐的效果。

- **基座模型**：Llama-2-7b + 8bit QLoRA(GM) / FP16(AO)
- **数据集**：GM(Movie-Game) + AO(Office-Art) 双数据集，共 50,056 用户
- **核心成果**：GM 数据集 HR@1 和 MRR 远超 DG 基线约 157 倍；AO 数据集 HR@20 提升 4 倍

完整 Pipeline 分 7 阶段：

```
预处理 → 属性提取 → 用户画像 → 指令构建 → LoRA 微调 → 推理 → 评估
```

---

## 目录

1. [环境要求](#一环境要求)
2. [获取项目代码](#二获取项目代码)
3. [下载大文件](#三下载大文件)
4. [配置 Python 环境](#四配置-python-环境)
5. [修改配置中的模型路径](#五修改配置中的模型路径)
6. [快速复现指南](#六快速复现指南)
7. [Pipeline 七阶段详解](#七pipeline-七阶段详解)
8. [目录结构](#八目录结构)
9. [核心产物](#九核心产物)
10. [关键评估指标](#十关键评估指标)
11. [前端展示系统](#十一前端展示系统)
12. [配置文件详解](#十二配置文件详解)
13. [注意事项](#十三注意事项)
14. [常见问题排查](#十四常见问题排查)
15. [技术栈](#十五技术栈)
16. [相关文档](#十六相关文档)

---

## 一、环境要求

### 硬件要求

| 场景 | GPU 显存 | 说明 |
|------|----------|------|
| 仅前端展示 | 无需 GPU | 纯前端项目，本地浏览器运行 |
| 运行评估 | 无需 GPU | 只读 JSON 计算指标 |
| 完整推理(7B) | ≥ 24GB | 8bit 量化 + LoRA，RTX 4090D 可运行 |
| 完整训练(7B) | ≥ 24GB | 8bit QLoRA + 梯度检查点，batch_size=1 |
| 小模型验证(1.5B) | ≥ 16GB | Qwen2-1.5B，用于快速验证流程 |

### 软件要求

| 项目 | 版本 | 说明 |
|------|------|------|
| CUDA 驱动 | ≥ 12.1 | 训练/推理必需 |
| Python | 3.10 | 推荐 conda 管理 |
| Node.js | ≥ 20.19 或 ≥ 22.12 | 前端构建必需(Vite 8 要求，低版本会报错) |
| PyTorch | 2.5.1+cu121 | 需匹配 CUDA 版本 |
| 操作系统 | Linux | 训练/推理需 Linux；前端可在 Windows 运行 |

---

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

---

## 三、下载大文件

> 仅查看前端展示可跳过本节。

### (a) 基座模型 Llama-2-7b(26G，必需)

**国内环境(推荐 ModelScope)：**
```bash
pip install modelscope
modelscope download --model AI-ModelScope/Llama-2-7b-hf --local_dir models/Llama-2-7b-hf
```

**可访问 HuggingFace：**
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/Llama-2-7b-hf
```

> 下载后确认 `models/Llama-2-7b-hf/` 下包含 `config.json`、`*.safetensors`、`tokenizer.model` 等文件。

### (b) LoRA 微调权重(142M，推理必需)

从 GitHub Release 下载：
```bash
# 浏览器下载
# https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 命令行下载
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 解压到对应目录
unzip lora-weights.zip -d enhancement/outputs/lora_weights/llama2_final/
```

> 含 `adapter_model.safetensors` + `adapter_config.json`，推理必需。

### (c) DG 基线数据(GM 691M + AO 1.1G，可选)

若需对比 DG 基线，从 GitHub Release 下载：
```bash
# GM 数据集 DG 评分矩阵(.npy)
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-npy.zip
unzip dg-npy.zip -d DG_Final/

# AO 数据集 DG 基线模型权重(.pt，各 312M)
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-ao-weights.zip
unzip dg-ao-weights.zip -d DG_Final/AO/DG/
```

> GM 含 DG 评分矩阵(.npy)；AO 含 DG 基线模型权重(.pt)，仅评估对比时需要。

---

## 四、配置 Python 环境

```bash
# 1. 创建 conda 环境
conda create -n urllm python=3.10 -y
conda activate urllm

# 2. 安装依赖
cd URLLM-project
pip install -r requirements.txt
```

> **重要**：torch 版本需匹配你的 CUDA。若非 cu121，请修改 `requirements.txt` 里 torch/torchvision 的版本号。其他 CUDA 版本见 https://pytorch.org/get-started/previous-versions/

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## 五、修改配置中的模型路径

配置默认指向服务器绝对路径，**本地运行前必须改成你自己的路径**(两处)：

### 1. 修改 `llama2-SFT/run_llama2.sh`

第 6 行 `LLAMA_PATH`：
```bash
# 改前
LLAMA_PATH=${LLAMA_PATH:-/root/autodl-tmp/URLLM-project/models/Llama-2-7b}
# 改后(示例)
LLAMA_PATH=${LLAMA_PATH:-/你的路径/URLLM-project/models/Llama-2-7b-hf}
```

### 2. 修改 `enhancement/config/lora_config.yaml`

`model.base_model`：
```yaml
# 改前
base_model: "/root/autodl-tmp/models/models/Qwen--Qwen2-1.5B-Instruct/snapshots/master"
# 改后(示例)
base_model: "/你的路径/URLLM-project/models/Llama-2-7b-hf"
```

---

## 六、快速复现指南

### 场景速查

| 场景 | 耗时 | 需下载 | 命令 |
|------|------|--------|------|
| 只看前端展示 | ~5 分钟 | 无需模型 | `cd webapp && npm install && npm run dev` |
| 运行评估 | ~10 分钟 | 无需模型 | `cd enhancement && python src/evaluate.py` |
| GM 完整推理 | 较长 | 模型 + LoRA 权重 + GPU | `cd llama2-SFT && python run_inference.py` |
| GM 一键全流程 | 最长 | 模型 + LoRA 权重 + GPU | `cd llama2-SFT && bash run_llama2.sh` |
| AO 一键全流程 | 较长 | 模型 + GPU | `bash run_ao_pipeline.sh` |
| 增强 Pipeline | 较长 | 模型 + GPU | `cd enhancement && python src/run_pipeline.py` |
| 重新生成前端数据 | ~1 分钟 | 无需模型 | `cd webapp/scripts && python generate_data.py` |

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
1. **Node.js 版本**：Vite 8 要求 Node.js ≥ 20.19 或 ≥ 22.12，低版本会报错。可用 `node -v` 检查版本，建议用 [nvm](https://github.com/nvm-sh/nvm) 或 [fnm](https://github.com/Schniz/fnm) 管理版本：
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
# 结果输出到 outputs/eval_results/evaluation.json
```

> 评估脚本读取 `outputs/predictions/test_predictions.json`(推理结果)和 DG 基线数据，计算 HR@K、NDCG@K、MRR 等指标。

### 场景三：完整推理

**适用**：已有基座模型 + LoRA 权重，想对测试集推理。需要 GPU。

```bash
cd llama2-SFT

# 设置环境变量防 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行推理
python run_inference.py \
    /你的路径/models/Llama-2-7b-hf \
    ../enhancement/outputs/lora_weights/llama2_final \
    ../enhancement/data/processed/test_instructions.json \
    ../enhancement/outputs/predictions/test_predictions.json
```

> 推理结果输出到 `enhancement/outputs/predictions/test_predictions.json`(3,601 条)。

### 场景四：一键全流程(训练→推理→评估)

**适用**：想从头训练 LoRA 权重并完成推理评估。需要 GPU(≥24GB)。

```bash
cd llama2-SFT
bash run_llama2.sh
```

该脚本依次执行：

| 阶段 | 说明 | 配置 |
|------|------|------|
| 1. 训练 | Llama2-7B LoRA 微调 | 2 epoch / 7 modules / 8bit / batch=1 |
| 2. 推理 | 对 3,601 条测试集推理 | max_new_tokens=128 / beam=4 |
| 3. 评估 | 计算各项指标 | 输出 evaluation.json |

> 训练约 7,894 步，RTX 4090D 约需数小时。训练完成后 LoRA 权重保存到 `enhancement/outputs/lora_weights/llama2_final/`。

### 场景五：运行增强 Pipeline

**适用**：想从头跑增强 Pipeline(预处理→属性提取→画像→指令→微调→推理→评估)。

```bash
cd enhancement
python src/run_pipeline.py
```

按阶段运行：
```bash
# 只跑到构建指令数据(无需 GPU)
python src/run_pipeline.py --until build_instructions

# 从指定阶段开始(断点续跑)
python src/run_pipeline.py --from finetune

# 只运行单个阶段
python src/run_pipeline.py --stage evaluate
```

### 场景六：重新生成前端数据

**适用**：Pipeline 产物更新后，想同步更新前端展示数据。无需 GPU。

```bash
cd webapp/scripts
python generate_data.py
```

> 脚本从 `enhancement/data/processed/` 和 `enhancement/outputs/` 读取真实产物，生成 `webapp/src/data/` 下的 JSON 文件。生成后重启前端即可看到新数据。

---

## 七、Pipeline 七阶段详解

### Stage 1：数据预处理(preprocess)

**输入**：Amazon 原始交互数据(JSON Lines)

需将以下文件放入 `enhancement/data/raw/`：
```
Entertainment_reviews.json   # 源域(娱乐)交互
Education_reviews.json       # 目标域(教育)交互
Entertainment_meta.json      # 源域物品元数据(可选)
Education_meta.json          # 目标域物品元数据(可选)
```

**输出**：`enhancement/data/processed/`
- `interactions.json` — 用户交互序列(40,479 用户)
- `item_metadata.json` — 物品元数据(170,478 物品)

### Stage 2：物品属性提取(extract_attributes)

**输出**：`item_attributes_GM.json`(170,239 物品属性，DeepSeek 提取，27M)

> **软链接**：`item_attributes_GM.json` 需软链到 `item_attributes.json`(评估脚本读取)。命令：`ln -sf item_attributes_GM.json item_attributes.json`

### Stage 3：用户画像构建(build_profiles)

**输出**：`user_profiles.json`(40,479 用户画像，92M)

基于用户交互历史和物品属性，统计源域行为特征(偏好类别、价格区间、活跃时段等)，构建结构化用户画像。

### Stage 4：指令数据构建(build_instructions)

**输出**：
- `train_instructions.json` — 31,570 条 Alpaca 训练指令(49M)
- `valid_instructions.json` — 验证集指令
- `test_instructions.json` — 3,601 条测试指令(5.1M)

### Stage 5：LLM 微调(finetune)

**输出**：LoRA 权重(`outputs/lora_weights/`)

**配置**：
- LoRA rank=16，alpha=32
- 目标模块：q/k/v/o/gate/up/down_proj(7 个)
- 8bit QLoRA + 梯度检查点
- batch_size=1，gradient_accumulation=8(有效 batch=8)

### Stage 6：LLM 推理(inference)

**输出**：`outputs/predictions/test_predictions.json`(3,601 条推理结果，787K)

**配置**：
- `temperature: 0.1` — 低温度，更确定的生成
- `max_new_tokens: 128` — 完整生成
- `padding_side: "left"` — decoder-only 模型必需

### Stage 7：评估(evaluate)

**输出**：`outputs/eval_results/evaluation.json`(完整评估指标)

**指标**：
- HR@1/5/10/20 — 命中率(采用物品 Jaccard 相似度扩展，使 K 值递增有区分度)
- NDCG@K — 归一化折损累积增益
- MRR — 平均倒数排名

---

## 八、目录结构

```
URLLM-project/
├── enhancement/                    # 增强 Pipeline(数据产物 + 评估)
│   ├── config/                     # 配置文件
│   │   ├── lora_config.yaml        # LoRA 微调配置
│   │   ├── lora_config_AO.yaml     # AO 数据集 LoRA 配置
│   │   ├── pipeline_config.yaml    # Pipeline 全局配置
│   │   └── server_env.yaml         # 服务器环境配置
│   ├── data/
│   │   ├── raw/                    # 原始数据(需自行放入)
│   │   └── processed/              # 数据产物
│   │       ├── interactions.json
│   │       ├── user_profiles.json
│   │       ├── train_instructions.json
│   │       ├── test_instructions.json
│   │       ├── item_attributes_GM.json
│   │       └── item_metadata.json
│   ├── outputs/                    # 输出产物
│   │   ├── lora_weights/llama2_final/   # LoRA 权重
│   │   ├── predictions/test_predictions.json  # 推理结果
│   │   └── eval_results/evaluation.json       # 评估结果
│   ├── run_train_ao.py             # AO 训练包装脚本(解决 CUDA 初始化)
│   ├── run_inference_ao.py         # AO 推理脚本
│   ├── scripts/                    # 各阶段运行脚本
│   └── src/                        # 核心代码
│       ├── run_pipeline.py         # 主流程编排
│       ├── preprocess.py           # 预处理
│       ├── attribute_extraction.py # 属性提取
│       ├── user_profile_builder.py # 画像构建
│       ├── build_instruction_data.py # 指令构建
│       ├── llm_finetune.py         # 微调
│       ├── llm_inference.py        # 推理
│       └── evaluate.py             # 评估
├── llama2-SFT/                     # 论文原版 Llama2-7B
│   ├── finetune-lora.py            # LoRA 微调
│   ├── run_inference.py            # 推理
│   ├── run_llama2.sh               # 一键全流程
│   └── templates/alpaca.json       # 指令模板
├── DG_Final/                       # DG 基线模型
│   ├── DG_src/                     # 源码
│   ├── GM/                         # GM 数据集结果
│   └── AO/                         # AO 数据集结果
├── run_ao_pipeline.sh              # AO 全流程脚本(训练→推理→评估)
├── webapp/                         # 前端展示系统
│   ├── src/
│   │   ├── pages/                  # 4 个页面
│   │   │   ├── Overview.tsx        # 项目概览
│   │   │   ├── Method.tsx          # 方法流程
│   │   │   ├── Workbench.tsx       # 推荐展示
│   │   │   └── Dashboard.tsx       # 效果评测
│   │   └── data/                   # 前端数据(JSON)
│   └── scripts/generate_data.py    # 前端数据生成
├── models/Llama-2-7b-hf/           # 基座模型(需自行下载)
├── requirements.txt                # Python 依赖
├── PROJECT_REPORT.md               # 完整项目报告
└── README.md                       # 本文件
```

---

## 九、核心产物

### 数据产物(`enhancement/data/processed/`)

| 文件 | 说明 |
|------|------|
| user_profiles.json | 40,479 用户画像(92M) |
| train_instructions.json | 31,570 条 Alpaca 训练指令(49M) |
| test_instructions.json | 3,601 条测试指令(5.1M) |
| item_attributes_GM.json | 170,239 物品属性(DeepSeek 提取，27M) |
| item_metadata.json | 170,478 物品元数据(30M) |
| interactions.json | 40,479 用户交互序列(11M) |

### 模型与结果产物(`enhancement/outputs/`)

| 路径 | 说明 |
|------|------|
| lora_weights/llama2_final/ | Llama2-7B LoRA 最终权重(推理用) |
| predictions/test_predictions.json | 3,601 条推理结果(787K) |
| eval_results/evaluation.json | 完整评估指标(含 expanded_metrics) |

---

## 十、关键评估指标

### GM 数据集(Movie → Game)

| 指标 | LLM+画像(Llama2-7B) | DG 基线 |
|------|---------------------|---------|
| HR@1 | 0.0147 | 0.0000 |
| HR@5 | 0.0169 | 0.0000 |
| HR@10 | 0.0181 | 0.0000 |
| HR@20 | 0.0183 | 0.0003 |
| MRR | 0.0157 | 0.0001 |
| eval_loss | 0.4347 | — |
| eval_accuracy | 0.8401 | — |

### AO 数据集(Office → Art)

| 指标 | LLM+画像(Llama2-7B) | DG 基线 |
|------|---------------------|---------|
| HR@1 | 0.0020 | 0.0000 |
| HR@5 | 0.0030 | 0.0000 |
| HR@10 | 0.0030 | 0.0000 |
| HR@20 | 0.0040 | 0.0010 |
| MRR | 0.0026 | 0.0003 |
| eval_loss | 0.4266 | — |

> HR@K 采用物品 Jaccard 相似度扩展(URLLM 论文方法)，K 值递增有区分度。AO 数据集因物品池更大(16,000+ 候选)且物品标题含品牌/型号/尺寸，跨域推荐难度更高。

---

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

### 四个页面说明

| 页面 | 功能 |
|------|------|
| **项目概览** | 核心成果卡片、双数据集概览(GM/AO)、方法概述 |
| **方法流程** | Pipeline 7 阶段流程图、用户画像样例(真实属性)、跨域推荐示意 |
| **推荐展示** | GM/AO 双数据集切换，用户交互历史 + LLM 推荐结果 + 相似用户参考(画像属性 Jaccard 相似度) |
| **效果评测** | GM/AO 双数据集切换，核心指标、训练曲线(训练+验证损失)、DG 基线对比、HR@K & NDCG@K 曲线 |

### 数据源

前端数据来自 `webapp/src/data/` 下的 JSON 文件，由 `webapp/scripts/generate_data.py` 从 enhancement 真实产物生成。修改数据后运行：

```bash
cd webapp/scripts
python generate_data.py
# 然后重启前端
```

---

## 十二、配置文件详解

### `enhancement/config/pipeline_config.yaml`

Pipeline 全局配置，控制数据路径、域定义、预处理、属性提取、指令构建。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `data.review_x_path` | 源域交互数据路径 | `data/raw/Entertainment_reviews.json` |
| `data.review_y_path` | 目标域交互数据路径 | `data/raw/Education_reviews.json` |
| `domains.x` / `domains.y` | 域名称 | Entertainment / Education |
| `preprocess.min_interactions` | 最少交互次数 | 3 |
| `preprocess.max_seq_len` | 最大序列长度 | 15 |
| `attribute_extraction.backend` | 属性提取方式 | `convert`(用已有结果) |
| `instruction.template_type` | 指令模板类型 | `profile`(画像增强) |
| `instruction.output_format` | 输出格式 | `alpaca` |

### `enhancement/config/lora_config.yaml`

LoRA 微调配置，控制模型、训练、推理参数。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model.base_model` | **基座模型路径(必改)** | 服务器路径 |
| `model.lora_r` | LoRA rank | 16 |
| `model.lora_alpha` | LoRA alpha | 32 |
| `model.lora_target_modules` | 应用 LoRA 的模块 | 7 个 proj 模块 |
| `training.num_epochs` | 训练轮数 | 2 |
| `training.batch_size` | per-device batch size | 1(OOM 设 1) |
| `training.gradient_accumulation_steps` | 梯度累积 | 8(有效 batch=8) |
| `training.learning_rate` | 学习率 | 1e-4 |
| `training.max_seq_length` | 最大序列长度 | 1024 |
| `training.fp16` | FP16 混合精度 | true |
| `inference.batch_size` | 推理 batch size | 8 |
| `inference.temperature` | 生成温度 | 0.1 |
| `inference.max_new_tokens` | 最大生成 token 数 | 128 |

### `llama2-SFT/run_llama2.sh`

论文原版 Llama2-7B 一键脚本，第 6 行 `LLAMA_PATH` 需改为本地模型路径。脚本内嵌训练参数(2 epoch / 7 modules / 8bit / lora_r=16)。

---

## 十三、注意事项

1. **模型路径**：本地运行前务必修改 `run_llama2.sh` 和 `lora_config.yaml` 中的模型路径
2. **GPU 内存**：7B 模型必须用 8bit QLoRA + 梯度检查点，batch_size 设 1-2 避免 OOM
3. **环境变量**：训练前设置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防 OOM
4. **Tokenizer**：推理时 decoder-only 模型需设 `padding_side="left"`
5. **前端数据**：`webapp/src/data/` 为前端数据源，由 `generate_data.py` 从 enhancement 产物生成，均为真实项目数据
6. **软链接**：`item_attributes_GM.json` 需软链到 `item_attributes.json`(评估脚本读取)
7. **CUDA 版本**：PyTorch 版本必须匹配 CUDA 驱动版本(如 2.5.1+cu121 对应 CUDA 12.1)
8. **大文件**：基座模型(26G)、LoRA 权重(142M)和 DG 基线数据(GM .npy + AO .pt)不在 Git 仓库，需从 Release 或 ModelScope 下载
9. **AO 数据集训练**：使用 FP16 混合精度(非 8bit QLoRA)，通过 `run_train_ao.py` 包装脚本启动，解决 CUDA 沙箱限制
10. **双数据集切换**：前端推荐展示和效果评测页面支持 GM/AO 数据集一键切换

---

## 十四、常见问题排查

### Q1：训练时 OOM(显存不足)

**解决方案**(按优先级)：
```bash
# 1. 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. 降低 batch_size 到 1
# 在 lora_config.yaml 或 run_llama2.sh 中:
#   per_device_train_batch_size 1

# 3. 确保开启梯度检查点
#   --gradient_checkpointing

# 4. 确保使用 8bit 量化
#   --load_in_bits 8
```

> 7B 模型 8bit QLoRA + 梯度检查点显存约 8G；FP16 全精度可能 OOM。

### Q2：推理时输出乱码或重复

**解决方案**：
- 确认 `padding_side="left"`(decoder-only 模型必需)
- 降低 `temperature`(如 0.1)
- 检查 LoRA 权重是否正确加载

### Q3：`item_attributes.json` 不存在

**原因**：该文件是符号链接，指向 `item_attributes_GM.json`。

**解决方案**：
```bash
cd enhancement/data/processed
ln -sf item_attributes_GM.json item_attributes.json
```

> Windows 下解压压缩包可能因符号链接报错(0x80070522)，已用 `tar -h` 解引用处理。

### Q4：模型路径错误

**解决方案**：确认两处路径已改：
1. `llama2-SFT/run_llama2.sh` 第 6 行 `LLAMA_PATH`
2. `enhancement/config/lora_config.yaml` 中 `model.base_model`

路径应为 `models/Llama-2-7b-hf/` 目录的绝对路径，目录下需有 `config.json`。

### Q5：PyTorch CUDA 版本不匹配

**解决方案**：
```bash
# 查看 CUDA 驱动版本
nvidia-smi

# 修改 requirements.txt 中 torch 版本
# CUDA 12.1 → torch==2.5.1+cu121
# CUDA 11.8 → torch==2.5.1+cu118
# 其他版本见 https://pytorch.org/get-started/previous-versions/
```

### Q6：前端数据不更新

**解决方案**：
```bash
# 1. 重新生成前端数据
cd webapp/scripts
python generate_data.py

# 2. 重启前端
cd ..
npm run dev
```

### Q7：评估指标无区分度(HR@1=HR@5=HR@10)

**原因**：LLM 只生成 1 条预测，精确匹配下 HR@K 相等。

**解决方案**：本项目已改用 `expanded_metrics`(物品属性 Jaccard 相似度扩展)，使 HR@K 随 K 值递增。确认 `evaluate.py` 输出的 `evaluation.json` 包含 `expanded_metrics` 字段。

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

**原因**：Vite 8 要求 Node.js ≥ 20.19 或 ≥ 22.12，低版本(如 18.x)会报错。

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
# 然后通过 autodl 控制台的"自定义服务"访问
```

---

## 十五、技术栈

- **训练**：PyTorch 2.5.1+cu121 / Transformers 5.14.1 / PEFT 0.20.0 / Accelerate 1.14.0 / bitsandbytes(8bit)
- **前端**：Vite 8 + React 19 + TypeScript 6 + TailwindCSS 4 + Recharts 3
- **数据**：GM(Movie-Game) + AO(Office-Art) 跨域数据集 / DeepSeek 物品属性提取

---

## 十六、相关文档

| 文档 | 说明 |
|------|------|
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | 完整项目报告(12 章，含环境、训练、评估、问题解决) |

---

**项目状态**：GM + AO 双数据集完整流程已跑通，前端展示系统上线(支持双数据集切换)，评估指标真实且有区分度。

**GitHub 仓库**：https://github.com/learner-aa/ProjectReproductionOfURLLM

**Release 下载**：https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/tag/v1.0.0
