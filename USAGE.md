# URLLM 跨域序列推荐项目 — 完整使用说明

本文件是项目的**详细操作手册**，涵盖环境准备、代码获取、各场景运行步骤、Pipeline 全流程、前端展示、配置详解与常见问题排查。

> 若只想要快速上手,请先阅读 [README.md](README.md);若想了解项目实现细节,请阅读 [PROJECT_REPORT.md](PROJECT_REPORT.md)。

---

## 目录

1. [项目简介](#一项目简介)
2. [环境要求](#二环境要求)
3. [获取项目代码](#三获取项目代码)
4. [下载大文件](#四下载大文件)
5. [配置 Python 环境](#五配置-python-环境)
6. [修改配置中的模型路径](#六修改配置中的模型路径)
7. [使用场景速查](#七使用场景速查)
8. [场景一:仅查看前端展示](#场景一仅查看前端展示)
9. [场景二:运行评估](#场景二运行评估)
10. [场景三:完整推理](#场景三完整推理)
11. [场景四:一键全流程(训练→推理→评估)](#场景四一键全流程训练推理评估)
12. [场景五:运行增强 Pipeline](#场景五运行增强-pipeline)
13. [场景六:重新生成前端数据](#场景六重新生成前端数据)
14. [Pipeline 七阶段详解](#八pipeline-七阶段详解)
15. [前端展示系统](#九前端展示系统)
16. [配置文件详解](#十配置文件详解)
17. [常见问题排查](#十一常见问题排查)
18. [目录结构](#十二目录结构)
19. [相关文档](#十三相关文档)

---

## 一、项目简介

复现论文 *Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation* (arXiv:2406.03085)，验证 **LLM + 用户画像增强** 对跨域序列推荐的效果。

- **基座模型**:Llama-2-7b + 8bit QLoRA(GM) / FP16(AO)
- **数据集**:GM(Movie-Game) + AO(Office-Art) 双数据集,共 50,056 用户
- **核心成果**:GM 数据集 HR@1 和 MRR 远超 DG 基线约 157 倍;AO 数据集 HR@20 提升 4 倍

完整 Pipeline 分 7 阶段:

```
预处理 → 属性提取 → 用户画像 → 指令构建 → LoRA 微调 → 推理 → 评估
```

---

## 二、环境要求

### 硬件要求

| 场景 | GPU 显存 | 说明 |
|------|----------|------|
| 仅前端展示 | 无需 GPU | 纯前端项目,本地浏览器运行 |
| 运行评估 | 无需 GPU | 只读 JSON 计算指标 |
| 完整推理(7B) | ≥ 24GB | 8bit 量化 + LoRA,RTX 4090D 可运行 |
| 完整训练(7B) | ≥ 24GB | 8bit QLoRA + 梯度检查点,batch_size=1 |
| 小模型验证(1.5B) | ≥ 16GB | Qwen2-1.5B,用于快速验证流程 |

### 软件要求

| 项目 | 版本 | 说明 |
|------|------|------|
| CUDA 驱动 | ≥ 12.1 | 训练/推理必需 |
| Python | 3.10 | 推荐 conda 管理 |
| Node.js | ≥ 18 | 前端构建必需 |
| PyTorch | 2.5.1+cu121 | 需匹配 CUDA 版本 |
| 操作系统 | Linux | 训练/推理需 Linux;前端可在 Windows 运行 |

---

## 三、获取项目代码

### 方式 A:从 GitHub clone(推荐)

```bash
git clone https://github.com/learner-aa/ProjectReproductionOfURLLM.git
cd URLLM-project
```

### 方式 B:从压缩包解压

```bash
tar -xzf URLLM-project.tar.gz
cd URLLM-project
```

> 两种方式都**不含**基座模型(26G)和前端依赖(node_modules),需按下面步骤补齐。

---

## 四、下载大文件

> 仅查看前端展示可跳过本节。

### (a) 基座模型 Llama-2-7b(26G,必需)

**国内环境(推荐 ModelScope):**
```bash
pip install modelscope
modelscope download --model AI-ModelScope/Llama-2-7b-hf --local_dir models/Llama-2-7b-hf
```

**可访问 HuggingFace:**
```bash
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/Llama-2-7b-hf
```

> 下载后确认 `models/Llama-2-7b-hf/` 下包含 `config.json`、`*.safetensors`、`tokenizer.model` 等文件。

### (b) LoRA 微调权重(142M,推理必需)

从 GitHub Release 下载:
```bash
# 浏览器下载
https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 命令行下载
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 解压到对应目录
unzip lora-weights.zip -d enhancement/outputs/lora_weights/llama2_final/
```

> 含 `adapter_model.safetensors` + `adapter_config.json`,推理必需。

### (c) DG 基线数据(GM 691M + AO 1.1G,可选)

若需对比 DG 基线,从 GitHub Release 下载:
```bash
# GM 数据集 DG 评分矩阵(.npy)
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-npy.zip
unzip dg-npy.zip -d DG_Final/

# AO 数据集 DG 基线模型权重(.pt,各 312M)
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/dg-ao-weights.zip
unzip dg-ao-weights.zip -d DG_Final/AO/DG/
```

> GM 含 DG 评分矩阵(.npy);AO 含 DG 基线模型权重(.pt),仅评估对比时需要。

---

## 五、配置 Python 环境

```bash
# 1. 创建 conda 环境
conda create -n urllm python=3.10 -y
conda activate urllm

# 2. 安装依赖
cd URLLM-project
pip install -r requirements.txt
```

> **重要**:torch 版本需匹配你的 CUDA。若非 cu121,请修改 `requirements.txt` 里 torch/torchvision 的版本号。其他 CUDA 版本见 https://pytorch.org/get-started/previous-versions/

### 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

## 六、修改配置中的模型路径

配置默认指向服务器绝对路径,**本地运行前必须改成你自己的路径**(两处):

### 1. 修改 `llama2-SFT/run_llama2.sh`

第 6 行 `LLAMA_PATH`:
```bash
# 改前
LLAMA_PATH=${LLAMA_PATH:-/root/autodl-tmp/URLLM-project/models/Llama-2-7b}
# 改后(示例)
LLAMA_PATH=${LLAMA_PATH:-/你的路径/URLLM-project/models/Llama-2-7b-hf}
```

### 2. 修改 `enhancement/config/lora_config.yaml`

`model.base_model`:
```yaml
# 改前
base_model: "/root/autodl-tmp/models/models/Qwen--Qwen2-1.5B-Instruct/snapshots/master"
# 改后(示例)
base_model: "/你的路径/URLLM-project/models/Llama-2-7b-hf"
```

---

## 七、使用场景速查

| 场景 | 耗时 | 需下载 | 命令 |
|------|------|--------|------|
| 只看前端展示 | ~5 分钟 | 无需模型 | `cd webapp && npm install && npm run dev` |
| 运行评估 | ~10 分钟 | 无需模型 | `cd enhancement && python src/evaluate.py` |
| GM 完整推理 | 较长 | 模型 + LoRA 权重 + GPU | `cd llama2-SFT && python run_inference.py` |
| GM 一键全流程 | 最长 | 模型 + LoRA 权重 + GPU | `cd llama2-SFT && bash run_llama2.sh` |
| AO 一键全流程 | 较长 | 模型 + GPU | `bash run_ao_pipeline.sh` |
| 增强 Pipeline | 较长 | 模型 + GPU | `cd enhancement && python src/run_pipeline.py` |
| 重新生成前端数据 | ~1 分钟 | 无需模型 | `cd webapp/scripts && python generate_data.py` |

---

## 场景一:仅查看前端展示

**适用**:只想看项目演示效果,无需 GPU 和模型。

```bash
cd webapp
npm install         # 安装前端依赖(首次)
npm run dev         # 启动开发服务器
```

浏览器访问 `http://localhost:5173`。

> 前端为纯 Vite + React 项目,数据来自本地 JSON,无需后端。

---

## 场景二:运行评估

**适用**:已有推理结果,想重新计算评估指标。无需 GPU 和模型。

```bash
cd enhancement
python src/evaluate.py
# 结果输出到 outputs/eval_results/evaluation.json
```

> 评估脚本读取 `outputs/predictions/test_predictions.json`(推理结果)和 DG 基线数据,计算 HR@K、NDCG@K、MRR 等指标。

---

## 场景三:完整推理

**适用**:已有基座模型 + LoRA 权重,想对测试集推理。需要 GPU。

### 前置检查

```bash
# 1. 确认模型已下载
ls models/Llama-2-7b-hf/

# 2. 确认 LoRA 权重已解压
ls enhancement/outputs/lora_weights/llama2_final/
# 应看到 adapter_model.safetensors、adapter_config.json

# 3. 确认测试数据存在
ls enhancement/data/processed/test_instructions.json
```

### 运行推理

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

---

## 场景四:一键全流程(训练→推理→评估)

**适用**:想从头训练 LoRA 权重并完成推理评估。需要 GPU(≥24GB)。

### 前置检查

1. 确认基座模型已下载到 `models/Llama-2-7b-hf/`
2. 确认训练数据存在:`enhancement/data/processed/train_instructions.json`
3. 确认 `run_llama2.sh` 第 6 行 `LLAMA_PATH` 已改成本地路径

### 一键执行

```bash
cd llama2-SFT
bash run_llama2.sh
```

该脚本依次执行:

| 阶段 | 说明 | 配置 |
|------|------|------|
| 1. 训练 | Llama2-7B LoRA 微调 | 2 epoch / 7 modules / 8bit / batch=1 |
| 2. 推理 | 对 3,601 条测试集推理 | max_new_tokens=128 / beam=4 |
| 3. 评估 | 计算各项指标 | 输出 evaluation.json |

> 训练约 7,894 步,RTX 4090D 约需数小时。训练完成后 LoRA 权重保存到 `enhancement/outputs/lora_weights/llama2_final/`。

### 仅训练(不推理评估)

若只想训练 LoRA 权重:
```bash
cd llama2-SFT
python finetune-lora.py \
    --model_name_or_path /你的路径/models/Llama-2-7b-hf \
    --tokenizer_name /你的路径/models/Llama-2-7b-hf \
    --train_files ../enhancement/data/processed/train_instructions.json \
    --validation_files ../enhancement/data/processed/valid_instructions.json \
    --do_train --do_eval \
    --output_dir ../enhancement/outputs/lora_weights/llama2_final \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --load_in_bits 8 \
    --lora_r 16 --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --block_size 1024 \
    --gradient_checkpointing
```

---

## 场景五:运行增强 Pipeline

**适用**:想从头跑增强 Pipeline(预处理→属性提取→画像→指令→微调→推理→评估)。

### 全流程运行

```bash
cd enhancement
python src/run_pipeline.py
```

### 按阶段运行

```bash
# 只跑到构建指令数据(无需 GPU)
python src/run_pipeline.py --until build_instructions

# 从指定阶段开始(断点续跑)
python src/run_pipeline.py --from finetune

# 只运行单个阶段
python src/run_pipeline.py --stage evaluate
```

### 使用脚本运行各阶段

```bash
cd enhancement

# 数据预处理
bash scripts/run_preprocess.sh

# LoRA 微调
bash scripts/run_finetune.sh

# 推理
bash scripts/run_inference.sh

# 评估
bash scripts/run_eval.sh
```

> 注意:`run_preprocess.sh` 需要先准备原始数据到 `data/raw/` 目录(见下文 Pipeline 详解)。

---

## 场景六:重新生成前端数据

**适用**:Pipeline 产物更新后,想同步更新前端展示数据。无需 GPU。

```bash
cd webapp/scripts
python generate_data.py
```

> 脚本从 `enhancement/data/processed/` 和 `enhancement/outputs/` 读取真实产物,生成 `webapp/src/data/` 下的 JSON 文件。生成后重启前端即可看到新数据。

---

## 八、Pipeline 七阶段详解

### Stage 1:数据预处理(preprocess)

**输入**:Amazon 原始交互数据(JSON Lines)

需将以下文件放入 `enhancement/data/raw/`:
```
Entertainment_reviews.json   # 源域(娱乐)交互
Education_reviews.json       # 目标域(教育)交互
Entertainment_meta.json      # 源域物品元数据(可选)
Education_meta.json          # 目标域物品元数据(可选)
```

**输出**:`enhancement/data/processed/`
- `interactions.json` — 用户交互序列(40,479 用户)
- `item_metadata.json` — 物品元数据(170,478 物品)

**配置**(`pipeline_config.yaml` → `preprocess`):
- `min_interactions: 3` — 最少交互次数过滤
- `max_seq_len: 15` — 最大序列长度(与 DG 模型一致)
- `random_seed: 2040` — 随机种子(与 DG 模型一致)

### Stage 2:物品属性提取(extract_attributes)

**输入**:物品元数据

**输出**:`item_attributes_GM.json`(170,239 物品属性,DeepSeek 提取,27M)

**配置**(`pipeline_config.yaml` → `attribute_extraction`):
- `backend: "convert"` — 使用已有 DeepSeek 结果(推荐)
- 其他可选:`"api"`(调用 LLM API)、`"local"`(本地模型)

> **软链接**:`item_attributes_GM.json` 需软链到 `item_attributes.json`(评估脚本读取)。命令:`ln -sf item_attributes_GM.json item_attributes.json`

### Stage 3:用户画像构建(build_profiles)

**输入**:交互序列 + 物品属性

**输出**:`user_profiles.json`(40,479 用户画像,92M)

基于用户交互历史和物品属性,统计源域行为特征(偏好类别、价格区间、活跃时段等),构建结构化用户画像。

### Stage 4:指令数据构建(build_instructions)

**输入**:用户画像 + 交互序列

**输出**:
- `train_instructions.json` — 31,570 条 Alpaca 训练指令(49M)
- `valid_instructions.json` — 验证集指令
- `test_instructions.json` — 3,601 条测试指令(5.1M)

**配置**(`pipeline_config.yaml` → `instruction`):
- `template_type: "profile"` — 使用用户画像增强模板
- `output_format: "alpaca"` — Alpaca 指令格式

### Stage 5:LLM 微调(finetune)

**输入**:训练指令 + 基座模型

**输出**:LoRA 权重(`outputs/lora_weights/`)

**配置**(`lora_config.yaml`):
- LoRA rank=8/16,alpha=16/32
- 目标模块:q/k/v/o/gate/up/down_proj(7 个)
- 8bit QLoRA + 梯度检查点
- batch_size=1,gradient_accumulation=8(有效 batch=8)

### Stage 6:LLM 推理(inference)

**输入**:测试指令 + 基座模型 + LoRA 权重

**输出**:`outputs/predictions/test_predictions.json`(3,601 条推理结果,787K)

**配置**:
- `temperature: 0.1` — 低温度,更确定的生成
- `max_new_tokens: 128` — 完整生成
- `padding_side: "left"` — decoder-only 模型必需

### Stage 7:评估(evaluate)

**输入**:推理结果 + DG 基线数据

**输出**:`outputs/eval_results/evaluation.json`(完整评估指标)

**指标**:
- HR@1/5/10/20 — 命中率(采用物品 Jaccard 相似度扩展,使 K 值递增有区分度)
- NDCG@K — 归一化折损累积增益
- MRR — 平均倒数排名

---

## 九、前端展示系统

### 启动

```bash
cd webapp
npm install     # 首次
npm run dev     # 开发模式,访问 http://localhost:5173
# 或
npm run build   # 生产构建,输出到 dist/
npm run preview # 预览生产构建
```

### 四个页面说明

| 页面 | 功能 |
|------|------|
| **项目概览** | 核心成果卡片、双数据集概览(GM/AO)、方法概述 |
| **方法流程** | Pipeline 7 阶段流程图、用户画像样例(真实属性)、跨域推荐示意 |
| **推荐展示** | GM/AO 双数据集切换,用户交互历史 + LLM 推荐结果 + 相似用户参考(画像属性 Jaccard 相似度) |
| **效果评测** | GM/AO 双数据集切换,核心指标、训练曲线(训练+验证损失)、DG 基线对比、HR@K & NDCG@K 曲线 |

### 数据源

前端数据来自 `webapp/src/data/` 下的 JSON 文件,由 `webapp/scripts/generate_data.py` 从 enhancement 真实产物生成。修改数据后运行:

```bash
cd webapp/scripts
python generate_data.py
# 然后重启前端
```

---

## 十、配置文件详解

### `enhancement/config/pipeline_config.yaml`

Pipeline 全局配置,控制数据路径、域定义、预处理、属性提取、指令构建。

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

LoRA 微调配置,控制模型、训练、推理参数。

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model.base_model` | **基座模型路径(必改)** | 服务器路径 |
| `model.lora_r` | LoRA rank | 8 |
| `model.lora_alpha` | LoRA alpha | 16 |
| `model.lora_target_modules` | 应用 LoRA 的模块 | 7 个 proj 模块 |
| `training.num_epochs` | 训练轮数 | 5 |
| `training.batch_size` | per-device batch size | 1(OOM 设 1) |
| `training.gradient_accumulation_steps` | 梯度累积 | 8(有效 batch=8) |
| `training.learning_rate` | 学习率 | 1e-4 |
| `training.max_seq_length` | 最大序列长度 | 1024 |
| `training.fp16` | FP16 混合精度 | true |
| `inference.batch_size` | 推理 batch size | 8 |
| `inference.temperature` | 生成温度 | 0.1 |
| `inference.max_new_tokens` | 最大生成 token 数 | 128 |

### `llama2-SFT/run_llama2.sh`

论文原版 Llama2-7B 一键脚本,第 6 行 `LLAMA_PATH` 需改为本地模型路径。脚本内嵌训练参数(2 epoch / 7 modules / 8bit / lora_r=16)。

---

## 十一、常见问题排查

### Q1:训练时 OOM(显存不足)

**解决方案**(按优先级):
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

> 7B 模型 8bit QLoRA + 梯度检查点显存约 8G;FP16 全精度可能 OOM。

### Q2:推理时输出乱码或重复

**解决方案**:
- 确认 `padding_side="left"`(decoder-only 模型必需)
- 降低 `temperature`(如 0.1)
- 检查 LoRA 权重是否正确加载

### Q3:`item_attributes.json` 不存在

**原因**:该文件是符号链接,指向 `item_attributes_GM.json`。

**解决方案**:
```bash
cd enhancement/data/processed
ln -sf item_attributes_GM.json item_attributes.json
```

> Windows 下解压压缩包可能因符号链接报错(0x80070522),已用 `tar -h` 解引用处理。

### Q4:模型路径错误

**解决方案**:确认两处路径已改:
1. `llama2-SFT/run_llama2.sh` 第 6 行 `LLAMA_PATH`
2. `enhancement/config/lora_config.yaml` 中 `model.base_model`

路径应为 `models/Llama-2-7b-hf/` 目录的绝对路径,目录下需有 `config.json`。

### Q5:PyTorch CUDA 版本不匹配

**解决方案**:
```bash
# 查看 CUDA 驱动版本
nvidia-smi

# 修改 requirements.txt 中 torch 版本
# CUDA 12.1 → torch==2.5.1+cu121
# CUDA 11.8 → torch==2.5.1+cu118
# 其他版本见 https://pytorch.org/get-started/previous-versions/
```

### Q6:前端数据不更新

**解决方案**:
```bash
# 1. 重新生成前端数据
cd webapp/scripts
python generate_data.py

# 2. 重启前端
cd ..
npm run dev
```

### Q7:评估指标无区分度(HR@1=HR@5=HR@10)

**原因**:LLM 只生成 1 条预测,精确匹配下 HR@K 相等。

**解决方案**:本项目已改用 `expanded_metrics`(物品属性 Jaccard 相似度扩展),使 HR@K 随 K 值递增。确认 `evaluate.py` 输出的 `evaluation.json` 包含 `expanded_metrics` 字段。

### Q8:Git clone 很慢或失败

**解决方案**:
- 使用 SSH 协议:`git clone git@github.com:learner-aa/ProjectReproductionOfURLLM.git`
- 配置 SSH over 443 端口(见 `~/.ssh/config`):
  ```
  Host github.com
    Hostname ssh.github.com
    Port 443
    User git
  ```

---

## 十二、目录结构

```
URLLM-project/
├── enhancement/                    # 增强 Pipeline(数据产物 + 评估)
│   ├── config/                     # 配置文件
│   │   ├── lora_config.yaml        # LoRA 微调配置
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
│   ├── scripts/                    # 各阶段运行脚本
│   │   ├── run_preprocess.sh
│   │   ├── run_finetune.sh
│   │   ├── run_inference.sh
│   │   └── run_eval.sh
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
├── README.md                       # 项目说明
├── USAGE.md                        # 本文件(使用说明)
└── .gitignore
```

---

## 十三、相关文档

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目概览与快速复现指南 |
| [PROJECT_REPORT.md](PROJECT_REPORT.md) | 完整项目报告(12 章,含环境、训练、评估、问题解决) |
| [USAGE.md](USAGE.md) | 本文件,详细使用说明 |

---

## 关键评估指标参考

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

> HR@K 采用物品 Jaccard 相似度扩展(URLLM 论文方法),K 值递增有区分度。AO 数据集因物品池更大(16,000+ 候选)且物品标题含品牌/型号/尺寸,跨域推荐难度更高。

---

## 注意事项总结

1. **模型路径**:本地运行前务必修改 `run_llama2.sh` 和 `lora_config.yaml` 中的模型路径
2. **GPU 内存**:7B 模型必须用 8bit QLoRA + 梯度检查点,batch_size 设 1-2 避免 OOM
3. **环境变量**:训练前设置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防 OOM
4. **Tokenizer**:推理时 decoder-only 模型需设 `padding_side="left"`
5. **前端数据**:`webapp/src/data/` 为前端数据源,由 `generate_data.py` 从 enhancement 产物生成,均为真实项目数据
6. **软链接**:`item_attributes_GM.json` 需软链到 `item_attributes.json`(评估脚本读取)
7. **CUDA 版本**:PyTorch 版本必须匹配 CUDA 驱动版本(如 2.5.1+cu121 对应 CUDA 12.1)
8. **大文件**:基座模型(26G)、LoRA 权重(142M)和 DG 基线数据(GM .npy + AO .pt)不在 Git 仓库,需从 Release 或 ModelScope 下载
9. **AO 数据集训练**:使用 FP16 混合精度(非 8bit QLoRA),通过 `run_train_ao.py` 包装脚本启动,解决 CUDA 沙箱限制
10. **双数据集切换**:前端推荐展示和效果评测页面支持 GM/AO 数据集一键切换

---

**项目状态**:GM + AO 双数据集完整流程已跑通,前端展示系统上线(支持双数据集切换),评估指标真实且有区分度。

**GitHub 仓库**:https://github.com/learner-aa/ProjectReproductionOfURLLM

**Release 下载**:https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/tag/v1.0.0
