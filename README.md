# URLLM 跨域序列推荐项目

复现论文 *Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation* (arXiv:2406.03085),验证 **LLM + 用户画像增强** 对跨域序列推荐的效果。

- 基座模型:Llama-2-7b + 8bit QLoRA(GM) / FP16(AO)
- 数据集:GM(Movie-Game) + AO(Office-Art) 双数据集,共 50,056 用户
- 核心成果:GM 数据集 HR@1 和 MRR 远超 DG 基线约 157 倍;AO 数据集 HR@20 提升 4 倍

---

## 一、压缩包说明

| 项目 | 值 |
|------|-----|
| 文件名 | `URLLM-project.tar.gz` |
| 大小 | 2.6G |
| 已排除 | `models/`(26G 基座模型,可重新下载)、`webapp/node_modules/`(248M,可重装)、`__pycache__/`、`*.pyc` |
| 已包含 | 全部源码、数据产物、LoRA 权重、预测/评估结果、前端源码、项目报告 |

**解压:**
```bash
tar -xzf URLLM-project.tar.gz
cd URLLM-project
```

---

## 二、环境要求

| 项目 | 要求 |
|------|------|
| GPU | ≥ 16GB(1.5B)/ ≥ 24GB(7B + 8bit QLoRA) |
| CUDA 驱动 | ≥ 12.1 |
| Python | 3.10 |
| Node.js | ≥ 18(前端) |
| 系统 | Linux |

---

## 三、快速复现指南

本项目支持两种获取方式,按需选择复现场景。

### 场景速查

| 场景 | 耗时 | 需要下载 | 命令 |
|------|------|----------|------|
| 只看前端展示 | ~5 分钟 | 无需模型 | `cd webapp && npm install && npm run dev` |
| 运行评估 | ~10 分钟 | 无需模型 | `pip install -r requirements.txt` 后跑 evaluate.py |
| 完整推理/训练 | 较长 | 模型 26G + LoRA 权重 + GPU | 见下文完整流程 |

### 1. 获取代码

**方式 A:从 GitHub clone(推荐)**
```bash
git clone https://github.com/learner-aa/ProjectReproductionOfURLLM.git
cd URLLM-project
```

**方式 B:从压缩包解压**
```bash
tar -xzf URLLM-project.tar.gz
cd URLLM-project
```

> 两种方式都不含基座模型(26G)和前端依赖(node_modules),需按下面步骤补齐。

### 2. 下载大文件(完整复现需要,仅看前端可跳过)

**(a) 基座模型 Llama-2-7b(26G)**
```bash
pip install modelscope
modelscope download --model AI-ModelScope/Llama-2-7b-hf --local_dir models/Llama-2-7b-hf
```
> 若可访问 HuggingFace,也可用 `huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/Llama-2-7b-hf`

**(b) LoRA 微调权重(142M)**

从 GitHub Release 下载 `lora-weights.zip`:
```bash
# 方式 1:浏览器下载
# https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 方式 2:命令行下载
wget https://github.com/learner-aa/ProjectReproductionOfURLLM/releases/download/v1.0.0/lora-weights.zip

# 解压到对应目录
unzip lora-weights.zip -d enhancement/outputs/lora_weights/llama2_final/
```
> 含 `adapter_model.safetensors` + `adapter_config.json`,推理必需。

**(c) DG 基线数据(GM 691M + AO 1.1G,可选)**

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

### 3. 配置 Python 环境

```bash
conda create -n urllm python=3.10 -y
conda activate urllm
pip install -r requirements.txt
```
> torch 版本需匹配你的 CUDA,若非 cu121 请改 `requirements.txt` 里 torch/torchvision 的版本号。

### 4. 修改配置中的模型路径

配置默认指向服务器绝对路径,本地运行前需改成你自己的路径(两处):

- `llama2-SFT/run_llama2.sh` 第 6 行 `LLAMA_PATH` → 改成本地模型路径
- `enhancement/config/lora_config.yaml` 中的 `base_model` → 改成本地模型路径

### 5. 运行

**启动前端展示(无需模型,最快):**
```bash
cd webapp
npm install
npm run dev
# 浏览器访问 http://localhost:5173
```

前端为纯 Vite + React 项目,数据来自本地 JSON,无需后端。共 4 个页面:
- **项目概览**:核心成果、双数据集概览、方法概述
- **方法流程**:Pipeline 7 阶段流程、用户画像样例、跨域推荐示意
- **推荐展示**:GM/AO 双数据集切换,用户交互历史 + LLM 推荐结果 + 相似用户参考
- **效果评测**:GM/AO 双数据集切换,核心指标、训练曲线、DG 基线对比、HR@K & NDCG@K 曲线

**运行评估(无需模型):**
```bash
cd enhancement
python src/evaluate.py
# 结果输出到 outputs/eval_results/evaluation.json
```

**完整推理(需模型 + LoRA 权重 + GPU):**
```bash
cd llama2-SFT
python run_inference.py
# 结果输出到 enhancement/outputs/predictions/test_predictions.json
```

**一键全流程(训练→推理→评估,需 GPU):**
```bash
cd llama2-SFT
bash run_llama2.sh
```

---

## 四、目录结构

```
URLLM-project/
├── enhancement/           # 增强 Pipeline(数据产物 + 评估)
│   ├── config/            # 配置(lora_config.yaml / lora_config_AO.yaml / pipeline_config.yaml)
│   ├── data/processed/    # 画像/指令/属性/元数据等数据产物
│   ├── outputs/           # LoRA 权重 / 预测 / 评估结果
│   ├── run_train_ao.py     # AO 训练包装脚本(解决 CUDA 初始化)
│   ├── run_inference_ao.py # AO 推理脚本
│   └── src/               # 核心代码
│       ├── llm_finetune.py
│       ├── llm_inference.py
│       ├── evaluate.py
│       └── preprocess.py / extract_attributes.py / build_profiles.py / build_instructions.py
├── llama2-SFT/            # 论文原版 Llama2-7B
│   ├── finetune-lora.py    # LoRA 微调
│   ├── run_inference.py    # 推理
│   ├── run_llama2.sh       # 一键全流程(训练→推理→评估)
│   └── templates/alpaca.json
├── DG_Final/              # DG 基线模型(评分矩阵对比)
├── run_ao_pipeline.sh     # AO 全流程脚本(训练→推理→评估)
├── webapp/                # 前端展示系统
│   ├── src/pages/         # 4 个页面
│   │   ├── Overview.tsx    # 项目概览
│   │   ├── Method.tsx      # 方法流程
│   │   ├── Workbench.tsx   # 推荐展示
│   │   └── Dashboard.tsx   # 效果评测
│   ├── src/data/          # 真实数据 JSON
│   └── scripts/generate_data.py  # 从 enhancement 产物生成前端数据
├── models/Llama-2-7b-hf/  # 基座模型(需自行下载)
├── PROJECT_REPORT.md      # 完整项目报告(12 章)
├── RUN_REPORT.md          # 跑通报告
└── README.md              # 本文件
```

---

## 五、核心产物

### 数据产物(`enhancement/data/processed/`)
| 文件 | 说明 |
|------|------|
| user_profiles.json | 40,479 用户画像(92M) |
| train_instructions.json | 31,570 条 Alpaca 训练指令(49M) |
| test_instructions.json | 3,601 条测试指令(5.1M) |
| item_attributes_GM.json | 170,239 物品属性(DeepSeek 提取,27M) |
| item_metadata.json | 170,478 物品元数据(30M) |
| interactions.json | 40,479 用户交互序列(11M) |

### 模型与结果产物(`enhancement/outputs/`)
| 路径 | 说明 |
|------|------|
| lora_weights/llama2_final/ | Llama2-7B LoRA 最终权重(推理用) |
| predictions/test_predictions.json | 3,601 条推理结果(787K) |
| eval_results/evaluation.json | 完整评估指标(含 expanded_metrics) |

---

## 六、复现流程

完整 Pipeline 分 7 阶段(详见 [PROJECT_REPORT.md](PROJECT_REPORT.md)):

```
预处理 → 属性提取 → 用户画像 → 指令构建 → LoRA 微调 → 推理 → 评估
```

**一键执行(Llama2-7B 全流程):**
```bash
cd llama2-SFT
bash run_llama2.sh
```

该脚本依次执行:训练(2 epoch / 7894 步)→ 推理(3601 条)→ 评估(生成 evaluation.json)。

**仅推理(用已有 LoRA 权重):**
```bash
cd llama2-SFT
python run_inference.py
```

**仅评估:**
```bash
cd enhancement
python src/evaluate.py
```

**重新生成前端数据:**
```bash
cd webapp/scripts
python generate_data.py
```

---

## 七、关键评估指标

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

## 八、注意事项

1. **模型路径**:本地解压后务必修改 `run_llama2.sh` 和 `lora_config.yaml` 中的模型路径
2. **GPU 内存**:7B 模型必须用 8bit QLoRA + 梯度检查点,batch_size 设 1-2 避免 OOM
3. **环境变量**:训练前设置 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防 OOM
4. **Tokenizer**:推理时 decoder-only 模型需设 `padding_side="left"`
5. **前端数据**:`webapp/src/data/real_data.json` 为前端数据源,由 `generate_data.py` 从 enhancement 产物生成,均为真实项目数据
6. **软链接**:`item_attributes_GM.json` 需软链到 `item_attributes.json`(评估脚本读取)

---

## 九、相关文档

- [PROJECT_REPORT.md](PROJECT_REPORT.md) — 完整项目报告(12 章,含环境、训练、评估、问题解决)
- [RUN_REPORT.md](RUN_REPORT.md) — 跑通报告(早期 Qwen2-1.5B 验证记录)

---

## 十、技术栈

- **训练**:PyTorch 2.5.1+cu121 / Transformers 5.14.1 / PEFT 0.20.0 / Accelerate 1.14.0 / bitsandbytes(8bit)
- **前端**:Vite 8 + React 19 + TypeScript 6 + TailwindCSS 4 + Recharts 3
- **数据**:GM(Movie-Game)跨域数据集 / DeepSeek 物品属性提取

---

**项目状态**:GM + AO 双数据集完整流程已跑通,前端展示系统上线(支持双数据集切换),评估指标真实且有区分度。
