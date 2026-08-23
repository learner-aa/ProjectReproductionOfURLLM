# URLLM 增强跨域推荐 (通用版)

基于论文 *"Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation"* (arXiv:2406.03085) 的方法，在已有 DG 双图序列模型基础上，利用大语言模型进行用户画像构建与推荐增强。

## 项目概述

本项目将 DG 模型产出的结构化特征（656维用户向量）转化为自然语言用户画像，通过 LoRA 微调 LLM 实现跨域推荐增强。支持 **AO** (Art→Office) 与 **GM** (Movie→Game) 两个数据集，两数据集共用一份 `config/pipeline_config.yaml`（切换时改 `dataset.name` 即可），数据与产物按数据集物理隔离（`data/processed/{AO|GM}/`、`outputs/{AO|GM}/`），ID 空间与 DG 索引完全对齐。

**核心流程** (9 阶段):
```
preprocess → extract_attributes → build_profiles → user_retrieval → build_instructions → finetune → inference → refinement → evaluate
```

## 数据说明

本项目 **不解析原始 JSON**, 而是直接复用 DG 模型已对齐的中间产物:

- 物品清单: `{dg_root}/{dataset}/item_list*.csv` (idAfter 即 DG 统一索引)
- 交互序列: `{dg_root}/{dataset}/{train,valid,test}_F*.txt` (物品 id 直接使用 DG 索引)
- 物品属性: `{dg_root}/{dataset}/item_prompt_{AO|GM}/*_exat_*.json` (qqid = DG 索引)
- 特征向量: `{dg_root}/{dataset}/DG/DG{NAME}_final_{train,test}_{x,y}_fea.npy` (行 = txt 行号, 即 dg_index)

预处理阶段将以上产物对齐解析到 `data/processed/{AO|GM}/`, 并产出 `id_mapping.json` 等文件。

## 快速开始

### 1. 环境准备

```bash
# 安装环境 (服务器执行)
bash scripts/setup_env.sh

# 或手动安装
conda create -n urllm python=3.10 -y
conda activate urllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 数据准备

无需单独准备原始数据。确保 DG 模型产物位于可访问路径, 并在 `config/pipeline_config.yaml` 中配置 `dataset.dg_root` 指向 DG 项目根目录:

```
{dg_root}/
├── AO/                     # 或 GM/
│   ├── item_listA_F.csv / item_listO_AA_F.csv
│   ├── train_F2.txt / valid_F2.txt / test_F2.txt
│   ├── DG/DGAO_final_train_{x,y}_fea.npy
│   ├── DG/DGAO_final_test_{x,y}_fea.npy
│   └── DG_src/dataset/item_prompt_AO/*_exat_*.json
└── config.json             # DG 训练配置 (可选)
```

### 3. 一键运行

```bash
# 运行全部阶段 (默认 AO, 由 config/pipeline_config.yaml 中 dataset.name 决定)
python src/run_pipeline.py

# 切换到 GM: 编辑 config/pipeline_config.yaml 中 dataset.name="GM" 等参数后
python src/run_pipeline.py

# 只运行到某个阶段
python src/run_pipeline.py --until build_instructions

# 只运行某个阶段
python src/run_pipeline.py --stage build_profiles

# 预览将要执行的阶段 (不实际运行)
python src/run_pipeline.py --dry-run
```

### 4. 分步执行

```bash
# Step 1: 数据预处理 (默认 AO; 换 GM 传参数 GM)
bash scripts/run_preprocess.sh            # 或: bash scripts/run_preprocess.sh GM

# Step 2: 物品属性提取 (复用 DG 预提取)
python src/run_pipeline.py --stage extract_attributes

# Step 3: 用户画像构建
python src/run_pipeline.py --stage build_profiles

# Step 4: 构建 Instruction 数据
python src/run_pipeline.py --stage build_instructions

# Step 5: LoRA 微调
bash scripts/run_finetune.sh             # 或: bash scripts/run_finetune.sh GM

# Step 6: 推理
bash scripts/run_inference.sh            # 或: bash scripts/run_inference.sh GM

# Step 7: 评估
bash scripts/run_eval.sh                 # 或: bash scripts/run_eval.sh GM
```

> 注: 分步脚本默认操作 AO 数据集, 传 `GM` 参数即切到 GM。所有脚本最终调用 `run_pipeline.py --stage <name> --config config/pipeline_config.yaml`, 确保数据集上下文 (set_dataset) 正确加载。

## 配置说明

### Pipeline 配置 (`config/pipeline_config.yaml`)

| 配置项 | 说明 | AO | GM |
|--------|------|----|----|
| `dataset.name` | 数据集 | `AO` | `GM` |
| `dataset.dg_root` | DG 项目根目录 | `/root/autodl-tmp/URLLM-project/DG_Final` | 同左 |
| `domains.x` | X域名称 | `Art` | `Movie` |
| `domains.y` | Y域名称 | `Office` | `Game` |
| `preprocess.min_interactions` | 最少交互过滤 | `3` | `3` |
| `preprocess.max_seq_len` | 最大序列长度 | `15` | `15` |
| `preprocess.train_file` | 训练文件 | `train_F2.txt` | `train_F.txt` |
| `attribute_extraction.use_precomputed` | 复用 DG 预提取属性 | `true` | `true` |
| `instruction.template_type` | Prompt 模板 | `paper` | `profile` |
| `instruction.output_format` | 数据格式 | `alpaca` | `alpaca` |
| `retrieval.k_train` | 训练检索 top-k | `2` | `2` |
| `retrieval.k_test` | 测试检索 top-k | `3` | `3` |

### LoRA 配置 (`config/lora_config.yaml`)

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `model.base_model` | 基座模型 | `meta-llama/Llama-2-7b-hf` (base 版, 非 chat) |
| `model.lora_r` | LoRA rank | `8` |
| `model.lora_alpha` | LoRA alpha | `16` |
| `training.num_epochs` | 训练轮数 | `3` |
| `training.batch_size` | 批大小 | `4` |
| `training.learning_rate` | 学习率 | `1e-4` |
| `training.max_seq_length` | 最大序列长度 | `1024` |
| `inference.batch_size` | 推理批大小 | `8` |
| `inference.temperature` | 生成温度 | `0.1` |

## 已有模型产物

本项目复用已训练的 DG 双图序列模型产物 (路径随数据集自动切换):

| 文件 | 形状 | 说明 |
|------|------|------|
| `{dg_root}/{dataset}/DG/DG{NAME}_final_train_x_fea.npy` | (N_train, 656) | 训练用户 X 域向量 |
| `{dg_root}/{dataset}/DG/DG{NAME}_final_train_y_fea.npy` | (N_train, 656) | 训练用户 Y 域向量 |
| `{dg_root}/{dataset}/DG/DG{NAME}_final_test_x_fea.npy` | (N_test, 656) | 测试用户 X 域向量 |
| `{dg_root}/{dataset}/DG/DG{NAME}_final_test_y_fea.npy` | (N_test, 656) | 测试用户 Y 域向量 |
| `{dg_root}/{dataset}/saver/best_trte_XORY_DG*.npy` | (N_test, N_item) | 测试评分矩阵 (回退用) |
| `{dg_root}/config.json` | — | DG 模型训练配置 |

特征矩阵**行号 = txt 行号 = `dg_index`**, 与 `train.json` / `test.json` 中的样本一一对应。

## 数据集切换

AO 与 GM 共用一份 `config/pipeline_config.yaml`, 切换时修改其中的 `dataset.name` / `domains` / `preprocess.*_file` 等参数即可。`data/processed` 与 `outputs` 由 `set_dataset` 自动切到对应子目录 (物理隔离)。

切换到 GM 的步骤:
1. 编辑 `config/pipeline_config.yaml`
2. 设置 `dataset.name: "GM"`
3. 设置 `domains.x: "Movie"`, `domains.y: "Game"`
4. 设置 `preprocess.train_file: "train_F.txt"` 等 (见文件内注释)
5. 运行 `python src/run_pipeline.py`

- `AO`: Art→Office, 8000 训练用户 × 2 行 (X/Y 目标各一) = 16000 训练样本, 2000 测试样本。目标 = 每行最后一个物品。模板 `paper`。
- `GM`: Movie→Game, 35941 训练用户 × 1 行, 3601 测试样本。目标 = `ts == user_ts` 的物品。模板 `profile`。

## Quick Reproduction Guide（快速复现指南）

克隆仓库后，基座模型与 DG 大文件不会随 Git 一起下载（受单文件 100MB 限制，通过 [GitHub Release](../../releases) 分发）。按以下两步补齐即可复现。

### Step A. 下载基座模型 (Llama-2-7b-hf, base 版)

训练与推理**必须使用 base 版**（非 chat 版），否则会出现乱码/0 命中。国内推荐用 ModelScope 镜像 `shakechen/Llama-2-7b-hf`：

```bash
# 在项目根目录执行
pip install -U modelscope
mkdir -p models/Llama-2-7b-hf/models/shakechen--Llama-2-7b-hf/snapshots
modelscope download --model shakechen/Llama-2-7b-hf \
  --local_dir models/Llama-2-7b-hf/models/shakechen--Llama-2-7b-hf/snapshots/master
```

下载完成后，`config/lora_config.yaml` 中 `model.base_model` 已默认指向该路径，无需改动：
```
/root/autodl-tmp/URLLM-project/models/Llama-2-7b-hf/models/shakechen--Llama-2-7b-hf/snapshots/master
```
> 如部署在其他机器，请把该路径改为本机实际绝对路径（必须与训练时一致，见 `outputs/AO/lora_weights/final/adapter_config.json` 的 `base_model_name_or_path`）。

### Step B. 下载大文件（LoRA 权重 / DG 向量 / GM 数据）

以下文件 >100MB 或属于 DG 产物，已从 Git 仓库排除，统一放在 GitHub Release：

| Release 附件 | 大小 | 用途 | 解压目标路径 |
|--------------|------|------|--------------|
| `adapter_model.safetensors` (AO) | ~17MB | AO LoRA 权重 | `enhancement/outputs/AO/lora_weights/final/` |
| `DGAO_final_train_x_fea.npy` 等 4 个 | ~百MB | AO 训练/测试用户向量 | `DG_Final/AO/DG/` |
| `best_trte_XORY_DG*.npy` | 大 | AO 测试评分矩阵(回退) | `DG_Final/AO/saver/` |
| `train.json` / `interactions.json` 等 (GM) | 60-88MB | GM 数据集 | `enhancement/data/processed/GM/` |

下载方式（在 Release 页面手动下载，或用 `gh` CLI）：

```bash
# 列出 Release 附件
gh release download <tag> --repo learner-aa/ProjectReproductionOfURLLM --pattern "*.safetensors" --dir enhancement/outputs/AO/lora_weights/final/

# 或浏览器打开: https://github.com/learner-aa/ProjectReproductionOfURLLM/releases
# 逐个下载并按上表路径放置
```

### Step C. 复现运行

补齐模型与大文件后，按"快速开始 → 一键运行"执行即可：

```bash
conda activate urllm
cd enhancement
python src/run_pipeline.py --dry-run          # 先预览 9 阶段是否齐全
python src/run_pipeline.py --stage inference  # 如仅想复现推理/评估，可跳过 finetune
python src/run_pipeline.py --stage refine_answers
python src/run_pipeline.py --stage evaluate
```

## 服务器部署

### 硬件要求

| 资源 | 最低 | 推荐 |
|------|------|------|
| GPU | 1× A100 40G | 2-4× A100 40G |
| 内存 | 32 GB | 64 GB |
| 磁盘 | 100 GB | 200 GB |

### 上传文件

```bash
# 将以下目录/文件上传到服务器
scp -r src/ config/ scripts/ requirements.txt README.md struct.md server:~/enhance/
```

DG 模型产物保留在原路径, 通过 `config/pipeline_config.yaml` 的 `dataset.dg_root` 指向即可, 无需拷贝。

### 服务器执行

```bash
ssh server
cd ~/enhance
bash scripts/setup_env.sh          # 安装环境
# 修改 config/pipeline_config.yaml 中 dataset.dg_root 指向 DG 产物, dataset.name 选择数据集
python src/run_pipeline.py   # 跑当前配置的数据集
```

## 项目结构

详见 [struct.md](struct.md)

## 参考

- 论文: [arXiv:2406.03085](https://arxiv.org/abs/2406.03085)
- 代码: [github.com/TingJShen/URLLM](https://github.com/TingJShen/URLLM)
