# AO 数据集改造方案:对齐原论文处理机制

> 本文档记录 URLLM 项目针对 AO(Office → Art)数据集 HR@1=0 问题的改造方案,
> 核心思路是仿照原论文的"DG 检索 + 相似用户序列拼接"机制,把 LLM 从"凭画像生成 SKU"
> 改造为"从相似用户已知序列中模仿生成"。
>
> 创建日期:2026-08-18

---

## 一、问题背景与现状

### 1.1 当前 AO 数据集效果(改造前)

| 指标 | LLM+画像+精炼 | DG 基线 |
|------|-------------|---------|
| exact HR@1 | **0.0000** | 0.0000 |
| fuzzy HR@1 | 0.0090 | — |
| partial HR@1 | 0.0045 | — |
| OOD 率(原始) | 49.95% | — |
| 精炼后 OOD 率 | 50.00% | — |
| fallback_to_dg | **97.9%** | — |
| 冷启动用户占比 | 94.1% | — |

**结论**:AO 数据集 exact HR@1=0 的根因是"生成式推荐 + 精确 SKU 标题 + 94% 冷启动 + 训练样本不足"四重困境叠加,LLM 在当前架构下无法学会从用户画像生成精确含品牌+SKU 的物品标题。

### 1.2 原论文的处理机制(关键差异)

原论文(`URLLM-master`)并不是让 LLM 凭空生成 SKU,而是用了一套 **"DG 检索 + 相似用户序列拼接"** 机制:

1. **DG 模型先做重活**:用对比学习 MLP 把"源域用户表征"映射到"目标域用户表征",最大化相似用户的相似度,生成 `matmul_trte` 相似度矩阵
2. **prompt 拼接相似用户序列**:把 top-1 相似用户买过的物品标题**全部塞进 prompt**,让 LLM 从已知物品列表中"模仿/挑选"
3. **训练充分**:20 epochs + DeepSpeed 多卡 + batch=256 + 8bit QLoRA
4. **评估口径宽松**:用 `ground_truth in full_seq`(检索式 HR),而非精确字符串匹配

### 1.3 当前项目已有但未启用的基础设施

| 模块 | 现状 | 是否启用 |
|------|------|---------|
| KNN 用户检索器 [knn_retriever.py](enhancement/src/knn_retriever.py) | 已实现,但用"物品均值近似"向量 | ❌ 未配置真实 DG 向量 |
| 检索 prompt 模板 `PROMPT_II_RECOMMEND_WITH_RETRIEVAL` | 已实现 [prompt_templates.py#L102](enhancement/src/prompt_templates.py#L102) | ❌ AO 用的是 `profile` 模板 |
| `load_real_vectors()` 接口 | 已实现 [knn_retriever.py#L175](enhancement/src/knn_retriever.py#L175) | ❌ 未调用 |
| 检索结果文件 `retrieval_results_AO.json` | 流程已打通 | ❌ 未生成 |

---

## 二、改造目标与预期

### 2.1 改造目标

| 指标 | 改造前 | 改造后(预期) | 依据 |
|------|--------|------------|------|
| exact HR@1 | 0.0000 | 0.001-0.005 | 原论文 AO 范围 |
| retrieval HR@1 | N/A | 0.01-0.05 | 原论文口径 |
| OOD 率 | 49.95% | <20% | 相似用户序列提供域内物品 |
| fallback_to_dg | 97.9% | <50% | LLM 有参考序列可模仿 |
| 训练步数 | 3,000 | 6,000+ | 对齐原论文 20 epochs |

### 2.2 不改造的边界

- 仍用单卡 RTX 4090D 24GB(不引入多卡/DeepSpeed)
- 仍用 Llama2-7B 基座(不换更大模型)
- 仍用 LoRA 微调(不做全参数 SFT)
- 物品 ID 仍用标题(不改为 ID 生成方案)

---

## 三、改造阶段与详细步骤

### 阶段 A:DG 真实向量接入(基础)

**目的**:把 KNN 从"物品均值近似"升级为"DG 对比学习真实输出"

#### A.1 盘点 DG 产物

检查 `URLLM-project/DG_Final/AO/` 下是否存在以下文件:

| 文件 | 用途 | 来源 |
|------|------|------|
| `DG_390/DGAO_final_train_x_fea.npy` | 训练用户 X 域向量 | DG 模型训练输出 |
| `DG_390/DGAO_final_train_y_fea.npy` | 训练用户 Y 域向量 | DG 模型训练输出 |
| `DGAO_final_test_x_fea.npy` | 测试用户 X 域向量 | DG 模型推理输出 |
| `DGAO_final_test_y_fea.npy` | 测试用户 Y 域向量 | DG 模型推理输出 |
| `XORY_train.npy` | 训练用户跨域标识 | 数据预处理 |
| `XORY_test.npy` | 测试用户跨域标识 | 数据预处理 |
| `best_trte_XORY_DG_390_.npy` | 对比学习相似度矩阵 | 对比学习 MLP 输出 |

**若缺失**:需先跑原论文的 [Final_train_contrasive_searcher.py](file:///root/autodl-tmp/ours/URLLM-master/DG_Final/AO/Final_train_contrasive_searcher.py)(对比学习 MLP,生成 matmul_trte)

#### A.2 修改配置

**文件**:`enhancement/config/pipeline_config_AO.yaml`

在末尾追加:

```yaml
# KNN 检索配置(对齐原论文)
retrieval:
  k_train: 1                              # 原论文只取 top-1 相似用户
  k_test: 1                               # 原论文只取 top-1
  real_vectors_x: "DG_Final/AO/DG_390/DGAO_final_train_x_fea.npy"
  real_vectors_y: "DG_Final/AO/DG_390/DGAO_final_train_y_fea.npy"
  use_real_vectors: true                  # 启用真实 DG 向量
```

#### A.3 重新跑检索阶段

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import yaml
from pathlib import Path
from src.knn_retriever import run_user_retrieval
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
run_user_retrieval(cfg)
"
```

**预期输出**:`enhancement/data/processed/retrieval_results_AO.json`

---

### 阶段 B:Prompt 改造(对齐原论文核心机制)

**目的**:把相似用户的**完整购买序列**拼进 prompt,让 LLM 从已知物品中"模仿"而非凭空生成

#### B.1 改造检索模板

**文件**:`enhancement/src/prompt_templates.py`

将 `PROMPT_II_RECOMMEND_WITH_RETRIEVAL`(L102)改造为对齐原论文格式:

```python
PROMPT_II_RECOMMEND_WITH_RETRIEVAL = """Instruction: Given a list of {target_domain} items the user has interacted with, along with some similar users who have similar interaction histories, please recommend a new {target_domain} item that the user would likely enjoy.

Input:
=== Similar Users ===
{retrieved_users_text}

=== Target User Interaction History ===
{interaction_sequence}

=== Target User Profile ===
{user_profile_text}

Please recommend ONE specific {target_domain} item title.
Output:"""
```

**关键变更**:强调"similar users"在前,作为 LLM 的主要参考来源。

#### B.2 改造相似用户序列文本生成

**文件**:`enhancement/src/knn_retriever.py`

修改 `get_retrieved_user_text` 方法(L279-L324):

```python
def get_retrieved_user_text(
    self,
    retrieved_user_ids: List[str],
    max_items_per_user: int = 5,   # 保留参数,但实际按 token 截断
    max_tokens: int = 512,          # 新增:对齐原论文 max_tk
) -> str:
    """
    将检索到的用户的交互序列格式化为 few-shot 文本(对齐原论文)。
    
    原论文格式:
      "There is another similar user who has played arts or office before: 
       art:item1 | office:item2 | art:item3 | ..."
    """
    lines = []
    current_tokens = 0
    for uid in retrieved_user_ids:
        items = self.interactions.get(uid, [])
        if not items:
            continue

        # 原论文:全序列拼接,按 token 长度截断
        item_texts = []
        for iid in items:  # 不再用 items[-max_items_per_user:],而是全序列
            meta = self.item_metadata.get(iid, {})
            title = meta.get("title", iid)
            domain = meta.get("domain", "")
            
            # 域标签:对齐原论文口径 "art" / "office"
            if domain in (self.domain_x_name, "X"):
                label = "art"      # AO 的 X 域是 Art
            elif domain in (self.domain_y_name, "Y"):
                label = "office"   # AO 的 Y 域是 Office
            else:
                label = "item"
            
            item_text = f"{label}:{title}"
            
            # token 长度截断(原论文 max_tk=512)
            estimated_tokens = len(item_text.split())
            if current_tokens + estimated_tokens > max_tokens:
                break
            current_tokens += estimated_tokens
            item_texts.append(item_text)

        if item_texts:
            lines.append(
                "There is another similar user who has interacted with: "
                + " | ".join(item_texts)
            )

    return "\n".join(lines) if lines else "(no similar users found)"
```

#### B.3 启用检索模板

**文件**:`enhancement/config/pipeline_config_AO.yaml`

修改 L34:

```yaml
instruction:
  template_type: "retrieval"     # 从 "profile" 改为 "retrieval"
  output_format: "alpaca"
```

#### B.4 重建 instruction 数据

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import yaml
from src.build_instruction_data import build_all_instruction_data
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
build_all_instruction_data(cfg)
"
```

**预期输出**:
- `enhancement/data/processed/train_instructions_AO.json`(含相似用户序列)
- `enhancement/data/processed/test_instructions_AO.json`

---

### 阶段 C:训练配置对齐原论文(单卡可行版)

**目的**:把训练规模从"3 epochs / batch=8"提升到与原论文可比的水平

#### C.1 修改训练配置

**文件**:`enhancement/config/lora_config_AO.yaml`

```yaml
# ============================================================
# LoRA 微调配置 (AO 数据集 - 对齐原论文)
# ============================================================

model:
  base_model: "/root/autodl-tmp/URLLM-project/models/Llama-2-7b-hf/models/shakechen--Llama-2-7b-hf/snapshots/master"
  lora_r: 8                    # 对齐原论文(从 16 改回 8)
  lora_alpha: 16              # 对齐原论文(从 32 改回 16)
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  load_in_bits: 8              # 启用 8bit QLoRA(从 FP16 改 8bit)

training:
  num_epochs: 20              # 对齐原论文(若显存不足先试 10)
  max_steps: -1
  batch_size: 1
  gradient_accumulation_steps: 8   # 单卡约束,等效 batch=8
  learning_rate: 1.0e-4
  warmup_steps: 400          # 对齐原论文(替代 ratio=0.03)
  max_seq_length: 1024
  weight_decay: 0.01
  fp16: false                # 8bit 模式下关闭
  bf16: false
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  save_total_limit: 5
  gradient_checkpointing: true

data:
  train_file: "train_instructions_AO.json"
  valid_file: "valid_instructions_AO.json"

inference:
  batch_size: 8
  temperature: 0.1
  max_new_tokens: 128
  num_beams: 4                # 新增:对齐原论文 beam=4

output:
  save_dir: "lora_weights_AO_v2"   # 新目录,不覆盖原权重
  log_dir: "logs_AO_v2"
```

#### C.2 修改训练脚本支持 8bit

**文件**:`enhancement/src/llm_finetune.py`

关键修改点:

```python
# 1. 加载 8bit 量化配置
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 2. 不要调用 prepare_model_for_kbit_training(已知 OOM 坑)
# 改为:
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 3. 设置环境变量防 OOM
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

#### C.3 重新训练

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_train_ao.py
```

**监控点**:
- eval_loss 应单调下降,无反弹
- 若 epoch 10+ 出现 eval_loss 反弹,可早停
- 显存监控:`nvidia-smi`(应稳定在 20-22GB)

---

### 阶段 D:推理配置对齐原论文

**目的**:用 beam search 提升生成质量

#### D.1 修改推理参数

**文件**:`enhancement/src/llm_inference.py`

```python
# 关键:启用 beam search
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,              # 从 1 改为 4(对齐原论文)
    max_new_tokens=128,
    return_dict_in_generate=True,
    output_scores=True,
)
```

#### D.2 重新推理

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python run_inference_ao.py
```

**预期输出**:`enhancement/outputs/predictions/test_predictions_AO_v2.json`

---

### 阶段 E:评估改造(增加原论文口径)

**目的**:增加"检索式 HR"评估,同时保留精确匹配作为对照

#### E.1 增加检索式 HR 指标

**文件**:`enhancement/src/evaluate.py`

新增函数:

```python
def retrieval_hit_rate(
    retrieved_user_seqs: List[List[str]],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    原论文口径:ground_truth 是否在 top-K 相似用户的训练序列里。
    
    Args:
        retrieved_user_seqs: 每个测试用户对应的相似用户购买序列
        ground_truths: 真实物品标题列表
    
    Returns:
        {"retrieval_HR@1": float, "retrieval_HR@5": float, ...}
    """
    metrics = {}
    for k in [1, 5, 10, 20]:
        hits = 0
        for seq, gt in zip(retrieved_user_seqs, ground_truths):
            # 取 top-K 相似用户的序列合集
            top_k_seq = seq[:k] if isinstance(seq[0], list) else seq
            if gt.lower().strip() in [s.lower().strip() for s in top_k_seq]:
                hits += 1
        metrics[f"retrieval_HR@{k}"] = hits / len(ground_truths) if ground_truths else 0.0
    return metrics
```

#### E.2 修改评估输出结构

**文件**:`enhancement/src/evaluate.py`

在 `evaluation_AO.json` 输出中新增字段:

```json
{
  "exact_metrics": {...},          // 保留:精确字符串匹配
  "fuzzy_metrics": {...},          // 保留:模糊匹配
  "retrieval_metrics": {           // 新增:原论文口径
    "retrieval_HR@1": 0.025,
    "retrieval_HR@5": 0.045,
    ...
  },
  "expanded_metrics": {...},       // 保留:BM25 扩展
  "cold_warm": {...},
  "out_of_domain": {...}
}
```

#### E.3 修复已知 bug

**文件**:`enhancement/src/build_instruction_data.py`

修复 `target_domain` 硬编码问题(见 [DATASETS_STATUS.md#L290-L296](DATASETS_STATUS.md#L290)):

```python
# 修复前(L60):
# target_domain = target_item.get("domain", domain_x_name)

# 修复后:从配置读取
target_domain = target_item.get("domain", "")
if not target_domain:
    target_domain = domain_x_name  # 从 config 传入,而非硬编码 "Entertainment"
```

**文件**:`enhancement/src/evaluate.py`

修复 `ood_count` 非整数问题:

```python
# 修复前:
# "ood_count": ood_count,   # 可能是 499.5

# 修复后:
"ood_count": int(round(ood_count)),
```

#### E.4 重新评估

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import yaml
from src.evaluate import run_evaluation
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
run_evaluation(cfg)
"
```

**预期输出**:`enhancement/outputs/eval_results/evaluation_AO_v2.json`

---

### 阶段 F:验证与文档更新

#### F.1 指标对比记录

| 指标 | 改造前(3 epochs) | 改造后(20 epochs + retrieval) | 变化 |
|------|-------------------|------------------------------|------|
| exact HR@1 | 0.0000 | (待填) | — |
| fuzzy HR@1 | 0.0090 | (待填) | — |
| partial HR@1 | 0.0045 | (待填) | — |
| retrieval HR@1 | N/A | (待填) | 新指标 |
| OOD 率 | 49.95% | (待填) | — |
| fallback_to_dg | 97.9% | (待填) | — |
| eval_loss | 0.3550 | (待填) | — |
| 训练步数 | 3,000 | (待填) | — |

#### F.2 文档更新清单

- [ ] `DATASETS_STATUS.md`:新增"九、AO 改造方案实施"章节
- [ ] `PROJECT_REPORT.md`:更新 AO 方法论和指标
- [ ] `README.md`:更新 pipeline 描述
- [ ] 前端 `webapp/src/data/eval_data.json`:同步 AO 新指标
- [ ] 前端 `webapp/src/data/training_logs.json`:同步 AO 新训练日志
- [ ] 前端 `webapp/src/data/datasets.json`:同步 AO 推荐展示数据

---

## 四、改造依赖关系与执行顺序

```
A. DG 向量接入  
   │
   ▼
B. Prompt 改造(相似用户序列拼接)
   │
   ▼
C. 训练(20 epochs + 8bit + r=8)   ← 最耗时阶段
   │
   ▼
D. 推理(beam=4)
   │
   ▼
E. 评估(新增检索式 HR 口径)
   │
   ▼
F. 文档与前端同步
```

**关键路径**:A → B → C → D → E → F(严格顺序,不可并行)

---

## 五、风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| DG 真实向量缺失 | 阶段 A 阻塞 | 先用"物品均值近似"跑通流程,再补真实向量 |
| 20 epochs 显存不足 | 阶段 C 失败 | 退而求其次用 10 epochs + 8bit QLoRA |
| 20 epochs 时间过长 | 阶段 C 耗时 | 用 10 epochs,监控 eval_loss 早停 |
| beam=4 推理慢 | 阶段 D 耗时 | 退而求其次用 beam=2 |
| 训练仍 OOM | 阶段 C 失败 | 用 GM 数据集配置(r=8/alpha=16)做参照 |
| 相似用户序列过长导致 prompt 溢出 | 阶段 B 失败 | 减小 max_tokens=256 或减少相似用户数 |

---

## 六、环境与命令速查

### 6.1 环境变量

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/tmp/node-v20.18.0-linux-x64/bin:$PATH"   # 前端构建用
```

### 6.2 关键路径

| 类别 | 路径 |
|------|------|
| AO 配置 | `enhancement/config/pipeline_config_AO.yaml` |
| AO 训练配置 | `enhancement/config/lora_config_AO.yaml` |
| AO 训练入口 | `enhancement/run_train_ao.py` |
| AO 推理入口 | `enhancement/run_inference_ao.py` |
| AO 检索结果 | `enhancement/data/processed/retrieval_results_AO.json` |
| AO 训练数据 | `enhancement/data/processed/train_instructions_AO.json` |
| AO 测试数据 | `enhancement/data/processed/test_instructions_AO.json` |
| AO LoRA 权重(新) | `enhancement/outputs/lora_weights_AO_v2/` |
| AO 预测结果(新) | `enhancement/outputs/predictions/test_predictions_AO_v2.json` |
| AO 评估结果(新) | `enhancement/outputs/eval_results/evaluation_AO_v2.json` |

### 6.3 一键执行命令(顺序执行)

```bash
cd /root/autodl-tmp/URLLM-project/enhancement

# 阶段 A:检索
python -c "import yaml; from src.knn_retriever import run_user_retrieval; run_user_retrieval(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"

# 阶段 B:重建 instruction
python -c "import yaml; from src.build_instruction_data import build_all_instruction_data; build_all_instruction_data(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"

# 阶段 C:训练(最耗时)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_train_ao.py

# 阶段 D:推理
python run_inference_ao.py

# 阶段 E:评估
python -c "import yaml; from src.evaluate import run_evaluation; run_evaluation(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"
```

---

## 七、参考资料

- 原论文仓库:`/root/autodl-tmp/ours/URLLM-master/`
- 原论文训练脚本:[finetune-lora.sh](file:///root/autodl-tmp/ours/URLLM-master/llama2-SFT/finetune-lora.sh)
- 原论文 prompt 构建:[jsinBuilder_testing_rc.py](file:///root/autodl-tmp/ours/URLLM-master/DG_Final/AO/jsinBuilder_testing_rc.py)
- 原论文对比学习:[Final_train_contrasive_searcher.py](file:///root/autodl-tmp/ours/URLLM-master/DG_Final/AO/Final_train_contrasive_searcher.py)
- 当前数据集状态:[DATASETS_STATUS.md](DATASETS_STATUS.md)

---

## 八、改造总结

**核心一句话**:把 AO 从"凭画像生成 SKU"改为"凭相似用户的购买序列模仿生成"——即启用 `retrieval` 模板 + 真实 DG 向量 + 全序列拼接 + beam=4 解码 + 原论文训练规模(20 epochs / r=8)+ 检索式 HR 评估口径,让 LLM 从"已知物品列表"中模仿,而非"凭空记忆 16000+ SKU"。

**预期效果**:exact HR@1 从 0.0000 提升至 0.001-0.005(原论文 AO 范围),OOD 率从 49.95% 降至 20% 以下,fallback_to_dg 从 97.9% 降至 50% 以下。

**实施原则**:严格按 A→B→C→D→E→F 顺序执行,每阶段完成后验证输出文件再进入下一阶段。
