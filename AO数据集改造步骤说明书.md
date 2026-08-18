# AO 数据集改造步骤详细说明书(5 epochs 版)

> 本文档是 URLLM 项目针对 AO(Office → Art)数据集 HR@1=0 问题的可执行改造说明书。
> 核心思路:启用 retrieval prompt + 真实 DG 向量(已存在)+ 全序列拼接 + beam=4 解码,
> 让 LLM 从"凭画像生成 SKU"改造为"从相似用户已知序列中模仿生成"。
>
> 创建日期:2026-08-18
> 训练轮数:5 epochs
> 预计总耗时:13-18 小时

---

## 一、背景与现状

### 1.1 AO 数据集当前效果(改造前)

| 指标 | LLM+画像+精炼 | DG 基线 |
|------|-------------|---------|
| exact HR@1 | **0.0000** | 0.0000 |
| fuzzy HR@1 | 0.0090 | — |
| partial HR@1 | 0.0045 | — |
| OOD 率(原始) | 49.95% | — |
| 精炼后 OOD 率 | 50.00% | — |
| fallback_to_dg | **97.9%** | — |
| 冷启动用户占比 | 94.1% | — |

### 1.2 根因分析

AO 数据集 HR@1=0 的根因是"生成式推荐 + 精确 SKU 标题 + 94% 冷启动 + 训练样本不足"四重困境叠加,LLM 在当前架构下无法学会从用户画像生成精确含品牌+SKU 的物品标题。

### 1.3 原论文的处理机制

原论文(`URLLM-master`)用 **"DG 检索 + 相似用户序列拼接"** 机制绕开精确记忆问题:
1. DG 模型先做重活:用对比学习 MLP 生成相似度矩阵
2. prompt 拼接相似用户序列:把 top-1 相似用户买过的物品标题全部塞进 prompt
3. LLM 从已知物品列表中"模仿/挑选",而非凭空生成

### 1.4 已有资源核查(关键)

| 资源 | 状态 | 路径 |
|------|------|------|
| AO DG 训练用户 X 域向量 | ✅ 已存在 | `DG_Final/AO/DG/DGAO_final_train_x_fea.npy` (41MB) |
| AO DG 训练用户 Y 域向量 | ✅ 已存在 | `DG_Final/AO/DG/DGAO_final_train_y_fea.npy` (41MB) |
| AO DG 测试用户 X 域向量 | ✅ 已存在 | `DG_Final/AO/DG/DGAO_final_test_x_fea.npy` (5.1MB) |
| AO DG 测试用户 Y 域向量 | ✅ 已存在 | `DG_Final/AO/DG/DGAO_final_test_y_fea.npy` (5.1MB) |
| AO 对比学习相似度矩阵 | ✅ 已存在 | `DG_Final/AO/saver/best_trte_XORY_DG_390_.npy` (123MB) |
| AO DG 推理结果矩阵 | ✅ 已存在 | `DG_Final/AO/DG/t4_G2_final_DGresult_matmul_trte_AO.npy` (245MB) |
| AO DG 候选物品矩阵 | ✅ 已存在 | `DG_Final/AO/DG/t4_G2_final_DGresult_test_candidate_AO.npy` (153MB) |
| 4 个 AO train/test 向量软链接 | ✅ 已配 | `enhancement/saved_models/` |
| best_trte_XORY_DG_390_.npy 软链接 | ❌ 未配 | 需补 |
| KNN 检索器 load_real_vectors 接口 | ✅ 已实现 | `knn_retriever.py#L175` |
| PROMPT_II_RECOMMEND_WITH_RETRIEVAL 模板 | ✅ 已实现 | `prompt_templates.py#L102` |
| AO 训练数据 | ✅ 已存在 | `data/processed/train_instructions_AO.json` (8000 条) |

---

## 二、改造目标

### 2.1 指标目标

| 指标 | 改造前 | 改造后(预期) |
|------|--------|------------|
| exact HR@1 | 0.0000 | 0.001-0.005 |
| fuzzy HR@1 | 0.0090 | 0.015-0.030 |
| OOD 率 | 49.95% | <30% |
| fallback_to_dg | 97.9% | <60% |

### 2.2 不改造的边界

- 仍用单卡 RTX 4090D 24GB
- 仍用 Llama2-7B 基座
- 仍用 LoRA 微调(不做全参数 SFT)
- 物品 ID 仍用标题(不改为 ID 生成方案)
- 不引入 8bit QLoRA(保留 FP16,避免破坏 GM 流程)

---

## 三、改造阶段总览

| 阶段 | 任务 | 时间 | 关键产物 |
|------|------|------|---------|
| 0 | 准备与备份 | 30 分钟 | git 提交 + 数据备份 |
| 1 | DG 向量接入与验证 | 1 小时 | 软链接 + shape 对齐验证 |
| 2 | Prompt 改造 | 30 分钟 | retrieval 模板启用 |
| 3 | 训练配置 | 30 分钟 | lora_config_AO.yaml |
| 4 | 训练(5 epochs) | 7-10 小时 | lora_weights_AO_v2/ |
| 5 | 推理(beam=4) | 2-3 小时 | test_predictions_AO_v2.json |
| 6 | 评估 | 30 分钟 | evaluation_AO_v2.json |
| 7 | 文档与前端同步 | 1-2 小时 | 6 个文件更新 |
| **总计** | — | **13-18 小时** | — |

**关键路径**:0 → 1 → 2 → 3 → 4 → 5 → 6 → 7(严格顺序)

---

## 四、阶段 0:准备与备份(30 分钟)

### 4.1 目的
确保改造可回滚,不破坏现有可用状态。

### 4.2 步骤

#### 步骤 0.1:git 提交当前状态
```bash
cd /root/autodl-tmp/URLLM-project
git status
git add -A
git commit -m "保存 AO 改造前的稳定状态"
```

#### 步骤 0.2:备份关键数据
```bash
cd /root/autodl-tmp/URLLM-project/enhancement/data/processed
cp train_instructions_AO.json train_instructions_AO.backup.json
cp test_instructions_AO.json test_instructions_AO.backup.json
cp retrieval_results_AO.json retrieval_results_AO.backup.json 2>/dev/null || echo "无 retrieval_results_AO.json"
```

#### 步骤 0.3:创建改造分支
```bash
git checkout -b feature/ao-retrieval-revamp
```

### 4.3 验证标准
- ✅ git log 显示最新提交
- ✅ 备份文件存在(train_instructions_AO.backup.json 等)
- ✅ 当前在 feature/ao-retrieval-revamp 分支

---

## 五、阶段 1:DG 向量接入与验证(1 小时)

### 5.1 目的
补齐 AO 软链接,验证用户 ID 顺序对齐(关键风险点)。

### 5.2 步骤

#### 步骤 1.1:补 2 个软链接(5 分钟)
```bash
cd /root/autodl-tmp/URLLM-project/enhancement/saved_models
ln -sf /root/autodl-tmp/URLLM-project/DG_Final/AO/saver/best_trte_XORY_DG_390_.npy best_trte_XORY_DG_390_.npy
ln -sf /root/autodl-tmp/URLLM-project/DG_Final/AO/DG/t4_G2_final_DGresult_test_candidate_AO.npy t4_G2_final_DGresult_test_candidate_AO.npy

# 验证
ls -lh best_trte_XORY_DG_390_.npy t4_G2_final_DGresult_test_candidate_AO.npy
```

#### 步骤 1.2:验证 .npy shape 与用户数对齐(15 分钟)
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import numpy as np
import json

# 加载 .npy 检查 shape
train_x = np.load('saved_models/DGAO_final_train_x_fea.npy')
train_y = np.load('saved_models/DGAO_final_train_y_fea.npy')
test_x = np.load('saved_models/DGAO_final_test_x_fea.npy')
test_y = np.load('saved_models/DGAO_final_test_y_fea.npy')

print(f'train_x shape: {train_x.shape}')   # 期望 (16000, 656)
print(f'train_y shape: {train_y.shape}')   # 期望 (16000, 656)
print(f'test_x shape: {test_x.shape}')     # 期望 (2000, 656)
print(f'test_y shape: {test_y.shape}')     # 期望 (2000, 656)

# 加载本项目的训练用户 ID 检查数量
with open('data/processed/id_mapping_AO.json') as f:
    id_map = json.load(f)
print(f'本项目 train_user_ids 数量: {len(id_map.get(\"user_id_to_index\", {}))}')

# 加载相似度矩阵检查
sims = np.load('saved_models/best_trte_XORY_DG_390_.npy')
print(f'相似度矩阵 shape: {sims.shape}')   # 期望 (2000, 16000)
"
```

**验证标准**:
- ✅ train_x.shape[0] == 16000
- ✅ test_x.shape[0] == 2000
- ✅ 相似度矩阵 shape = (2000, 16000)
- ❌ 若 shape 不对齐,**停止改造**,排查 id_mapping

#### 步骤 1.3:修改 pipeline_config_AO.yaml(5 分钟)

**文件**:`enhancement/config/pipeline_config_AO.yaml`

在末尾追加:
```yaml
# KNN 检索配置(对齐原论文)
retrieval:
  k_train: 1                              # 原论文只取 top-1 相似用户
  k_test: 1                               # 原论文只取 top-1
  use_real_vectors: true                  # 启用真实 DG 向量
  real_vectors_x: "saved_models/DGAO_final_train_x_fea.npy"
  real_vectors_y: "saved_models/DGAO_final_train_y_fea.npy"
```

#### 步骤 1.4:跑 KNN 检索(30 分钟)
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import yaml
from src.knn_retriever import run_user_retrieval
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
run_user_retrieval(cfg)
"
```

### 5.3 验证标准
- ✅ 输出 `data/processed/retrieval_results_AO.json`
- ✅ 每个 test user 有 1 个相似用户
- ✅ 相似用户序列不为空

---

## 六、阶段 2:Prompt 改造(30 分钟)

### 6.1 目的
把相似用户的全序列拼进 prompt,启用 retrieval 模板。

### 6.2 步骤

#### 步骤 2.1:改造 get_retrieved_user_text(15 分钟)

**文件**:`enhancement/src/knn_retriever.py` 第 279-324 行

替换 `get_retrieved_user_text` 方法为:

```python
def get_retrieved_user_text(
    self,
    retrieved_user_ids: List[str],
    max_items_per_user: int = 5,   # 保留参数
    max_tokens: int = 512,         # 新增:对齐原论文 max_tk
) -> str:
    """
    将检索到的用户的交互序列格式化为文本(对齐原论文)。
    """
    lines = []
    for uid in retrieved_user_ids:
        items = self.interactions.get(uid, [])
        if not items:
            continue

        item_texts = []
        current_tokens = 0
        for iid in items:  # 全序列拼接
            meta = self.item_metadata.get(iid, {})
            title = meta.get("title", iid)
            domain = meta.get("domain", "")
            if domain in (self.domain_x_name, "X"):
                label = "art"
            elif domain in (self.domain_y_name, "Y"):
                label = "office"
            else:
                label = "item"
            item_text = f"{label}:{title}"
            est_tokens = len(item_text.split())
            if current_tokens + est_tokens > max_tokens:
                break
            current_tokens += est_tokens
            item_texts.append(item_text)

        if item_texts:
            lines.append(
                "There is another similar user who has interacted with: "
                + " | ".join(item_texts)
            )

    return "\n".join(lines) if lines else "(no similar users found)"
```

#### 步骤 2.2:启用 retrieval 模板(5 分钟)

**文件**:`enhancement/config/pipeline_config_AO.yaml` 第 34 行

修改:
```yaml
instruction:
  template_type: "retrieval"   # 从 "profile" 改为 "retrieval"
```

#### 步骤 2.3:重建 instruction 数据(10 分钟)
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
python -c "
import yaml
from src.build_instruction_data import build_all_instruction_data
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
build_all_instruction_data(cfg)
"
```

### 6.3 验证标准
- ✅ 检查 `train_instructions_AO.json` 第一条,确认 prompt 含 "similar user who has interacted with"
- ✅ 文件大小应增大(因 prompt 变长)

---

## 七、阶段 3:训练配置(30 分钟)

### 7.1 目的
配置 5 epochs + 启用早停 + 保留 FP16(不改 8bit)。

### 7.2 步骤

#### 步骤 3.1:修改 lora_config_AO.yaml(15 分钟)

**文件**:`enhancement/config/lora_config_AO.yaml`

完整内容:
```yaml
# ============================================================
# LoRA 微调配置 (AO 数据集 - retrieval 改造版 5 epochs)
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

training:
  num_epochs: 5                                  # 用户指定 5 轮
  max_steps: -1
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  warmup_ratio: 0.03                             # 保留原 ratio
  max_seq_length: 1024
  weight_decay: 0.01
  fp16: true                                     # 保留 FP16(不改 8bit)
  bf16: false
  logging_steps: 10
  save_steps: 200                                # 更密集保存,便于回溯
  eval_steps: 200
  save_total_limit: 10                           # 多保留 checkpoint
  gradient_checkpointing: true
  early_stopping:                                # 新增:启用早停
    patience: 5                                  # eval_loss 连续 5 次不下降则停
    threshold: 0.001

data:
  train_file: "train_instructions_AO.json"
  valid_file: "valid_instructions_AO.json"

inference:
  batch_size: 8
  temperature: 0.1
  max_new_tokens: 128
  num_beams: 4                                   # 新增:对齐原论文 beam=4

output:
  save_dir: "lora_weights_AO_v2"                 # 新目录,不覆盖原权重
  log_dir: "logs_AO_v2"
```

#### 步骤 3.2:设置环境变量(5 分钟)
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi   # 确认显存空闲
```

#### 步骤 3.3:验证训练入口(10 分钟)
```bash
# 确认 run_train_ao.py 或对应入口存在
ls /root/autodl-tmp/URLLM-project/enhancement/run_train*.py
# 若不存在,直接用 llm_finetune.py
python -c "
import yaml
cfg = yaml.safe_load(open('config/lora_config_AO.yaml'))
print('epochs:', cfg['training']['num_epochs'])
print('lora_r:', cfg['model']['lora_r'])
print('fp16:', cfg['training']['fp16'])
"
```

### 7.3 验证标准
- ✅ yaml 语法正确
- ✅ num_epochs=5, lora_r=8, fp16=true
- ✅ 显存空闲

---

## 八、阶段 4:训练(7-10 小时)

### 8.1 目的
用 retrieval prompt + 真实 DG 向量训练 5 epochs。

### 8.2 步骤

#### 步骤 4.1:启动训练
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATASET_SUFFIX=_AO

# 后台运行,日志输出到文件
nohup python -u -m src.llm_finetune > logs_AO_v2/train_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $!  # 记录 PID
```

#### 步骤 4.2:监控训练(每 30 分钟检查)

**监控命令**:
```bash
# 1. 查看训练日志
tail -50 logs_AO_v2/train_*.log

# 2. 检查 eval_loss 趋势(关键!判断是否过拟合)
grep "eval_loss" logs_AO_v2/train_*.log | tail -20

# 3. 检查显存
nvidia-smi

# 4. 检查进程存活
ps -ef | grep llm_finetune | grep -v grep
```

**eval_loss 监控决策表**:

| Epoch | eval_loss 预期 | 决策 |
|-------|-------------|------|
| 1.0 | 0.55-0.65 | 正常下降 |
| 2.0 | 0.45-0.55 | 继续训练 |
| 3.0 | 0.40-0.50 | 接近拐点,密切关注 |
| 4.0 | 0.38-0.48 | 若反弹则早停 |
| 5.0 | 0.36-0.46 | 训练完成 |

**若 epoch 3-4 出现 eval_loss 反弹**(过拟合信号):
- 立即停止训练(Ctrl+C 或 kill PID)
- 使用 epoch 3 的 checkpoint 继续后续步骤

#### 步骤 4.3:训练完成验证
```bash
ls -lh /root/autodl-tmp/URLLM-project/enhancement/outputs/lora_weights_AO_v2/
# 应有 checkpoint-XXX 目录,内含 adapter_model.safetensors
```

### 8.3 验证标准
- ✅ 训练正常完成,无 OOM
- ✅ eval_loss 单调下降,无明显反弹
- ✅ checkpoint 目录存在,含 adapter_model.safetensors

---

## 九、阶段 5:推理(2-3 小时)

### 9.1 目的
用 beam=4 生成预测。

### 9.2 步骤

#### 步骤 5.1:修改推理配置

**文件**:`enhancement/src/llm_inference.py`

找到 GenerationConfig,改为:
```python
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,              # 从 1 改为 4
    max_new_tokens=128,
    do_sample=False,         # beam search 模式
    return_dict_in_generate=True,
    output_scores=True,
)
```

#### 步骤 5.2:启动推理
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATASET_SUFFIX=_AO

# 后台运行
nohup python -u -m src.llm_inference > logs_AO_v2/inference_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $!
```

#### 步骤 5.3:监控推理
```bash
tail -30 logs_AO_v2/inference_*.log
# 检查输出文件
ls -lh outputs/predictions/test_predictions_AO_v2.json
```

### 9.3 验证标准
- ✅ 推理正常完成,无 OOM
- ✅ test_predictions_AO_v2.json 生成
- ✅ 预测条数 = 1000(测试用户数)

**预计耗时**:1000 测试用户 × beam=4 ≈ 2-3 小时

---

## 十、阶段 6:评估(30 分钟)

### 10.1 目的
生成 evaluation_AO_v2.json。

### 10.2 步骤

#### 步骤 6.1:运行评估
```bash
cd /root/autodl-tmp/URLLM-project/enhancement
export DATASET_SUFFIX=_AO
python -c "
import yaml
from src.evaluate import run_evaluation
cfg = yaml.safe_load(open('config/pipeline_config_AO.yaml'))
run_evaluation(cfg)
"
```

#### 步骤 6.2:对比指标
```bash
python -c "
import json

# 改造前
with open('outputs/eval_results/evaluation_AO.json') as f:
    before = json.load(f)

# 改造后
with open('outputs/eval_results/evaluation_AO_v2.json') as f:
    after = json.load(f)

print('=== AO 改造前后对比 ===')
print(f'exact HR@1: {before.get(\"exact_HR@1\", 0)} → {after.get(\"exact_HR@1\", 0)}')
print(f'fuzzy HR@1: {before.get(\"fuzzy_HR@1\", 0)} → {after.get(\"fuzzy_HR@1\", 0)}')
print(f'OOD 率: {before.get(\"ood_rate\", 0)} → {after.get(\"ood_rate\", 0)}')
"
```

### 10.3 验证标准
- ✅ exact HR@1 > 0(从 0.0000 提升)
- ✅ OOD 率 < 30%(从 49.95% 下降)
- ❌ 若 HR@1 仍为 0,改造失败,执行阶段 7 回滚

---

## 十一、阶段 7:文档与前端同步(1-2 小时)

### 11.1 目的
更新所有文档和前端展示数据,反映改造结果。

### 11.2 步骤

#### 步骤 7.1:更新文档

| 文件 | 更新内容 |
|------|---------|
| `DATASETS_STATUS.md` | 新增"九、AO retrieval 改造实施"章节 |
| `PROJECT_REPORT.md` | 更新 AO 方法论和指标 |
| `README.md` | 更新 pipeline 描述 |

#### 步骤 7.2:更新前端数据

| 文件 | 更新内容 |
|------|---------|
| `webapp/src/data/eval_data.json` | 同步 AO 新指标 |
| `webapp/src/data/training_logs.json` | 同步 AO 新训练日志 |
| `webapp/src/data/datasets.json` | 同步 AO 推荐展示数据 |

#### 步骤 7.3:重启前端
```bash
cd /root/autodl-tmp/URLLM-project/webapp
export PATH="/tmp/node-v20.18.0-linux-x64/bin:$PATH"
npm run dev -- --host 0.0.0.0 --port 6006
```

#### 步骤 7.4:git 提交改造结果
```bash
cd /root/autodl-tmp/URLLM-project
git add -A
git commit -m "feat(AO): 启用 retrieval prompt + 真实 DG 向量,5 epochs 训练,HR@1=XXX"
git checkout main
git merge feature/ao-retrieval-revamp
```

### 11.3 验证标准
- ✅ 前端 AO 页面显示新指标
- ✅ 文档指标一致
- ✅ git 提交完成,合并到 main 分支

---

## 十二、风险与应对汇总

| 风险 | 阶段 | 概率 | 严重度 | 应对 |
|------|------|------|-------|------|
| 用户 ID 顺序不匹配 | 1 | 30% | 高 | 验证 shape 后停止,排查 id_mapping |
| 5 epochs 过拟合 | 4 | 50% | 高 | 监控 eval_loss,反弹即早停 |
| 训练 OOM | 4 | 20% | 中 | 减小 batch_size 或用 gradient_checkpointing |
| beam=4 推理慢 | 5 | 60% | 中 | 退而求其次 beam=2 |
| HR@1 仍为 0 | 6 | 40% | 高 | 检查 prompt 格式,回滚到改造前 |
| 改造破坏 GM 流程 | 7 | 10% | 中 | 用 DATASET_SUFFIX 隔离,独立配置 |

---

## 十三、关键决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 训练轮数 | **5 epochs** | 用户指定;retrieval 模板可推迟过拟合 |
| 精度 | **FP16** | 不改 8bit,避免破坏 GM 流程 |
| LoRA r | **8**(从 16 改回) | 对齐原论文 |
| LoRA alpha | **16**(从 32 改回) | 对齐原论文 |
| beam | **4** | 对齐原论文,提升生成质量 |
| 早停 | **patience=5** | 防 epoch 4-5 过拟合 |
| 输出目录 | **lora_weights_AO_v2** | 不覆盖原权重 |
| 评估口径 | 保留 exact + fuzzy + BM25 expanded | 不引入新口径,保持与 GM 一致 |

---

## 十四、时间汇总

| 阶段 | 时间 | 累计 |
|------|------|------|
| 0 准备 | 0.5h | 0.5h |
| 1 DG 接入 | 1h | 1.5h |
| 2 Prompt | 0.5h | 2h |
| 3 训练配置 | 0.5h | 2.5h |
| 4 训练 | 7-10h | 9.5-12.5h |
| 5 推理 | 2-3h | 11.5-15.5h |
| 6 评估 | 0.5h | 12-16h |
| 7 文档 | 1-2h | **13-18h** |

**预计总耗时:13-18 小时**

---

## 十五、环境与命令速查

### 15.1 环境变量
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/tmp/node-v20.18.0-linux-x64/bin:$PATH"   # 前端构建用
export DATASET_SUFFIX=_AO                                # 切换 AO 数据集
```

### 15.2 关键路径

| 类别 | 路径 |
|------|------|
| AO 配置 | `enhancement/config/pipeline_config_AO.yaml` |
| AO 训练配置 | `enhancement/config/lora_config_AO.yaml` |
| AO 训练入口 | `enhancement/src/llm_finetune.py` |
| AO 推理入口 | `enhancement/src/llm_inference.py` |
| AO 检索结果 | `enhancement/data/processed/retrieval_results_AO.json` |
| AO 训练数据 | `enhancement/data/processed/train_instructions_AO.json` |
| AO 测试数据 | `enhancement/data/processed/test_instructions_AO.json` |
| AO LoRA 权重(新) | `enhancement/outputs/lora_weights_AO_v2/` |
| AO 预测结果(新) | `enhancement/outputs/predictions/test_predictions_AO_v2.json` |
| AO 评估结果(新) | `enhancement/outputs/eval_results/evaluation_AO_v2.json` |
| AO DG 向量 | `DG_Final/AO/DG/DGAO_final_*.npy` |
| AO 相似度矩阵 | `DG_Final/AO/saver/best_trte_XORY_DG_390_.npy` |
| 备份数据 | `enhancement/data/processed/*.backup.json` |

### 15.3 一键执行命令(顺序执行)

```bash
cd /root/autodl-tmp/URLLM-project/enhancement
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DATASET_SUFFIX=_AO

# 阶段 1:检索
python -c "import yaml; from src.knn_retriever import run_user_retrieval; run_user_retrieval(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"

# 阶段 2:重建 instruction
python -c "import yaml; from src.build_instruction_data import build_all_instruction_data; build_all_instruction_data(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"

# 阶段 4:训练(最耗时)
nohup python -u -m src.llm_finetune > logs_AO_v2/train.log 2>&1 &

# 阶段 5:推理
nohup python -u -m src.llm_inference > logs_AO_v2/inference.log 2>&1 &

# 阶段 6:评估
python -c "import yaml; from src.evaluate import run_evaluation; run_evaluation(yaml.safe_load(open('config/pipeline_config_AO.yaml')))"
```

---

## 十六、回滚方案

若改造失败(HR@1 仍为 0 或更差):

```bash
# 1. 切回主分支
cd /root/autodl-tmp/URLLM-project
git checkout main

# 2. 删除改造分支
git branch -D feature/ao-retrieval-revamp

# 3. 恢复备份的数据
cd enhancement/data/processed
cp train_instructions_AO.backup.json train_instructions_AO.json
cp test_instructions_AO.backup.json test_instructions_AO.json

# 4. 恢复配置
# 把 pipeline_config_AO.yaml 中 template_type 改回 "profile"
# 把 lora_config_AO.yaml 中 num_epochs 改回 3,lora_r 改回 16

# 5. 删除新的权重和预测
rm -rf enhancement/outputs/lora_weights_AO_v2/
rm -f enhancement/outputs/predictions/test_predictions_AO_v2.json
rm -f enhancement/outputs/eval_results/evaluation_AO_v2.json
```

---

## 十七、改造总结

**核心一句话**:把 AO 从"凭画像生成 SKU"改为"凭相似用户的购买序列模仿生成"——即启用 `retrieval` 模板 + 真实 DG 向量(已存在)+ 全序列拼接 + beam=4 解码 + 5 epochs 训练 + FP16 保留 + 早停机制,让 LLM 从"已知物品列表"中模仿,而非"凭空记忆 16000+ SKU"。

**预期效果**:exact HR@1 从 0.0000 提升至 0.001-0.005,OOD 率从 49.95% 降至 30% 以下,fallback_to_dg 从 97.9% 降至 60% 以下。

**实施原则**:严格按 0→1→2→3→4→5→6→7 顺序执行,每阶段完成后验证输出文件再进入下一阶段。

**关键验证点**:
1. 阶段 1 的用户 ID 顺序对齐(shape 验证)
2. 阶段 4 的 eval_loss 是否反弹(过拟合监控)
3. 阶段 6 的 HR@1 是否提升(改造成败判定)

**失败应对**:若改造后 HR@1 仍为 0,通过 git checkout 回滚到改造前状态,在文档中如实说明 AO 数据集因 94% 冷启动 + SKU 标题难度,当前架构无法解决。
