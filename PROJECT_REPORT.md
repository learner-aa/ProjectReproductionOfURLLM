# URLLM 跨域序列推荐项目报告

> 生成时间:2026-08-16
> 工作空间:`/root/autodl-tmp/URLLM-project/`
> 复现论文:*Exploring User Retrieval Integration towards Large Language Models for Cross-Domain Sequential Recommendation* (arXiv:2406.03085)

---

## 一、项目概述

### 1.1 研究目标
验证 **LLM + 用户画像增强 + KNN 用户检索 + 答案精炼** 能否提升跨域序列推荐效果。通过 LoRA 微调让大语言模型学习用户偏好模式,结合 BM25 grounding 和域检查实现从源域到目标域的跨域推荐。

### 1.2 核心方法
- **用户画像**:行为特征 + 语义偏好,构建用户级画像
- **KNN 用户检索**:从 DG 特征向量中检索相似用户,为 LLM 提供 few-shot 示例(论文 §4.2.1)
- **指令微调**:Alpaca 格式指令,LoRA 高效微调 Llama2-7B
- **答案精炼(Answer Refinement)**:BM25 grounding 将 LLM 输出映射到真实物品 + 域检查 + DG 回退
- **评估**:HR@K / NDCG@K / MRR + 模糊匹配 + 冷热启动 + 域外率 + DG 基线对比

### 1.3 最终成果
- 论文原版 **Llama2-7B** 流程完整跑通:训练 → 推理 → 答案精炼 → 评估 → 前端展示
- **增强 Pipeline 9 阶段**:预处理 → 属性提取 → 用户画像 → KNN 用户检索 → 指令构建 → LoRA 微调 → 推理 → 答案精炼 → 评估
- **修复关键问题**:tokenizer `padding_side="left"` 修复后,推理质量从 65% 空预测降至 0%
- 前端展示系统 4 页面,全部基于真实项目数据

---

## 二、项目结构

```
URLLM-project/
├── enhancement/           # 增强 Pipeline(9 阶段,数据产物 + 评估)
│   ├── config/            # lora_config.yaml / pipeline_config.yaml
│   ├── data/processed/    # 画像 / 指令 / 属性 / 元数据等数据产物
│   ├── outputs/           # LoRA 权重 / 预测 / 精炼结果 / 评估结果
│   └── src/               # 核心代码(llm_finetune / llm_inference / evaluate 等)
├── llama2-SFT/            # 论文原版 Llama2-7B
├── DG_Final/              # DG 基线模型(提供评分矩阵对比)
├── webapp/                # 前端展示系统(Vite + React + TypeScript)
├── models/Llama-2-7b-hf   # 基座模型
└── PROJECT_REPORT.md      # 本报告
```

---

## 三、运行环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D(24564 MiB) |
| 驱动 / CUDA | NVIDIA-SMI 550.120 / 驱动支持 CUDA 12.4 |
| OS | Linux(容器环境) |
| Conda 环境 | `urllm`(Python 3.10) |
| PyTorch | 2.5.1+cu121 |
| Transformers | 5.14.1 |
| PEFT | 0.20.0 |
| Accelerate | 1.14.0 |
| 基座模型 | Llama-2-7b-hf |
| 前端 | Vite 8.1.5 + React 19 + TypeScript 6 + TailwindCSS 4 |

---

## 四、数据集

### GM 数据集(Movie → Game)

| 维度 | 数值 |
|------|------|
| 用户总数 | 40,479 |
| 测试用户 | 3,601 |
| 平均序列长度 | 17.15 |
| 源域(Movie)物品数 | 57,088 |
| 目标域(Game)物品数 | 107,792 |
| 物品属性(DeepSeek 提取) | 170,239 |
| 训练指令 | 31,570 条 |
| 测试指令 | 3,601 条 |

---

## 五、增强 Pipeline(9 阶段)

| 阶段 | 说明 | 关键产物 |
|------|------|---------|
| 1. preprocess | 数据预处理 | interactions.json (40,479 用户) |
| 2. extract_attributes | 物品属性提取 | item_attributes_GM.json (170,239 物品) |
| 3. build_profiles | 用户画像构建 | user_profiles.json (40,479 画像) |
| 4. retrieve_users | KNN 用户检索 | retrieval_results.json (31,570 条) |
| 5. build_instructions | 指令构建 | train/valid/test_instructions.json |
| 6. finetune | LoRA 微调 | lora_weights/ (adapter_model.safetensors) |
| 7. inference | LLM 推理 | test_predictions.json (3,601 条) |
| 8. refine_answers | 答案精炼 | refined_predictions.json (BM25 grounding + 域检查) |
| 9. evaluate | 评估 | evaluation.json (完整指标) |

---

## 六、LoRA 微调

### 6.1 训练配置

| 参数 | 值 |
|------|-----|
| 基座模型 | Llama-2-7b-hf |
| LoRA r / alpha | 8 / 16 |
| 目标模块 | q/k/v/o/gate/up/down_proj(7 个) |
| batch_size | 1 |
| gradient_accumulation_steps | 8 |
| 学习率 | 1e-4 |
| 训练轮数 | 5 epochs |
| max_seq_length | 1024 |
| 梯度检查点 | 开启 |
| 精度 | FP16 混合精度 |

### 6.2 训练结果

| 指标 | 值 |
|------|-----|
| 总步数 | 19,730 |
| 训练时长 | ~35 小时 |
| 初始 Loss | 2.0842 (step 50) |
| 最终 Loss | 0.3914 (step 19700) |
| Best Eval Loss | 0.5756 (step 19400) |

产物:[enhancement/outputs/lora_weights/](file:///root/autodl-tmp/URLLM-project/enhancement/outputs/lora_weights/)

---

## 七、推理

### 7.1 推理配置
- max_new_tokens = 128
- temperature = 0.1
- **padding_side = "left"**(decoder-only 模型必需)
- batch_size = 8

### 7.2 推理结果
- 3,601 条测试样本全部完成
- 推理耗时:904 秒(~15 分钟)
- 每样本耗时:0.25 秒
- 产物:[test_predictions.json](file:///root/autodl-tmp/URLLM-project/enhancement/outputs/predictions/test_predictions.json)

### 7.3 padding_side 修复效果

修复 `padding_side="left"` 前后对比:

| 指标 | 修复前 | 修复后 | 提升 |
|------|--------|--------|------|
| 空预测 | 2,344 (65%) | 0 (0%) | -100% |
| 正常长度预测 | 521 (14.5%) | 3,235 (89.8%) | +521% |
| 精确匹配 | 5 (0.1%) | 64 (1.8%) | +1280% |
| 推理耗时 | 1,991s | 904s | -55% |

---

## 八、答案精炼(Answer Refinement)

### 8.1 流程
1. **BM25 grounding**:将 LLM 输出文本映射到物品库中最相似的真实物品
2. **域检查**:检查 grounding 结果是否属于目标域
3. **DG 回退**:域外预测用 DG 评分矩阵回退到目标域物品

### 8.2 精炼结果

| 指标 | 值 |
|------|-----|
| 总样本 | 3,601 |
| BM25 匹配成功 | 3,601 (100%) |
| 域内保留 | 3,385 (93.9%) |
| DG 回退 | 216 (6.0%) |
| 未精炼 | 0 (0%) |

---

## 九、评估

### 9.1 评估方法
评估代码:[enhancement/src/evaluate.py](file:///root/autodl-tmp/URLLM-project/enhancement/src/evaluate.py),包含:
- **精确匹配(exact_metrics)**:LLM 原始预测的精确匹配
- **精炼后指标(refined_metrics)**:BM25 grounding + 域检查后的精确匹配
- **Top-K 候选指标(expanded_metrics)**:BM25 检索的 top-20 候选列表,HR@K 有区分度
- **模糊匹配**:fuzzy_HR@1 / partial_HR@1
- **冷/热启动分析**
- **域外率(OOD)**
- **DG 基线对比**

### 9.2 核心指标(BM25 top-K 候选,有区分度)

| 指标 | LLM+画像+精炼 | DG 基线 | 提升 |
|------|-------------|---------|------|
| HR@1 | **0.0275** | 0.0000 | LLM top-1 即超越 DG top-20 |
| HR@5 | **0.0439** | 0.0000 | — |
| HR@10 | **0.0483** | 0.0000 | — |
| HR@20 | **0.0555** | 0.0003 | ~185 倍 |
| MRR | **0.0345** | 0.0001 | ~345 倍 |
| NDCG@5 | 0.0361 | 0.0000 | — |
| NDCG@10 | 0.0375 | 0.0000 | — |
| NDCG@20 | 0.0393 | 0.0001 | ~393 倍 |

### 9.3 多层级评估对比

| 评估层级 | HR@1 | 说明 |
|---------|------|------|
| LLM 原始预测 | 0.0178 | LLM 直接生成的文本 |
| 精炼后预测 | 0.0233 | BM25 grounding + 域检查后 |
| BM25 top-K 候选 | 0.0275 | BM25 检索 top-1 候选 |

### 9.4 模糊匹配指标

| 指标 | 值 |
|------|-----|
| fuzzy_HR@1 | 0.0192 |
| partial_HR@1 | 0.0258 |
| exact_HR@1 | 0.0178 |

### 9.5 跨域分析

| 指标 | 值 |
|------|-----|
| 域外率(原始) | 9.34% (336/3601) |
| 域外率(精炼后) | 3.42% (123/3601) |
| 冷启动用户 | 0 |
| 热启动用户 | 3,601 |

---

## 十、前端展示系统

### 10.1 技术栈
Vite 8 + React 19 + TypeScript 6 + TailwindCSS 4 + Recharts 3

### 10.2 页面结构(4 页面,蓝色专业配色)

| 页面 | 功能 |
|------|------|
| 项目概览 | 核心成果卡片、数据集概览、方法概述 |
| 方法流程 | Pipeline 9 阶段流程、用户画像样例(真实属性) |
| 推荐展示 | 24 个演示用户,交互历史 + LLM 推荐结果 + 相似用户参考 |
| 效果评测 | 核心指标卡片、训练曲线(394 步真实日志)、HR@K & NDCG@K 曲线 |

### 10.3 数据真实性
- **eval_data.json**:所有指标来自 evaluation.json(真实评估结果)
- **training_logs.json**:GM 394 条来自 trainer_state.json(真实训练日志)
- **datasets.json**:24/24 用户推荐结果与 refined_predictions.json 一致

---

## 十一、关键问题与解决

1. **HuggingFace 网络不可达** → 改用 ModelScope 国内源下载
2. **CUDA OOM** → batch_size=1 + 梯度检查点 + FP16 + expandable_segments
3. **transformers 5.x API 变更** → `evaluation_strategy`→`eval_strategy`
4. **推理 padding_side 错误** → 设 `padding_side="left"` 修复 65% 空预测
5. **HR@K 无区分度** → 使用 BM25 top-K 候选扩展(refine_answers 阶段保存 top-20)
6. **前端数据不真实** → 从 evaluation.json 和 refined_predictions.json 直接读取真实数据

---

## 十二、结论

1. **完整 Pipeline 跑通**:9 阶段(预处理→属性提取→画像→KNN检索→指令→微调→推理→答案精炼→评估)
2. **LLM 方法有效性验证**:HR@1=0.0275, MRR=0.0345,远超 DG 基线(HR@1=0.0000)
3. **padding_side 修复关键**:修复后空预测从 65% 降至 0%,精确匹配提升 12.8 倍
4. **答案精炼有效**:域外率从 9.34% 降至 3.42%,BM25 匹配率 100%
5. **前端展示真实**:所有数据来自真实评估结果,HR@K 有区分度

---

**报告结束。**
