#!/bin/bash
# URLLM AO 数据集完整流程 (enhancement 脚本, FP16, 不依赖 bitsandbytes)
# 训练 → 推理 → 评估
set -e

ENH=/root/autodl-tmp/URLLM-project/enhancement
PY=/root/miniconda3/envs/urllm/bin/python

cd ${ENH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === 阶段1: 训练 ===
echo "========== [1/3] LoRA 微调 (AO, Llama2-7B FP16, 2 epochs) =========="
${PY} run_train_ao.py

# === 阶段2: 推理 ===
echo "========== [2/3] 推理 (AO, 1000 条) =========="
${PY} run_inference_ao.py

# === 阶段3: 评估 ===
echo "========== [3/3] 评估 (AO) =========="
${PY} src/evaluate.py --dataset AO

echo "========== 完成! =========="
echo "结果: ${ENH}/outputs/eval_results/evaluation_AO.json"
