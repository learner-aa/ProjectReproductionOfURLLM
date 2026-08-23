#!/bin/bash
# ============================================================
# LLM LoRA 微调脚本
# 用法: bash scripts/run_finetune.sh [AO|GM]
# ============================================================

set -e

DATASET="${1:-AO}"
DATASET="${DATASET^^}"
CONFIG="config/pipeline_config.yaml"

echo "=========================================="
echo "URLLM 增强推荐 - LoRA 微调 [数据集: ${DATASET}]"
echo "=========================================="

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查 Instruction 数据
if [ ! -f "data/processed/${DATASET}/train_instructions.json" ]; then
    echo "错误: 训练 Instruction 数据不存在: data/processed/${DATASET}/train_instructions.json"
    echo "请先运行: bash scripts/run_preprocess.sh ${DATASET}"
    echo "再运行: python src/run_pipeline.py --until build_instructions --config $CONFIG"
    exit 1
fi

# 检查 GPU
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，请确保有 GPU'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 启动微调 (lora_config 自动定位 config/lora_config.yaml)
python src/run_pipeline.py --stage finetune --config "$CONFIG"

echo "=========================================="
echo "微调完成！LoRA 权重在: outputs/${DATASET}/lora_weights/final/"
echo "=========================================="
