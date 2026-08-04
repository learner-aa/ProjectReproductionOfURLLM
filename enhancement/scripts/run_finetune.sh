#!/bin/bash
# ============================================================
# LLM LoRA 微调脚本
# ============================================================

set -e

echo "=========================================="
echo "URLLM 增强推荐 - LoRA 微调"
echo "=========================================="

# 检查 Instruction 数据
if [ ! -f "data/processed/train_instructions.json" ]; then
    echo "错误: 训练 Instruction 数据不存在"
    echo "请先运行: python src/run_pipeline.py --until build_instructions"
    exit 1
fi

# 检查 GPU
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用，请确保有 GPU'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# 启动微调
python src/llm_finetune.py

echo "=========================================="
echo "微调完成！"
echo "LoRA 权重在: outputs/lora_weights/final/"
echo "=========================================="
