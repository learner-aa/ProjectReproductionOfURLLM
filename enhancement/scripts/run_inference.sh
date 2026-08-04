#!/bin/bash
# ============================================================
# LLM 推理脚本
# ============================================================

set -e

echo "=========================================="
echo "URLLM 增强推荐 - LLM 推理"
echo "=========================================="

# 检查微调权重
if [ ! -d "outputs/lora_weights/final" ]; then
    echo "警告: LoRA 权重目录不存在，将使用基座模型直接推理"
    echo "如需微调，请先运行: bash scripts/run_finetune.sh"
fi

# 检查测试数据
if [ ! -f "data/processed/test_instructions.json" ]; then
    echo "错误: 测试 Instruction 数据不存在"
    exit 1
fi

# 运行推理
python src/llm_inference.py

echo "=========================================="
echo "推理完成！"
echo "预测结果在: outputs/predictions/test_predictions.json"
echo "=========================================="
