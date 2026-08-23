#!/bin/bash
# ============================================================
# LLM 推理脚本
# 用法: bash scripts/run_inference.sh [AO|GM]
# ============================================================

set -e

DATASET="${1:-AO}"
DATASET="${DATASET^^}"
CONFIG="config/pipeline_config.yaml"

echo "=========================================="
echo "URLLM 增强推荐 - LLM 推理 [数据集: ${DATASET}]"
echo "=========================================="

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查微调权重
if [ ! -d "outputs/${DATASET}/lora_weights/final" ]; then
    echo "警告: LoRA 权重目录不存在，将使用基座模型直接推理"
    echo "如需微调，请先运行: bash scripts/run_finetune.sh ${DATASET}"
fi

# 检查测试数据
if [ ! -f "data/processed/${DATASET}/test_instructions.json" ]; then
    echo "错误: 测试 Instruction 数据不存在: data/processed/${DATASET}/test_instructions.json"
    exit 1
fi

# 运行推理 (lora_config 自动定位 config/lora_config.yaml)
python src/run_pipeline.py --stage inference --config "$CONFIG"

echo "=========================================="
echo "推理完成！预测结果在: outputs/${DATASET}/predictions/test_predictions.json"
echo "=========================================="
