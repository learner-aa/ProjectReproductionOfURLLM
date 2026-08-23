#!/bin/bash
# ============================================================
# 评估脚本
# 用法: bash scripts/run_eval.sh [AO|GM]
# ============================================================

set -e

DATASET="${1:-AO}"
DATASET="${DATASET^^}"
CONFIG="config/pipeline_config.yaml"

echo "=========================================="
echo "URLLM 增强推荐 - 评估 [数据集: ${DATASET}]"
echo "=========================================="

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

# 检查推理结果
if [ ! -f "outputs/${DATASET}/predictions/test_predictions.json" ]; then
    echo "错误: 推理结果不存在: outputs/${DATASET}/predictions/test_predictions.json"
    echo "请先运行: bash scripts/run_inference.sh ${DATASET}"
    exit 1
fi

# 运行评估
python src/run_pipeline.py --stage evaluate --config "$CONFIG"

echo "=========================================="
echo "评估完成！评估结果在: outputs/${DATASET}/eval_results/evaluation.json"
echo "=========================================="
