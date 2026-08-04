#!/bin/bash
# ============================================================
# 评估脚本
# ============================================================

set -e

echo "=========================================="
echo "URLLM 增强推荐 - 评估"
echo "=========================================="

# 检查推理结果
if [ ! -f "outputs/predictions/test_predictions.json" ]; then
    echo "错误: 推理结果不存在"
    echo "请先运行: bash scripts/run_inference.sh"
    exit 1
fi

# 运行评估
python src/evaluate.py

echo "=========================================="
echo "评估完成！"
echo "评估结果在: outputs/eval_results/evaluation.json"
echo "=========================================="
