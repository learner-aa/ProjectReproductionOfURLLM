#!/bin/bash
# ============================================================
# 数据预处理脚本 (解析 DG 产物)
# 用法: bash scripts/run_preprocess.sh [AO|GM]
# ============================================================

set -e

DATASET="${1:-AO}"
DATASET="${DATASET^^}"                      # 转大写
CONFIG="config/pipeline_config.yaml"

echo "=========================================="
echo "URLLM 增强推荐 - 数据预处理 [数据集: ${DATASET}]"
echo "=========================================="

if [ ! -f "$CONFIG" ]; then
    echo "错误: 配置文件不存在: $CONFIG"
    exit 1
fi

python src/run_pipeline.py --stage preprocess --config "$CONFIG"

echo "=========================================="
echo "数据预处理完成！产出文件在: data/processed/${DATASET}/"
echo "=========================================="
