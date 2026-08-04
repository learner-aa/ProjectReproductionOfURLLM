#!/bin/bash
# ============================================================
# 数据预处理脚本
# ============================================================

set -e

echo "=========================================="
echo "URLLM 增强推荐 - 数据预处理"
echo "=========================================="

# 检查原始数据
if [ ! -f "data/raw/Entertainment_reviews.json" ] || [ ! -f "data/raw/Education_reviews.json" ]; then
    echo "错误: 请先将原始数据放入 data/raw/ 目录"
    echo "需要以下文件:"
    echo "  - data/raw/Entertainment_reviews.json"
    echo "  - data/raw/Education_reviews.json"
    echo "  - data/raw/Entertainment_meta.json (可选)"
    echo "  - data/raw/Education_meta.json (可选)"
    exit 1
fi

# 运行预处理
python src/preprocess.py

echo "=========================================="
echo "数据预处理完成！"
echo "产出文件在: data/processed/"
echo "=========================================="
