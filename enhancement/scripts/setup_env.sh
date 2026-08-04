#!/bin/bash
# ============================================================
# 环境安装脚本
# ============================================================

set -e

echo "=========================================="
echo "URLLM 增强推荐 - 环境安装"
echo "=========================================="

# 1. 创建 conda 环境
echo "[Step 1] 创建 conda 环境..."
conda create -n urllm python=3.10 -y 2>/dev/null || echo "环境已存在，跳过创建"
eval "$(conda shell.bash hook)"
conda activate urllm

# 2. 安装 PyTorch (CUDA 12.1)
echo "[Step 2] 安装 PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装依赖
echo "[Step 3] 安装项目依赖..."
pip install -r requirements.txt

# 4. 验证安装
echo "[Step 4] 验证安装..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers
print(f'Transformers: {transformers.__version__}')

import peft
print(f'PEFT: {peft.__version__}')
"

echo "=========================================="
echo "环境安装完成！"
echo "使用: conda activate urllm"
echo "=========================================="
