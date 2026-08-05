"""AO 训练脚本 - 先确认 CUDA，再训练"""
import sys
sys.path.insert(0, 'src')

import torch
print(f'1. import torch 后 CUDA: {torch.cuda.is_available()}', flush=True)

from llm_finetune import train
print(f'2. from llm_finetune import train 后 CUDA: {torch.cuda.is_available()}', flush=True)

if not torch.cuda.is_available():
    print('ERROR: CUDA 不可用!', flush=True)
    sys.exit(1)

print(f'3. GPU: {torch.cuda.get_device_name(0)}', flush=True)

import yaml
from pathlib import Path

config_path = Path('config/lora_config_AO.yaml')
with open(config_path) as f:
    cfg = yaml.safe_load(f)
print(f'4. 配置加载完成, 开始训练...', flush=True)
train(cfg)
