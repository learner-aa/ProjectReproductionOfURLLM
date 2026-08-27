#!/bin/bash
# 等待 finetune 进程结束后自动运行 inference -> refine_answers -> evaluate
# 用法: nohup bash _continue_after_finetune.sh <FINETUNE_PID> > /dev/null 2>&1 &
set -u
FINETUNE_PID="${1:?usage: $0 <finetune_pid>}"
LOG=/tmp/pipeline_rest.log
PY=/root/miniconda3/envs/urllm/bin/python
# 治 CUDA 显存碎片化 (OOM 时 reserved-but-unallocated 部分可被复用)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/URLLM-project/enhancement

{
  echo "=== $(date) waiting for finetune PID $FINETUNE_PID ==="
  while kill -0 "$FINETUNE_PID" 2>/dev/null; do sleep 60; done
  echo "=== $(date) finetune process exited ==="

  # 守护: 确认最终 LoRA 权重存在, 否则中止 (防止用陈旧权重跑下游)
  FINAL=outputs/AO/lora_weights/final
  if [ ! -e "$FINAL/adapter_model.safetensors" ] && [ ! -e "$FINAL/adapter_model.bin" ]; then
    echo "!!! FINAL LoRA weights missing under $FINAL — aborting rest"
    exit 1
  fi

  echo "=== $(date) running inference ==="
  $PY src/run_pipeline.py --stage inference || { echo "!!! inference failed"; exit 1; }
  echo "=== $(date) running refine_answers ==="
  $PY src/run_pipeline.py --stage refine_answers || { echo "!!! refine failed"; exit 1; }
  echo "=== $(date) running evaluate ==="
  $PY src/run_pipeline.py --stage evaluate || { echo "!!! evaluate failed"; exit 1; }
  echo "=== $(date) ALL DONE ==="
} >> "$LOG" 2>&1
