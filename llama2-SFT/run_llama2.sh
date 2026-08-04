#!/bin/bash
# URLLM 论文原版 Llama2-7B 完整流程: 训练 → 推理 → 评估
set -e

# === 路径配置 (LLAMA_PATH 下载后确认) ===
LLAMA_PATH=${LLAMA_PATH:-/root/autodl-tmp/URLLM-project/models/Llama-2-7b}
DATA=/root/autodl-tmp/URLLM-project/enhancement/data/processed
ENH=/root/autodl-tmp/URLLM-project/enhancement
LORA_OUT=${ENH}/outputs/lora_weights/llama2_final
PRED=${ENH}/outputs/predictions/test_predictions.json
SCRIPT_DIR=/root/autodl-tmp/URLLM-project/llama2-SFT
PY=/root/miniconda3/envs/urllm/bin/python

cd ${SCRIPT_DIR}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/root/autodl-tmp/URLLM-project/.triton_cache
mkdir -p ${TRITON_CACHE_DIR}

# === 阶段1: 训练 ===
echo "========== [1/3] Llama2-7B LoRA 微调 (2 epochs, 7 modules, 8bit) =========="
rm -rf ${LORA_OUT}
mkdir -p ${LORA_OUT}
CUDA_VISIBLE_DEVICES=0 ${PY} finetune-lora.py \
    --model_name_or_path ${LLAMA_PATH} \
    --tokenizer_name ${LLAMA_PATH} \
    --train_files ${DATA}/train_instructions.json \
    --validation_files ${DATA}/valid_instructions.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --do_train --do_eval \
    --use_fast_tokenizer true \
    --output_dir ${LORA_OUT} \
    --eval_strategy steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --warmup_steps 100 \
    --load_in_bits 8 \
    --lora_r 16 --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${LORA_OUT}/logs \
    --logging_strategy steps --logging_steps 10 \
    --save_strategy steps --preprocessing_num_workers 10 \
    --save_steps 500 --eval_steps 500 --save_total_limit 3 \
    --seed 42 --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --ignore_data_skip true \
    --gradient_checkpointing

# === 阶段2: 推理 ===
echo "========== [2/3] 推理 (max_new_tokens=128, beam=4) =========="
mkdir -p $(dirname ${PRED})
CUDA_VISIBLE_DEVICES=0 ${PY} run_inference.py \
    ${LLAMA_PATH} \
    ${LORA_OUT} \
    ${DATA}/test_instructions.json \
    ${PRED}

# === 阶段3: 评估 ===
echo "========== [3/3] 评估 =========="
cd ${ENH}
${PY} src/evaluate.py
echo "========== 完成! =========="
echo "结果: ${ENH}/outputs/eval_results/evaluation.json"
