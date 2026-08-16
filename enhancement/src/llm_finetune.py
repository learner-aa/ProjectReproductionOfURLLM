"""
LLM LoRA 微调模块

基于 PEFT 库对基座 LLM 进行参数高效微调:
- 加载 Instruction 数据集
- 应用 LoRA adapter
- 训练并保存 adapter 权重
- 支持多 GPU (DeepSpeed / FSDP)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data_utils import OUTPUT_DIR, PROCESSED_DIR, DATASET_SUFFIX, ensure_dirs

logger = logging.getLogger(__name__)


# ============================================================
# 默认配置
# ============================================================

DEFAULT_LORA_CONFIG = {
    "model": {
        "base_model": "meta-llama/Llama-2-7b-chat-hf",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.03,
        "max_seq_length": 1024,
        "weight_decay": 0.01,
        "fp16": True,
        "bf16": False,
        "logging_steps": 50,
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 3,
    },
    "data": {
        "train_file": f"train_instructions{DATASET_SUFFIX}.json",
        "valid_file": f"valid_instructions{DATASET_SUFFIX}.json",
    },
}


# ============================================================
# 数据加载
# ============================================================

def load_instruction_dataset(
    file_path: str,
    tokenizer,
    max_seq_length: int = 1024,
) -> List[Dict]:
    """
    加载 Instruction 数据集并 tokenize。

    支持两种格式:
    - Alpaca: {"instruction": str, "input": str, "output": str}
    - ChatML: {"messages": [...]}

    Returns:
        list of tokenized examples
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        if "messages" in item:
            # ChatML 格式
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # Alpaca 格式
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            text = prompt + tokenizer.eos_token

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()

        # 对 padding token 设置 -100 (不计算 loss)
        if "attention_mask" in tokenized:
            for i in range(len(tokenized["labels"])):
                if tokenized["attention_mask"][i] == 0:
                    tokenized["labels"][i] = -100

        examples.append(tokenized)

    logger.info(f"加载 {file_path}: {len(examples)} 条样本")
    return examples


# ============================================================
# 模型加载与 LoRA 应用
# ============================================================

def load_model_and_tokenizer(model_config: Dict, training_config: Dict):
    """
    加载基座模型和 tokenizer，应用 LoRA。

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "请安装必要依赖:\n"
            "  pip install transformers peft accelerate bitsandbytes"
        )

    base_model = model_config["base_model"]
    logger.info(f"加载基座模型: {base_model}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # 应用 LoRA
    lora_config = LoraConfig(
        r=model_config.get("lora_r", 8),
        lora_alpha=model_config.get("lora_alpha", 16),
        lora_dropout=model_config.get("lora_dropout", 0.05),
        target_modules=model_config.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 打印可训练参数统计
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================
# 训练
# ============================================================

def train(config: Optional[Dict] = None):
    """
    执行 LoRA 微调训练。

    Args:
        config: 完整配置字典 (含 model, training, data 子配置)
    """
    try:
        from transformers import Trainer, TrainingArguments
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError("请安装 transformers 和 torch")

    ensure_dirs()

    if config is None:
        config = DEFAULT_LORA_CONFIG
    else:
        # 合并默认配置
        merged = {}
        for section in ["model", "training", "data"]:
            default = DEFAULT_LORA_CONFIG.get(section, {})
            override = config.get(section, {})
            merged[section] = {**default, **override}
        config = merged

    model_config = config["model"]
    training_config = config["training"]
    data_config = config["data"]

    # 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config, training_config)

    # 加载数据
    max_seq_len = training_config.get("max_seq_length", 1024)

    train_file = str(PROCESSED_DIR / data_config["train_file"])
    train_data = load_instruction_dataset(train_file, tokenizer, max_seq_len)

    valid_data = None
    valid_file = str(PROCESSED_DIR / data_config.get("valid_file", ""))
    if Path(valid_file).exists():
        valid_data = load_instruction_dataset(valid_file, tokenizer, max_seq_len)

    # 包装为 Dataset
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = SimpleDataset(train_data)
    eval_dataset = SimpleDataset(valid_data) if valid_data else None

    # 训练参数
    save_dir = str(OUTPUT_DIR / f"lora_weights{DATASET_SUFFIX}")
    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=training_config.get("num_epochs", 3),
        per_device_train_batch_size=training_config.get("batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=training_config.get("learning_rate", 1e-4),
        warmup_ratio=training_config.get("warmup_ratio", 0.03),
        weight_decay=training_config.get("weight_decay", 0.01),
        fp16=training_config.get("fp16", True),
        bf16=training_config.get("bf16", False),
        logging_steps=training_config.get("logging_steps", 50),
        save_steps=training_config.get("save_steps", 200),
        eval_steps=training_config.get("eval_steps", 200),
        save_total_limit=training_config.get("save_total_limit", 3),
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=eval_dataset is not None,
        logging_dir=str(OUTPUT_DIR / "logs"),
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # 训练
    logger.info("=" * 60)
    logger.info("开始 LoRA 微调训练")
    logger.info(f"  训练样本: {len(train_data)}")
    logger.info(f"  验证样本: {len(valid_data) if valid_data else 'N/A'}")
    logger.info(f"  总步数: ~{len(train_data) // (training_config['batch_size'] * training_config['gradient_accumulation_steps']) * training_config['num_epochs']}")
    logger.info(f"  保存目录: {save_dir}")
    logger.info("=" * 60)

    trainer.train()

    # 保存最终模型
    final_path = str(OUTPUT_DIR / f"lora_weights{DATASET_SUFFIX}" / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    logger.info(f"微调完成！LoRA 权重已保存至: {final_path}")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "lora_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        train(cfg)
    else:
        logger.info("使用默认配置训练")
        train()
