"""
LLM 推理模块

加载微调后的 LoRA 模型，对测试集进行批量推理生成推荐。
支持:
- 单条推理 (交互式)
- 批量推理 (测试集)
- 结果保存与解析
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data_utils import get_output_dir, get_processed_dir, load_json, save_json, ensure_dirs

logger = logging.getLogger(__name__)


# ============================================================
# 模型加载
# ============================================================

class LLMRecommender:
    """LLM 推荐推理器"""

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        lora_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 128,
    ):
        """
        Args:
            base_model: 基座模型名称 或 路径
            lora_path: LoRA adapter 权重路径 (None 则不加载 adapter)
            device: 推理设备
            max_new_tokens: 最大生成 token 数
        """
        self.base_model = base_model
        self.lora_path = lora_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load(self):
        """加载模型和 tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        logger.info(f"加载基座模型: {self.base_model}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True, padding_side="left"
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 4bit 量化加载 (与训练端一致): 全量 fp16 7B ≈14GB, 4bit ≈4GB, 24GB 卡上否则推理必 OOM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # 加载 LoRA adapter。保留 PeftModel 不 merge_and_unload:
        # merge 会把 4bit 基座去量化回 fp16 (~14GB), 24GB 卡必 OOM; 未 merge 直接 generate 同样生效。
        if self.lora_path and Path(self.lora_path).exists():
            try:
                from peft import PeftModel
                logger.info(f"加载 LoRA adapter: {self.lora_path}")
                self._model = PeftModel.from_pretrained(self._model, self.lora_path)
                logger.info("LoRA adapter 已加载 (4bit, 未 merge)")
            except ImportError:
                raise ImportError("请安装 peft: pip install peft")

        self._model.eval()
        logger.info("模型加载完成")

    def _ensure_loaded(self):
        if self._model is None:
            self.load()

    @staticmethod
    def _extract_title(generated: str) -> str:
        """从生成文本提取推荐标题: 去 Output: 前缀, 取最后一个非空行 (COT 最后一行是标题)"""
        lines = [ln.strip() for ln in generated.split("\n")]
        lines = [re.sub(r'^Output:\s*', '', ln, flags=re.IGNORECASE).strip() for ln in lines]
        lines = [ln for ln in lines if ln]
        return lines[-1] if lines else ""

    def generate(
        self,
        instruction_text: str,
        temperature: float = 0.1,
        top_p: float = 0.9,
    ) -> str:
        """
        单条推理。

        Args:
            instruction_text: 完整的 prompt 文本
            temperature: 生成温度
            top_p: nucleus sampling 参数

        Returns:
            生成的推荐物品标题
        """
        import torch

        self._ensure_loaded()

        inputs = self._tokenizer(
            instruction_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._extract_title(generated)

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        temperature: float = 0.1,
    ) -> List[str]:
        """
        批量推理。

        Args:
            prompts: prompt 文本列表
            batch_size: 批处理大小
            temperature: 生成温度

        Returns:
            生成结果列表
        """
        import torch

        self._ensure_loaded()
        results = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            inputs = self._tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            for j, output in enumerate(outputs):
                # left/right padding 都适用: 用输入总长度 (含 pad) 切片, 跳过整个输入前缀
                new_tokens = output[inputs["input_ids"].shape[1]:]
                generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                results.append(self._extract_title(generated))

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"推理进度: {min(i + batch_size, len(prompts))}/{len(prompts)}")

        return results


# ============================================================
# 测试集推理
# ============================================================

def build_test_prompts(
    test_instructions: List[Dict],
    format_type: str = "alpaca",
) -> List[str]:
    """
    从 Instruction 数据构建推理 prompt。

    Args:
        test_instructions: 测试集 instruction 数据
        format_type: "alpaca" 或 "chatml"

    Returns:
        prompt 文本列表
    """
    prompts = []
    for item in test_instructions:
        if format_type == "chatml" and "messages" in item:
            # ChatML: 取 user message
            for msg in item["messages"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"])
                    break
        else:
            # Alpaca
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
            prompts.append(prompt)
    return prompts


def run_inference(
    test_file: Optional[str] = None,
    lora_path: Optional[str] = None,
    config: Optional[Dict] = None,
):
    """
    执行测试集推理。

    Args:
        test_file: 测试集 instruction 文件路径
        lora_path: LoRA 权重路径
        config: 配置字典
    """
    ensure_dirs()

    if test_file is None:
        test_file = str(get_processed_dir() / "test_instructions.json")
    if lora_path is None:
        lora_path = str(get_output_dir() / "lora_weights" / "final")

    cfg = config or {}
    base_model = cfg.get("model", {}).get("base_model", "meta-llama/Llama-2-7b-hf")
    batch_size = cfg.get("inference", {}).get("batch_size", 8)
    temperature = cfg.get("inference", {}).get("temperature", 0.1)

    # 加载测试数据
    test_data = load_json(test_file)
    logger.info(f"测试集: {len(test_data)} 条")

    # 构建 prompts
    format_type = "chatml" if test_data and "messages" in test_data[0] else "alpaca"
    prompts = build_test_prompts(test_data, format_type)

    # 加载模型
    recommender = LLMRecommender(
        base_model=base_model,
        lora_path=lora_path,
    )
    recommender.load()

    # 批量推理
    logger.info("=" * 60)
    logger.info(f"开始推理: {len(prompts)} 条, batch_size={batch_size}")
    logger.info("=" * 60)

    start_time = time.time()
    predictions = recommender.generate_batch(
        prompts,
        batch_size=batch_size,
        temperature=temperature,
    )
    elapsed = time.time() - start_time

    # 组装结果
    results = []
    for i, (inst, pred) in enumerate(zip(test_data, predictions)):
        results.append({
            "user_id": inst.get("user_id"),
            "sample_id": inst.get("sample_id"),
            "dg_index": inst.get("dg_index"),
            "actual_user_id": inst.get("actual_user_id"),
            "target_item_id": inst.get("target_item_id"),
            "target_domain": inst.get("target_domain"),
            "ground_truth": inst.get("output", ""),
            "prediction": pred,
        })

    # 保存
    output_file = str(get_output_dir() / "predictions" / "test_predictions.json")
    save_json(results, output_file)

    # 统计
    exact_match = sum(
        1 for r in results
        if r["prediction"].lower().strip() == r["ground_truth"].lower().strip()
    )
    logger.info("=" * 60)
    logger.info(f"推理完成:")
    logger.info(f"  总样本: {len(results)}")
    logger.info(f"  耗时: {elapsed:.1f}s ({elapsed/len(results):.2f}s/样本)")
    logger.info(f"  精确匹配: {exact_match}/{len(results)} ({exact_match/len(results)*100:.1f}%)")
    logger.info(f"  结果保存: {output_file}")
    logger.info("=" * 60)

    return results


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import yaml

    config_path = Path(__file__).parent.parent / "config" / "lora_config.yaml"
    cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    run_inference(config=cfg)
