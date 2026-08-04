"""
LLM 物品属性提取模块 (Prompt I)

利用大模型的 Chain-of-Thought 能力从物品文本描述中提取结构化属性标签。
支持两种后端:
- OpenAI/兼容 API (GPT-3.5/4, 通义千问 API)
- 本地模型 (通过 transformers)
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_utils import PROCESSED_DIR, load_json, save_json
from prompt_templates import (
    PROMPT_I_ATTRIBUTE_EXTRACTION_EN,
    PROMPT_I_ATTRIBUTE_EXTRACTION_ZH,
    PROMPT_I_BATCH,
)

logger = logging.getLogger(__name__)


# ============================================================
# API 调用后端
# ============================================================

class APIClient:
    """OpenAI 兼容 API 客户端"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo-0613",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, prompt: str, temperature: float = 0.3) -> str:
        """发送单轮对话请求"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError("请安装 openai: pip install openai")

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=512,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"API 调用失败 (尝试 {attempt+1}): {e}, {self.retry_delay}s 后重试")
                    time.sleep(self.retry_delay)
                else:
                    raise


class LocalModelClient:
    """本地 transformers 模型客户端"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                raise ImportError("请安装 transformers: pip install transformers")

            logger.info(f"加载本地模型: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype="auto",
            )
            self._model.eval()

    def chat(self, prompt: str, temperature: float = 0.3) -> str:
        import torch

        self._load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 属性解析
# ============================================================

def parse_attribute_response(response: str) -> Dict[str, Any]:
    """
    解析 LLM 返回的属性提取结果。

    支持多种格式:
    - ATTRIBUTES: ["a", "b", "c"]
    - 属性: ["a", "b", "c"]
    - JSON 格式 {"attributes": [...]}

    Returns:
        {"intro": str, "attributes": [str, ...]}
    """
    result = {"intro": "", "attributes": []}

    # 尝试提取 JSON 列表
    list_patterns = [
        r'ATTRIBUTES:\s*(\[.*?\])',
        r'属性:\s*(\[.*?\])',
        r'(\[["\'][^]]*["\']\])',
    ]
    for pattern in list_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                attrs = json.loads(match.group(1))
                if isinstance(attrs, list):
                    result["attributes"] = [str(a).strip() for a in attrs]
                    break
            except json.JSONDecodeError:
                continue

    # 尝试提取介绍部分
    intro_patterns = [
        r'INTRODUCTION:\s*(.*?)(?=ATTRIBUTES:|$)',
        r'介绍:\s*(.*?)(?=属性:|$)',
    ]
    for pattern in intro_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            result["intro"] = match.group(1).strip()
            break

    # 如果没提取到属性，尝试从整个回复中找列表
    if not result["attributes"]:
        match = re.search(r'\[(.*?)\]', response)
        if match:
            try:
                attrs = json.loads(f"[{match.group(1)}]")
                result["attributes"] = [str(a).strip() for a in attrs]
            except json.JSONDecodeError:
                # 最后尝试逗号分隔
                items = match.group(1).split(",")
                result["attributes"] = [
                    i.strip().strip('"').strip("'").strip()
                    for i in items if i.strip()
                ]

    return result


# ============================================================
# 批量属性提取
# ============================================================

def extract_attributes_single(
    client: Any,
    item_id: str,
    title: str,
    description: str,
    category: str = "",
    language: str = "en",
) -> Dict[str, Any]:
    """提取单个物品的属性"""
    template = PROMPT_I_ATTRIBUTE_EXTRACTION_EN if language == "en" else PROMPT_I_ATTRIBUTE_EXTRACTION_ZH
    prompt = template.format(
        title=title,
        description=description or "N/A",
        category=category or "N/A",
    )
    response = client.chat(prompt)
    result = parse_attribute_response(response)
    result["item_id"] = item_id
    result["raw_response"] = response
    return result


def extract_all_attributes(
    item_metadata: Dict[str, Dict],
    client: Any,
    output_path: Optional[str] = None,
    batch_size: int = 50,
    language: str = "en",
    resume_from: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    批量提取所有物品的属性。

    Args:
        item_metadata: {item_id: {"title": str, "description": str, ...}}
        client: APIClient 或 LocalModelClient
        output_path: 输出路径 (支持断点续传)
        batch_size: 每处理 N 个物品保存一次
        language: "en" 或 "zh"
        resume_from: 断点续传文件路径

    Returns:
        {item_id: {"intro": str, "attributes": [str], "item_id": str}}
    """
    if output_path is None:
        output_path = str(PROCESSED_DIR / "item_attributes.json")

    # 断点续传: 加载已处理结果
    results = {}
    if resume_from and Path(resume_from).exists():
        results = load_json(resume_from)
        logger.info(f"断点续传: 已加载 {len(results)} 个已处理物品")

    total = len(item_metadata)
    processed = 0
    skipped = 0

    for item_id, meta in item_metadata.items():
        if item_id in results:
            skipped += 1
            continue

        try:
            result = extract_attributes_single(
                client, item_id,
                title=meta.get("title", item_id),
                description=meta.get("description", ""),
                category=str(meta.get("category", "")),
                language=language,
            )
            results[item_id] = {
                "intro": result["intro"],
                "attributes": result["attributes"],
                "item_id": item_id,
            }
            processed += 1

        except Exception as e:
            logger.error(f"处理 {item_id} 失败: {e}")
            results[item_id] = {
                "intro": "",
                "attributes": [],
                "item_id": item_id,
                "error": str(e),
            }

        # 定期保存
        if processed % batch_size == 0 and processed > 0:
            save_json(results, output_path)
            logger.info(f"进度: {processed}/{total - skipped} (已保存)")

    # 最终保存
    save_json(results, output_path)
    logger.info(
        f"属性提取完成: 处理={processed}, 跳过={skipped}, "
        f"总计={len(results)}, 保存至={output_path}"
    )
    return results


# ============================================================
# 入口
# ============================================================

def run_attribute_extraction(config: Dict):
    """
    运行属性提取流程。

    Args:
        config: 配置字典，需包含:
            - backend: "api" 或 "local"
            - api_key, base_url, model (API模式)
            - model_name, device (本地模式)
            - language: "en" 或 "zh"
    """
    item_metadata = load_json(PROCESSED_DIR / "item_metadata.json")

    if config.get("backend", "api") == "api":
        client = APIClient(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            model=config.get("model", "gpt-3.5-turbo-0613"),
        )
    else:
        client = LocalModelClient(
            model_name=config.get("model_name", "meta-llama/Llama-2-7b-chat-hf"),
            device=config.get("device", "cuda"),
        )

    extract_all_attributes(
        item_metadata=item_metadata,
        client=client,
        language=config.get("language", "en"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # 示例: 从 config 文件加载配置
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        run_attribute_extraction(cfg.get("attribute_extraction", {}))
    else:
        logger.error(f"配置文件不存在: {config_path}")
