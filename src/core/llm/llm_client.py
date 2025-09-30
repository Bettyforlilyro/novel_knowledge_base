# src/core/llm_client.py
import hashlib
import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

import yaml
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 缓存目录
CACHE_DIR = Path("cache/llm_responses")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_key(messages: List[Dict[str, str]], model: str, response_format: str = "text") -> str:
    """基于输入生成唯一缓存键，以 model_name + messages + response_format 作为唯一key值"""
    payload = {
        "model": model,
        "messages": messages,
        "format": response_format
    }
    key_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def file_cache(func):
    """装饰器：自动缓存 LLM 响应到本地文件"""

    @wraps(func)
    def wrapper(self, messages: List[Dict[str, str]], response_format: str = "text", **kwargs):
        if self.use_cache:
            cache_key = get_cache_key(messages, self.model, response_format)
            cache_file = CACHE_DIR / f"{cache_key}.json"
            if cache_file.exists():
                logger.debug(f"Cache hit for key: {cache_key[:8]}...")
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)

        # 没有命中缓存或者不适用缓存，则调用原始函数
        result = func(self, messages, response_format, **kwargs)

        # 保存缓存
        if self.use_cache:
            cache_file = CACHE_DIR / f"{cache_key}.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    return wrapper


class QwenClient:
    def __init__(
            self,
            config_path: str = "/home/zhy/workspace/novel_knowledge_base/config/model.yaml",
            use_cache: bool = True,
            tokenizer_path: Optional[str] = None
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg["model"]

        # vLLM 兼容 OpenAI API，但不需要真实 API key
        self.client = OpenAI(
            base_url=model_cfg["base_url"],
            api_key="token-abc123",  # vLLM 忽略此字段，但 SDK 要求提供
        )
        self.model = model_cfg["model_name"]
        self.max_tokens = model_cfg.get("max_tokens", 32768)
        self.temperature = model_cfg.get("temperature", 0.3)
        self.timeout = model_cfg.get("timeout", 120)
        self.use_cache = use_cache

        # 加载tokenizer，方便计算消耗的token数量
        tokenizer_source = tokenizer_path or model_cfg.get("tokenizer_path", self.model)
        logger.info(f"Loading tokenizer from : {tokenizer_source}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True, use_fast=True)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e} from {tokenizer_source}")
            raise e

    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """
        如果是str，直接encode计算
        如果是messages（List[Dict[str, str]]）：按照Qwen3聊天模板拼接后encode
        """
        if isinstance(text, str):
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        elif isinstance(text, list):
            # 使用qwen3的模板拼接
            try:
                prompt = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
                return len(self.tokenizer.encode(prompt, add_special_tokens=False))
            except Exception as e:
                logger.error(f"Failed to apply chat template: {e}")
                raise e
        else:
            raise ValueError("Invalid input type. Must be str or List[Dict[str, str]]")

    @file_cache
    def chat_completion(
            self,
            messages: List[Dict[str, str]],
            response_format: str = "text",  # "text" 或 "json"
            max_retries: int = 3,
            retry_delay: float = 2.0,
    ) -> Dict[str, Any]:
        """
        调用 LLM，带重试和缓存。
        返回完整 response dict（含 content、usage 等）
        """
        input_tokens = self.count_tokens(messages)
        if input_tokens > self.max_tokens:
            raise ValueError(f"Input tokens exceed max_tokens: the input tokens is {input_tokens},"
                             f" which is greater than max_tokens: {self.max_tokens}")
        # 构建请求参数
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": min(4096, self.max_tokens // 2),  # 保守估计
            "timeout": self.timeout,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)

                # 提取关键信息
                result = {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "local_prompt_tokens": input_tokens
                    },
                    "model": response.model,
                    "timestamp": time.time(),
                }
                logger.info(f"LLM call succeeded. Tokens: {result['usage']['total_tokens']}")
                return result

            except (APIConnectionError, RateLimitError, APIStatusError) as e:
                last_exception = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * (2 ** attempt))  # 指数退避
                else:
                    logger.error("Max retries exceeded.")
                    raise e
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise e

        raise last_exception