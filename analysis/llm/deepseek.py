# analysis/llm/deepseek.py
"""
DeepSeek LLM 客户端
"""

import os
from typing import List, Optional
import requests


class DeepSeekClient:
    """DeepSeek API Client"""

    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = "https://api.deepseek.com"

    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送对话请求"""
        if not self.api_key:
            raise Exception("DeepSeek API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        system_prompt = ""
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        messages_payload = []
        if system_prompt:
            messages_payload.append({"role": "system", "content": system_prompt})
        messages_payload.extend(filtered_messages)

        data = {
            "model": self.model,
            "messages": messages_payload,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120,
        )

        if resp.status_code != 200:
            raise Exception(f"DeepSeek API error: {resp.status_code} - {resp.text}")

        result = resp.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise Exception(f"Invalid response: {result}")

    def is_available(self) -> bool:
        """检查是否可用"""
        if not self.api_key:
            return False
        try:
            self.chat([{"role": "user", "content": "hi"}], max_tokens=5)
            return True
        except:
            return False

    def get_provider_name(self) -> str:
        """获取提供商名称"""
        return "deepseek"


_client: Optional[DeepSeekClient] = None


def get_client() -> DeepSeekClient:
    """获取 DeepSeek 客户端"""
    global _client
    if _client is None:
        from dotenv import load_dotenv

        load_dotenv()
        _client = DeepSeekClient()
    return _client


def reset_client():
    """重置客户端（用于重新加载环境变量）"""
    global _client
    _client = None
    return get_client()


def is_available() -> bool:
    """检查客户端是否可用"""
    client = get_client()
    return client.is_available() if client else False
