# analysis/llm/client.py
"""
MiniMax M2.5 大模型客户端

仅支持 MiniMax M2.5 模型
"""

import os
from typing import Optional


class MiniMaxClient:
    """MiniMax API Client - M2.5 Model"""

    def __init__(self, api_key: str = None, model: str = "MiniMax-M2.5"):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.model = model
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")

    def chat(self, messages: list, **kwargs) -> str:
        """发送对话请求"""
        import requests

        # 提取 system 消息
        system_prompt = ""
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        # 构建 prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)

        for msg in filtered_messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")

        full_prompt = "\n\n".join(prompt_parts)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        resp = requests.post(
            f"{self.base_url}/text/chatcompletion_v2",
            headers=headers,
            json=data,
            timeout=120,
        )

        if resp.status_code != 200:
            raise Exception(f"MiniMax API error: {resp.status_code} - {resp.text}")

        result = resp.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise Exception(f"Invalid response: {result}")

    def is_available(self) -> bool:
        """检查是否可用"""
        if not self.api_key:
            return False
        try:
            # 简单测试请求
            self.chat([{"role": "user", "content": "hi"}], max_tokens=5)
            return True
        except:
            return False


# 全局客户端
_client: Optional[MiniMaxClient] = None


def get_client() -> Optional[MiniMaxClient]:
    """获取 MiniMax 客户端"""
    global _client
    if _client is None:
        from dotenv import load_dotenv

        load_dotenv()
        _client = MiniMaxClient()
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
