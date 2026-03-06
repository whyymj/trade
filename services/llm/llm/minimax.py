#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax LLM 客户端
"""

import os
from typing import List, Optional
import requests


class MiniMaxClient:
    """MiniMax API Client"""

    def __init__(self, api_key: Optional[str] = None, model: str = "abab5.5-chat"):
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        self.model = model
        self.base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")

    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送对话请求"""
        system_prompt = ""
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

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
        return bool(self.api_key)

    def get_provider_name(self) -> str:
        """获取提供商名称"""
        return "minimax"
