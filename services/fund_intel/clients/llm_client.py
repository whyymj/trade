import requests
import os
import json
import hashlib
from shared.cache import get_cache


class LLMClient:
    def __init__(self):
        self.base_url = os.getenv("LLM_SERVICE_URL", "http://llm-service:8006")
        self.cache = get_cache()

    def _get_cache_key(self, messages: list, provider: str) -> str:
        content = json.dumps(messages, sort_keys=True)
        return f"llm:{provider}:{hashlib.md5(content.encode()).hexdigest()}"

    def chat(
        self, messages: list, provider: str = "deepseek", use_cache: bool = True
    ) -> str:
        if use_cache:
            cache_key = self._get_cache_key(messages, provider)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        resp = requests.post(
            f"{self.base_url}/api/llm/chat",
            json={"provider": provider, "messages": messages},
        )

        if resp.status_code == 200:
            data = resp.json()
            result = data.get("data")

            if use_cache:
                cache_key = self._get_cache_key(messages, provider)
                self.cache.set(cache_key, result, ttl=86400)

            return result
        return None

    def analyze_news(self, news_list: list) -> dict:
        messages = [
            {
                "role": "system",
                "content": "你是一个金融新闻分析助手，请对新闻进行深度分析。",
            },
            {"role": "user", "content": f"请分析以下新闻：{news_list[:5]}"},
        ]
        result = self.chat(messages, provider="deepseek")
        return {"analysis": result}

    def classify_industry(self, text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是一个行业分类专家，请将文本分类到以下行业之一：宏观、行业、全球、政策、公司。",
            },
            {"role": "user", "content": text},
        ]
        result = self.chat(messages, provider="deepseek")
        return result.strip()

    def generate_investment_advice(
        self, fund_info: dict, news: list, industry: dict
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": "你是一个资深的投资顾问，请基于基金信息、新闻和行业分析生成投资建议。",
            },
            {
                "role": "user",
                "content": f"""
基金信息：{fund_info}
相关新闻：{news[:3]}
行业分析：{industry}

请生成投资建议。
            """,
            },
        ]
        result = self.chat(messages, provider="deepseek")
        return result
