import requests
import os
from shared.cache import get_cache


class NewsClient:
    def __init__(self):
        self.base_url = os.getenv("NEWS_SERVICE_URL", "http://news-service:8003")
        self.cache = get_cache()

    def get_news(self, days: int = 1, category: str = None, limit: int = 100) -> list:
        cache_key = f"news_list_{days}_{category}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {"days": days, "limit": limit}
        if category:
            params["category"] = category

        resp = requests.get(f"{self.base_url}/api/news/list", params=params)
        if resp.status_code == 200:
            data = resp.json().get("data")
            self.cache.set(cache_key, data, ttl=1800)
            return data
        return []

    def get_news_by_industry(self, industry: str, days: int = 7) -> list:
        return self.get_news(days=days, category=industry)

    def get_news_detail(self, news_id: int) -> dict:
        cache_key = f"news_detail_{news_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/news/detail/{news_id}")
        if resp.status_code == 200:
            data = resp.json().get("data")
            self.cache.set(cache_key, data, ttl=3600)
            return data
        return None
