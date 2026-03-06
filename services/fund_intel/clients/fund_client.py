import requests
import os
from shared.cache import get_cache


class FundClient:
    def __init__(self):
        self.base_url = os.getenv("FUND_SERVICE_URL", "http://fund-service:8002")
        self.cache = get_cache()

    def get_fund_info(self, fund_code: str) -> dict:
        cache_key = f"fund_info_{fund_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/fund/{fund_code}")
        if resp.status_code == 200:
            data = resp.json().get("data")
            self.cache.set(cache_key, data, ttl=3600)
            return data
        return None

    def get_fund_nav(self, fund_code: str, days: int = 30) -> list:
        cache_key = f"fund_nav_{fund_code}_{days}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/fund/{fund_code}/nav?days={days}")
        if resp.status_code == 200:
            data = resp.json().get("data")
            self.cache.set(cache_key, data, ttl=1800)
            return data
        return []

    def get_fund_list(self, page: int = 1, size: int = 50) -> dict:
        cache_key = f"fund_list_{page}_{size}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        resp = requests.get(f"{self.base_url}/api/fund/list?page={page}&size={size}")
        if resp.status_code == 200:
            data = resp.json()
            self.cache.set(cache_key, data, ttl=1800)
            return data
        return {}
