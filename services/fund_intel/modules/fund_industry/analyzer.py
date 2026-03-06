from typing import List
from services.fund_intel.clients import FundClient


class FundIndustryAnalyzer:
    def __init__(self):
        self.fund_client = FundClient()

    def analyze(self, fund_code: str) -> dict:
        fund_info = self.fund_client.get_fund_info(fund_code)
        if not fund_info:
            return []

        fund_name = fund_info.get("fund_name", "")
        fund_type = fund_info.get("fund_type", "")

        from shared.cache import get_cache

        cache = get_cache()

        cache_key = f"fund_industry_{fund_code}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        industries = self._classify_by_keywords(fund_name, fund_type)
        cache.set(cache_key, industries, ttl=3600)
        return industries

    def _classify_by_keywords(self, fund_name: str, fund_type: str) -> List[dict]:
        text = f"{fund_name} {fund_type}"

        industry_keywords = {
            "新能源": [
                "光伏",
                "锂电",
                "储能",
                "电动车",
                "新能源汽车",
                "锂电池",
                "太阳能",
                "风电",
            ],
            "半导体": [
                "芯片",
                "集成电路",
                "AI",
                "算力",
                "半导体",
                "晶圆",
                "GPU",
                "CPU",
            ],
            "医药": [
                "创新药",
                "医疗器械",
                "中药",
                "生物医药",
                "疫苗",
                "医疗服务",
                "医药",
                "制药",
            ],
            "消费": [
                "白酒",
                "食品饮料",
                "零售",
                "家电",
                "纺织服装",
                "餐饮",
                "旅游",
                "酒店",
            ],
            "金融": ["银行", "保险", "券商", "信托", "租赁", "担保", "金融科技"],
            "军工": ["航空航天", "船舶", "军工", "国防", "导弹", "无人机", "卫星"],
            "TMT": ["互联网", "软件", "云服务", "游戏", "传媒", "通信", "5G"],
            "基建": ["建筑", "建材", "工程机械", "房地产", "园林", "装饰"],
            "农业": ["种植", "养殖", "农林牧渔", "种子", "饲料", "农药", "化肥"],
            "化工": ["石化", "化工新材料", "精细化工", "化学制品", "塑料", "橡胶"],
        }

        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                industry_scores[industry] = score

        if not industry_scores:
            return []

        max_score = max(industry_scores.values())
        result = []
        for industry, score in sorted(industry_scores.items(), key=lambda x: -x[1])[:3]:
            confidence = (score / max_score) * 50 + 50
            confidence = min(95, confidence)
            result.append(
                {
                    "industry": industry,
                    "confidence": round(confidence, 2),
                    "source": "keyword",
                }
            )

        return result
