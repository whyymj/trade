from typing import Optional
from services.fund_intel.clients import FundClient, NewsClient, LLMClient
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer
from services.fund_intel.modules.fund_news_association import FundNewsMatcher


class InvestmentAdviceGenerator:
    def __init__(self):
        self.fund_client = FundClient()
        self.news_client = NewsClient()
        self.llm_client = LLMClient()
        self.industry_analyzer = FundIndustryAnalyzer()
        self.news_matcher = FundNewsMatcher()

    def generate(self, fund_code: str) -> Optional[dict]:
        fund_info = self.fund_client.get_fund_info(fund_code)
        if not fund_info:
            return None

        fund_nav = self.fund_client.get_fund_nav(fund_code, days=30)
        industries = self.industry_analyzer.analyze(fund_code)
        news_list = self.news_matcher.match_fund_news(fund_code, days=7)

        industry_names = [ind["industry"] for ind in industries[:3]]
        news_summaries = [
            {
                "title": n.get("title", ""),
                "industry": n.get("match_industry", ""),
                "score": n.get("match_score", 0),
            }
            for n in news_list[:5]
        ]

        advice = self._generate_advice(
            fund_code, fund_info, industry_names, news_summaries
        )
        return advice

    def _generate_advice(
        self, fund_code: str, fund_info: dict, industries: list, news: list
    ) -> dict:
        fund_name = fund_info.get("fund_name", "")
        fund_type = fund_info.get("fund_type", "")

        news_text = (
            "\n".join(
                [
                    f"- {n['title']} (行业: {n['industry']}, 相关度: {n['score']:.0%})"
                    for n in news
                    if n.get("title")
                ]
            )
            if news
            else "暂无相关新闻"
        )

        industries_text = ", ".join(industries) if industries else "未知"

        prompt = f"""你是一位专业的基金投资顾问。请根据以下信息，为投资者提供投资建议。

基金代码: {fund_code}
基金名称: {fund_name}
基金类型: {fund_type}
主要持仓行业: {industries_text}

相关新闻:
{news_text}

请从专业角度分析以下内容并返回JSON格式的建议:

1. **短期建议 (1周内)**: 基于近期新闻和市场情绪，分析是否应该买入/卖出/持有
2. **中期建议 (1-3个月)**: 基于行业趋势和政策动向，分析未来1-3个月的表现预期
3. **长期建议 (6个月以上)**: 基于行业发展周期和基金经理能力，分析长期投资价值
4. **风险等级**: 评估该基金的风险程度 (低/中/中高/高)
5. **信心指数**: 你对上述分析的信心程度 (0-100的数值)
6. **关键因素**: 影响该基金表现的关键因素列表 (3-5个)

请返回以下JSON格式 (只返回JSON，不要其他内容):
{{
    "short_term": "短期建议内容",
    "medium_term": "中期建议",
    "long_term": "长期建议",
    "risk_level": "风险等级",
    "confidence": 信心指数,
    "key_factors": ["因素1", "因素2", "因素3"]
}}"""

        try:
            response = self.llm_client.chat(
                [{"role": "user", "content": prompt}],
                provider="deepseek",
                use_cache=True,
            )
            return self._parse_advice_response(response)
        except:
            return self._fallback_advice(industries, news)

    def _parse_advice_response(self, response: str) -> dict:
        import json
        import re

        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if not json_match:
            return self._fallback_advice([], [])

        try:
            data = json.loads(json_match.group())
            return {
                "short_term": data.get("short_term", "暂无建议"),
                "medium_term": data.get("medium_term", "暂无建议"),
                "long_term": data.get("long_term", "暂无建议"),
                "risk_level": data.get("risk_level", "中"),
                "confidence": float(data.get("confidence", 50)),
                "key_factors": data.get("key_factors", []),
            }
        except:
            return self._fallback_advice([], [])

    def _fallback_advice(self, industries: list, news: list) -> dict:
        sentiment = "中性"
        if news:
            high_score = sum(1 for n in news if n.get("score", 0) >= 0.7)
            low_score = sum(1 for n in news if n.get("score", 0) < 0.4)
            if high_score > low_score:
                sentiment = "偏正面"
            elif low_score > high_score:
                sentiment = "偏负面"

        return {
            "short_term": f"近期新闻情绪{sentiment}，建议谨慎关注",
            "medium_term": f"关注{industries[0] if industries else '相关'}行业发展趋势",
            "long_term": "长期投资价值需综合评估",
            "risk_level": "中",
            "confidence": 50,
            "key_factors": ["市场情绪", "行业走势", "政策影响"],
        }
