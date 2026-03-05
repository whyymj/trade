from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from modules.fund_industry import FundIndustryRepo
from modules.news_classification import ClassificationRepo
from modules.fund_news_association import FundNewsMatcher
from analysis.llm.deepseek import DeepSeekClient


@dataclass
class InvestmentAdvice:
    """投资建议"""

    short_term: str
    medium_term: str
    long_term: str
    risk_level: str
    confidence: float
    key_factors: List[str]
    generated_at: datetime


class InvestmentAdvisor:
    """投资顾问 - 基于新闻和行业分析提供投资建议"""

    def __init__(self):
        self.fund_industry_repo = FundIndustryRepo()
        self.news_classification_repo = ClassificationRepo()
        self.fund_news_matcher = FundNewsMatcher()
        self.llm = DeepSeekClient()

    def get_advice(self, fund_code: str, days: int = 7) -> Optional[InvestmentAdvice]:
        """获取投资建议"""
        fund_industries = self.fund_industry_repo.get_industries(fund_code)

        if not fund_industries:
            return None

        associations = self.fund_news_matcher.match_fund_news(fund_code, days)

        industry_names = [ind["industry"] for ind in fund_industries[:3]]

        news_summaries = []
        for assoc in associations[:10]:
            news_summaries.append(
                {
                    "title": assoc.news_title,
                    "industry": assoc.industry,
                    "score": assoc.match_score,
                }
            )

        advice = self._generate_advice(fund_code, industry_names, news_summaries)

        return advice

    def _generate_advice(
        self, fund_code: str, industries: List[str], news: List[dict]
    ) -> InvestmentAdvice:
        """使用LLM生成投资建议"""

        news_text = (
            "\n".join(
                [
                    f"- {n['title']} (行业: {n['industry']}, 相关度: {n['score']:.0%})"
                    for n in news[:8]
                ]
            )
            if news
            else "暂无相关新闻"
        )

        industries_text = ", ".join(industries) if industries else "未知"

        prompt = f"""你是一位专业的基金投资顾问。请根据以下信息，为投资者提供投资建议。

基金代码: {fund_code}
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
    "medium_term": "中期建议内容", 
    "long_term": "长期建议内容",
    "risk_level": "风险等级",
    "confidence": 信心指数(0-100),
    "key_factors": ["因素1", "因素2", "因素3"]
}}

确保:
- 短期建议要具体可操作
- 中期建议要结合行业趋势
- 长期建议要分析行业发展周期
- 风险等级要客观评估
- 关键因素要与新闻和行业相关"""

        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            return self._parse_advice_response(response)
        except Exception as e:
            print(f"[InvestmentAdvisor] generate advice error: {e}")
            return self._fallback_advice(industries, news)

    def _parse_advice_response(self, response: str) -> InvestmentAdvice:
        """解析LLM响应"""
        import json
        import re

        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if not json_match:
            return self._fallback_advice([], [])

        try:
            data = json.loads(json_match.group())
            return InvestmentAdvice(
                short_term=data.get("short_term", "暂无建议"),
                medium_term=data.get("medium_term", "暂无建议"),
                long_term=data.get("long_term", "暂无建议"),
                risk_level=data.get("risk_level", "中"),
                confidence=float(data.get("confidence", 50)),
                key_factors=data.get("key_factors", []),
                generated_at=datetime.now(),
            )
        except:
            return self._fallback_advice([], [])

    def _fallback_advice(
        self, industries: List[str], news: List[dict]
    ) -> InvestmentAdvice:
        """兜底建议"""
        sentiment = "中性"
        if news:
            high_score = sum(1 for n in news if n.get("score", 0) >= 0.7)
            low_score = sum(1 for n in news if n.get("score", 0) < 0.4)
            if high_score > low_score:
                sentiment = "偏正面"
            elif low_score > high_score:
                sentiment = "偏负面"

        return InvestmentAdvice(
            short_term=f"近期新闻情绪{sentiment}，建议谨慎关注",
            medium_term=f"关注{industries[0] if industries else '相关'}行业发展趋势",
            long_term="长期投资价值需综合评估",
            risk_level="中",
            confidence=50,
            key_factors=["市场情绪", "行业走势", "政策影响"],
            generated_at=datetime.now(),
        )


def get_advisor() -> InvestmentAdvisor:
    return InvestmentAdvisor()
