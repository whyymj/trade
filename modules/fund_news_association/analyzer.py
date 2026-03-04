from datetime import datetime
from typing import List, Optional

from modules.fund_industry import FundIndustryRepo
from modules.news_classification import ClassificationRepo
from modules.fund_news_association.interfaces import (
    FundNewsAssociation,
    FundNewsSummary,
)
from data import fund_repo


class FundNewsMatcher:
    """基金-新闻匹配引擎"""

    def __init__(self):
        self.fund_industry_repo = FundIndustryRepo()
        self.news_classification_repo = ClassificationRepo()

    def match_fund_news(
        self, fund_code: str, days: int = 7, min_confidence: float = 0.5
    ) -> List[FundNewsAssociation]:
        """为指定基金匹配相关新闻"""
        fund_industries = self.fund_industry_repo.get_industries(fund_code)

        if not fund_industries:
            return []

        industry_codes = [
            ind["industry_code"]
            for ind in fund_industries
            if ind.get("confidence", 0) >= min_confidence * 100
        ]

        if not industry_codes:
            return []

        associations = []
        for code in industry_codes:
            news_list = self.news_classification_repo.get_by_industry(code, days)

            for news in news_list:
                match_type = "industry" if news.confidence >= 0.8 else "partial"
                score = news.confidence

                assoc = FundNewsAssociation(
                    id=0,
                    fund_code=fund_code,
                    fund_name=self._get_fund_name(fund_code),
                    news_id=news.news_id,
                    news_title=news.title,
                    news_source=news.source,
                    news_url=news.url,
                    industry=news.industry,
                    industry_code=news.industry_code,
                    match_type=match_type,
                    match_score=score,
                    created_at=datetime.now(),
                )
                associations.append(assoc)

        associations.sort(key=lambda x: x.match_score, reverse=True)
        return associations

    def match_all_funds(self, days: int = 7) -> dict:
        """为所有有行业信息的基金匹配新闻"""
        all_funds = fund_repo.get_fund_list(page=1, size=500)
        fund_codes = [f["fund_code"] for f in all_funds.get("data", [])]

        results = {}
        for code in fund_codes:
            associations = self.match_fund_news(code, days)
            if associations:
                results[code] = associations

        return results

    def get_fund_news_summary(
        self, fund_code: str, days: int = 7
    ) -> Optional[FundNewsSummary]:
        """获取基金新闻摘要"""
        fund_industries = self.fund_industry_repo.get_industries(fund_code)

        if not fund_industries:
            return None

        associations = self.match_fund_news(fund_code, days)

        latest_news = [
            {
                "title": a.news_title,
                "source": a.news_source,
                "url": a.news_url,
                "industry": a.industry,
                "match_score": a.match_score,
            }
            for a in associations[:5]
        ]

        positive = sum(1 for a in associations if a.match_score >= 0.7)
        negative = sum(1 for a in associations if a.match_score < 0.4)
        sentiment = (
            "positive"
            if positive > negative
            else "negative"
            if negative > positive
            else "neutral"
        )

        return FundNewsSummary(
            fund_code=fund_code,
            fund_name=self._get_fund_name(fund_code),
            industries=fund_industries,
            news_count=len(associations),
            latest_news=latest_news,
            sentiment=sentiment,
            updated_at=datetime.now(),
        )

    def _get_fund_name(self, fund_code: str) -> str:
        """获取基金名称"""
        fund = fund_repo.get_fund_by_code(fund_code)
        return fund.get("fund_name", "") if fund else ""


def get_matcher() -> FundNewsMatcher:
    return FundNewsMatcher()
