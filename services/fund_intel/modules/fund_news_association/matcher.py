from typing import List
from services.fund_intel.clients import NewsClient
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer


class FundNewsMatcher:
    def __init__(self):
        self.news_client = NewsClient()
        self.industry_analyzer = FundIndustryAnalyzer()

    def match_fund_news(self, fund_code: str, days: int = 7) -> List[dict]:
        industries = self.industry_analyzer.analyze(fund_code)
        if not industries:
            return []

        all_news = []
        for ind in industries:
            news_list = self.news_client.get_news_by_industry(
                ind["industry"], days=days
            )
            for news in news_list:
                news["match_industry"] = ind["industry"]
                news["match_score"] = ind["confidence"] / 100
            all_news.extend(news_list)

        seen = set()
        unique_news = []
        for news in all_news:
            news_id = news.get("id", news.get("title"))
            if news_id not in seen:
                seen.add(news_id)
                unique_news.append(news)

        unique_news.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return unique_news[:20]

    def get_fund_industries(self, fund_code: str) -> List[str]:
        industries = self.industry_analyzer.analyze(fund_code)
        return [ind["industry"] for ind in industries]
