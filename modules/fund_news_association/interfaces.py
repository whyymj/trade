from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional


@dataclass
class FundNewsAssociation:
    """基金-新闻关联"""

    id: int
    fund_code: str
    fund_name: str
    news_id: int
    news_title: str
    news_source: str
    news_url: str
    industry: str
    industry_code: str
    match_type: str
    match_score: float
    created_at: datetime


@dataclass
class FundNewsSummary:
    """基金新闻摘要"""

    fund_code: str
    fund_name: str
    industries: List[dict]
    news_count: int
    latest_news: List[dict]
    sentiment: str
    updated_at: datetime
