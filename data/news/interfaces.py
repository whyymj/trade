# data/news/interfaces.py
"""
新闻模块接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime, date


@dataclass
class NewsItem:
    """新闻条目"""

    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str = "general"
    news_date: date = None
    id: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.news_date is None:
            self.news_date = (
                self.published_at.date() if self.published_at else date.today()
            )


@dataclass
class AnalysisResult:
    """分析结果"""

    news_count: int
    summary: str
    deep_analysis: str
    market_impact: str
    key_events: List[dict]
    investment_advice: str
    analyzed_at: datetime


class NewsCrawlerPort(ABC):
    """新闻爬虫端口"""

    @abstractmethod
    def fetch_cailian(self) -> List[NewsItem]:
        """抓取财联社"""
        pass

    @abstractmethod
    def fetch_wallstreet(self) -> List[NewsItem]:
        """抓取华尔街见闻"""
        pass

    @abstractmethod
    def fetch_today(self) -> List[NewsItem]:
        """抓取当天新闻（增量）"""
        pass

    @abstractmethod
    def can_fetch(self) -> bool:
        """检查是否可以爬取（频率控制）"""
        pass


class NewsRepoPort(ABC):
    """新闻仓储端口"""

    @abstractmethod
    def save_news(self, news_list: List[NewsItem]) -> int:
        """保存新闻（去重）"""
        pass

    @abstractmethod
    def get_news(
        self, days: int = 1, category: str = None, limit: int = 100
    ) -> List[NewsItem]:
        """获取新闻"""
        pass

    @abstractmethod
    def get_today_news(self) -> List[NewsItem]:
        """获取当天新闻"""
        pass

    @abstractmethod
    def cleanup_old_news(self, keep_days: int = 30) -> int:
        """清理过期新闻"""
        pass

    @abstractmethod
    def save_analysis(self, result: AnalysisResult) -> bool:
        """保存分析结果"""
        pass

    @abstractmethod
    def get_latest_analysis(self) -> Optional[AnalysisResult]:
        """获取最新分析结果"""
        pass
