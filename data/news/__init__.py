# data/news/__init__.py
"""新闻模块"""

from .interfaces import NewsItem, AnalysisResult, NewsCrawlerPort, NewsRepoPort
from .crawler import NewsCrawler
from .repo import NewsRepo

__all__ = [
    "NewsItem",
    "AnalysisResult",
    "NewsCrawlerPort",
    "NewsRepoPort",
    "NewsCrawler",
    "NewsRepo",
]
