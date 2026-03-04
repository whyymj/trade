# data/market/interfaces.py
"""
市场数据模块接口定义
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class MarketCrawlerPort(ABC):
    """市场爬虫端口"""

    @abstractmethod
    def fetch_macro(self) -> pd.DataFrame:
        """抓取宏观经济数据"""
        pass

    @abstractmethod
    def fetch_money_flow(self) -> pd.DataFrame:
        """抓取资金流向数据"""
        pass

    @abstractmethod
    def fetch_sentiment(self) -> pd.DataFrame:
        """抓取市场情绪数据"""
        pass

    @abstractmethod
    def fetch_global(self) -> pd.DataFrame:
        """抓取全球宏观数据"""
        pass


class MarketRepoPort(ABC):
    """市场仓储端口"""

    @abstractmethod
    def save_macro_data(self, df: pd.DataFrame) -> int:
        """保存宏观数据"""
        pass

    @abstractmethod
    def get_macro_data(
        self, indicator: str = None, days: int = 30
    ) -> Optional[pd.DataFrame]:
        """获取宏观数据"""
        pass

    @abstractmethod
    def save_money_flow(self, df: pd.DataFrame) -> int:
        """保存资金流向"""
        pass

    @abstractmethod
    def get_money_flow(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取资金流向"""
        pass

    @abstractmethod
    def save_sentiment(self, df: pd.DataFrame) -> int:
        """保存市场情绪"""
        pass

    @abstractmethod
    def get_sentiment(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取市场情绪"""
        pass

    @abstractmethod
    def get_market_features(self, days: int = 30) -> Optional[pd.DataFrame]:
        """获取合并后的市场特征"""
        pass
