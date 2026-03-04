# modules/fund_industry/interfaces.py
"""
基金行业模块接口定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FundIndustry:
    """基金行业"""
    fund_code: str
    industry: str
    confidence: float = 0.0
    source: str = "llm"


class FundIndustryRepoPort(ABC):
    """基金行业仓储端口"""

    @abstractmethod
    def save_industries(self, fund_code: str, industries: List[dict]) -> bool:
        """保存基金行业"""
        pass

    @abstractmethod
    def get_industries(self, fund_code: str) -> List[dict]:
        """获取基金行业"""
        pass

    @abstractmethod
    def delete_industries(self, fund_code: str) -> int:
        """删除基金行业"""
        pass


class FundIndustryAnalyzerPort(ABC):
    """基金行业分析器端口"""

    @abstractmethod
    def analyze(self, fund_code: str) -> List[dict]:
        """分析基金行业"""
        pass
