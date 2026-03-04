# modules/fund_industry/__init__.py
"""
基金行业分析模块

提供基金行业分类功能：
- 基于LLM的行业分析
- 基于关键词的行业匹配
- 行业数据持久化
"""

from .interfaces import FundIndustry, FundIndustryRepoPort, FundIndustryAnalyzerPort
from .repo import FundIndustryRepo, get_repo
from .analyzer import FundIndustryAnalyzer, get_analyzer
from .schema import create_fund_industry_table, drop_fund_industry_table

__all__ = [
    "FundIndustry",
    "FundIndustryRepoPort",
    "FundIndustryAnalyzerPort",
    "FundIndustryRepo",
    "get_repo",
    "FundIndustryAnalyzer",
    "get_analyzer",
    "create_fund_industry_table",
    "drop_fund_industry_table",
]
