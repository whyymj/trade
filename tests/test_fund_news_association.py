# tests/test_fund_news_association.py
"""
基金-新闻关联模块测试
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime

from modules.fund_news_association import (
    FundNewsMatcher,
    AssociationRepo,
    FundNewsAssociation,
    FundNewsSummary,
)
from modules.fund_news_association.interfaces import (
    FundNewsAssociation,
    FundNewsSummary,
)


class TestFundNewsMatcher:
    """测试基金-新闻匹配器"""

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    @patch("modules.fund_news_association.analyzer.fund_repo")
    def test_match_fund_news_no_industry(
        self, mock_fund_repo, mock_news_repo, mock_industry_repo
    ):
        """测试基金无行业信息时返回空"""
        mock_industry_repo.return_value.get_industries.return_value = []

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value
        matcher.news_classification_repo = mock_news_repo.return_value

        result = matcher.match_fund_news("000001")

        assert result == []

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    @patch("modules.fund_news_association.analyzer.fund_repo")
    def test_match_fund_news_with_industry(
        self, mock_fund_repo, mock_news_repo, mock_industry_repo
    ):
        """测试基金有行业信息时匹配新闻"""
        mock_industry_repo.return_value.get_industries.return_value = [
            {"industry": "新能源汽车", "industry_code": "I001", "confidence": 90.0}
        ]

        mock_news = MagicMock()
        mock_news.news_id = 1
        mock_news.title = "比亚迪发布新车"
        mock_news.source = "东方财富"
        mock_news.url = "http://example.com"
        mock_news.industry = "新能源汽车"
        mock_news.industry_code = "I001"
        mock_news.confidence = 0.85
        mock_news_repo.return_value.get_by_industry.return_value = [mock_news]
        mock_fund_repo.get_fund_info.return_value = {
            "fund_name": "测试基金"
        }

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value
        matcher.news_classification_repo = mock_news_repo.return_value

        result = matcher.match_fund_news("000001")

        assert len(result) == 1
        assert result[0].fund_code == "000001"
        assert result[0].industry == "新能源汽车"

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    def test_get_fund_news_summary_no_industry(self, mock_industry_repo):
        """测试无行业信息时返回None"""
        mock_industry_repo.return_value.get_industries.return_value = []

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value

        result = matcher.get_fund_news_summary("000001")

        assert result is None


class TestAssociationRepo:
    """测试关联仓储"""

    def test_get_funds_with_news(self):
        """测试获取有关联新闻的基金"""
        repo = AssociationRepo()
        funds = repo.get_funds_with_news(days=7)

        assert isinstance(funds, list)


class TestFundNewsAssociation:
    """测试关联数据类"""

    def test_creation(self):
        """测试创建关联对象"""
        assoc = FundNewsAssociation(
            id=1,
            fund_code="000001",
            fund_name="测试基金",
            news_id=100,
            news_title="新闻标题",
            news_source="东方财富",
            news_url="http://example.com",
            industry="新能源汽车",
            industry_code="I001",
            match_type="industry",
            match_score=0.85,
            created_at=datetime.now(),
        )

        assert assoc.fund_code == "000001"
        assert assoc.industry_code == "I001"
        assert assoc.match_score == 0.85


class TestFundNewsSummary:
    """测试摘要数据类"""

    def test_creation(self):
        """测试创建摘要对象"""
        summary = FundNewsSummary(
            fund_code="000001",
            fund_name="测试基金",
            industries=[{"industry": "新能源汽车", "confidence": 90.0}],
            news_count=10,
            latest_news=[],
            sentiment="positive",
            updated_at=datetime.now(),
        )

        assert summary.fund_code == "000001"
        assert summary.news_count == 10
        assert summary.sentiment == "positive"
