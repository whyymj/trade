# tests/test_fund_news_matcher.py
"""
基金-新闻匹配器测试
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from modules.fund_news_association import FundNewsMatcher
from modules.news_classification import get_industry_code, get_industry_by_code


class TestIndustryCodeMapping:
    """测试行业代码映射"""

    def test_get_industry_code_by_name(self):
        """根据行业名称获取代码"""
        assert get_industry_code("新能源汽车") == "I001"
        assert get_industry_code("半导体") == "I002"
        assert get_industry_code("医药生物") == "I003"
        assert get_industry_code("银行") == "I006"

    def test_get_industry_code_unknown(self):
        """未知行业返回默认代码"""
        assert get_industry_code("未知行业") == "I000"
        assert get_industry_code("") == "I000"

    def test_get_industry_by_code(self):
        """根据代码获取行业信息"""
        result = get_industry_by_code("I001")
        assert result is not None
        assert result["name"] == "新能源汽车"
        assert "比亚迪" in result["keywords"]

    def test_get_industry_by_code_unknown(self):
        """未知代码返回None"""
        assert get_industry_by_code("I999") is None


class TestFundNewsMatcherIndustryCode:
    """测试匹配器处理行业代码"""

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    def test_match_fund_news_without_industry_code(
        self, mock_classification_repo, mock_industry_repo
    ):
        """测试基金行业数据没有industry_code字段时能正常处理"""
        mock_industry_repo.return_value.get_industries.return_value = [
            {"industry": "新能源汽车", "confidence": 85.0},
            {"industry": "半导体", "confidence": 75.0},
        ]

        mock_classification_repo.return_value.get_by_industry.return_value = []

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value
        matcher.news_classification_repo = mock_classification_repo.return_value

        result = matcher.match_fund_news("000001")

        assert result == []
        mock_classification_repo.return_value.get_by_industry.assert_called()

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    def test_match_fund_news_with_mixed_industry_format(
        self, mock_classification_repo, mock_industry_repo
    ):
        """测试混合行业数据（有code和无code）"""
        mock_industry_repo.return_value.get_industries.return_value = [
            {"industry": "新能源汽车", "confidence": 85.0},
            {"industry": "半导体", "industry_code": "I002", "confidence": 75.0},
        ]

        mock_classification_repo.return_value.get_by_industry.return_value = []

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value
        matcher.news_classification_repo = mock_classification_repo.return_value

        result = matcher.match_fund_news("000001")

        assert result == []

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    @patch("modules.fund_news_association.analyzer.fund_repo")
    def test_get_fund_name(
        self, mock_fund_repo, mock_classification_repo, mock_industry_repo
    ):
        """测试获取基金名称"""
        mock_fund_repo.get_fund_info.return_value = {
            "fund_code": "000001",
            "fund_name": "测试基金",
        }

        mock_industry_repo.return_value.get_industries.return_value = []

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value

        name = matcher._get_fund_name("000001")

        assert name == "测试基金"


class TestFundNewsMatcherWithNews:
    """测试匹配器与新闻数据"""

    @patch("modules.fund_news_association.analyzer.FundIndustryRepo")
    @patch("modules.fund_news_association.analyzer.ClassificationRepo")
    @patch("modules.fund_news_association.analyzer.fund_repo")
    def test_match_with_classified_news(
        self, mock_fund_repo, mock_classification_repo, mock_industry_repo
    ):
        """测试匹配到分类新闻"""
        mock_industry_repo.return_value.get_industries.return_value = [
            {"industry": "新能源汽车", "confidence": 85.0},
        ]

        mock_news = MagicMock()
        mock_news.news_id = 1
        mock_news.title = "比亚迪发布新车"
        mock_news.source = "东方财富"
        mock_news.url = "http://example.com"
        mock_news.industry = "新能源汽车"
        mock_news.industry_code = "I001"
        mock_news.confidence = 0.85

        mock_classification_repo.return_value.get_by_industry.return_value = [mock_news]
        mock_fund_repo.get_fund_info.return_value = {"fund_name": "测试基金"}

        matcher = FundNewsMatcher()
        matcher.fund_industry_repo = mock_industry_repo.return_value
        matcher.news_classification_repo = mock_classification_repo.return_value

        result = matcher.match_fund_news("000001")

        assert len(result) == 1
        assert result[0].fund_code == "000001"
        assert result[0].industry == "新能源汽车"
        assert result[0].match_score == 0.85
