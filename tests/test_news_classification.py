# tests/test_news_classification.py
"""
新闻行业分类模块测试
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import date, datetime

from modules.news_classification import (
    NewsClassifier,
    ClassificationRepo,
    IndustryClassification,
    INDUSTRY_CATEGORIES,
)


class TestIndustryCategories:
    """测试行业分类常量"""

    def test_industry_categories_not_empty(self):
        """行业分类列表不应为空"""
        assert len(INDUSTRY_CATEGORIES) > 0

    def test_industry_categories_has_required_fields(self):
        """每个行业分类应有 code, name, keywords"""
        for cat in INDUSTRY_CATEGORIES:
            assert "code" in cat
            assert "name" in cat
            assert "keywords" in cat
            assert cat["code"].startswith("I")
            assert len(cat["code"]) == 4

    def test_industry_categories_unique_codes(self):
        """行业代码应唯一"""
        codes = [c["code"] for c in INDUSTRY_CATEGORIES]
        assert len(codes) == len(set(codes))


class TestNewsClassifier:
    """测试新闻分类器"""

    @patch("modules.news_classification.analyzer.DeepSeekClient")
    def test_classify_news_llm_response(self, mock_client_cls):
        """测试 LLM 分类响应"""
        mock_client = MagicMock()
        mock_client.chat.return_value = """
        {
            "industry": "新能源汽车",
            "industry_code": "I001",
            "confidence": 0.85,
            "reasoning": "包含比亚迪、锂电池等关键词"
        }
        """
        mock_client_cls.return_value = mock_client

        classifier = NewsClassifier(use_deepseek=True)
        result = classifier.classify_news(
            "比亚迪发布新款新能源汽车", "比亚迪今日发布...", "东方财富"
        )

        assert result.industry == "新能源汽车"
        assert result.industry_code == "I001"
        assert result.confidence > 0

    def test_fallback_classify_by_keywords(self):
        """测试关键词兜底分类"""
        classifier = NewsClassifier()

        result = classifier._fallback_classify(
            "宁德时代发布最新电池技术", "宁德时代今日发布..."
        )

        assert result.industry in ["新能源汽车", "新能源", "其他"]
        assert result.confidence > 0

    def test_fallback_classify_no_match(self):
        """测试无匹配时的兜底分类"""
        classifier = NewsClassifier()

        result = classifier._fallback_classify("今天天气很好", "...")

        assert result.industry == "其他"
        assert result.industry_code == "I000"
        assert result.confidence < 1.0


class TestClassificationRepo:
    """测试分类仓储"""

    def test_get_industry_stats(self):
        """测试获取行业统计"""
        repo = ClassificationRepo()
        stats = repo.get_industry_stats(days=7)

        assert isinstance(stats, list)

    def test_get_by_industry(self):
        """测试按行业获取分类结果"""
        repo = ClassificationRepo()
        news_list = repo.get_by_industry("I001", days=7)

        assert isinstance(news_list, list)


class TestIndustryClassification:
    """测试行业分类数据类"""

    def test_industry_classification_creation(self):
        """测试创建行业分类对象"""
        ic = IndustryClassification(
            industry="半导体",
            industry_code="I002",
            confidence=0.9,
            reasoning="包含中芯国际等关键词",
        )

        assert ic.industry == "半导体"
        assert ic.industry_code == "I002"
        assert ic.confidence == 0.9
        assert ic.reasoning == "包含中芯国际等关键词"
