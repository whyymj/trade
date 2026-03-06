import pytest
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from unittest.mock import Mock, patch, MagicMock
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer
from services.fund_intel.modules.news_classification import NewsClassifier
from services.fund_intel.modules.fund_news_association import FundNewsMatcher
from services.fund_intel.modules.investment_advice import InvestmentAdviceGenerator


@pytest.fixture
def mock_cache():
    with patch("shared.cache.get_cache") as mock:
        cache = MagicMock()
        cache.get.return_value = None
        mock.return_value = cache
        yield cache


class TestFundIndustryAnalyzer:
    @pytest.fixture
    def analyzer(self, mock_cache):
        return FundIndustryAnalyzer()

    def test_analyze_success(self, analyzer):
        with patch.object(analyzer.fund_client, "get_fund_info") as mock_fund:
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源产业基金",
                "fund_type": "股票型",
            }

            result = analyzer.analyze("001")
            assert isinstance(result, list)
            assert len(result) >= 0
            if result:
                assert "industry" in result[0]
                assert "confidence" in result[0]

    def test_analyze_with_cache(self, analyzer, mock_cache):
        cached_data = [{"industry": "新能源", "confidence": 85.0, "source": "keyword"}]
        mock_cache.get.return_value = cached_data

        result = analyzer.analyze("001")
        assert result == cached_data

    def test_analyze_fund_not_found(self, analyzer):
        with patch.object(analyzer.fund_client, "get_fund_info") as mock_fund:
            mock_fund.return_value = None

            result = analyzer.analyze("999")
            assert result == []

    def test_classify_new_energy(self, analyzer):
        fund_name = "新能源产业基金"
        fund_type = "股票型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert len(result) >= 1
        if result:
            assert result[0]["industry"] == "新能源"
            assert result[0]["source"] == "keyword"

    def test_classify_semi_conductor(self, analyzer):
        fund_name = "半导体芯片基金"
        fund_type = "股票型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert len(result) > 0
        assert any(r["industry"] == "半导体" for r in result)

    def test_classify_medicine(self, analyzer):
        fund_name = "生物医药创新药基金"
        fund_type = "混合型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert len(result) > 0
        assert any(r["industry"] == "医药" for r in result)

    def test_classify_multiple_industries(self, analyzer):
        fund_name = "新能源半导体混合基金"
        fund_type = "混合型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert len(result) >= 1

    def test_classify_no_match(self, analyzer):
        fund_name = "普通成长基金"
        fund_type = "股票型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert len(result) == 0

    def test_confidence_calculation(self, analyzer):
        fund_name = "新能源光伏基金"
        fund_type = "股票型"

        result = analyzer._classify_by_keywords(fund_name, fund_type)
        assert result[0]["confidence"] > 0
        assert result[0]["confidence"] <= 95


class TestNewsClassifier:
    @pytest.fixture
    def classifier(self, mock_cache):
        return NewsClassifier()

    def test_classify_success(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "行业"

            result = classifier.classify("新能源汽车销量创新高")
            assert result["industry"] == "行业"
            assert result["confidence"] == 0.8

    def test_classify_with_invalid_result(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "不存在的行业"

            result = classifier.classify("测试文本")
            assert result["industry"] == "其他"
            assert result["confidence"] == 0.3

    def test_classify_with_exception(self, classifier):
        with patch.object(
            classifier.llm_client, "chat", side_effect=Exception("API error")
        ):
            result = classifier.classify("测试文本")
            assert result["industry"] == "其他"
            assert result["confidence"] == 0.3

    def test_classify_macro_news(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "宏观"

            result = classifier.classify("GDP增速达到预期")
            assert result["industry"] == "宏观"

    def test_classify_policy_news(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "政策"

            result = classifier.classify("央行降准0.5个百分点")
            assert result["industry"] == "政策"

    def test_classify_company_news(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "公司"

            result = classifier.classify("特斯拉发布新款车型")
            assert result["industry"] == "公司"

    def test_classify_global_news(self, classifier):
        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "全球"

            result = classifier.classify("美联储宣布加息")
            assert result["industry"] == "全球"


class TestFundNewsMatcher:
    @pytest.fixture
    def matcher(self, mock_cache):
        return FundNewsMatcher()

    def test_match_fund_news_success(self, matcher):
        with (
            patch.object(matcher.industry_analyzer, "analyze") as mock_analyze,
            patch.object(matcher.news_client, "get_news_by_industry") as mock_news,
        ):
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_news.return_value = [
                {"id": 1, "title": "新能源汽车销量大增"},
                {"id": 2, "title": "光伏产业政策支持"},
            ]

            result = matcher.match_fund_news("001", days=7)
            assert len(result) == 2
            assert result[0]["match_industry"] == "新能源"
            assert result[0]["match_score"] == 0.85

    def test_match_fund_news_no_industries(self, matcher):
        with patch.object(matcher.industry_analyzer, "analyze") as mock_analyze:
            mock_analyze.return_value = []

            result = matcher.match_fund_news("001", days=7)
            assert result == []

    def test_match_fund_news_deduplication(self, matcher):
        with (
            patch.object(matcher.industry_analyzer, "analyze") as mock_analyze,
            patch.object(matcher.news_client, "get_news_by_industry") as mock_news,
        ):
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_news.return_value = [
                {"id": 1, "title": "News 1"},
                {"id": 1, "title": "News 1"},
            ]

            result = matcher.match_fund_news("001", days=7)
            assert len(result) == 1

    def test_match_fund_news_sorting(self, matcher):
        with (
            patch.object(matcher.industry_analyzer, "analyze") as mock_analyze,
            patch.object(matcher.news_client, "get_news_by_industry") as mock_news,
        ):
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0},
                {"industry": "半导体", "confidence": 70.0},
            ]
            mock_news.side_effect = [
                [{"id": 1, "title": "Low score news"}],
                [{"id": 2, "title": "High score news"}],
            ]

            result = matcher.match_fund_news("001", days=7)
            assert result[0]["match_score"] >= result[-1]["match_score"]

    def test_match_fund_news_limit(self, matcher):
        with (
            patch.object(matcher.industry_analyzer, "analyze") as mock_analyze,
            patch.object(matcher.news_client, "get_news_by_industry") as mock_news,
        ):
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_news.return_value = [
                {"id": i, "title": f"News {i}"} for i in range(1, 31)
            ]

            result = matcher.match_fund_news("001", days=7)
            assert len(result) <= 20

    def test_get_fund_industries_success(self, matcher):
        with patch.object(matcher.industry_analyzer, "analyze") as mock_analyze:
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0},
                {"industry": "半导体", "confidence": 70.0},
            ]

            result = matcher.get_fund_industries("001")
            assert result == ["新能源", "半导体"]

    def test_get_fund_industries_empty(self, matcher):
        with patch.object(matcher.industry_analyzer, "analyze") as mock_analyze:
            mock_analyze.return_value = []

            result = matcher.get_fund_industries("001")
            assert result == []


class TestInvestmentAdviceGenerator:
    @pytest.fixture
    def generator(self, mock_cache):
        return InvestmentAdviceGenerator()

    def test_generate_success(self, generator):
        with (
            patch.object(generator.fund_client, "get_fund_info") as mock_fund,
            patch.object(generator.fund_client, "get_fund_nav") as mock_nav,
            patch.object(generator.industry_analyzer, "analyze") as mock_analyze,
            patch.object(generator.news_matcher, "match_fund_news") as mock_news,
            patch.object(generator.llm_client, "chat") as mock_llm,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
                "fund_type": "股票型",
            }
            mock_nav.return_value = [{"date": "2024-01-01", "nav": 1.0}]
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_news.return_value = [
                {
                    "title": "新能源汽车销量大增",
                    "match_industry": "新能源",
                    "match_score": 0.8,
                }
            ]
            mock_llm.return_value = """{
                "short_term": "建议持有",
                "medium_term": "关注新能源板块",
                "long_term": "长期看好",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["政策", "市场", "技术"]
            }"""

            result = generator.generate("001")
            assert result is not None
            assert "short_term" in result
            assert "medium_term" in result
            assert "long_term" in result
            assert "risk_level" in result
            assert "confidence" in result
            assert "key_factors" in result

    def test_generate_fund_not_found(self, generator):
        with patch.object(generator.fund_client, "get_fund_info") as mock_fund:
            mock_fund.return_value = None

            result = generator.generate("999")
            assert result is None

    def test_generate_with_fallback(self, generator):
        with (
            patch.object(generator.fund_client, "get_fund_info") as mock_fund,
            patch.object(generator.fund_client, "get_fund_nav") as mock_nav,
            patch.object(generator.industry_analyzer, "analyze") as mock_analyze,
            patch.object(generator.news_matcher, "match_fund_news") as mock_news,
            patch.object(
                generator.llm_client, "chat", side_effect=Exception("API error")
            ),
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
                "fund_type": "股票型",
            }
            mock_nav.return_value = [{"date": "2024-01-01", "nav": 1.0}]
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_news.return_value = []

            result = generator.generate("001")
            assert result is not None
            assert "short_term" in result

    def test_parse_advice_response_success(self, generator):
        response = """{
            "short_term": "建议持有",
            "medium_term": "关注板块",
            "long_term": "长期看好",
            "risk_level": "中",
            "confidence": 75,
            "key_factors": ["政策", "市场"]
        }"""

        result = generator._parse_advice_response(response)
        assert result["short_term"] == "建议持有"
        assert result["risk_level"] == "中"
        assert result["confidence"] == 75

    def test_parse_advice_response_invalid_json(self, generator):
        response = "这不是JSON格式"

        result = generator._parse_advice_response(response)
        assert "短期建议" in result["short_term"]

    def test_parse_advice_response_no_json_match(self, generator):
        response = "这是一段文本，没有JSON"

        result = generator._parse_advice_response(response)
        assert "short_term" in result

    def test_fallback_advice_positive_sentiment(self, generator):
        industries = ["新能源"]
        news = [
            {"title": "News 1", "score": 0.8},
            {"title": "News 2", "score": 0.75},
            {"title": "News 3", "score": 0.3},
        ]

        result = generator._fallback_advice(industries, news)
        assert "偏正面" in result["short_term"]

    def test_fallback_advice_negative_sentiment(self, generator):
        industries = ["新能源"]
        news = [
            {"title": "News 1", "score": 0.3},
            {"title": "News 2", "score": 0.35},
            {"title": "News 3", "score": 0.8},
        ]

        result = generator._fallback_advice(industries, news)
        assert "偏负面" in result["short_term"]

    def test_fallback_advice_neutral_sentiment(self, generator):
        industries = ["新能源"]
        news = [
            {"title": "News 1", "score": 0.5},
            {"title": "News 2", "score": 0.55},
        ]

        result = generator._fallback_advice(industries, news)
        assert "中性" in result["short_term"]

    def test_fallback_advice_no_news(self, generator):
        industries = ["新能源"]
        news = []

        result = generator._fallback_advice(industries, news)
        assert result["risk_level"] == "中"

    def test_fallback_advice_no_industries(self, generator):
        industries = []
        news = []

        result = generator._fallback_advice(industries, news)
        assert "相关" in result["medium_term"]
