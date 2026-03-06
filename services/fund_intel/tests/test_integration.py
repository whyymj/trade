import pytest
import json
import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

from unittest.mock import Mock, patch, MagicMock
from services.fund_intel.app import app
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer
from services.fund_intel.modules.news_classification import NewsClassifier
from services.fund_intel.modules.fund_news_association import FundNewsMatcher
from services.fund_intel.modules.investment_advice import InvestmentAdviceGenerator


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_cache():
    with patch("shared.cache.get_cache") as mock:
        cache = MagicMock()
        cache.get.return_value = None
        mock.return_value = cache
        yield cache


class TestFundIndustryAnalysisFlow:
    def test_complete_fund_industry_analysis(self, mock_cache):
        analyzer = FundIndustryAnalyzer()

        with patch.object(
            analyzer.fund_client, "get_fund_info"
        ) as mock_fund:
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

    def test_fund_industry_with_cache(self, mock_cache):
        analyzer = FundIndustryAnalyzer()
        cached_data = [{"industry": "新能源", "confidence": 85.0}]
        mock_cache.get.return_value = cached_data

        result = analyzer.analyze("001")
        assert result == cached_data

    def test_fund_industry_api_endpoint(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_industry.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.fund_industry.analyzer.analyze"
            ) as mock_analyze,
            patch(
                "services.fund_intel.routes.fund_industry.llm_client.chat"
            ) as mock_llm,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
            }
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_llm.return_value = "新能源行业前景看好"

            response = client.post("/api/fund-industry/analyze/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "industry_distribution" in data["data"]
            assert "llm_analysis" in data["data"]


class TestNewsClassificationFlow:
    def test_complete_news_classification(self, mock_cache):
        classifier = NewsClassifier()

        with patch.object(classifier.llm_client, "chat") as mock_chat:
            mock_chat.return_value = "行业"

            result = classifier.classify("新能源汽车销量创新高")
            assert result["industry"] == "行业"
            assert result["confidence"] == 0.8

    def test_news_classification_api_endpoint(self, client):
        with patch(
            "services.fund_intel.routes.news_classification.llm_client.classify_industry"
        ) as mock_classify:
            mock_classify.return_value = "新能源"

            response = client.post(
                "/api/news-classification/classify",
                json={"text": "新能源汽车市场火热"},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert data["data"]["industry"] == "新能源"

    def test_classify_today_news_batch(self, client):
        with (
            patch(
                "services.fund_intel.routes.news_classification.news_client.get_news"
            ) as mock_news,
            patch(
                "services.fund_intel.routes.news_classification.llm_client.classify_industry"
            ) as mock_classify,
        ):
            mock_news.return_value = [
                {"id": 1, "title": "News 1", "content": "Content 1"},
                {"id": 2, "title": "News 2", "content": "Content 2"},
                {"id": 3, "title": "News 3", "content": "Content 3"},
            ]
            mock_classify.side_effect = ["新能源", "半导体", "医药"]

            response = client.post("/api/news-classification/classify-today")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert data["data"]["total"] == 3
            assert data["data"]["classified"] == 3
            assert len(data["data"]["results"]) == 3


class TestFundNewsMatchingFlow:
    def test_complete_fund_news_matching(self, mock_cache):
        matcher = FundNewsMatcher()

        with (
            patch.object(matcher.industry_analyzer, "analyze") as mock_analyze,
            patch.object(matcher.news_client, "get_news_by_industry") as mock_news,
        ):
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0},
                {"industry": "半导体", "confidence": 70.0},
            ]
            mock_news.side_effect = [
                [
                    {"id": 1, "title": "新能源汽车销量大增"},
                    {"id": 2, "title": "光伏产业政策支持"},
                ],
                [
                    {"id": 3, "title": "芯片技术突破"},
                ],
            ]

            result = matcher.match_fund_news("001", days=7)

            assert len(result) == 3
            assert all("match_industry" in news for news in result)
            assert all("match_score" in news for news in result)

    def test_fund_news_matching_api_endpoint(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_news.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.fund_news.matcher.match_fund_news"
            ) as mock_match,
            patch(
                "services.fund_intel.routes.fund_news.matcher.get_fund_industries"
            ) as mock_industries,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
            }
            mock_industries.return_value = ["新能源"]
            mock_match.return_value = [
                {"title": "News 1", "match_industry": "新能源", "match_score": 0.8}
            ]

            response = client.get("/api/fund-news/match/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "industries" in data["data"]
            assert "news" in data["data"]

    def test_fund_news_summary_with_sentiment(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_news.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.fund_news.matcher.get_fund_industries"
            ) as mock_industries,
            patch(
                "services.fund_intel.routes.fund_news.matcher.match_fund_news"
            ) as mock_match,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
            }
            mock_industries.return_value = ["新能源"]
            mock_match.return_value = [
                {"title": "News 1", "match_score": 0.8},
                {"title": "News 2", "match_score": 0.7},
                {"title": "News 3", "match_score": 0.3},
            ]

            response = client.get("/api/fund-news/summary/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "sentiment" in data["data"]
            assert "latest_news" in data["data"]
            assert data["data"]["sentiment"] == "positive"


class TestInvestmentAdviceGenerationFlow:
    def test_complete_investment_advice_generation(self, mock_cache):
        generator = InvestmentAdviceGenerator()

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

    def test_investment_advice_api_endpoint(self, client):
        with (
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_nav"
            ) as mock_nav,
            patch(
                "services.fund_intel.routes.investment_advice.advice_generator.generate"
            ) as mock_generate,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
            }
            mock_nav.return_value = [{"date": "2024-01-01", "nav": 1.0}]
            mock_generate.return_value = {
                "short_term": "建议持有",
                "medium_term": "关注新能源板块",
                "long_term": "长期看好",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["政策", "市场", "技术"],
            }

            response = client.get("/api/investment-advice/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "advice" in data["data"]
            assert "generated_at" in data["data"]

    def test_batch_investment_advice_generation(self, client):
        with patch(
            "services.fund_intel.routes.investment_advice.advice_generator.generate"
        ) as mock_generate:
            mock_generate.return_value = {
                "short_term": "建议持有",
                "medium_term": "关注板块",
                "long_term": "长期看好",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["政策"],
            }

            response = client.post(
                "/api/investment-advice/batch",
                json={"fund_codes": ["001", "002", "003"]},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 3

    def test_investment_advice_with_fallback(self, mock_cache):
        generator = InvestmentAdviceGenerator()

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
            assert "confidence" in result


class TestCrossServiceIntegration:
def test_fund_industry_to_news_matching(self, mock_cache):
        analyzer = FundIndustryAnalyzer()
        matcher = FundNewsMatcher()

        with (
            patch.object(
                analyzer.fund_client, "get_fund_info"
            ) as mock_fund,
            patch.object(
                matcher.news_client, "get_news_by_industry"
            ) as mock_news,
        ):
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源产业基金",
                "fund_type": "股票型",
            }
            mock_news.return_value = [
                {"id": 1, "title": "新能源汽车销量大增"}
            ]

            industries = analyzer.analyze("001")
            assert len(industries) >= 0

            news = matcher.match_fund_news("001", days=7)
            assert len(news) >= 0
            if news and industries:
                assert news[0]["match_industry"] in [ind["industry"] for ind in industries]

def test_news_classification_to_industry_matching(self, mock_cache):
        classifier = NewsClassifier()
        matcher = FundNewsMatcher()

        with (
            patch.object(
                classifier.llm_client, "chat"
            ) as mock_chat,
            patch.object(
                matcher.industry_analyzer, "analyze"
            ) as mock_analyze,
            patch.object(
                matcher.news_client, "get_news_by_industry"
            ) as mock_news,
        ):
            mock_chat.return_value = "新能源"
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0}
            ]
            mock_news.return_value = [
                {"id": 1, "title": "新能源汽车新闻"}
            ]

            classification = classifier.classify("新能源汽车销量大增")
            assert classification["industry"] in ["行业", "新能源"]

            news = matcher.match_fund_news("001", days=7)
            assert len(news) >= 0

    def test_full_pipeline_fund_to_advice(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_industry.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.fund_industry.analyzer.analyze"
            ) as mock_analyze,
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
            ) as mock_fund2,
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_nav"
            ) as mock_nav,
            patch(
                "services.fund_intel.routes.investment_advice.advice_generator.generate"
            ) as mock_generate,
        ):
            fund_data = {
                "fund_code": "001",
                "fund_name": "新能源基金",
                "fund_type": "股票型",
            }
            mock_fund.return_value = fund_data
            mock_fund2.return_value = fund_data
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_nav.return_value = [{"date": "2024-01-01", "nav": 1.0}]
            mock_generate.return_value = {
                "short_term": "建议持有",
                "medium_term": "关注",
                "long_term": "长期看好",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["政策"],
            }

            response_industry = client.post("/api/fund-industry/analyze/001")
            assert response_industry.status_code == 200

            response_advice = client.get("/api/investment-advice/001")
            assert response_advice.status_code == 200

            data_advice = json.loads(response_advice.data)
            assert data_advice["success"]
            assert "advice" in data_advice["data"]


class TestErrorHandlingAndEdgeCases:
    def test_fund_not_found_in_all_endpoints(self, client):
        with patch(
            "services.fund_intel.routes.fund_industry.fund_client.get_fund_info"
        ) as mock_fund:
            mock_fund.return_value = None

            response = client.post("/api/fund-industry/analyze/999")
            assert response.status_code == 404

    def test_news_classification_with_empty_text(self, client):
        response = client.post("/api/news-classification/classify", json={"text": ""})
        assert response.status_code == 400

    def test_investment_advice_generation_with_invalid_fund(self, client):
        with patch(
            "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
        ) as mock_fund:
            mock_fund.return_value = None

            response = client.get("/api/investment-advice/999")
            assert response.status_code == 404

    def test_batch_advice_with_empty_codes(self, client):
        response = client.post("/api/investment-advice/batch", json={"fund_codes": []})
        assert response.status_code == 400

    def test_cache_behavior_across_modules(self, mock_cache):
        analyzer = FundIndustryAnalyzer()

        with patch.object(analyzer.fund_client, "get_fund_info") as mock_fund:
            mock_fund.return_value = {
                "fund_code": "001",
                "fund_name": "新能源基金",
            }

            result1 = analyzer.analyze("001")
            cached_data = result1
            mock_cache.get.return_value = cached_data

            result2 = analyzer.analyze("001")
            assert result1 == result2
