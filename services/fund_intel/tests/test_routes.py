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


class TestFundIndustryRoutes:
    def test_analyze_fund_industry_success(self, client):
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
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0, "source": "keyword"}
            ]
            mock_llm.return_value = "This is a good fund for new energy sector."

            response = client.post("/api/fund-industry/analyze/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "industry_distribution" in data["data"]
            assert "llm_analysis" in data["data"]

    def test_analyze_fund_industry_not_found(self, client):
        with patch(
            "services.fund_intel.routes.fund_industry.fund_client.get_fund_info"
        ) as mock_fund:
            mock_fund.return_value = None

            response = client.post("/api/fund-industry/analyze/999")
            assert response.status_code == 404

            data = json.loads(response.data)
            assert not data["success"]

    def test_get_fund_industry(self, client, mock_cache):
        with patch(
            "services.fund_intel.routes.fund_industry.analyzer.analyze"
        ) as mock_analyze:
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]

            response = client.get("/api/fund-industry/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "data" in data

    def test_get_fund_industry_cached(self, client, mock_cache):
        cached_data = [{"industry": "新能源", "confidence": 85.0}]
        mock_cache.get.return_value = cached_data

        response = client.get("/api/fund-industry/001")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["data"] == cached_data

    def test_get_primary_industry(self, client):
        with patch(
            "services.fund_intel.routes.fund_industry.analyzer.analyze"
        ) as mock_analyze:
            mock_analyze.return_value = [
                {"industry": "新能源", "confidence": 85.0},
                {"industry": "半导体", "confidence": 75.0},
                {"industry": "医药", "confidence": 60.0},
            ]

            response = client.get("/api/fund-industry/primary/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 3

    def test_get_primary_industry_empty(self, client):
        with patch(
            "services.fund_intel.routes.fund_industry.analyzer.analyze"
        ) as mock_analyze:
            mock_analyze.return_value = []

            response = client.get("/api/fund-industry/primary/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["data"] == []


class TestNewsClassificationRoutes:
    def test_classify_news_success(self, client):
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

    def test_classify_news_missing_text(self, client):
        response = client.post("/api/news-classification/classify", json={})
        assert response.status_code == 400

        data = json.loads(response.data)
        assert not data["success"]

    def test_classify_today_news(self, client):
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
            ]
            mock_classify.side_effect = ["新能源", "半导体"]

            response = client.post("/api/news-classification/classify-today")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert data["data"]["total"] == 2
            assert data["data"]["classified"] == 2

    def test_get_industries(self, client):
        response = client.get("/api/news-classification/industries")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["success"]
        assert "行业" in data["data"]

    def test_get_news_by_industry(self, client):
        with patch(
            "services.fund_intel.routes.news_classification.news_client.get_news_by_industry"
        ) as mock_news:
            mock_news.return_value = [{"id": 1, "title": "新能源新闻"}]

            response = client.get("/api/news-classification/industry/新能源?days=7")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 1

    def test_get_industry_stats(self, client):
        with patch(
            "services.fund_intel.routes.news_classification.news_client.get_news_by_industry"
        ) as mock_news:
            mock_news.return_value = [{"id": 1}, {"id": 2}, {"id": 3}]

            response = client.get("/api/news-classification/stats")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert isinstance(data["data"], dict)

    def test_get_today_classified(self, client):
        with (
            patch(
                "services.fund_intel.routes.news_classification.news_client.get_news"
            ) as mock_news,
            patch(
                "services.fund_intel.routes.news_classification.llm_client.classify_industry"
            ) as mock_classify,
        ):
            mock_news.return_value = [
                {"id": 1, "title": "News 1", "content": "Content 1"}
            ]
            mock_classify.return_value = "新能源"

            response = client.get("/api/news-classification/today")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) > 0


class TestFundNewsRoutes:
    def test_match_fund_news_success(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_news.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.fund_news.matcher.match_fund_news"
            ) as mock_match,
            patch(
                "services.fund_intel.routes.fund_news.matcher.get_fund_industries"
            ) as mock_get_industries,
        ):
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
            mock_get_industries.return_value = ["新能源", "半导体"]
            mock_match.return_value = [
                {"title": "News 1", "match_industry": "新能源", "match_score": 0.8}
            ]

            response = client.get("/api/fund-news/match/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "news" in data["data"]
            assert data["data"]["news_count"] == 1

    def test_match_fund_news_not_found(self, client):
        with patch(
            "services.fund_intel.routes.fund_news.fund_client.get_fund_info"
        ) as mock_fund:
            mock_fund.return_value = None

            response = client.get("/api/fund-news/match/999")
            assert response.status_code == 404

            data = json.loads(response.data)
            assert not data["success"]

    def test_get_fund_news_summary(self, client):
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
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
            mock_industries.return_value = ["新能源"]
            mock_match.return_value = [
                {"title": "News 1", "match_score": 0.8},
                {"title": "News 2", "match_score": 0.6},
                {"title": "News 3", "match_score": 0.3},
            ]

            response = client.get("/api/fund-news/summary/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "sentiment" in data["data"]
            assert "latest_news" in data["data"]

    def test_get_funds_with_news(self, client):
        with (
            patch(
                "services.fund_intel.routes.fund_news.fund_client.get_fund_list"
            ) as mock_fund_list,
            patch(
                "services.fund_intel.routes.fund_news.matcher.match_fund_news"
            ) as mock_match,
        ):
            mock_fund_list.return_value = {
                "data": [
                    {"fund_code": "001", "fund_name": "Fund 1"},
                    {"fund_code": "002", "fund_name": "Fund 2"},
                ]
            }
            mock_match.side_effect = [
                [{"title": "News 1"}],
                [],
            ]

            response = client.get("/api/fund-news/list")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 1


class TestInvestmentAdviceRoutes:
    def test_get_investment_advice_success(self, client):
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
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
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

    def test_get_investment_advice_not_found(self, client):
        with patch(
            "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
        ) as mock_fund:
            mock_fund.return_value = None

            response = client.get("/api/investment-advice/999")
            assert response.status_code == 404

            data = json.loads(response.data)
            assert not data["success"]

def test_get_investment_advice_generation_failed(self, client):
        with (
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.investment_advice.advice_generator.generate"
            ) as mock_generate,
            patch.object(
                client.application, "test_request_context"
            ) as mock_ctx,
        ):
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
            mock_generate.return_value = None

            response = client.get("/api/investment-advice/001")
            assert response.status_code == 500

            data = json.loads(response.data)
            assert not data["success"]

    def test_get_batch_investment_advice_success(self, client):
        with patch(
            "services.fund_intel.routes.investment_advice.advice_generator.generate"
        ) as mock_generate:
            mock_generate.return_value = {
                "short_term": "建议持有",
                "medium_term": "关注",
                "long_term": "长期看好",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["政策"],
            }

            response = client.post(
                "/api/investment-advice/batch",
                json={"fund_codes": ["001", "002"]},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 2

    def test_get_batch_investment_advice_missing_codes(self, client):
        response = client.post("/api/investment-advice/batch", json={})
        assert response.status_code == 400

        data = json.loads(response.data)
        assert not data["success"]

    def test_get_batch_investment_advice_partial_failure(self, client):
        with patch(
            "services.fund_intel.routes.investment_advice.advice_generator.generate"
        ) as mock_generate:
            mock_generate.side_effect = [
                {"short_term": "建议持有"},
                Exception("API error"),
            ]

            response = client.post(
                "/api/investment-advice/batch",
                json={"fund_codes": ["001", "002"]},
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert len(data["data"]) == 2
            assert data["data"][0]["success"]
            assert not data["data"][1]["success"]


class TestHealthAndMetrics:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "fund-intel-service" in data["service"]

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "fund-intel-service" in data["service"]
        assert "uptime" in data
