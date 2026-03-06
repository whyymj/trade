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
    def test_analyze_fund_industry(self, client, mock_cache):
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
            mock_analyze.return_value = [{"industry": "新能源", "confidence": 85.0}]
            mock_llm.return_value = "This is a good fund for new energy sector."

            response = client.post("/api/fund-industry/analyze/001")
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert "industry_distribution" in data["data"]


class TestNewsClassificationRoutes:
    def test_classify_news(self, client):
        with patch(
            "services.fund_intel.routes.news_classification.llm_client.classify_industry"
        ) as mock_classify:
            mock_classify.return_value = "新能源"

            response = client.post(
                "/api/news-classification/classify", json={"text": "新能源汽车市场火热"}
            )
            assert response.status_code == 200

            data = json.loads(response.data)
            assert data["success"]
            assert data["data"]["industry"] == "新能源"


class TestFundNewsRoutes:
    def test_match_fund_news(self, client, mock_cache):
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


class TestInvestmentAdviceRoutes:
    def test_get_investment_advice(self, client, mock_cache):
        with (
            patch(
                "services.fund_intel.routes.investment_advice.fund_client.get_fund_info"
            ) as mock_fund,
            patch(
                "services.fund_intel.routes.investment_advice.advice_generator.generate"
            ) as mock_generate,
        ):
            mock_fund.return_value = {"fund_code": "001", "fund_name": "Test Fund"}
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
