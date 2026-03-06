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
from services.fund_intel.clients import FundClient, NewsClient, LLMClient


@pytest.fixture
def mock_cache():
    with patch("shared.cache.get_cache") as mock:
        cache = MagicMock()
        cache.get.return_value = None
        mock.return_value = cache
        yield cache


class TestFundClient:
    def test_get_fund_info(self, mock_cache):
        with patch("services.fund_intel.clients.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": {"fund_code": "001", "fund_name": "Test Fund"}
            }
            mock_get.return_value = mock_response

            client = FundClient()
            result = client.get_fund_info("001")
            assert result is not None
            assert result["fund_code"] == "001"

    def test_get_fund_nav(self, mock_cache):
        with patch("services.fund_intel.clients.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"date": "2024-01-01", "nav": 1.0}]
            }
            mock_get.return_value = mock_response

            client = FundClient()
            result = client.get_fund_nav("001", days=30)
            assert isinstance(result, list)


class TestNewsClient:
    def test_get_news(self, mock_cache):
        with patch("services.fund_intel.clients.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"id": 1, "title": "Test News"}]
            }
            mock_get.return_value = mock_response

            client = NewsClient()
            result = client.get_news(days=1)
            assert isinstance(result, list)


class TestLLMClient:
    def test_chat(self, mock_cache):
        with patch("services.fund_intel.clients.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "Test response"}
            mock_post.return_value = mock_response

            client = LLMClient()
            result = client.chat(
                [{"role": "user", "content": "test"}], provider="deepseek"
            )
            assert result == "Test response"

    def test_classify_industry(self):
        with patch("services.fund_intel.clients.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "新能源"}
            mock_post.return_value = mock_response

            client = LLMClient()
            result = client.classify_industry("新能源汽车市场火热")
            assert result == "新能源"
