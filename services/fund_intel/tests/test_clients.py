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


class TestFundClient:
    def test_get_fund_info_success(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

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
            assert result["fund_name"] == "Test Fund"

    def test_get_fund_info_not_found(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            client = FundClient()
            result = client.get_fund_info("999")
            assert result is None

    def test_get_fund_nav_success(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"date": "2024-01-01", "nav": 1.0}]
            }
            mock_get.return_value = mock_response

            client = FundClient()
            result = client.get_fund_nav("001", days=30)
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["nav"] == 1.0

    def test_get_fund_list_success(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"fund_code": "001", "fund_name": "Fund 1"},
                    {"fund_code": "002", "fund_name": "Fund 2"},
                ]
            }
            mock_get.return_value = mock_response

            client = FundClient()
            result = client.get_fund_list(page=1, size=50)
            assert "data" in result
            assert len(result["data"]) == 2


class TestNewsClient:
    def test_get_news_success(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": 1, "title": "News 1"},
                    {"id": 2, "title": "News 2"},
                ]
            }
            mock_get.return_value = mock_response

            client = NewsClient()
            result = client.get_news(days=1)
            assert isinstance(result, list)
            assert len(result) == 2

    def test_get_news_with_category(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.get") as mock_get,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": 1, "title": "News 1"}]}
            mock_get.return_value = mock_response

            client = NewsClient()
            result = client.get_news(days=1, category="新能源")
            assert len(result) == 1


class TestLLMClient:
    def test_chat_success(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.post") as mock_post,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "Test response"}
            mock_post.return_value = mock_response

            client = LLMClient()
            result = client.chat(
                [{"role": "user", "content": "test"}], provider="deepseek"
            )
            assert result == "Test response"

    def test_chat_error(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.post") as mock_post,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 500
            mock_post.return_value = mock_response

            client = LLMClient()
            result = client.chat(
                [{"role": "user", "content": "test"}], provider="deepseek"
            )
            assert result is None

    def test_classify_industry(self):
        with (
            patch("shared.cache.get_cache") as mock_cache,
            patch("requests.post") as mock_post,
        ):
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "新能源"}
            mock_post.return_value = mock_response

            client = LLMClient()
            result = client.classify_industry("新能源汽车市场火热")
            assert result == "新能源"

    def test_cache_key_generation(self):
        with patch("shared.cache.get_cache") as mock_cache:
            cache = MagicMock()
            cache.get.return_value = None
            mock_cache.return_value = cache

            client = LLMClient()
            messages = [{"role": "user", "content": "test"}]
            cache_key = client._get_cache_key(messages, "deepseek")
            assert "llm:deepseek:" in cache_key
