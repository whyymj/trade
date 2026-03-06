import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import pytest
import json
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_cache():
    """Mock cache for testing"""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set = MagicMock()
    return cache


@pytest.fixture
def client(mock_cache):
    """创建测试客户端"""
    with (
        patch("shared.cache.RedisCache", return_value=mock_cache),
        patch("shared.cache.get_cache", return_value=mock_cache),
    ):
        from services.market.app import app

        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


class TestHealthEndpoints:
    """测试健康检查端点"""

    def test_health_endpoint(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "market-service"

    def test_metrics_endpoint(self, client):
        """测试指标端点"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["service"] == "market-service"


class TestMacroDataEndpoints:
    """测试宏观数据端点"""

    def test_get_macro_data_default(self, client):
        """测试获取宏观数据（默认参数）"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": 1,
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                    "trade_date": None,
                }
            ]

            response = client.get("/api/market/macro")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert "data" in data
            assert isinstance(data["data"], list)

    def test_get_macro_data_with_indicator(self, client):
        """测试按指标获取宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": 1,
                    "indicator": "CPI",
                    "period": "2024-03",
                    "value": 0.1,
                    "unit": "%",
                    "source": "国家统计局",
                    "publish_date": "2024-04-11",
                    "trade_date": None,
                }
            ]

            response = client.get("/api/market/macro?indicator=CPI&days=30")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_macro_data_with_days(self, client):
        """测试按天数获取宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []

            response = client.get("/api/market/macro?days=7")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_macro_data_cached(self, client, mock_cache):
        """测试宏观数据缓存"""
        mock_cache.get.return_value = [
            {"indicator": "GDP", "period": "2024Q1", "value": 29.6}
        ]

        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            response = client.get("/api/market/macro")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["cached"] is True
            mock_cache.get.assert_called()


class TestMoneyFlowEndpoints:
    """测试资金流向端点"""

    def test_get_money_flow_default(self, client):
        """测试获取资金流向（默认参数）"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": "2024-04-15",
                    "north_money": -50.5,
                    "north_buy": 120.3,
                    "north_sell": 170.8,
                    "main_money": 30.2,
                    "margin_balance": 15000.5,
                }
            ]

            response = client.get("/api/market/money-flow")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert isinstance(data["data"], list)

    def test_get_money_flow_with_days(self, client):
        """测试按天数获取资金流向"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []

            response = client.get("/api/market/money-flow?days=7")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_money_flow_cached(self, client, mock_cache):
        """测试资金流向缓存"""
        mock_cache.get.return_value = [
            {"trade_date": "2024-04-15", "north_money": -50.5}
        ]

        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            response = client.get("/api/market/money-flow")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["cached"] is True


class TestSentimentEndpoints:
    """测试市场情绪端点"""

    def test_get_sentiment_default(self, client):
        """测试获取市场情绪（默认参数）"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": "2024-04-15",
                    "volume": 850000000000,
                    "up_count": 1200,
                    "down_count": 800,
                    "turnover_rate": 1.2,
                    "advance_count": 3000,
                    "decline_count": 1500,
                }
            ]

            response = client.get("/api/market/sentiment")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert isinstance(data["data"], list)

    def test_get_sentiment_with_days(self, client):
        """测试按天数获取市场情绪"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []

            response = client.get("/api/market/sentiment?days=7")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_sentiment_cached(self, client, mock_cache):
        """测试市场情绪缓存"""
        mock_cache.get.return_value = [
            {"trade_date": "2024-04-15", "volume": 850000000000}
        ]

        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            response = client.get("/api/market/sentiment")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["cached"] is True


class TestGlobalMacroEndpoints:
    """测试全球宏观端点"""

    def test_get_global_macro_default(self, client):
        """测试获取全球宏观（默认参数）"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": "2024-04-15",
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                }
            ]

            response = client.get("/api/market/global")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert isinstance(data["data"], list)

    def test_get_global_macro_with_symbol(self, client):
        """测试按符号获取全球宏观"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": "2024-04-15",
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                }
            ]

            response = client.get("/api/market/global?symbol=USD/CNY")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_global_macro_with_days(self, client):
        """测试按天数获取全球宏观"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []

            response = client.get("/api/market/global?days=7")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_global_macro_cached(self, client, mock_cache):
        """测试全球宏观缓存"""
        mock_cache.get.return_value = [{"symbol": "USD/CNY", "close_price": 7.24}]

        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            response = client.get("/api/market/global")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["cached"] is True


class TestMarketFeaturesEndpoints:
    """测试市场特征端点"""

    def test_get_market_features_default(self, client):
        """测试获取市场特征（默认参数）"""
        mock_repo = MagicMock()
        mock_repo.get_market_features.return_value = {
            "macro": [{"indicator": "GDP", "value": 29.6}],
            "money_flow": [{"north_money": -50.5}],
            "sentiment": [{"volume": 850000000000}],
            "global": [{"symbol": "USD/CNY"}],
        }

        with patch("services.market.routes.market.repo", mock_repo):
            response = client.get("/api/market/features")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert "macro" in data["data"]
            assert "money_flow" in data["data"]
            assert "sentiment" in data["data"]
            assert "global" in data["data"]

    def test_get_market_features_with_days(self, client):
        """测试按天数获取市场特征"""
        mock_repo = MagicMock()
        mock_repo.get_market_features.return_value = {
            "macro": [],
            "money_flow": [],
            "sentiment": [],
            "global": [],
        }

        with patch("services.market.routes.market.repo", mock_repo):
            response = client.get("/api/market/features?days=7")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_get_market_features_cached(self, client, mock_cache):
        """测试市场特征缓存"""
        mock_cache.get.return_value = {
            "macro": [],
            "money_flow": [],
            "sentiment": [],
            "global": [],
        }

        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            response = client.get("/api/market/features")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["cached"] is True


class TestSyncEndpoints:
    """测试同步端点"""

    def test_sync_market_data_success(self, client):
        """测试同步市场数据成功"""
        mock_crawler = MagicMock()
        mock_crawler.sync_all.return_value = {
            "macro": [{"indicator": "GDP"}],
            "money_flow": [{"trade_date": "2024-04-15"}],
            "sentiment": [{"trade_date": "2024-04-15"}],
            "global": [{"symbol": "USD/CNY"}],
        }

        mock_repo = MagicMock()
        mock_repo.save_macro.return_value = 1
        mock_repo.save_money_flow.return_value = 1
        mock_repo.save_sentiment.return_value = 1
        mock_repo.save_global_macro.return_value = 1

        with (
            patch(
                "services.market.data.market_crawler.MarketCrawler",
                return_value=mock_crawler,
            ),
            patch("services.market.routes.market.repo", mock_repo),
        ):
            response = client.post("/api/market/sync")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert "data" in data
            assert data["data"]["saved"] == 4
            assert data["data"]["macro_records"] == 1
            assert data["data"]["money_flow_records"] == 1
            assert data["data"]["sentiment_records"] == 1
            assert data["data"]["global_records"] == 1

    def test_sync_market_data_with_empty_data(self, client):
        """测试同步市场数据（空数据）"""
        mock_crawler = MagicMock()
        mock_crawler.sync_all.return_value = {
            "macro": None,
            "money_flow": None,
            "sentiment": None,
            "global": None,
        }

        mock_repo = MagicMock()
        mock_repo.save_macro.return_value = 0
        mock_repo.save_money_flow.return_value = 0
        mock_repo.save_sentiment.return_value = 0
        mock_repo.save_global_macro.return_value = 0

        with (
            patch(
                "services.market.data.market_crawler.MarketCrawler",
                return_value=mock_crawler,
            ),
            patch("services.market.routes.market.repo", mock_repo),
        ):
            response = client.post("/api/market/sync")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert data["data"]["saved"] == 0

    def test_sync_market_data_with_partial_data(self, client):
        """测试同步市场数据（部分数据）"""
        mock_crawler = MagicMock()
        mock_crawler.sync_all.return_value = {
            "macro": [{"indicator": "GDP"}],
            "money_flow": None,
            "sentiment": [{"trade_date": "2024-04-15"}],
            "global": None,
        }

        mock_repo = MagicMock()
        mock_repo.save_macro.return_value = 1
        mock_repo.save_money_flow.return_value = 0
        mock_repo.save_sentiment.return_value = 1
        mock_repo.save_global_macro.return_value = 0

        with (
            patch(
                "services.market.data.market_crawler.MarketCrawler",
                return_value=mock_crawler,
            ),
            patch("services.market.routes.market.repo", mock_repo),
        ):
            response = client.post("/api/market/sync")
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert data["data"]["saved"] == 2
            assert data["data"]["macro_records"] == 1
            assert data["data"]["money_flow_records"] == 0
            assert data["data"]["sentiment_records"] == 1
            assert data["data"]["global_records"] == 0


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_days_parameter(self, client):
        """测试无效的天数参数"""
        response = client.get("/api/market/macro?days=invalid")
        assert response.status_code == 500

    def test_get_macro_data_not_cached_fetch_error(self, client):
        """测试宏观数据获取失败"""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with (
            patch("services.market.routes.market.cache", mock_cache),
            patch("services.market.data.market_repo.fetch_all") as mock_fetch,
        ):
            mock_fetch.side_effect = Exception("Database error")
            response = client.get("/api/market/macro")
            assert response.status_code == 500
