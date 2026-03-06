import sys
import os

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import pytest
import pandas as pd
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock
from services.market.data.market_crawler import MarketCrawler
from services.market.data.market_repo import MarketRepo


class TestMarketRepo:
    """测试市场仓储"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    def test_save_macro_with_empty_df(self, repo):
        """测试保存空宏观数据"""
        result = repo.save_macro(pd.DataFrame())
        assert result == 0

        result = repo.save_macro(None)
        assert result == 0

    def test_save_macro_success(self, repo):
        """测试成功保存宏观数据"""
        df = pd.DataFrame(
            [
                {
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": datetime(2024, 4, 16),
                    "trade_date": None,
                },
                {
                    "indicator": "CPI",
                    "period": "2024-03",
                    "value": 0.1,
                    "unit": "%",
                    "source": "国家统计局",
                    "publish_date": datetime(2024, 4, 11),
                    "trade_date": datetime(2024, 4, 11),
                },
            ]
        )

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.return_value = 2
            result = repo.save_macro(df)
            assert result == 2
            mock_conn.assert_called_once()

    def test_get_macro_all(self, repo):
        """测试获取所有宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "id": 1,
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": date(2024, 4, 16),
                    "trade_date": None,
                }
            ]
            result = repo.get_macro(days=30)
            assert isinstance(result, list)
            assert len(result) == 1
            mock_fetch.assert_called_once()

    def test_get_macro_by_indicator(self, repo):
        """测试按指标获取宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []
            result = repo.get_macro(indicator="GDP", days=30)
            assert isinstance(result, list)
            mock_fetch.assert_called_once()

    def test_get_latest_macro(self, repo):
        """测试获取最新宏观数据"""
        with patch("services.market.data.market_repo.fetch_one") as mock_fetch:
            mock_fetch.return_value = {
                "id": 1,
                "indicator": "GDP",
                "period": "2024Q1",
                "value": 29.6,
                "unit": "万亿元",
                "source": "国家统计局",
                "publish_date": date(2024, 4, 16),
            }
            result = repo.get_latest_macro(indicator="GDP")
            assert result is not None
            assert result["indicator"] == "GDP"
            assert result["value"] == 29.6

    def test_get_latest_macro_not_found(self, repo):
        """测试获取最新宏观数据-未找到"""
        with patch("services.market.data.market_repo.fetch_one") as mock_fetch:
            mock_fetch.return_value = None
            result = repo.get_latest_macro(indicator="UNKNOWN")
            assert result is None

    def test_save_money_flow_with_empty_df(self, repo):
        """测试保存空资金流向数据"""
        result = repo.save_money_flow(pd.DataFrame())
        assert result == 0

        result = repo.save_money_flow(None)
        assert result == 0

    def test_save_money_flow_success(self, repo):
        """测试成功保存资金流向数据"""
        df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "north_money": -50.5,
                    "north_buy": 120.3,
                    "north_sell": 170.8,
                    "main_money": 30.2,
                    "margin_balance": 15000.5,
                }
            ]
        )

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.return_value = 1
            result = repo.save_money_flow(df)
            assert result == 1

    def test_get_money_flow(self, repo):
        """测试获取资金流向数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": date(2024, 4, 15),
                    "north_money": -50.5,
                    "north_buy": 120.3,
                    "north_sell": 170.8,
                    "main_money": 30.2,
                    "margin_balance": 15000.5,
                }
            ]
            result = repo.get_money_flow(days=30)
            assert isinstance(result, list)
            assert len(result) == 1

    def test_get_latest_money_flow(self, repo):
        """测试获取最新资金流向"""
        with patch("services.market.data.market_repo.fetch_one") as mock_fetch:
            mock_fetch.return_value = {
                "trade_date": date(2024, 4, 15),
                "north_money": -50.5,
                "north_buy": 120.3,
                "north_sell": 170.8,
                "main_money": 30.2,
                "margin_balance": 15000.5,
            }
            result = repo.get_latest_money_flow()
            assert result is not None
            assert result["north_money"] == -50.5

    def test_save_sentiment_with_empty_df(self, repo):
        """测试保存空市场情绪数据"""
        result = repo.save_sentiment(pd.DataFrame())
        assert result == 0

        result = repo.save_sentiment(None)
        assert result == 0

    def test_save_sentiment_success(self, repo):
        """测试成功保存市场情绪数据"""
        df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "volume": 850000000000,
                    "up_count": 1200,
                    "down_count": 800,
                    "turnover_rate": 1.2,
                    "advance_count": 3000,
                    "decline_count": 1500,
                }
            ]
        )

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.return_value = 1
            result = repo.save_sentiment(df)
            assert result == 1

    def test_get_sentiment(self, repo):
        """测试获取市场情绪数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": date(2024, 4, 15),
                    "volume": 850000000000,
                    "up_count": 1200,
                    "down_count": 800,
                    "turnover_rate": 1.2,
                    "advance_count": 3000,
                    "decline_count": 1500,
                }
            ]
            result = repo.get_sentiment(days=30)
            assert isinstance(result, list)
            assert len(result) == 1

    def test_get_latest_sentiment(self, repo):
        """测试获取最新市场情绪"""
        with patch("services.market.data.market_repo.fetch_one") as mock_fetch:
            mock_fetch.return_value = {
                "trade_date": date(2024, 4, 15),
                "volume": 850000000000,
                "up_count": 1200,
                "down_count": 800,
                "turnover_rate": 1.2,
                "advance_count": 3000,
                "decline_count": 1500,
            }
            result = repo.get_latest_sentiment()
            assert result is not None
            assert result["volume"] == 850000000000
            assert result["advance_count"] == 3000

    def test_save_global_macro_with_empty_df(self, repo):
        """测试保存空全球宏观数据"""
        result = repo.save_global_macro(pd.DataFrame())
        assert result == 0

        result = repo.save_global_macro(None)
        assert result == 0

    def test_save_global_macro_success(self, repo):
        """测试成功保存全球宏观数据"""
        df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                },
                {
                    "trade_date": "2024-04-15",
                    "symbol": "USDX",
                    "close_price": 106.5,
                    "change_pct": 0.2,
                },
            ]
        )

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.return_value = 2
            result = repo.save_global_macro(df)
            assert result == 2

    def test_get_global_macro_by_symbol(self, repo):
        """测试按符号获取全球宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": date(2024, 4, 15),
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                }
            ]
            result = repo.get_global_macro(symbol="USD/CNY")
            assert isinstance(result, list)
            assert len(result) == 1

    def test_get_global_macro_all(self, repo):
        """测试获取所有全球宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []
            result = repo.get_global_macro(days=30)
            assert isinstance(result, list)

    def test_get_latest_global(self, repo):
        """测试获取最新全球宏观数据"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = [
                {
                    "trade_date": date(2024, 4, 15),
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                },
                {
                    "trade_date": date(2024, 4, 15),
                    "symbol": "USDX",
                    "close_price": 106.5,
                    "change_pct": 0.2,
                },
            ]
            result = repo.get_latest_global()
            assert result is not None
            assert len(result) == 2
            assert result[0]["symbol"] == "USD/CNY"

    def test_get_latest_global_not_found(self, repo):
        """测试获取最新全球宏观数据-未找到"""
        with patch("services.market.data.market_repo.fetch_all") as mock_fetch:
            mock_fetch.return_value = []
            result = repo.get_latest_global()
            assert result is None

    def test_get_market_features(self, repo):
        """测试获取市场特征"""
        with (
            patch.object(repo, "get_macro") as mock_macro,
            patch.object(repo, "get_money_flow") as mock_money,
            patch.object(repo, "get_sentiment") as mock_sentiment,
            patch.object(repo, "get_global_macro") as mock_global,
        ):
            mock_macro.return_value = [{"indicator": "GDP", "value": 29.6}]
            mock_money.return_value = [{"north_money": -50.5}]
            mock_sentiment.return_value = [{"volume": 850000000000}]
            mock_global.return_value = [{"symbol": "USD/CNY"}]

            result = repo.get_market_features(days=30)
            assert "macro" in result
            assert "money_flow" in result
            assert "sentiment" in result
            assert "global" in result
            assert len(result["macro"]) == 1
            assert len(result["money_flow"]) == 1

    def test_get_sentiment_summary(self, repo):
        """测试获取情绪摘要"""
        with patch.object(repo, "get_latest_sentiment") as mock_latest:
            mock_latest.return_value = {
                "trade_date": "2024-04-15",
                "volume": 850000000000,
                "up_count": 1200,
                "down_count": 800,
                "turnover_rate": 1.2,
                "advance_count": 3000,
                "decline_count": 1500,
            }
            result = repo.get_sentiment_summary()
            assert result["volume"] == 850000000000
            assert result["advance_count"] == 3000

    def test_get_sentiment_summary_empty(self, repo):
        """测试获取情绪摘要-无数据"""
        with patch.object(repo, "get_latest_sentiment") as mock_latest:
            mock_latest.return_value = None
            result = repo.get_sentiment_summary()
            assert result["volume"] is None
            assert result["up_count"] == 0
            assert result["down_count"] == 0


class TestMarketRepoErrors:
    """测试市场仓储错误处理"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    def test_save_macro_error_handling(self, repo):
        """测试宏观数据保存错误处理"""
        df = pd.DataFrame([{"indicator": "GDP", "period": "2024Q1", "value": 29.6}])

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.side_effect = Exception("Database error")
            result = repo.save_macro(df)
            assert result == 0

    def test_save_money_flow_error_handling(self, repo):
        """测试资金流向保存错误处理"""
        df = pd.DataFrame([{"trade_date": pd.Timestamp("2024-04-15")}])

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.side_effect = Exception("Database error")
            result = repo.save_money_flow(df)
            assert result == 0

    def test_save_sentiment_error_handling(self, repo):
        """测试市场情绪保存错误处理"""
        df = pd.DataFrame([{"trade_date": pd.Timestamp("2024-04-15")}])

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.side_effect = Exception("Database error")
            result = repo.save_sentiment(df)
            assert result == 0

    def test_save_global_macro_error_handling(self, repo):
        """测试全球宏观保存错误处理"""
        df = pd.DataFrame(
            [{"trade_date": pd.Timestamp("2024-04-15"), "symbol": "USD/CNY"}]
        )

        with patch("services.market.data.market_repo.run_connection") as mock_conn:
            mock_conn.side_effect = Exception("Database error")
            result = repo.save_global_macro(df)
            assert result == 0
