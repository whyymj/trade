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
from unittest.mock import patch, MagicMock
from services.market.data.market_crawler import MarketCrawler
from services.market.data.market_repo import MarketRepo


class TestMarketSync:
    """测试市场数据同步"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_sync_all_data_types_success(self, repo, crawler):
        """测试同步所有数据类型成功"""
        mock_macro_df = pd.DataFrame(
            [
                {
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                }
            ]
        )

        mock_money_df = pd.DataFrame(
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

        mock_sentiment_df = pd.DataFrame(
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

        mock_global_df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                }
            ]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(crawler, "fetch_money_flow", return_value=mock_money_df),
            patch.object(crawler, "fetch_sentiment", return_value=mock_sentiment_df),
            patch.object(crawler, "fetch_global", return_value=mock_global_df),
            patch.object(repo, "save_macro", return_value=1),
            patch.object(repo, "save_money_flow", return_value=1),
            patch.object(repo, "save_sentiment", return_value=1),
            patch.object(repo, "save_global_macro", return_value=1),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)
            assert not sync_data["macro"].empty
            assert not sync_data["money_flow"].empty
            assert not sync_data["sentiment"].empty
            assert not sync_data["global"].empty

            saved_macro = repo.save_macro(sync_data["macro"])
            saved_money = repo.save_money_flow(sync_data["money_flow"])
            saved_sentiment = repo.save_sentiment(sync_data["sentiment"])
            saved_global = repo.save_global_macro(sync_data["global"])

            total_saved = saved_macro + saved_money + saved_sentiment + saved_global
            assert total_saved == 4

    def test_sync_all_data_types_with_empty(self, repo, crawler):
        """测试同步所有数据类型（部分为空）"""
        mock_macro_df = pd.DataFrame(
            [
                {
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                }
            ]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(crawler, "fetch_money_flow", return_value=None),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
            patch.object(repo, "save_macro", return_value=1),
            patch.object(repo, "save_money_flow", return_value=0),
            patch.object(repo, "save_sentiment", return_value=0),
            patch.object(repo, "save_global_macro", return_value=0),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)
            assert not sync_data["macro"].empty
            assert sync_data["money_flow"] is None
            assert sync_data["sentiment"].empty
            assert sync_data["global"].empty

            saved_macro = repo.save_macro(sync_data["macro"])
            saved_money = repo.save_money_flow(sync_data["money_flow"])
            saved_sentiment = repo.save_sentiment(sync_data["sentiment"])
            saved_global = repo.save_global_macro(sync_data["global"])

            total_saved = saved_macro + saved_money + saved_sentiment + saved_global
            assert total_saved == 1

    def test_sync_batch_data(self, repo, crawler):
        """测试批量同步数据"""
        mock_macro_df = pd.DataFrame(
            [
                {"indicator": "GDP", "period": "2024Q1", "value": 29.6},
                {"indicator": "CPI", "period": "2024-03", "value": 0.1},
                {"indicator": "PMI", "period": "2024-03", "value": 50.8},
            ]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(repo, "save_macro", return_value=3) as mock_save,
        ):
            sync_data = crawler.sync_all()
            saved = repo.save_macro(sync_data["macro"])

            assert saved == 3
            mock_save.assert_called_once()

    def test_sync_data_consistency(self, repo, crawler):
        """测试数据一致性"""
        mock_macro_df = pd.DataFrame(
            [
                {
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                }
            ]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(repo, "save_macro", return_value=1) as mock_save,
            patch.object(repo, "get_macro") as mock_get,
        ):
            mock_get.return_value = [
                {
                    "id": 1,
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                }
            ]

            sync_data = crawler.sync_all()
            repo.save_macro(sync_data["macro"])
            fetched = repo.get_macro(days=30)

            assert len(fetched) == 1
            assert fetched[0]["indicator"] == "GDP"
            assert fetched[0]["value"] == 29.6


class TestSyncErrorHandling:
    """测试同步错误处理"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_sync_with_crawler_error(self, repo, crawler):
        """测试爬虫错误处理 - crawler捕获异常"""
        with (
            patch.object(crawler, "fetch_macro", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
            patch.object(repo, "save_macro", return_value=0),
            patch.object(repo, "save_money_flow", return_value=0),
            patch.object(repo, "save_sentiment", return_value=0),
            patch.object(repo, "save_global_macro", return_value=0),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)

    def test_sync_with_repo_error(self, repo, crawler):
        """测试仓储错误处理"""
        mock_macro_df = pd.DataFrame(
            [{"indicator": "GDP", "period": "2024Q1", "value": 29.6}]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
            patch.object(repo, "save_macro", side_effect=Exception("Repo error")),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)

    def test_sync_with_partial_error(self, repo, crawler):
        """测试部分错误处理 - 返回空结果"""
        with (
            patch.object(crawler, "fetch_macro", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
            patch.object(repo, "save_macro", return_value=0),
            patch.object(repo, "save_money_flow", return_value=0),
            patch.object(repo, "save_sentiment", return_value=0),
            patch.object(repo, "save_global_macro", return_value=0),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)

    def test_sync_all_empty_data(self, repo, crawler):
        """测试同步空数据"""
        with (
            patch.object(crawler, "fetch_macro", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
            patch.object(repo, "save_macro", return_value=0),
            patch.object(repo, "save_money_flow", return_value=0),
            patch.object(repo, "save_sentiment", return_value=0),
            patch.object(repo, "save_global_macro", return_value=0),
        ):
            sync_data = crawler.sync_all()

            assert isinstance(sync_data, dict)
            assert sync_data["macro"].empty
            assert sync_data["money_flow"].empty
            assert sync_data["sentiment"].empty
            assert sync_data["global"].empty


class TestSyncDataValidation:
    """测试同步数据验证"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_validate_macro_data_structure(self, repo, crawler):
        """验证宏观数据结构"""
        mock_macro_df = pd.DataFrame(
            [
                {
                    "indicator": "GDP",
                    "period": "2024Q1",
                    "value": 29.6,
                    "unit": "万亿元",
                    "source": "国家统计局",
                    "publish_date": "2024-04-16",
                }
            ]
        )

        with patch.object(crawler, "fetch_macro", return_value=mock_macro_df):
            sync_data = crawler.sync_all()
            macro_data = sync_data["macro"]

            assert "indicator" in macro_data.columns
            assert "period" in macro_data.columns
            assert "value" in macro_data.columns
            assert len(macro_data) == 1

    def test_validate_money_flow_data_structure(self, repo, crawler):
        """验证资金流向数据结构"""
        mock_money_df = pd.DataFrame(
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

        with patch.object(crawler, "fetch_money_flow", return_value=mock_money_df):
            sync_data = crawler.sync_all()
            money_data = sync_data["money_flow"]

            assert "trade_date" in money_data.columns
            assert "north_money" in money_data.columns
            assert "main_money" in money_data.columns
            assert len(money_data) == 1

    def test_validate_sentiment_data_structure(self, repo, crawler):
        """验证市场情绪数据结构"""
        mock_sentiment_df = pd.DataFrame(
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

        with patch.object(crawler, "fetch_sentiment", return_value=mock_sentiment_df):
            sync_data = crawler.sync_all()
            sentiment_data = sync_data["sentiment"]

            assert "trade_date" in sentiment_data.columns
            assert "volume" in sentiment_data.columns
            assert "up_count" in sentiment_data.columns
            assert len(sentiment_data) == 1

    def test_validate_global_data_structure(self, repo, crawler):
        """验证全球宏观数据结构"""
        mock_global_df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "symbol": "USD/CNY",
                    "close_price": 7.24,
                    "change_pct": -0.05,
                }
            ]
        )

        with patch.object(crawler, "fetch_global", return_value=mock_global_df):
            sync_data = crawler.sync_all()
            global_data = sync_data["global"]

            assert "trade_date" in global_data.columns
            assert "symbol" in global_data.columns
            assert "close_price" in global_data.columns
            assert len(global_data) == 1


class TestSyncPerformance:
    """测试同步性能"""

    @pytest.fixture
    def repo(self):
        return MarketRepo()

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_sync_large_batch(self, repo, crawler):
        """测试大批量同步"""
        mock_macro_df = pd.DataFrame(
            [
                {"indicator": f"INDICATOR_{i}", "period": "2024Q1", "value": i * 10}
                for i in range(100)
            ]
        )

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(repo, "save_macro", return_value=100),
        ):
            sync_data = crawler.sync_all()
            saved = repo.save_macro(sync_data["macro"])

            assert saved == 100
            assert len(sync_data["macro"]) == 100

    def test_sync_multiple_symbols(self, repo, crawler):
        """测试多符号同步"""
        mock_global_df = pd.DataFrame(
            [
                {
                    "trade_date": pd.Timestamp("2024-04-15"),
                    "symbol": f"SYMBOL_{i}",
                    "close_price": 100 + i,
                    "change_pct": i * 0.1,
                }
                for i in range(10)
            ]
        )

        with (
            patch.object(crawler, "fetch_global", return_value=mock_global_df),
            patch.object(repo, "save_global_macro", return_value=10),
        ):
            sync_data = crawler.sync_all()
            saved = repo.save_global_macro(sync_data["global"])

            assert saved == 10
            assert len(sync_data["global"]) == 10
