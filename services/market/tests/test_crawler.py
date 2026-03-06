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
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from services.market.data.market_crawler import MarketCrawler


class TestMarketCrawler:
    """测试市场数据爬虫"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_crawler_init(self, crawler):
        """测试爬虫初始化"""
        assert crawler is not None
        assert hasattr(crawler, "fetch_money_flow")
        assert hasattr(crawler, "fetch_sentiment")
        assert hasattr(crawler, "fetch_macro")
        assert hasattr(crawler, "fetch_global")
        assert hasattr(crawler, "sync_all")


class TestMoneyFlowCrawler:
    """测试资金流向爬虫"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_fetch_money_flow_akshare_not_available(self, crawler):
        """测试 akshare 不可用时抓取资金流向"""
        with patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", False):
            result = crawler.fetch_money_flow()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_money_flow_success(self, crawler):
        """测试成功抓取资金流向"""
        mock_df = pd.DataFrame(
            [
                {
                    "日期": "2024-04-15",
                    "主力净流入-净额": 30.2,
                    "北向净流入-净额": -50.5,
                    "融资融券-融资买入额": 120.3,
                }
            ]
        )

        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_individual_fund_flow.return_value = mock_df
            result = crawler.fetch_money_flow()

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert "trade_date" in result.columns
            assert "main_money" in result.columns
            assert "north_money" in result.columns

    def test_fetch_money_flow_empty_result(self, crawler):
        """测试抓取资金流向返回空结果"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_individual_fund_flow.return_value = None
            result = crawler.fetch_money_flow()

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_money_flow_exception(self, crawler):
        """测试抓取资金流向异常"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_individual_fund_flow.side_effect = Exception("Network error")
            result = crawler.fetch_money_flow()

            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestSentimentCrawler:
    """测试市场情绪爬虫"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_fetch_sentiment_akshare_not_available(self, crawler):
        """测试 akshare 不可用时抓取市场情绪"""
        with patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", False):
            result = crawler.fetch_sentiment()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_sentiment_success(self, crawler):
        """测试成功抓取市场情绪"""
        mock_df = pd.DataFrame(
            [
                {
                    "日期": "2024-04-15",
                    "成交量": 850000000000,
                    "成交额": 9500000000000,
                    "涨跌幅": 1.5,
                    "振幅": 2.3,
                    "换手率": 1.2,
                }
            ]
        )

        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_zh_a_hist.return_value = mock_df
            result = crawler.fetch_sentiment()

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert "trade_date" in result.columns
            assert "volume" in result.columns
            assert "turnover_rate" in result.columns
            assert "up_count" in result.columns
            assert "down_count" in result.columns

    def test_fetch_sentiment_empty_result(self, crawler):
        """测试抓取市场情绪返回空结果"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_zh_a_hist.return_value = None
            result = crawler.fetch_sentiment()

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_sentiment_exception(self, crawler):
        """测试抓取市场情绪异常"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.stock_zh_a_hist.side_effect = Exception("Network error")
            result = crawler.fetch_sentiment()

            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestMacroCrawler:
    """测试宏观经济爬虫"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_fetch_macro_akshare_not_available(self, crawler):
        """测试 akshare 不可用时抓取宏观经济"""
        with patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", False):
            result = crawler.fetch_macro()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_macro_success(self, crawler):
        """测试成功抓取宏观经济"""
        mock_df = pd.DataFrame(
            [
                {"月份": "2024-03", "广义货币(M2)": 304.8},
                {"月份": "2024-02", "广义货币(M2)": 299.5},
            ]
        )

        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.macro_china_人民资产负债表.return_value = mock_df
            result = crawler.fetch_macro()

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert "period" in result.columns
            assert "m2" in result.columns

    def test_fetch_macro_empty_result(self, crawler):
        """测试抓取宏观经济返回空结果"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.macro_china_人民资产负债表.return_value = None
            result = crawler.fetch_macro()

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_macro_exception(self, crawler):
        """测试抓取宏观经济异常"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.macro_china_人民资产负债表.side_effect = Exception("Network error")
            result = crawler.fetch_macro()

            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestGlobalMacroCrawler:
    """测试全球宏观爬虫"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_fetch_global_akshare_not_available(self, crawler):
        """测试 akshare 不可用时抓取全球宏观"""
        with patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", False):
            result = crawler.fetch_global()
            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_global_success(self, crawler):
        """测试成功抓取全球宏观"""
        mock_df = pd.DataFrame(
            [
                {"名称": "美元/人民币", "最新价": 7.24, "涨跌幅": -0.05},
                {"名称": "美元指数", "最新价": 106.5, "涨跌幅": 0.2},
            ]
        )

        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.currency_latest.return_value = mock_df
            result = crawler.fetch_global()

            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            assert "trade_date" in result.columns

    def test_fetch_global_empty_result(self, crawler):
        """测试抓取全球宏观返回空结果"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.currency_latest.return_value = None
            result = crawler.fetch_global()

            assert isinstance(result, pd.DataFrame)
            assert result.empty

    def test_fetch_global_exception(self, crawler):
        """测试抓取全球宏观异常"""
        with (
            patch("services.market.data.market_crawler.AKSHARE_AVAILABLE", True),
            patch("services.market.data.market_crawler.ak") as mock_ak,
        ):
            mock_ak.currency_latest.side_effect = Exception("Network error")
            result = crawler.fetch_global()

            assert isinstance(result, pd.DataFrame)
            assert result.empty


class TestSyncAll:
    """测试同步全部数据"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_sync_all_success(self, crawler):
        """测试成功同步全部数据"""
        mock_macro_df = pd.DataFrame([{"月份": "2024-03", "广义货币(M2)": 304.8}])
        mock_money_df = pd.DataFrame([{"日期": "2024-04-15", "主力净流入-净额": 30.2}])
        mock_sentiment_df = pd.DataFrame(
            [{"日期": "2024-04-15", "成交量": 850000000000}]
        )
        mock_global_df = pd.DataFrame([{"名称": "美元/人民币", "最新价": 7.24}])

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(crawler, "fetch_money_flow", return_value=mock_money_df),
            patch.object(crawler, "fetch_sentiment", return_value=mock_sentiment_df),
            patch.object(crawler, "fetch_global", return_value=mock_global_df),
        ):
            result = crawler.sync_all()

            assert isinstance(result, dict)
            assert "macro" in result
            assert "money_flow" in result
            assert "sentiment" in result
            assert "global" in result
            assert not result["macro"].empty
            assert not result["money_flow"].empty
            assert not result["sentiment"].empty
            assert not result["global"].empty

    def test_sync_all_partial_empty(self, crawler):
        """测试同步部分数据为空"""
        mock_macro_df = pd.DataFrame([{"月份": "2024-03", "广义货币(M2)": 304.8}])

        with (
            patch.object(crawler, "fetch_macro", return_value=mock_macro_df),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=None),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
        ):
            result = crawler.sync_all()

            assert isinstance(result, dict)
            assert not result["macro"].empty
            assert result["money_flow"].empty
            assert result["sentiment"] is None
            assert result["global"].empty

    def test_sync_all_all_empty(self, crawler):
        """测试同步所有数据为空"""
        with (
            patch.object(crawler, "fetch_macro", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
        ):
            result = crawler.sync_all()

            assert isinstance(result, dict)
            assert result["macro"].empty
            assert result["money_flow"].empty
            assert result["sentiment"].empty
            assert result["global"].empty


class TestCrawlerErrorHandling:
    """测试爬虫错误处理"""

    @pytest.fixture
    def crawler(self):
        return MarketCrawler()

    def test_sync_all_with_partial_failure(self, crawler):
        """测试同步时部分失败 - 返回空结果"""
        with (
            patch.object(crawler, "fetch_macro", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_money_flow", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_sentiment", return_value=pd.DataFrame()),
            patch.object(crawler, "fetch_global", return_value=pd.DataFrame()),
        ):
            result = crawler.sync_all()

            assert isinstance(result, dict)
            assert "macro" in result
            assert "money_flow" in result
            assert "sentiment" in result
            assert "global" in result
