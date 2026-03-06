#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据抓取模块单元测试
"""

from unittest.mock import MagicMock, patch, Mock
import pytest
import pandas as pd
import numpy as np

from services.stock.data import get_stock_data, get_stock_list


class TestStockData:
    """股票数据测试"""

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_success(self, mock_get_df):
        """测试成功获取股票数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100)
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000000, 5000000, 100)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        assert result is not None
        assert len(result) == 60
        assert "收盘" in result.columns
        mock_get_df.assert_called_once()

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_not_found(self, mock_get_df):
        """测试股票数据未找到"""
        mock_get_df.return_value = None

        result = get_stock_data("999999", days=60)

        assert result is None
        mock_get_df.assert_called_once()

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_truncate(self, mock_get_df):
        """测试数据截断"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100)
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000000, 5000000, 100)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=30)

        assert result is not None
        assert len(result) == 30

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_sorting(self, mock_get_df):
        """测试数据按日期排序"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100)
        close = np.cumsum(np.random.randn(100)) + 100
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000000, 5000000, 100)

        # 创建乱序的数据
        shuffled_indices = np.random.permutation(100)
        df = pd.DataFrame(
            {
                "日期": dates[shuffled_indices],
                "收盘": close[shuffled_indices],
                "最高": high[shuffled_indices],
                "最低": low[shuffled_indices],
                "开盘": open_price[shuffled_indices],
                "成交量": volume[shuffled_indices],
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=100)

        assert result is not None
        # 检查是否按日期排序
        assert all(
            result["日期"].iloc[i] <= result["日期"].iloc[i + 1]
            for i in range(len(result) - 1)
        )

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_empty(self, mock_get_df):
        """测试空数据"""
        # 返回带有必要列的空 DataFrame
        mock_get_df.return_value = pd.DataFrame(
            {"日期": [], "收盘": [], "最高": [], "最低": [], "开盘": [], "成交量": []}
        )

        result = get_stock_data("000001", days=60)

        assert result is not None
        assert len(result) == 0

    @patch("data.stock_repo.get_stock_daily_df")
    def test_get_stock_data_missing_columns(self, mock_get_df):
        """测试缺少必要列的数据"""
        df = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=50),
                "收盘": np.arange(50) + 100,
                "成交量": np.arange(50) + 1000000,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        assert result is not None
        assert len(result) == 50


class TestStockList:
    """股票列表测试"""

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_get_stock_list_default(self, mock_fetch_one, mock_fetch_all):
        """测试获取股票列表 - 默认参数"""
        mock_fetch_all.return_value = [
            {"symbol": "000001", "name": "平安银行", "market": "sz"},
            {"symbol": "000002", "name": "万科A", "market": "sz"},
        ]
        mock_fetch_one.return_value = {"total": 2}

        result = get_stock_list()

        assert result["items"] is not None
        assert result["total"] == 2
        assert result["page"] == 1
        assert result["size"] == 20
        assert result["pages"] == 1
        mock_fetch_all.assert_called_once()
        mock_fetch_one.assert_called_once()

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_get_stock_list_pagination(self, mock_fetch_one, mock_fetch_all):
        """测试股票列表分页"""
        mock_fetch_all.return_value = [
            {"symbol": "000021", "name": "深科技A", "market": "sz"},
        ]
        mock_fetch_one.return_value = {"total": 100}

        result = get_stock_list(page=2, size=10)

        assert result["items"] is not None
        assert result["total"] == 100
        assert result["page"] == 2
        assert result["size"] == 10
        assert result["pages"] == 10
        mock_fetch_all.assert_called_once()
        # 检查 offset 是否正确
        assert "OFFSET" in mock_fetch_all.call_args[0][0]
        assert "LIMIT" in mock_fetch_all.call_args[0][0]

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_get_stock_list_empty(self, mock_fetch_one, mock_fetch_all):
        """测试空股票列表"""
        mock_fetch_all.return_value = []
        mock_fetch_one.return_value = {"total": 0}

        result = get_stock_list()

        assert result["items"] == []
        assert result["total"] == 0
        assert result["pages"] == 0

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_get_stock_list_large_size(self, mock_fetch_one, mock_fetch_all):
        """测试大分页大小"""
        mock_fetch_all.return_value = [
            {"symbol": f"{i:06d}", "name": f"股票{i}"} for i in range(100)
        ]
        mock_fetch_one.return_value = {"total": 500}

        result = get_stock_list(page=1, size=100)

        assert len(result["items"]) == 100
        assert result["pages"] == 5

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_get_stock_list_boundary_page(self, mock_fetch_one, mock_fetch_all):
        """测试边界页码"""
        mock_fetch_all.return_value = [{"symbol": "000500", "name": "股票500"}]
        mock_fetch_one.return_value = {"total": 500}

        result = get_stock_list(page=5, size=100)

        assert result["items"] is not None
        assert result["page"] == 5
        assert result["pages"] == 5


class TestDataValidation:
    """数据验证测试"""

    @patch("data.stock_repo.get_stock_daily_df")
    def test_validate_required_columns(self, mock_get_df):
        """测试验证必要列"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60)
        close = np.cumsum(np.random.randn(60)) + 100
        high = close + np.random.rand(60) * 2
        low = close - np.random.rand(60) * 2
        open_price = close + np.random.randn(60) * 0.5
        volume = np.random.randint(1000000, 5000000, 60)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        # 验证所有必要列都存在
        required_cols = ["日期", "收盘", "最高", "最低", "开盘", "成交量"]
        for col in required_cols:
            assert col in result.columns

    @patch("data.stock_repo.get_stock_daily_df")
    def test_validate_no_null_values(self, mock_get_df):
        """测试验证无空值"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60)
        close = np.cumsum(np.random.randn(60)) + 100
        high = close + np.random.rand(60) * 2
        low = close - np.random.rand(60) * 2
        open_price = close + np.random.randn(60) * 0.5
        volume = np.random.randint(1000000, 5000000, 60)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        # 检查关键列是否有空值
        for col in ["收盘", "最高", "最低", "开盘", "成交量"]:
            assert not result[col].isna().any()


class TestExceptionHandling:
    """异常处理测试"""

    @patch("data.stock_repo.get_stock_daily_df")
    def test_database_connection_error(self, mock_get_df):
        """测试数据库连接错误"""
        mock_get_df.side_effect = Exception("数据库连接失败")

        with pytest.raises(Exception) as exc_info:
            get_stock_data("000001", days=60)

        assert "数据库连接失败" in str(exc_info.value)

    @patch("services.stock.data.fetch_all")
    @patch("services.stock.data.fetch_one")
    def test_query_error(self, mock_fetch_one, mock_fetch_all):
        """测试查询错误"""
        mock_fetch_all.side_effect = Exception("查询超时")

        with pytest.raises(Exception) as exc_info:
            get_stock_list()

        assert "查询超时" in str(exc_info.value)

    @patch("data.stock_repo.get_stock_daily_df")
    def test_invalid_symbol(self, mock_get_df):
        """测试无效股票代码"""
        mock_get_df.return_value = None

        result = get_stock_data("INVALID@#", days=60)

        assert result is None


class TestDataIntegrity:
    """数据完整性测试"""

    @patch("data.stock_repo.get_stock_daily_df")
    def test_price_relationships(self, mock_get_df):
        """测试价格关系"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60)
        base_price = 100
        close = base_price + np.cumsum(np.random.randn(60))

        # 确保价格关系正确
        high = close + np.abs(np.random.randn(60)) * 2
        low = close - np.abs(np.random.randn(60)) * 2
        open_price = np.minimum(close, high)
        open_price = np.maximum(open_price, low)

        # 确保没有负数
        high = np.maximum(high, 0.01)
        low = np.maximum(low, 0.01)
        open_price = np.maximum(open_price, 0.01)
        close = np.maximum(close, 0.01)

        # 最后确保关系
        high = np.maximum.reduce([high, open_price, close])
        low = np.minimum.reduce([low, open_price, close])

        volume = np.random.randint(1000000, 5000000, 60)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        # 验证价格关系: 最高价 >= 开盘价,收盘价 >= 最低价
        for idx, row in result.iterrows():
            assert row["最高"] >= row["开盘"]
            assert row["最高"] >= row["收盘"]
            assert row["最低"] <= row["开盘"]
            assert row["最低"] <= row["收盘"]

    @patch("data.stock_repo.get_stock_daily_df")
    def test_positive_volume(self, mock_get_df):
        """测试成交量为正"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60)
        close = np.cumsum(np.random.randn(60)) + 100
        high = close + np.random.rand(60) * 2
        low = close - np.random.rand(60) * 2
        open_price = close + np.random.randn(60) * 0.5
        volume = np.random.randint(1000000, 5000000, 60)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        # 验证成交量为正
        assert (result["成交量"] > 0).all()

    @patch("data.stock_repo.get_stock_daily_df")
    def test_positive_prices(self, mock_get_df):
        """测试价格为正"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=60)
        close = np.cumsum(np.random.randn(60)) + 100
        high = close + np.random.rand(60) * 2
        low = close - np.random.rand(60) * 2
        open_price = close + np.random.randn(60) * 0.5
        volume = np.random.randint(1000000, 5000000, 60)

        df = pd.DataFrame(
            {
                "日期": dates,
                "收盘": close,
                "最高": high,
                "最低": low,
                "开盘": open_price,
                "成交量": volume,
            }
        )
        mock_get_df.return_value = df

        result = get_stock_data("000001", days=60)

        # 验证所有价格为正
        for col in ["开盘", "收盘", "最高", "最低"]:
            assert (result[col] > 0).all()
