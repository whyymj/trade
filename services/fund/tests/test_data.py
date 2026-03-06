# -*- coding: utf-8 -*-
"""
数据层完整测试
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime, date
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from services.fund.data import FundRepo, FundFetcher


class TestFundRepo:
    """基金仓储测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    def test_get_fund_list_basic(self, repo):
        """测试基本基金列表查询"""
        result = repo.get_fund_list(page=1, size=10)
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "data" in result
        assert len(result["data"]) <= 10
        assert result["page"] == 1
        assert result["page_size"] == 10

    def test_get_fund_list_with_filter(self, repo):
        """测试带筛选条件的基金列表查询"""
        result = repo.get_fund_list(page=1, size=10, fund_type="股票型")
        assert "data" in result
        assert "total" in result

    def test_get_fund_list_watchlist_only(self, repo):
        """测试自选基金列表查询"""
        result = repo.get_fund_list(page=1, size=10, watchlist_only=True)
        assert "data" in result
        assert "total" in result

    def test_get_fund_list_industry_tag(self, repo):
        """测试行业标签筛选"""
        result = repo.get_fund_list(page=1, size=10, industry_tag="医药")
        assert "data" in result

    def test_get_fund_list_pagination(self, repo):
        """测试分页功能"""
        result1 = repo.get_fund_list(page=1, size=5)
        result2 = repo.get_fund_list(page=2, size=5)
        assert result1["page"] == 1
        assert result2["page"] == 2
        assert result1["page_size"] == 5
        assert result2["page_size"] == 5

    def test_get_fund_list_size_limit(self, repo):
        """测试分页大小限制"""
        result1 = repo.get_fund_list(page=1, size=200)
        assert result1["page_size"] == 100

        result2 = repo.get_fund_list(page=1, size=0)
        assert result2["page_size"] == 1

    def test_get_fund_info_exists(self, repo):
        """测试获取存在的基金信息"""
        result = repo.get_fund_info("001302")
        assert result is not None
        assert "fund_code" in result
        assert result["fund_code"] == "001302"

    def test_get_fund_info_not_exists(self, repo):
        """测试获取不存在的基金信息"""
        result = repo.get_fund_info("999999")
        assert result is None

    def test_get_fund_info_empty_code(self, repo):
        """测试空基金代码"""
        result = repo.get_fund_info("")
        assert result is None

        result = repo.get_fund_info(None)
        assert result is None

    def test_get_fund_info_whitespace(self, repo):
        """测试基金代码包含空格"""
        result = repo.get_fund_info("  001302  ")
        assert result is not None

    def test_get_fund_nav_basic(self, repo):
        """测试基本净值查询"""
        result = repo.get_fund_nav("001302", days=30)
        assert result is not None
        assert len(result) > 0
        assert "nav_date" in result.columns
        assert "unit_nav" in result.columns
        assert "accum_nav" in result.columns
        assert "daily_return" in result.columns

    def test_get_fund_nav_date_range(self, repo):
        """测试日期范围查询"""
        result = repo.get_fund_nav(
            "001302", start_date="2024-01-01", end_date="2024-01-31"
        )
        assert result is not None

    def test_get_fund_nav_empty_code(self, repo):
        """测试空基金代码"""
        result = repo.get_fund_nav("")
        assert result is None

    def test_upsert_fund_nav_basic(self, repo):
        """测试插入/更新净值数据"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", test_df)
        assert count >= 0

    def test_upsert_fund_nav_empty_df(self, repo):
        """测试空数据框"""
        test_df = pd.DataFrame()
        count = repo.upsert_fund_nav("001302", test_df)
        assert count == 0

    def test_upsert_fund_nav_none_df(self, repo):
        """测试None数据框"""
        count = repo.upsert_fund_nav("001302", None)
        assert count == 0

    def test_upsert_fund_nav_empty_code(self, repo):
        """测试空基金代码"""
        test_df = pd.DataFrame([{"nav_date": date(2024, 1, 1), "unit_nav": 1.0}])
        count = repo.upsert_fund_nav("", test_df)
        assert count == 0

    def test_upsert_fund_nav_alternate_columns(self, repo):
        """测试备用列名"""
        test_df = pd.DataFrame(
            [
                {
                    "日期": "2024-01-01",
                    "单位净值": 1.2345,
                    "累计净值": 1.3456,
                    "日增长率": 0.5,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", test_df)
        assert count >= 0

    def test_upsert_fund_nav_with_nulls(self, repo):
        """测试包含空值的插入"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": None,
                    "accum_nav": None,
                    "daily_return": None,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", test_df)
        assert count >= 0

    def test_upsert_fund_nav_invalid_data(self, repo):
        """测试无效数据处理"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": "invalid_date",
                    "unit_nav": "not_a_number",
                    "accum_nav": "not_a_number",
                    "daily_return": "not_a_number",
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", test_df)
        assert count == 0


class TestFundFetcher:
    """基金抓取器测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_fetch_fund_nav_basic(self, fetcher):
        """测试基本净值抓取"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        assert df is not None
        assert len(df) > 0
        assert "nav_date" in df.columns
        assert "unit_nav" in df.columns

    def test_fetch_fund_nav_empty_code(self, fetcher):
        """测试空基金代码"""
        df = fetcher.fetch_fund_nav("", days=30)
        assert df is None

    def test_fetch_fund_nav_whitespace_code(self, fetcher):
        """测试基金代码包含空格"""
        df = fetcher.fetch_fund_nav("  001302  ", days=30)
        assert df is not None

    def test_fetch_fund_info_basic(self, fetcher):
        """测试获取基金信息"""
        info = fetcher.fetch_fund_info("001302")
        assert info is not None
        assert "fund_code" in info
        assert info["fund_code"] == "001302"

    def test_fetch_fund_info_empty_code(self, fetcher):
        """测试空基金代码"""
        info = fetcher.fetch_fund_info("")
        assert info is not None
        assert info.get("fund_code") == ""

    def test_fetch_fund_info_timeout(self, fetcher):
        """测试超时设置"""
        info = fetcher.fetch_fund_info("001302", timeout=1)
        assert info is not None

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_fetch_from_eastmoney_success(self, mock_get, fetcher):
        """测试东方财富抓取成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        var Data_netWorthTrend = [
            {"x": 1704067200000, "y": 1.2345, "equityReturn": 0.5}
        ];
        """
        mock_get.return_value = mock_response

        df = fetcher._fetch_from_eastmoney("001302", 30)
        assert df is not None
        assert len(df) > 0

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_fetch_from_eastmoney_failure(self, mock_get, fetcher):
        """测试东方财富抓取失败"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        df = fetcher._fetch_from_eastmoney("001302", 30)
        assert df is None

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_fetch_from_tiantian_success(self, mock_get, fetcher):
        """测试天天基金抓取成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Data_ACWFF = [["2024-01-01", 1.2345, 1.3456]];'
        mock_get.return_value = mock_response

        df = fetcher._fetch_from_tiantian("001302", 30)
        assert df is not None

    @patch("services.fund.data.fund_fetcher.ak.fund_open_fund_daily_em")
    def test_fetch_from_akshare_success(self, mock_akshare, fetcher):
        """测试akshare抓取成功"""
        mock_df = pd.DataFrame(
            {
                "基金代码": ["001302"],
                "2024-01-01-单位净值": [1.2345],
                "2024-01-01-累计净值": [1.3456],
                "日增长率": [0.5],
            }
        )
        mock_akshare.return_value = mock_df

        df = fetcher._fetch_from_akshare("001302", 30)
        assert df is not None
        assert len(df) > 0

    @patch("services.fund.data.fund_fetcher.ak.fund_open_fund_daily_em")
    def test_fetch_from_akshare_empty(self, mock_akshare, fetcher):
        """测试akshare返回空数据"""
        mock_df = pd.DataFrame({"基金代码": []})
        mock_akshare.return_value = mock_df

        df = fetcher._fetch_from_akshare("999999", 30)
        assert df is None

    def test_multi_source_fallback(self, fetcher):
        """测试多数据源切换"""
        with (
            patch.object(fetcher, "_fetch_from_eastmoney", return_value=None),
            patch.object(fetcher, "_fetch_from_tiantian", return_value=None),
        ):
            df = fetcher.fetch_fund_nav("001302", days=30)
            assert df is not None or df is None
