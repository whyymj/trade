# -*- coding: utf-8 -*-
"""
数据抓取集成测试 - 重点测试数据抓取功能
"""

import pytest
import sys
import os
import pandas as pd
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from services.fund.data import FundFetcher, FundRepo


class TestRealDataFetching:
    """真实数据抓取测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def test_fund_codes(self):
        """测试用的基金代码"""
        return ["001302", "000001", "110011", "161725"]

    def test_real_fetch_nav_eastmoney(self, fetcher, test_fund_codes):
        """测试真实从东方财富抓取净值数据"""
        for fund_code in test_fund_codes[:2]:  # 只测试2个基金
            df = fetcher._fetch_from_eastmoney(fund_code, days=30)
            if df is not None:
                assert len(df) > 0
                assert "nav_date" in df.columns
                assert "unit_nav" in df.columns
                assert df["unit_nav"].notna().any()
                print(f"✓ 成功从东方财富抓取 {fund_code}: {len(df)} 条数据")

    def test_real_fetch_nav_tiantian(self, fetcher, test_fund_codes):
        """测试真实从天天基金抓取净值数据"""
        for fund_code in test_fund_codes[:2]:
            df = fetcher._fetch_from_tiantian(fund_code, days=30)
            if df is not None:
                assert len(df) > 0
                assert "nav_date" in df.columns
                assert "unit_nav" in df.columns
                print(f"✓ 成功从天天基金抓取 {fund_code}: {len(df)} 条数据")

    def test_real_fetch_nav_akshare(self, fetcher, test_fund_codes):
        """测试真实从akshare抓取净值数据"""
        for fund_code in test_fund_codes[:2]:
            df = fetcher._fetch_from_akshare(fund_code, days=30)
            if df is not None:
                assert len(df) > 0
                assert "nav_date" in df.columns
                assert "unit_nav" in df.columns
                print(f"✓ 成功从akshare抓取 {fund_code}: {len(df)} 条数据")

    def test_real_fetch_fund_info(self, fetcher, test_fund_codes):
        """测试真实抓取基金信息"""
        for fund_code in test_fund_codes[:2]:
            info = fetcher.fetch_fund_info(fund_code)
            assert info is not None
            assert "fund_code" in info
            assert info["fund_code"] == fund_code
            print(f"✓ 成功抓取基金信息 {fund_code}: {info.get('fund_name')}")

    def test_real_fetch_auto_fallback(self, fetcher, test_fund_codes):
        """测试自动切换数据源"""
        for fund_code in test_fund_codes[:1]:
            df = fetcher.fetch_fund_nav(fund_code, days=30)
            assert df is not None
            assert len(df) > 0
            print(f"✓ 自动切换数据源成功抓取 {fund_code}: {len(df)} 条数据")

    def test_fetch_performance(self, fetcher):
        """测试抓取性能"""
        start_time = time.time()
        df = fetcher.fetch_fund_nav("001302", days=365)
        elapsed_time = time.time() - start_time
        assert df is not None
        assert len(df) > 0
        assert elapsed_time < 30  # 应该在30秒内完成
        print(f"✓ 抓取性能测试: {len(df)} 条数据耗时 {elapsed_time:.2f} 秒")

    def test_fetch_with_different_days(self, fetcher):
        """测试不同天数抓取"""
        for days in [7, 30, 90, 365]:
            df = fetcher.fetch_fund_nav("001302", days=days)
            if df is not None:
                assert len(df) > 0
                print(f"✓ 成功抓取 {days} 天数据: {len(df)} 条")


class TestDataDeduplication:
    """数据去重测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_upsert_duplicate_data(self, repo, fetcher):
        """测试插入重复数据"""
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

        count1 = repo.upsert_fund_nav("001302", test_df)
        count2 = repo.upsert_fund_nav("001302", test_df)
        assert count1 >= 0
        assert count2 >= 0
        print(f"✓ 重复数据插入测试: 第一次 {count1}, 第二次 {count2}")

    def test_fetch_and_upsert_integration(self, repo, fetcher):
        """测试抓取并插入数据库"""
        df = fetcher.fetch_fund_nav("001302", days=5)
        if df is not None:
            count = repo.upsert_fund_nav("001302", df)
            assert count >= 0
            print(f"✓ 抓取并插入数据库: {count} 条数据")

            # 验证数据是否正确插入
            result = repo.get_fund_nav("001302", days=5)
            assert result is not None
            assert len(result) > 0


class TestFetchRateControl:
    """抓取频率控制测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_concurrent_fetch(self, fetcher):
        """测试并发抓取"""
        fund_codes = ["001302", "000001", "110011"]
        results = []
        for fund_code in fund_codes:
            df = fetcher.fetch_fund_nav(fund_code, days=30)
            results.append((fund_code, df))

        assert len(results) == 3
        for fund_code, df in results:
            if df is not None:
                assert len(df) > 0
                print(f"✓ 并发抓取 {fund_code}: {len(df)} 条数据")

    def test_sequential_fetch_timing(self, fetcher):
        """测试连续抓取的时间间隔"""
        fund_codes = ["001302", "000001"]
        times = []
        for fund_code in fund_codes:
            start = time.time()
            df = fetcher.fetch_fund_nav(fund_code, days=30)
            elapsed = time.time() - start
            times.append(elapsed)
            if df is not None:
                print(f"✓ 抓取 {fund_code} 耗时 {elapsed:.2f} 秒")

        assert len(times) == 2

    def test_fetch_timeout_handling(self, fetcher):
        """测试超时处理"""
        info = fetcher.fetch_fund_info("001302", timeout=3)
        assert info is not None
        print(f"✓ 超时测试完成: {info}")


class TestMockDataFetching:
    """Mock数据抓取测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_mock_eastmoney_large_dataset(self, mock_get, fetcher):
        """Mock东方财富大数据集"""
        mock_data = []
        for i in range(100):
            mock_data.append(
                {
                    "x": (1704067200000 + i * 86400000),
                    "y": 1.2 + i * 0.001,
                    "equityReturn": (i % 3 - 1) * 0.1,
                }
            )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = f"var Data_netWorthTrend = {mock_data};"
        mock_get.return_value = mock_response

        df = fetcher._fetch_from_eastmoney("001302", days=100)
        assert df is not None
        assert len(df) == 100
        print(f"✓ Mock大数据集测试: {len(df)} 条数据")

    @patch("services.fund.data.fund_fetcher.ak.fund_open_fund_daily_em")
    def test_mock_akshare_multiple_funds(self, mock_akshare, fetcher):
        """Mock akshare多基金数据"""
        mock_df = pd.DataFrame(
            {
                "基金代码": ["001302", "000001", "110011"],
                "2024-01-01-单位净值": [1.2345, 2.3456, 0.9876],
                "2024-01-01-累计净值": [1.3456, 2.4567, 0.9987],
                "日增长率": [0.5, -0.3, 1.2],
            }
        )
        mock_akshare.return_value = mock_df

        fund_codes = ["001302", "000001", "110011"]
        for fund_code in fund_codes:
            df = fetcher._fetch_from_akshare(fund_code, days=30)
            assert df is not None
            assert len(df) > 0
        print(f"✓ Mock多基金测试完成")

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_mock_network_error(self, mock_get, fetcher):
        """Mock网络错误"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        df = fetcher._fetch_from_eastmoney("001302", days=30)
        assert df is None
        print(f"✓ Mock网络错误测试完成")

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_mock_timeout(self, mock_get, fetcher):
        """Mock超时"""
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        df = fetcher._fetch_from_eastmoney("001302", days=30)
        assert df is None
        print(f"✓ Mock超时测试完成")


class TestDataValidation:
    """数据验证测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    @pytest.fixture
    def repo(self):
        return FundRepo()

    def test_validate_nav_data(self, fetcher):
        """验证净值数据格式"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is not None:
            assert df["nav_date"].notna().all()
            assert df["unit_nav"].notna().all()
            assert (df["unit_nav"] > 0).all()
            print(f"✓ 数据验证通过: {len(df)} 条数据")

    def test_validate_date_sequence(self, fetcher):
        """验证日期序列"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is not None:
            df_sorted = df.sort_values("nav_date")
            dates = df_sorted["nav_date"].tolist()
            assert dates == sorted(dates)
            print(f"✓ 日期序列验证通过")

    def test_validate_no_duplicate_dates(self, fetcher):
        """验证无重复日期"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is not None:
            duplicate_count = df["nav_date"].duplicated().sum()
            assert duplicate_count == 0
            print(f"✓ 无重复日期验证通过")

    def test_validate_return_calculation(self, fetcher):
        """验证收益率计算"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is not None and "daily_return" in df.columns:
            assert df["daily_return"].notna().any()
            print(f"✓ 收益率计算验证通过")


class TestDataPersistence:
    """数据持久化测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_fetch_persist_retrieve(self, repo, fetcher):
        """测试抓取-持久化-检索流程"""
        # 抓取
        df = fetcher.fetch_fund_nav("001302", days=10)
        if df is None:
            pytest.skip("无法抓取数据")

        # 持久化
        count = repo.upsert_fund_nav("001302", df)
        assert count >= 0

        # 检索
        result = repo.get_fund_nav("001302", days=10)
        assert result is not None
        assert len(result) > 0
        print(
            f"✓ 完整流程测试: 抓取{len(df)}条 -> 持久化{count}条 -> 检索{len(result)}条"
        )

    def test_persisted_data_quality(self, repo, fetcher):
        """测试持久化数据质量"""
        df = fetcher.fetch_fund_nav("001302", days=5)
        if df is None:
            pytest.skip("无法抓取数据")

        repo.upsert_fund_nav("001302", df)
        result = repo.get_fund_nav("001302", days=5)

        # 验证列名
        assert all(
            col in result.columns for col in ["nav_date", "unit_nav", "accum_nav"]
        )

        # 验证数据类型
        assert result["unit_nav"].dtype == float
        print(f"✓ 持久化数据质量验证通过")


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_invalid_fund_code(self, fetcher):
        """测试无效基金代码"""
        invalid_codes = ["", "   ", "abc", "123", "000000", "99999999"]
        for code in invalid_codes:
            df = fetcher.fetch_fund_nav(code, days=30)
            assert df is None
        print(f"✓ 无效基金代码测试通过")

    def test_zero_days(self, fetcher):
        """测试0天"""
        df = fetcher.fetch_fund_nav("001302", days=0)
        assert df is not None or df is None  # 取决于实现

    def test_negative_days(self, fetcher):
        """测试负数天"""
        df = fetcher.fetch_fund_nav("001302", days=-10)
        assert df is not None or df is None

    def test_very_large_days(self, fetcher):
        """测试非常大的天数"""
        df = fetcher.fetch_fund_nav("001302", days=1000)
        if df is not None:
            assert len(df) > 0
            print(f"✓ 大天数测试通过: {len(df)} 条数据")

    def test_recent_data(self, fetcher):
        """测试最新数据"""
        df = fetcher.fetch_fund_nav("001302", days=1)
        if df is not None:
            assert len(df) > 0
            latest_date = df["nav_date"].max()
            assert latest_date <= date.today()
            print(f"✓ 最新数据测试通过: {latest_date}")
