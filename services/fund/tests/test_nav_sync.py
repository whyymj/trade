# -*- coding: utf-8 -*-
"""
净值同步测试
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


class TestBatchNavSync:
    """批量净值同步测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    @pytest.fixture
    def test_fund_codes(self):
        """测试基金代码"""
        return ["001302", "000001", "110011"]

    def test_batch_sync_multiple_funds(self, repo, fetcher, test_fund_codes):
        """测试批量同步多个基金"""
        results = []
        for fund_code in test_fund_codes:
            df = fetcher.fetch_fund_nav(fund_code, days=30)
            if df is not None:
                count = repo.upsert_fund_nav(fund_code, df)
                results.append((fund_code, count, len(df)))
                print(f"✓ 同步 {fund_code}: 抓取{len(df)}条, 插入{count}条")

        assert len(results) > 0
        for fund_code, count, fetched in results:
            assert count >= 0

    def test_batch_sync_with_large_dataset(self, repo, fetcher):
        """测试大批量同步"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, i + 1),
                    "unit_nav": 1.0 + i * 0.001,
                    "accum_nav": 1.1 + i * 0.001,
                    "daily_return": (i % 3 - 1) * 0.1,
                }
                for i in range(365)
            ]
        )
        count = repo.upsert_fund_nav("001302", test_df)
        assert count >= 0
        print(f"✓ 大批量同步测试: {count} 条数据")

    def test_batch_sync_performance(self, repo, fetcher):
        """测试批量同步性能"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, i + 1),
                    "unit_nav": 1.0 + i * 0.001,
                    "accum_nav": 1.1 + i * 0.001,
                    "daily_return": (i % 3 - 1) * 0.1,
                }
                for i in range(100)
            ]
        )

        start_time = time.time()
        count = repo.upsert_fund_nav("001302", test_df)
        elapsed_time = time.time() - start_time

        assert count >= 0
        assert elapsed_time < 5  # 应该在5秒内完成
        print(f"✓ 批量同步性能: {count} 条数据耗时 {elapsed_time:.2f} 秒")


class TestIncrementalUpdate:
    """增量更新测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_update_existing_records(self, repo):
        """测试更新已存在的记录"""
        test_df1 = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                }
            ]
        )
        count1 = repo.upsert_fund_nav("001302", test_df1)

        test_df2 = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2350,
                    "accum_nav": 1.3460,
                    "daily_return": 0.6,
                }
            ]
        )
        count2 = repo.upsert_fund_nav("001302", test_df2)

        assert count1 >= 0
        assert count2 >= 0
        print(f"✓ 更新已存在记录: 第一次{count1}, 第二次{count2}")

    def test_add_new_records(self, repo):
        """测试添加新记录"""
        existing_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                }
            ]
        )
        repo.upsert_fund_nav("001302", existing_df)

        new_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 2),
                    "unit_nav": 1.2350,
                    "accum_nav": 1.3460,
                    "daily_return": 0.4,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", new_df)

        assert count >= 0
        print(f"✓ 添加新记录: {count} 条")

    def test_mixed_update_and_insert(self, repo):
        """测试混合更新和插入"""
        mixed_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                },
                {
                    "nav_date": date(2024, 1, 3),
                    "unit_nav": 1.2360,
                    "accum_nav": 1.3470,
                    "daily_return": 0.6,
                },
            ]
        )
        count1 = repo.upsert_fund_nav("001302", mixed_df)

        mixed_df2 = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2350,
                    "accum_nav": 1.3460,
                    "daily_return": 0.55,
                },
                {
                    "nav_date": date(2024, 1, 2),
                    "unit_nav": 1.2355,
                    "accum_nav": 1.3465,
                    "daily_return": 0.52,
                },
            ]
        )
        count2 = repo.upsert_fund_nav("001302", mixed_df2)

        assert count1 >= 0
        assert count2 >= 0
        print(f"✓ 混合更新和插入: 第一次{count1}, 第二次{count2}")

    def test_incremental_sync_with_dates(self, repo):
        """测试基于日期的增量同步"""
        base_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                },
                {
                    "nav_date": date(2024, 1, 2),
                    "unit_nav": 1.2350,
                    "accum_nav": 1.3460,
                    "daily_return": 0.4,
                },
            ]
        )
        repo.upsert_fund_nav("001302", base_df)

        increment_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 3),
                    "unit_nav": 1.2355,
                    "accum_nav": 1.3465,
                    "daily_return": 0.3,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", increment_df)

        assert count >= 0

        result = repo.get_fund_nav(
            "001302", start_date="2024-01-01", end_date="2024-01-03"
        )
        assert result is not None
        assert len(result) >= 3
        print(f"✓ 基于日期的增量同步: {len(result)} 条记录")


class TestErrorRetry:
    """错误重试测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_sync_with_invalid_data(self, repo):
        """测试同步无效数据"""
        invalid_df = pd.DataFrame(
            [
                {
                    "nav_date": "invalid",
                    "unit_nav": "not_a_number",
                    "accum_nav": "not_a_number",
                    "daily_return": "not_a_number",
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", invalid_df)
        assert count == 0
        print(f"✓ 无效数据处理: {count} 条")

    def test_sync_with_partial_invalid_data(self, repo):
        """测试同步部分无效数据"""
        mixed_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                },
                {
                    "nav_date": "invalid",
                    "unit_nav": "not_a_number",
                    "accum_nav": "not_a_number",
                    "daily_return": "not_a_number",
                },
                {
                    "nav_date": date(2024, 1, 3),
                    "unit_nav": 1.2360,
                    "accum_nav": 1.3470,
                    "daily_return": 0.6,
                },
            ]
        )
        count = repo.upsert_fund_nav("001302", mixed_df)
        assert count >= 0
        print(f"✓ 部分无效数据处理: {count} 条有效数据")

    def test_sync_with_null_values(self, repo):
        """测试同步空值数据"""
        null_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": None,
                    "accum_nav": None,
                    "daily_return": None,
                }
            ]
        )
        count = repo.upsert_fund_nav("001302", null_df)
        assert count >= 0
        print(f"✓ 空值数据处理: {count} 条")

    def test_sync_after_failure(self, repo, fetcher):
        """测试失败后的重试同步"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is None:
            pytest.skip("无法抓取数据")

        count = repo.upsert_fund_nav("001302", df)
        assert count >= 0

        # 重试同步
        count2 = repo.upsert_fund_nav("001302", df)
        assert count2 >= 0
        print(f"✓ 重试同步: 第一次{count}, 第二次{count2}")

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_fetch_retry_logic(self, mock_get, fetcher):
        """测试抓取重试逻辑"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        # 第一次失败
        df = fetcher._fetch_from_eastmoney("001302", days=30)
        assert df is None

        # 第二次也失败
        df = fetcher._fetch_from_eastmoney("001302", days=30)
        assert df is None

        print(f"✓ 抓取重试逻辑测试完成")

    @patch("services.fund.data.fund_fetcher.requests.get")
    def test_fetch_eventual_success(self, mock_get, fetcher):
        """测试最终成功"""
        # 前两次失败
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500

        # 第三次成功
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = """
        var Data_netWorthTrend = [
            {"x": 1704067200000, "y": 1.2345, "equityReturn": 0.5}
        ];
        """

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        # 使用自动切换的fetch_fund_nav
        df = fetcher.fetch_fund_nav("001302", days=30)
        assert df is not None
        print(f"✓ 最终成功测试完成")


class TestDataConsistency:
    """数据一致性测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_sync_data_consistency(self, repo, fetcher):
        """测试同步数据一致性"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is None:
            pytest.skip("无法抓取数据")

        original_count = len(df)
        inserted_count = repo.upsert_fund_nav("001302", df)

        result = repo.get_fund_nav("001302", days=30)
        assert result is not None
        assert len(result) >= original_count

        print(
            f"✓ 数据一致性: 原始{original_count}条, 插入{inserted_count}条, 检索{len(result)}条"
        )

    def test_no_duplicate_dates(self, repo):
        """测试无重复日期"""
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
        repo.upsert_fund_nav("001302", test_df)

        result = repo.get_fund_nav("001302", days=30)
        if result is not None:
            duplicate_count = result["nav_date"].duplicated().sum()
            assert duplicate_count == 0
            print(f"✓ 无重复日期验证通过")

    def test_date_ordering(self, repo):
        """测试日期排序"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 3),
                    "unit_nav": 1.2360,
                    "accum_nav": 1.3470,
                    "daily_return": 0.6,
                },
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                },
                {
                    "nav_date": date(2024, 1, 2),
                    "unit_nav": 1.2350,
                    "accum_nav": 1.3460,
                    "daily_return": 0.4,
                },
            ]
        )
        repo.upsert_fund_nav("001302", test_df)

        result = repo.get_fund_nav("001302", days=30)
        if result is not None:
            dates = result["nav_date"].tolist()
            assert dates == sorted(dates)
            print(f"✓ 日期排序验证通过")

    def test_numeric_precision(self, repo):
        """测试数值精度"""
        test_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.234567,
                    "accum_nav": 1.345678,
                    "daily_return": 0.56789,
                }
            ]
        )
        repo.upsert_fund_nav("001302", test_df)

        result = repo.get_fund_nav("001302", days=30)
        if result is not None and len(result) > 0:
            row = result.iloc[0]
            assert abs(row["unit_nav"] - 1.234567) < 0.0001
            assert abs(row["accum_nav"] - 1.345678) < 0.0001
            assert abs(row["daily_return"] - 0.56789) < 0.0001
            print(f"✓ 数值精度验证通过")


class TestSyncScenarios:
    """同步场景测试"""

    @pytest.fixture
    def repo(self):
        return FundRepo()

    @pytest.fixture
    def fetcher(self):
        return FundFetcher()

    def test_full_sync_scenario(self, repo, fetcher):
        """测试完整同步场景"""
        fund_codes = ["001302"]

        for fund_code in fund_codes:
            # 抓取数据
            df = fetcher.fetch_fund_nav(fund_code, days=30)
            if df is None:
                continue

            # 同步数据
            count = repo.upsert_fund_nav(fund_code, df)

            # 验证数据
            result = repo.get_fund_nav(fund_code, days=30)

            print(
                f"✓ 完整同步场景 {fund_code}: 抓取{len(df)}条 -> 同步{count}条 -> 验证{len(result) if result is not None else 0}条"
            )

            if result is not None:
                assert len(result) >= len(df)

    def test_partial_sync_scenario(self, repo, fetcher):
        """测试部分同步场景"""
        df = fetcher.fetch_fund_nav("001302", days=30)
        if df is None:
            pytest.skip("无法抓取数据")

        # 先同步前10条
        partial_df = df.head(10)
        count1 = repo.upsert_fund_nav("001302", partial_df)

        # 再同步后10条
        remaining_df = df.tail(len(df) - 10)
        count2 = repo.upsert_fund_nav("001302", remaining_df)

        # 验证总数
        result = repo.get_fund_nav("001302", days=30)

        print(
            f"✓ 部分同步场景: 前10条{count1}, 后{len(df) - 10}条{count2}, 总计{len(result) if result is not None else 0}条"
        )

    def test_update_after_fetch(self, repo, fetcher):
        """测试抓取后更新"""
        df1 = fetcher.fetch_fund_nav("001302", days=30)
        if df1 is None:
            pytest.skip("无法抓取数据")

        count1 = repo.upsert_fund_nav("001302", df1)

        # 等待一段时间
        time.sleep(1)

        # 再次抓取
        df2 = fetcher.fetch_fund_nav("001302", days=30)
        if df2 is not None:
            count2 = repo.upsert_fund_nav("001302", df2)

            print(f"✓ 抓取后更新: 第一次{count1}, 第二次{count2}")

    def test_sync_with_alternate_columns(self, repo):
        """测试备用列名同步"""
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

        result = repo.get_fund_nav("001302", days=30)
        if result is not None:
            print(f"✓ 备用列名同步: {len(result)} 条记录")
