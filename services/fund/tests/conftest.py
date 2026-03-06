# -*- coding: utf-8 -*-
"""
测试配置文件
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


@pytest.fixture
def mock_cache():
    """Mock缓存对象"""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = None
    cache.delete.return_value = None
    return cache


@pytest.fixture
def mock_redis_client():
    """Mock Redis 客户端"""
    client = MagicMock()
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.setex.return_value = True
    return client


@pytest.fixture(autouse=True)
def mock_shared_cache(mock_redis_client):
    """自动mock共享缓存"""
    with patch("shared.cache.RedisCache") as mock_cache_class:
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache_instance.set.return_value = None
        mock_cache_instance.delete.return_value = None
        mock_cache_instance.clear.return_value = 0
        mock_cache_instance.cleanup_expired.return_value = 0
        mock_cache_class.return_value = mock_cache_instance

        with patch("shared.cache._cache_instance", mock_cache_instance):
            with patch("shared.cache.get_cache", return_value=mock_cache_instance):
                yield


@pytest.fixture(autouse=True)
def mock_shared_db():
    """自动mock共享数据库"""
    with patch("shared.db.fetch_one", return_value=None):
        with patch("shared.db.fetch_all", return_value=[]):
            with patch("shared.db.execute", return_value=0):
                with patch("shared.db.execute_many", return_value=0):
                    yield


@pytest.fixture
def sample_fund_data():
    """示例基金数据"""
    return {
        "fund_code": "001302",
        "fund_name": "测试基金",
        "fund_type": "股票型",
        "manager": "测试经理",
        "establishment_date": "2020-01-01",
        "fund_scale": 10.5,
        "watchlist": 0,
    }


@pytest.fixture
def sample_nav_data():
    """示例净值数据"""
    import pandas as pd
    from datetime import date

    return pd.DataFrame(
        [
            {
                "nav_date": date(2024, 1, i + 1),
                "unit_nav": 1.2 + i * 0.001,
                "accum_nav": 1.3 + i * 0.001,
                "daily_return": (i % 3 - 1) * 0.1,
            }
            for i in range(10)
        ]
    )
