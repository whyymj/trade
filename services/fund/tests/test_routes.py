# -*- coding: utf-8 -*-
"""
API路由完整测试
"""

import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from services.fund.app import app


@pytest.fixture
def client():
    """测试客户端"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_health(self, client):
        """测试健康检查"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "fund-service"

    def test_metrics(self, client):
        """测试指标"""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "fund-service"
        assert "uptime" in data


class TestFundListAPI:
    """基金列表API测试"""

    def test_fund_list_basic(self, client):
        """测试基本基金列表"""
        resp = client.get("/api/fund/list?page=1&size=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data
        assert "success" in data
        assert data["success"] is True
        assert "total" in data["data"]
        assert "page" in data["data"]
        assert "page_size" in data["data"]
        assert "data" in data["data"]

    def test_fund_list_pagination(self, client):
        """测试分页"""
        resp1 = client.get("/api/fund/list?page=1&size=5")
        resp2 = client.get("/api/fund/list?page=2&size=5")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        data1 = resp1.get_json()
        data2 = resp2.get_json()
        assert data1["data"]["page"] == 1
        assert data2["data"]["page"] == 2

    def test_fund_list_with_fund_type(self, client):
        """测试按类型筛选"""
        resp = client.get("/api/fund/list?page=1&size=10&fund_type=股票型")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_fund_list_watchlist_only(self, client):
        """测试自选基金"""
        resp = client.get("/api/fund/list?watchlist_only=true")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_fund_list_industry_tag(self, client):
        """测试行业标签筛选"""
        resp = client.get("/api/fund/list?industry_tag=医药")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

    def test_fund_list_invalid_page(self, client):
        """测试无效页码"""
        resp = client.get("/api/fund/list?page=0&size=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["page"] == 1

    def test_fund_list_large_size(self, client):
        """测试大分页大小"""
        resp = client.get("/api/fund/list?page=1&size=200")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["page_size"] <= 100

    def test_fund_list_empty_params(self, client):
        """测试空参数"""
        resp = client.get("/api/fund/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


class TestFundDetailAPI:
    """基金详情API测试"""

    def test_fund_detail_exists(self, client):
        """测试存在的基金"""
        resp = client.get("/api/fund/001302")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "success" in data
        assert "data" in data
        if data["success"]:
            assert data["data"]["fund_code"] == "001302"

    def test_fund_detail_not_exists(self, client):
        """测试不存在的基金"""
        resp = client.get("/api/fund/999999")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False
        assert "message" in data

    def test_fund_detail_whitespace(self, client):
        """测试空格"""
        resp = client.get("/api/fund/%20%20001302%20%20")
        assert resp.status_code in [200, 404]

    def test_fund_detail_cache(self, client):
        """测试缓存功能"""
        resp1 = client.get("/api/fund/001302")
        assert resp1.status_code == 200
        resp2 = client.get("/api/fund/001302")
        assert resp2.status_code == 200
        data1 = resp1.get_json()
        data2 = resp2.get_json()
        assert data1 == data2


class TestFundNavAPI:
    """基金净值API测试"""

    def test_fund_nav_basic(self, client):
        """测试基本净值查询"""
        resp = client.get("/api/fund/001302/nav?days=30")
        assert resp.status_code in [200, 404]
        data = resp.get_json()
        assert "success" in data

    def test_fund_nav_not_exists(self, client):
        """测试不存在的基金"""
        resp = client.get("/api/fund/999999/nav?days=30")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["success"] is False

    def test_fund_nav_date_range(self, client):
        """测试日期范围"""
        resp = client.get(
            "/api/fund/001302/nav?start_date=2024-01-01&end_date=2024-01-31"
        )
        assert resp.status_code in [200, 404]
        data = resp.get_json()
        assert "success" in data

    def test_fund_nav_only_days(self, client):
        """测试仅使用天数"""
        resp = client.get("/api/fund/001302/nav?days=7")
        assert resp.status_code in [200, 404]
        data = resp.get_json()
        assert "success" in data

    def test_fund_nav_large_days(self, client):
        """测试大天数"""
        resp = client.get("/api/fund/001302/nav?days=365")
        assert resp.status_code in [200, 404]
        data = resp.get_json()
        assert "success" in data

    def test_fund_nav_cache(self, client):
        """测试缓存功能"""
        resp1 = client.get("/api/fund/001302/nav?days=30")
        resp2 = client.get("/api/fund/001302/nav?days=30")
        assert resp1.status_code == resp2.status_code


class TestFundPredictAPI:
    """基金预测API测试"""

    def test_fund_predict_basic(self, client):
        """测试基本预测"""
        resp = client.get("/api/fund/001302/predict")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "success" in data
        assert data["success"] is True
        assert "data" in data
        assert "fund_code" in data["data"]
        assert data["data"]["fund_code"] == "001302"

    def test_fund_predict_fields(self, client):
        """测试预测字段"""
        resp = client.get("/api/fund/001302/predict")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data
        pred = data["data"]
        required_fields = [
            "fund_code",
            "direction",
            "direction_label",
            "magnitude",
            "prob_up",
            "predict_date",
        ]
        for field in required_fields:
            assert field in pred

    def test_fund_predict_cache(self, client):
        """测试预测缓存"""
        resp1 = client.get("/api/fund/001302/predict")
        resp2 = client.get("/api/fund/001302/predict")
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        data1 = resp1.get_json()
        data2 = resp2.get_json()
        assert data1 == data2


class TestAPIValidation:
    """API验证测试"""

    def test_invalid_params_fund_list(self, client):
        """测试无效参数"""
        resp = client.get("/api/fund/list?page=abc&size=xyz")
        assert resp.status_code == 400 or resp.status_code == 200

    def test_negative_params(self, client):
        """测试负数参数"""
        resp = client.get("/api/fund/list?page=-1&size=-10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["data"]["page"] == 1
        assert data["data"]["page_size"] == 1

    def test_empty_fund_code(self, client):
        """测试空基金代码"""
        resp = client.get("/api/fund//nav")
        assert resp.status_code in [404, 405]

    def test_special_characters(self, client):
        """测试特殊字符"""
        resp = client.get("/api/fund/abc123/nav")
        assert resp.status_code in [200, 404]


class TestCacheBehavior:
    """缓存行为测试"""

    @patch("services.fund.routes.fund.repo")
    def test_fund_list_caching(self, mock_repo, client):
        """测试基金列表缓存"""
        mock_result = {"total": 100, "page": 1, "page_size": 10, "data": []}
        mock_repo.get_fund_list.return_value = mock_result

        resp1 = client.get("/api/fund/list?page=1&size=10")
        resp2 = client.get("/api/fund/list?page=1&size=10")

        assert resp1.status_code == 200
        assert resp2.status_code == 200

    @patch("services.fund.routes.fund.repo")
    def test_fund_detail_caching(self, mock_repo, client):
        """测试基金详情缓存"""
        mock_result = {
            "fund_code": "001302",
            "fund_name": "测试基金",
            "fund_type": "股票型",
        }
        mock_repo.get_fund_info.return_value = mock_result

        resp1 = client.get("/api/fund/001302")
        resp2 = client.get("/api/fund/001302")

        assert resp1.status_code == 200
        assert resp2.status_code == 200

    @patch("services.fund.routes.fund.repo")
    def test_fund_nav_caching(self, mock_repo, client):
        """测试基金净值缓存"""
        import pandas as pd
        from datetime import date

        mock_df = pd.DataFrame(
            [
                {
                    "nav_date": date(2024, 1, 1),
                    "unit_nav": 1.2345,
                    "accum_nav": 1.3456,
                    "daily_return": 0.5,
                }
            ]
        )
        mock_repo.get_fund_nav.return_value = mock_df

        resp1 = client.get("/api/fund/001302/nav?days=30")
        resp2 = client.get("/api/fund/001302/nav?days=30")

        assert resp1.status_code == 200
        assert resp2.status_code == 200


class TestErrorHandling:
    """错误处理测试"""

    @patch("services.fund.routes.fund.repo")
    def test_database_error(self, mock_repo, client):
        """测试数据库错误"""
        mock_repo.get_fund_list.side_effect = Exception("Database error")

        resp = client.get("/api/fund/list?page=1&size=10")
        assert resp.status_code == 500

    @patch("services.fund.routes.fund.repo")
    def test_fund_not_found(self, mock_repo, client):
        """测试基金不存在"""
        mock_repo.get_fund_info.return_value = None

        resp = client.get("/api/fund/999999")
        assert resp.status_code == 404

    @patch("services.fund.routes.fund.repo")
    def test_nav_not_found(self, mock_repo, client):
        """测试净值不存在"""
        mock_repo.get_fund_nav.return_value = None

        resp = client.get("/api/fund/999999/nav?days=30")
        assert resp.status_code == 404
