#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock 服务路由单元测试
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.stock.app import app


@pytest.fixture
def client():
    """测试客户端"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_cache():
    """Mock 缓存"""
    with patch("shared.cache.get_cache") as mock:
        cache_instance = MagicMock()
        cache_instance.get.return_value = None
        cache_instance.set.return_value = True
        mock.return_value = cache_instance
        yield cache_instance


@pytest.fixture
def mock_stock_data():
    """Mock 股票数据"""
    import pandas as pd
    import numpy as np

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
    return df


class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_health(self, client):
        """测试健康检查"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "stock-service"

    def test_metrics(self, client):
        """测试指标端点"""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "stock-service"

    def test_lstm_health(self, client):
        """测试 LSTM 健康检查"""
        resp = client.get("/api/lstm/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "lstm"
        assert data["status"] == "healthy"

    def test_stock_health(self, client):
        """测试 Stock 健康检查"""
        resp = client.get("/api/stock/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "stock"
        assert data["status"] == "healthy"


class TestStockAPI:
    """Stock API 测试"""

    def test_stock_indicators_missing_symbol(self, client):
        """测试技术指标 - 缺少 symbol"""
        resp = client.get("/api/stock/indicators")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "Symbol required" in data["message"]

    @patch("services.stock.data.get_stock_data")
    def test_stock_indicators_success(self, mock_get_data, client, mock_stock_data):
        """测试技术指标成功"""
        mock_get_data.return_value = mock_stock_data

        resp = client.get("/api/stock/indicators?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["symbol"] == "000001"

        # 检查所有指标都存在
        required_fields = [
            "ma5",
            "ma10",
            "ma20",
            "ma60",
            "macd",
            "rsi",
            "bollinger",
            "volatility",
            "obv",
            "mfi",
            "aroon",
        ]
        for field in required_fields:
            assert field in data["data"]

    @patch("services.stock.data.get_stock_data")
    def test_stock_indicators_insufficient_data(self, mock_get_data, client):
        """测试技术指标 - 数据不足"""
        import pandas as pd

        mock_get_data.return_value = pd.DataFrame({"收盘": [1, 2, 3]})

        resp = client.get("/api/stock/indicators?symbol=000001")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "Insufficient data" in data["message"]

    @patch("services.stock.data.get_stock_data")
    def test_stock_indicators_cache_hit(
        self, mock_get_data, client, mock_cache, mock_stock_data
    ):
        """测试技术指标 - 缓存命中"""
        cached_data = {"symbol": "000001", "ma5": 100.0}
        mock_cache.get.return_value = cached_data

        resp = client.get("/api/stock/indicators?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["cached"] is True
        assert data["data"] == cached_data

        # 确保没有调用数据获取函数
        mock_get_data.assert_not_called()

    @patch("services.stock.data.get_stock_data")
    def test_stock_indicators_cache_miss(
        self, mock_get_data, client, mock_cache, mock_stock_data
    ):
        """测试技术指标 - 缓存未命中"""
        mock_cache.get.return_value = None
        mock_get_data.return_value = mock_stock_data

        resp = client.get("/api/stock/indicators?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["cached"] is False

        # 确保调用了数据获取函数
        mock_get_data.assert_called_once()

    @patch("services.stock.data.get_stock_list")
    def test_stock_list_default_params(self, mock_get_list, client):
        """测试股票列表 - 默认参数"""
        mock_get_list.return_value = {
            "items": [],
            "total": 0,
            "page": 1,
            "size": 20,
            "pages": 0,
        }

        resp = client.get("/api/stock/list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True

        mock_get_list.assert_called_with(page=1, size=20)

    @patch("services.stock.data.get_stock_list")
    def test_stock_list_custom_params(self, mock_get_list, client):
        """测试股票列表 - 自定义参数"""
        mock_get_list.return_value = {
            "items": [{"symbol": "000001", "name": "平安银行"}],
            "total": 1,
            "page": 2,
            "size": 10,
            "pages": 1,
        }

        resp = client.get("/api/stock/list?page=2&size=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["data"]["items"]) == 1

        mock_get_list.assert_called_with(page=2, size=10)


class TestLSTMTrainingAPI:
    """LSTM 训练 API 测试"""

    def test_lstm_train_missing_symbol(self, client):
        """测试 LSTM 训练 - 缺少 symbol"""
        resp = client.post("/api/lstm/train", json={})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "Symbol required" in data["message"]

    def test_lstm_train_empty_symbol(self, client):
        """测试 LSTM 训练 - 空符号"""
        resp = client.post("/api/lstm/train", json={"symbol": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    @patch("services.stock.routes.lstm._acquire_lock")
    def test_lstm_train_lock_failed(self, mock_lock, client):
        """测试 LSTM 训练 - 获取锁失败"""
        mock_lock.return_value = False

        resp = client.post("/api/lstm/train", json={"symbol": "000001"})
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["success"] is False
        assert "Training in progress" in data["message"]

    @patch("services.stock.routes.lstm._acquire_lock")
    @patch("services.stock.routes.lstm._release_lock")
    @patch("services.stock.analysis.train_model")
    def test_lstm_train_success(self, mock_train, mock_release, mock_lock, client):
        """测试 LSTM 训练成功"""
        mock_lock.return_value = True
        mock_train.return_value = {
            "symbol": "000001",
            "version_id": "v1",
            "model_path": "/path/to/model.pt",
            "metrics": {"accuracy": 0.85, "mse": 0.01},
        }

        resp = client.post("/api/lstm/train", json={"symbol": "000001"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["symbol"] == "000001"
        assert data["data"]["status"] == "completed"
        assert "metrics" in data["data"]

        mock_lock.assert_called_once()
        mock_release.assert_called_once()

    @patch("services.stock.routes.lstm._acquire_lock")
    @patch("services.stock.routes.lstm._release_lock")
    @patch("services.stock.analysis.train_model")
    def test_lstm_train_error(self, mock_train, mock_release, mock_lock, client):
        """测试 LSTM 训练错误"""
        mock_lock.return_value = True
        mock_train.side_effect = Exception("训练失败")

        resp = client.post("/api/lstm/train", json={"symbol": "000001"})
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False
        assert "训练失败" in data["message"]

        mock_lock.assert_called_once()
        mock_release.assert_called_once()


class TestLSMTPredictAPI:
    """LSTM 预测 API 测试"""

    def test_lstm_predict_missing_symbol(self, client):
        """测试 LSTM 预测 - 缺少 symbol"""
        resp = client.get("/api/lstm/predict")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "Symbol required" in data["message"]

    @patch("services.stock.analysis.predict")
    def test_lstm_predict_success(self, mock_predict, client):
        """测试 LSTM 预测成功"""
        mock_predict.return_value = {
            "symbol": "000001",
            "direction": "上涨",
            "direction_code": 1,
            "magnitude": 0.02,
            "confidence": 0.75,
            "daily_magnitude": [0.01, 0.02, 0.01, 0.02, 0.01],
        }

        resp = client.get("/api/lstm/predict?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "data" in data
        assert data["data"]["symbol"] == "000001"
        assert data["data"]["direction"] == "上涨"
        assert data["cached"] is False

    @patch("services.stock.analysis.predict")
    def test_lstm_predict_cache_hit(self, mock_predict, client, mock_cache):
        """测试 LSTM 预测 - 缓存命中"""
        cached_result = {"symbol": "000001", "direction": "上涨", "confidence": 0.75}
        mock_cache.get.return_value = cached_result

        resp = client.get("/api/lstm/predict?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["cached"] is True
        assert data["data"] == cached_result

        mock_predict.assert_not_called()

    @patch("services.stock.analysis.predict")
    def test_lstm_predict_error(self, mock_predict, client):
        """测试 LSTM 预测错误"""
        mock_predict.side_effect = Exception("预测失败")

        resp = client.get("/api/lstm/predict?symbol=000001")
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False
        assert "预测失败" in data["message"]

    @patch("services.stock.analysis.predict")
    def test_lstm_predict_model_not_found(self, mock_predict, client):
        """测试 LSTM 预测 - 模型未找到"""
        mock_predict.return_value = {
            "symbol": "000001",
            "error": "模型未训练",
            "direction": None,
            "confidence": None,
        }

        resp = client.get("/api/lstm/predict?symbol=000001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["error"] == "模型未训练"
        assert data["data"]["direction"] is None


class TestParameterValidation:
    """参数验证测试"""

    def test_invalid_page_parameter(self, client):
        """测试无效的页码参数"""
        with patch("services.stock.data.get_stock_list") as mock:
            mock.return_value = {
                "items": [],
                "total": 0,
                "page": -1,
                "size": 20,
                "pages": 0,
            }
            resp = client.get("/api/stock/list?page=-1")
            assert resp.status_code == 200

    def test_invalid_size_parameter(self, client):
        """测试无效的大小参数"""
        with patch("services.stock.data.get_stock_list") as mock:
            mock.return_value = {
                "items": [],
                "total": 0,
                "page": 1,
                "size": -1,
                "pages": 0,
            }
            resp = client.get("/api/stock/list?size=-1")
            assert resp.status_code == 200

    def test_large_size_parameter(self, client):
        """测试过大的大小参数"""
        with patch("services.stock.data.get_stock_list") as mock:
            mock.return_value = {
                "items": [],
                "total": 0,
                "page": 1,
                "size": 1000,
                "pages": 0,
            }
            resp = client.get("/api/stock/list?size=1000")
            assert resp.status_code == 200


class TestErrorHandling:
    """错误处理测试"""

    @patch("services.stock.data.get_stock_data")
    def test_internal_server_error(self, mock_get_data, client):
        """测试内部服务器错误"""
        mock_get_data.side_effect = Exception("数据库连接失败")

        resp = client.get("/api/stock/indicators?symbol=000001")
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["success"] is False
        assert "数据库连接失败" in data["message"]


class TestCaching:
    """缓存测试"""

    def test_cache_functionality(self, client, mock_cache):
        """测试缓存功能"""
        mock_cache.get.return_value = None

        # 设置缓存
        mock_cache.set.return_value = True

        with patch("services.stock.data.get_stock_data") as mock_get_data:
            import pandas as pd
            import numpy as np

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
            mock_get_data.return_value = df

            resp = client.get("/api/stock/indicators?symbol=000001")
            assert resp.status_code == 200

            # 检查缓存是否被设置
            assert mock_cache.set.called
            call_args = mock_cache.set.call_args
            cache_key = call_args[0][0]
            assert "stock:indicators:000001" in cache_key
