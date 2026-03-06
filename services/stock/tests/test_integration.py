#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock 服务集成测试
测试完整的训练流程、预测流程和 Redis 集成
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

from services.stock.analysis.lstm_training import (
    train_model,
    build_features_from_df,
    SEQ_LEN,
    FORECAST_DAYS,
)
from services.stock.analysis.lstm_predict import predict, load_model


@pytest.fixture
def sample_training_data():
    """创建样本训练数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=600)
    close = np.cumsum(np.random.randn(600)) + 100
    high = close + np.random.rand(600) * 2
    low = close - np.random.rand(600) * 2
    open_price = close + np.random.randn(600) * 0.5
    volume = np.random.randint(1000000, 5000000, 600)

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


@pytest.fixture
def temp_model_dir():
    """临时模型目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCompleteTrainingWorkflow:
    """完整训练流程测试"""

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    def test_complete_training(
        self, mock_get_data, sample_training_data, temp_model_dir
    ):
        """测试完整训练流程"""
        mock_get_data.return_value = sample_training_data

        result = train_model(
            symbol="000001",
            lr=1e-3,
            hidden_size=32,
            epochs=5,
            batch_size=32,
            save_dir=temp_model_dir,
        )

        assert result["symbol"] == "000001"
        assert result["n_samples"] > 0
        assert result["n_features"] > 0
        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert "model_path" in result

        # 检查模型文件是否创建
        model_path = Path(result["model_path"])
        assert model_path.exists()

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    def test_training_with_insufficient_data(self, mock_get_data):
        """测试数据不足的训练"""
        short_data = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=50),
                "收盘": np.arange(50) + 100,
                "最高": np.arange(50) + 102,
                "最低": np.arange(50) + 98,
                "开盘": np.arange(50) + 100,
                "成交量": np.arange(50) + 1000000,
            }
        )
        mock_get_data.return_value = short_data

        with pytest.raises(ValueError) as exc_info:
            train_model(symbol="000001")

        assert "数据不足" in str(exc_info.value)

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    def test_training_metrics_quality(
        self, mock_get_data, sample_training_data, temp_model_dir
    ):
        """测试训练指标质量"""
        mock_get_data.return_value = sample_training_data

        result = train_model(
            symbol="000001",
            lr=1e-3,
            hidden_size=32,
            epochs=10,
            batch_size=32,
            save_dir=temp_model_dir,
        )

        metrics = result["metrics"]

        # 检查指标是否在合理范围内
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert metrics["mse"] >= 0


class TestCompletePredictionWorkflow:
    """完整预测流程测试"""

    @patch("services.stock.analysis.lstm_predict.load_model")
    @patch("services.stock.analysis.lstm_predict.get_stock_data")
    @patch("services.stock.analysis.lstm_predict.build_features_from_df")
    def test_complete_prediction(
        self, mock_build_features, mock_get_data, mock_load_model
    ):
        """测试完整预测流程"""
        # Mock 模型
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        # Mock 前向传播
        direction_logits = torch.tensor([[0.3, 0.7]])
        magnitude_pred = torch.tensor([[0.01, 0.02, 0.01, 0.02, 0.01]])
        mock_model.return_value = (direction_logits, magnitude_pred)

        mock_load_model.return_value = (
            mock_model,
            {"n_features": 10, "hidden_size": 64},
        )

        # Mock 数据
        mock_get_data.return_value = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=100),
                "收盘": np.arange(100) + 100,
                "最高": np.arange(100) + 102,
                "最低": np.arange(100) + 98,
                "开盘": np.arange(100) + 100,
                "成交量": np.arange(100) + 1000000,
            }
        )

        # Mock 特征
        X = np.random.randn(40, SEQ_LEN, 10).astype(np.float32)
        mock_build_features.return_value = (
            X,
            [],
            pd.Series(),
            np.array([]),
            np.array([]),
        )

        result = predict("000001")

        assert result["symbol"] == "000001"
        assert "direction" in result
        assert "confidence" in result
        assert "magnitude" in result
        assert result["direction"] in ["上涨", "下跌"]
        assert 0 <= result["confidence"] <= 1

    @patch("services.stock.analysis.lstm_predict.load_model")
    def test_prediction_model_not_found(self, mock_load_model):
        """测试模型未找到的预测"""
        mock_load_model.side_effect = FileNotFoundError("模型未训练")

        result = predict("000001")

        assert result["symbol"] == "000001"
        assert result["error"] == "模型未训练"
        assert result["direction"] is None
        assert result["confidence"] is None

    @patch("services.stock.analysis.lstm_predict.load_model")
    @patch("services.stock.analysis.lstm_predict.get_stock_data")
    def test_prediction_insufficient_data(self, mock_get_data, mock_load_model):
        """测试数据不足的预测"""
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_model, {"n_features": 10})
        mock_get_data.return_value = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=30),
                "收盘": np.arange(30) + 100,
            }
        )

        result = predict("000001")

        assert result["symbol"] == "000001"
        assert result["error"] == "数据不足"


class TestRedisIntegration:
    """Redis 集成测试"""

    @patch("services.stock.routes.lstm._acquire_lock")
    @patch("services.stock.routes.lstm._release_lock")
    @patch("services.stock.analysis.train_model")
    def test_redis_lock_acquire(self, mock_train, mock_release, mock_lock):
        """测试 Redis 锁获取"""
        from services.stock.app import app

        mock_lock.return_value = True
        mock_train.return_value = {"symbol": "000001", "metrics": {"accuracy": 0.85}}

        with app.test_client() as client:
            resp = client.post("/api/lstm/train", json={"symbol": "000001"})
            assert resp.status_code == 200

        mock_lock.assert_called_once_with("000001")
        mock_release.assert_called_once_with("000001")

    @patch("services.stock.routes.lstm._acquire_lock")
    def test_redis_lock_conflict(self, mock_lock):
        """测试 Redis 锁冲突"""
        from services.stock.app import app

        mock_lock.return_value = False

        with app.test_client() as client:
            resp = client.post("/api/lstm/train", json={"symbol": "000001"})
            assert resp.status_code == 409

        data = resp.get_json()
        assert "Training in progress" in data["message"]

    @patch("services.stock.routes.stock.cache")
    @patch("services.stock.data.get_stock_data")
    def test_redis_cache_set(self, mock_get_data, mock_cache):
        """测试 Redis 缓存设置"""
        from services.stock.app import app

        mock_cache.get.return_value = None
        mock_cache.set.return_value = True

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

        with app.test_client() as client:
            resp = client.get("/api/stock/indicators?symbol=000001")
            assert resp.status_code == 200

        # 检查缓存是否被设置
        assert mock_cache.set.called
        call_args = mock_cache.set.call_args
        assert call_args[0][0] == "stock:indicators:000001"
        assert call_args[1]["ttl"] == 3600

    @patch("services.stock.routes.stock.cache")
    def test_redis_cache_get(self, mock_cache):
        """测试 Redis 缓存获取"""
        from services.stock.app import app

        cached_data = {"symbol": "000001", "ma5": 100.5, "ma10": 101.2, "ma20": 102.1}
        mock_cache.get.return_value = cached_data

        with app.test_client() as client:
            resp = client.get("/api/stock/indicators?symbol=000001")
            assert resp.status_code == 200

        data = resp.get_json()
        assert data["success"] is True
        assert data["cached"] is True
        assert data["data"] == cached_data


class TestEndToEndWorkflow:
    """端到端工作流测试"""

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    @patch("services.stock.analysis.lstm_training.save_lstm_model")
    @patch("services.stock.analysis.lstm_predict.load_model")
    @patch("services.stock.analysis.lstm_predict.get_stock_data")
    @patch("services.stock.analysis.lstm_predict.build_features_from_df")
    def test_train_then_predict(
        self,
        mock_build_features_pred,
        mock_get_data_pred,
        mock_load_model,
        mock_save_model,
        mock_get_data_train,
        sample_training_data,
        temp_model_dir,
    ):
        """测试训练后预测的完整流程"""
        # 1. 训练
        mock_get_data_train.return_value = sample_training_data
        mock_save_model.return_value = "v1"

        train_result = train_model(
            symbol="000001",
            lr=1e-3,
            hidden_size=32,
            epochs=5,
            batch_size=32,
            save_dir=temp_model_dir,
        )

        assert train_result["symbol"] == "000001"

        # 2. 预测
        mock_model = MagicMock()
        mock_model.eval = MagicMock()

        direction_logits = torch.tensor([[0.3, 0.7]])
        magnitude_pred = torch.tensor([[0.01, 0.02, 0.01, 0.02, 0.01]])
        mock_model.return_value = (direction_logits, magnitude_pred)

        mock_load_model.return_value = (
            mock_model,
            {"n_features": 10, "hidden_size": 64},
        )

        mock_get_data_pred.return_value = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=100),
                "收盘": np.arange(100) + 100,
                "最高": np.arange(100) + 102,
                "最低": np.arange(100) + 98,
                "开盘": np.arange(100) + 100,
                "成交量": np.arange(100) + 1000000,
            }
        )

        X = np.random.randn(40, SEQ_LEN, 10).astype(np.float32)
        mock_build_features_pred.return_value = (
            X,
            [],
            pd.Series(),
            np.array([]),
            np.array([]),
        )

        pred_result = predict("000001")

        assert pred_result["symbol"] == "000001"
        assert pred_result["direction"] in ["上涨", "下跌"]
        assert 0 <= pred_result["confidence"] <= 1


class TestConcurrentOperations:
    """并发操作测试"""

    @patch("services.stock.routes.lstm._acquire_lock")
    @patch("services.stock.routes.lstm._release_lock")
    @patch("services.stock.analysis.train_model")
    def test_concurrent_training(self, mock_train, mock_release, mock_lock):
        """测试并发训练"""
        from services.stock.app import app

        lock_calls = []

        def mock_acquire_side_effect(symbol):
            lock_calls.append(("acquire", symbol))
            return True

        def mock_release_side_effect(symbol):
            lock_calls.append(("release", symbol))

        mock_lock.side_effect = mock_acquire_side_effect
        mock_release.side_effect = mock_release_side_effect
        mock_train.return_value = {"symbol": "000001", "metrics": {"accuracy": 0.85}}

        with app.test_client() as client:
            # 模拟两个并发请求
            resp1 = client.post("/api/lstm/train", json={"symbol": "000001"})
            resp2 = client.post("/api/lstm/train", json={"symbol": "000002"})

        assert resp1.status_code == 200
        assert resp2.status_code == 200

        # 验证锁的获取和释放
        assert len(lock_calls) == 4  # 2 acquire + 2 release


class TestErrorRecovery:
    """错误恢复测试"""

    @patch("services.stock.routes.lstm._acquire_lock")
    @patch("services.stock.routes.lstm._release_lock")
    @patch("services.stock.analysis.train_model")
    def test_training_error_lock_release(self, mock_train, mock_release, mock_lock):
        """测试训练错误时释放锁"""
        from services.stock.app import app

        mock_lock.return_value = True
        mock_train.side_effect = Exception("训练失败")

        with app.test_client() as client:
            resp = client.post("/api/lstm/train", json={"symbol": "000001"})
            assert resp.status_code == 500

        # 确保锁被释放
        mock_release.assert_called_once()


class TestModelPersistence:
    """模型持久化测试"""

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    def test_model_save_and_load(
        self, mock_get_data, sample_training_data, temp_model_dir
    ):
        """测试模型保存和加载"""
        mock_get_data.return_value = sample_training_data

        # 训练并保存模型
        train_result = train_model(
            symbol="000001",
            lr=1e-3,
            hidden_size=32,
            epochs=5,
            batch_size=32,
            save_dir=temp_model_dir,
        )

        model_path = train_result["model_path"]

        # 验证模型文件存在
        assert Path(model_path).exists()

        # 加载模型
        model_state = torch.load(model_path)
        assert isinstance(model_state, dict)
        assert len(model_state) > 0

    @patch("services.stock.analysis.lstm_training.get_stock_data")
    def test_model_metadata(self, mock_get_data, sample_training_data, temp_model_dir):
        """测试模型元数据"""
        mock_get_data.return_value = sample_training_data

        result = train_model(
            symbol="000001",
            lr=1e-3,
            hidden_size=64,
            epochs=10,
            batch_size=32,
            save_dir=temp_model_dir,
        )

        # 验证元数据
        assert result["n_features"] > 0
        assert result["n_samples"] > 0
        assert result["metrics"]["accuracy"] >= 0
        assert result["metrics"]["mse"] >= 0
