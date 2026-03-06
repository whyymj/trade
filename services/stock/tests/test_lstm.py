#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 模块单元测试
"""

import pytest
import torch
import numpy as np
import pandas as pd

from services.stock.analysis.lstm_model import LSTMModel, train_epoch, evaluate_model
from services.stock.analysis.lstm_training import build_features_from_df
from services.stock.analysis.indicators import (
    calculate_ma,
    calculate_ema,
    calculate_macd_values,
    calculate_rsi_values,
    calculate_bollinger_bands_values,
    calculate_volatility,
    calculate_obv_values,
    calculate_mfi_values,
    calculate_aroon_values,
)


@pytest.fixture
def sample_stock_data():
    """创建样本股票数据"""
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


class TestLSTMModel:
    """LSTM 模型测试"""

    def test_lstm_model_creation(self):
        """测试 LSTM 模型创建"""
        model = LSTMModel(input_size=10, hidden_size=64, num_layers=2)
        assert model.hidden_size == 64
        assert model.num_layers == 2

    def test_lstm_model_forward_single(self):
        """测试单样本前向传播"""
        model = LSTMModel(input_size=10, hidden_size=64, num_layers=2)
        x = torch.randn(1, 60, 10)
        direction_logits, magnitude = model(x)

        assert direction_logits.shape == (1, 2)
        assert magnitude.shape == (1, 5)
        assert not torch.isnan(direction_logits).any()
        assert not torch.isnan(magnitude).any()

    def test_lstm_model_forward_batch(self):
        """测试批量前向传播"""
        model = LSTMModel(input_size=10, hidden_size=64, num_layers=2)
        x = torch.randn(8, 60, 10)
        direction_logits, magnitude = model(x)

        assert direction_logits.shape == (8, 2)
        assert magnitude.shape == (8, 5)

    def test_lstm_model_no_nan_inf(self):
        """测试输出无 NaN 或 Inf"""
        model = LSTMModel(input_size=5, hidden_size=32, num_layers=1)
        x = torch.randn(2, 60, 5)
        direction_logits, magnitude = model(x)

        assert not torch.isnan(direction_logits).any()
        assert not torch.isnan(magnitude).any()
        assert not torch.isinf(direction_logits).any()
        assert not torch.isinf(magnitude).any()

    def test_lstm_model_different_configs(self):
        """测试不同配置的模型"""
        configs = [
            (10, 32, 1),
            (15, 64, 2),
            (20, 128, 3),
        ]
        for input_size, hidden_size, num_layers in configs:
            model = LSTMModel(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
            )
            x = torch.randn(1, 60, input_size)
            direction_logits, magnitude = model(x)
            assert direction_logits.shape == (1, 2)
            assert magnitude.shape == (1, 5)


class TestIndicators:
    """技术指标计算测试"""

    def test_calculate_ma(self, sample_stock_data):
        """测试移动平均线"""
        close = sample_stock_data["收盘"]
        ma5 = calculate_ma(close, 5)
        ma20 = calculate_ma(close, 20)

        assert isinstance(ma5, float)
        assert isinstance(ma20, float)
        assert not np.isnan(ma5)
        assert not np.isnan(ma20)

    def test_calculate_ema(self, sample_stock_data):
        """测试指数移动平均线"""
        close = sample_stock_data["收盘"]
        ema12 = calculate_ema(close, 12)
        ema26 = calculate_ema(close, 26)

        assert isinstance(ema12, float)
        assert isinstance(ema26, float)
        assert not np.isnan(ema12)
        assert not np.isnan(ema26)

    def test_calculate_macd(self, sample_stock_data):
        """测试 MACD 指标"""
        close = sample_stock_data["收盘"]
        macd_values = calculate_macd_values(close)

        assert isinstance(macd_values, dict)
        assert "macd" in macd_values
        assert "signal" in macd_values
        assert "hist" in macd_values
        assert all(isinstance(v, float) for v in macd_values.values())

    def test_calculate_rsi(self, sample_stock_data):
        """测试 RSI 指标"""
        close = sample_stock_data["收盘"]
        rsi = calculate_rsi_values(close, 14)

        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100
        assert not np.isnan(rsi)

    def test_calculate_bollinger_bands(self, sample_stock_data):
        """测试布林带"""
        close = sample_stock_data["收盘"]
        bb = calculate_bollinger_bands_values(close, 20, 2.0)

        assert isinstance(bb, dict)
        assert "mid" in bb
        assert "upper" in bb
        assert "lower" in bb
        assert bb["upper"] > bb["mid"]
        assert bb["mid"] > bb["lower"]

    def test_calculate_volatility(self, sample_stock_data):
        """测试波动率"""
        close = sample_stock_data["收盘"]
        vol = calculate_volatility(close, 20)

        assert isinstance(vol, float)
        assert vol >= 0
        assert not np.isnan(vol)

    def test_calculate_obv(self, sample_stock_data):
        """测试 OBV"""
        close = sample_stock_data["收盘"]
        volume = sample_stock_data["成交量"]
        obv = calculate_obv_values(close, volume)

        assert isinstance(obv, float)
        assert not np.isnan(obv)

    def test_calculate_mfi(self, sample_stock_data):
        """测试 MFI"""
        high = sample_stock_data["最高"]
        low = sample_stock_data["最低"]
        close = sample_stock_data["收盘"]
        volume = sample_stock_data["成交量"]
        mfi = calculate_mfi_values(high, low, close, volume, 14)

        assert isinstance(mfi, float)
        assert 0 <= mfi <= 100
        assert not np.isnan(mfi)

    def test_calculate_aroon(self, sample_stock_data):
        """测试 Aroon 指标"""
        high = sample_stock_data["最高"]
        low = sample_stock_data["最低"]
        aroon = calculate_aroon_values(high, low, 20)

        assert isinstance(aroon, dict)
        assert "aroon_up" in aroon
        assert "aroon_down" in aroon
        assert 0 <= aroon["aroon_up"] <= 100
        assert 0 <= aroon["aroon_down"] <= 100


class TestFeatureBuilding:
    """特征构建测试"""

    def test_build_features_success(self, sample_stock_data):
        """测试成功构建特征"""
        X, feature_names, dates, y_dir, y_mag = build_features_from_df(
            sample_stock_data
        )

        assert len(X) > 0
        assert X.shape[1] == 60
        assert X.shape[2] == 10
        assert len(feature_names) == 10
        assert len(y_dir) == len(X)
        assert len(y_mag) == len(X)

    def test_build_features_insufficient_data(self):
        """测试数据不足的情况"""
        short_df = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=50),
                "收盘": np.cumsum(np.random.randn(50)) + 100,
                "最高": np.arange(50) + 102,
                "最低": np.arange(50) + 98,
                "开盘": np.arange(50) + 100,
                "成交量": np.arange(50) + 1000000,
            }
        )

        X, _, _, _, _ = build_features_from_df(short_df)

        assert X.shape[0] == 0

    def test_build_features_empty_data(self):
        """测试空数据"""
        # 至少需要一些数据来创建空的 Series
        empty_df = pd.DataFrame(
            {
                "日期": pd.date_range(start="2024-01-01", periods=1),
                "收盘": [100.0],
                "最高": [102.0],
                "最低": [98.0],
                "开盘": [100.0],
                "成交量": [1000000],
            }
        )

        X, _, _, _, _ = build_features_from_df(empty_df)

        assert X.shape[0] == 0


class TestTrainingAndEvaluation:
    """训练和评估测试"""

    def test_train_epoch(self):
        """测试训练一个 epoch"""
        model = LSTMModel(input_size=10, hidden_size=32, num_layers=2)

        X = torch.randn(100, 60, 10)
        y_dir = torch.randint(0, 2, (100,))
        y_mag = torch.randn(100, 5)

        dataset = torch.utils.data.TensorDataset(X, y_dir, y_mag)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        loss = train_epoch(model, loader, optimizer, device)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluate_model(self):
        """测试模型评估"""
        model = LSTMModel(input_size=10, hidden_size=32, num_layers=2)
        model.eval()

        X = torch.randn(50, 60, 10)
        y_dir = torch.randint(0, 2, (50,))
        y_mag = torch.randn(50, 5)

        dataset = torch.utils.data.TensorDataset(X, y_dir, y_mag)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        device = torch.device("cpu")

        metrics = evaluate_model(model, loader, device)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "mse" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert metrics["mse"] >= 0


class TestPredict:
    """预测功能测试"""

    def test_prediction_output_format(self):
        """测试预测输出格式"""
        from services.stock.analysis.lstm_predict import predict

        # 测试未训练模型的预测
        result = predict("999999")

        assert isinstance(result, dict)
        assert "symbol" in result
        assert "direction" in result
        assert "confidence" in result
        assert result["symbol"] == "999999"
