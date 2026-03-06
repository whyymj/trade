#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 训练模块
提供模型训练功能
"""

import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from services.stock.analysis.lstm_model import (
    LSTMModel,
    evaluate_model,
    train_epoch,
)
from analysis.technical import (
    calc_bollinger_bands,
    calc_macd,
    calc_mfi,
    calc_obv,
    calc_rsi,
    calc_rolling_volatility,
)
from data.stock_repo import get_stock_daily_df as repo_get_stock_data

SEQ_LEN = 60
FORECAST_DAYS = 5


def build_features_from_df(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.Series, np.ndarray, np.ndarray]:
    """从 DataFrame 构建 LSTM 特征

    Returns:
        (X, feature_names, y_info, y_direction, y_magnitude)
    """
    close_col = "收盘" if "收盘" in df.columns else "close"
    vol_col = "成交量" if "成交量" in df.columns else "volume"
    high_col = "最高" if "最高" in df.columns else "high"
    low_col = "最低" if "最低" in df.columns else "low"
    open_col = "开盘" if "开盘" in df.columns else "open"

    def _to_series(s):
        if isinstance(s, pd.DataFrame):
            out = s.squeeze(axis=1)
            return out if isinstance(out, pd.Series) else pd.Series(out)
        return s

    close = _to_series(df[close_col].astype(float))
    volume = _to_series(df[vol_col].astype(float))
    high = _to_series(df[high_col].astype(float))
    low = _to_series(df[low_col].astype(float))
    open_ = _to_series(df[open_col].astype(float))

    rsi = calc_rsi(close, period=14)
    macd = calc_macd(close, 12, 26, 9)
    bb = calc_bollinger_bands(close, period=20, num_std=2.0)
    returns = close.pct_change()
    vol = calc_rolling_volatility(returns, window=20, annualize=True)
    obv = calc_obv(close, volume)
    mfi = calc_mfi(high, low, close, volume, period=14)

    close_norm = (close - close.rolling(SEQ_LEN, min_periods=1).min()) / (
        close.rolling(SEQ_LEN, min_periods=1).max()
        - close.rolling(SEQ_LEN, min_periods=1).min()
        + 1e-8
    )

    vol_min = volume.rolling(SEQ_LEN, min_periods=1).min()
    vol_max = volume.rolling(SEQ_LEN, min_periods=1).max()
    volume_norm = (volume - vol_min) / (vol_max - vol_min + 1e-8)

    bb_mid, bb_upper, bb_lower = bb["mid"], bb["upper"], bb["lower"]
    bb_position = (close - bb_mid) / (bb_upper - bb_lower + 1e-8)

    obv_min = obv.rolling(SEQ_LEN, min_periods=1).min()
    obv_max = obv.rolling(SEQ_LEN, min_periods=1).max()
    obv_norm = (obv - obv_min) / (obv_max - obv_min + 1e-8)

    rsi = rsi.fillna(50)
    macd_hist = macd["hist"].fillna(0)
    bb_position = bb_position.fillna(0)
    vol = vol.fillna(0)
    obv_norm = obv_norm.fillna(0.5)
    mfi = mfi.fillna(50)

    n = len(df)
    need = SEQ_LEN + FORECAST_DAYS
    if n < need:
        return (
            np.zeros((0, SEQ_LEN, 10)),
            [
                "close_norm",
                "volume_norm",
                "rsi",
                "macd_hist",
                "bb_position",
                "volatility",
                "obv_norm",
                "mfi",
                "aroon_up",
                "volatility_up",
            ],
            pd.Series(dtype=object),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    feature_list = [
        close_norm.values,
        volume_norm.values,
        np.asarray(rsi.values, dtype=np.float64) / 100.0,
        np.asarray(macd_hist.values, dtype=np.float64),
        np.asarray(bb_position.values, dtype=np.float64),
        np.asarray(vol.values, dtype=np.float64),
        obv_norm.values,
        np.asarray(mfi.values, dtype=np.float64) / 100.0,
        np.zeros_like(close.values) * 0.5,
        np.zeros_like(close.values) * 0.5,
    ]

    F = np.stack(feature_list, axis=1)

    X_list = []
    y_direction_list = []
    y_magnitude_list = []
    end_dates = []

    for i in range(SEQ_LEN, n - FORECAST_DAYS):
        X_list.append(F[i - SEQ_LEN : i])

        daily_rets = []
        for d in range(FORECAST_DAYS):
            ret_d = (float(close.iloc[i + d]) / float(close.iloc[i + d - 1])) - 1.0
            daily_rets.append(ret_d)

        cum_ret = (
            float(close.iloc[i + FORECAST_DAYS - 1]) / float(close.iloc[i - 1])
        ) - 1.0
        y_direction_list.append(1 if cum_ret > 0 else 0)
        y_magnitude_list.append(daily_rets)

        date_val = df["日期"].iloc[i] if "日期" in df.columns else df.index[i]
        end_dates.append(date_val)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_dir = np.array(y_direction_list, dtype=np.int64)
    y_mag = np.array(y_magnitude_list, dtype=np.float32)

    feature_names = [
        "close_norm",
        "volume_norm",
        "rsi",
        "macd_hist",
        "bb_position",
        "volatility",
        "obv_norm",
        "mfi",
        "aroon_up",
        "volatility_up",
    ]

    return X, feature_names, pd.Series(end_dates), y_dir, y_mag


def get_stock_data(symbol: str, days: int = 500):
    """获取股票数据"""
    return repo_get_stock_data(symbol, end_date=None, start_date=None)


def train_model(
    symbol: str,
    *,
    lr: float = 5e-4,
    hidden_size: int = 64,
    epochs: int = 30,
    batch_size: int = 32,
    save_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    stop_event: Optional[Any] = None,
) -> dict[str, Any]:
    """训练 LSTM 模型

    Args:
        symbol: 股票代码
        lr: 学习率
        hidden_size: 隐藏层大小
        epochs: 训练轮数
        batch_size: 批大小
        save_dir: 保存目录
        device: 设备
        stop_event: 停止事件

    Returns:
        训练结果字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = get_stock_data(symbol, days=500)
    if df is None or len(df) < SEQ_LEN + FORECAST_DAYS:
        raise ValueError(f"数据不足，需要至少 {SEQ_LEN + FORECAST_DAYS} 个交易日数据")

    X, feature_names, _, y_dir, y_mag = build_features_from_df(df)
    if len(X) == 0:
        raise ValueError("无法构建特征，样本不足")

    n_features = X.shape[2]
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_dir),
        torch.from_numpy(y_mag),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=2,
        n_magnitude_outputs=FORECAST_DAYS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_loss_history = []
    for epoch in range(epochs):
        if stop_event is not None and stop_event.is_set():
            break

        loss = train_epoch(
            model,
            loader,
            optimizer,
            device,
            weight_cls=0.9,
            weight_reg=4.0,
        )
        training_loss_history.append(loss)

    metrics = evaluate_model(model, loader, device)

    save_dir = Path(save_dir or "models/lstm")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{symbol}_lstm.pt"
    torch.save(model.state_dict(), model_path)

    try:
        from data.lstm_repo import save_lstm_model

        version_id = save_lstm_model(
            symbol=symbol,
            model_path=str(model_path),
            feature_names=feature_names,
            metadata={
                "hidden_size": hidden_size,
                "n_features": n_features,
                "lr": lr,
                "epochs": epochs,
                "metrics": metrics,
            },
        )
    except Exception:
        version_id = None

    return {
        "symbol": symbol,
        "version_id": version_id,
        "model_path": str(model_path),
        "metrics": metrics,
        "n_samples": len(X),
        "n_features": n_features,
    }
