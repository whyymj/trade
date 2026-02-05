# -*- coding: utf-8 -*-
"""
预测回退机制：LSTM -> ARIMA -> 技术指标，确保在任一模型异常时仍可返回预测。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 未来 5 日预测
FORECAST_DAYS = 5


def _predict_arima(close_series: pd.Series) -> Optional[dict[str, Any]]:
    """用 ARIMA 预测未来 5 日价格，推导方向与涨跌幅。"""
    try:
        from analysis.arima_model import build_arima_model
    except ImportError:
        return None
    if len(close_series) < 30:
        return None
    close_series = close_series.dropna().sort_index()
    try:
        result = build_arima_model(
            close_series,
            forecast_days=FORECAST_DAYS,
            show_plots=False,
            verbose=False,
        )
    except Exception as e:
        logger.debug("ARIMA 预测失败: %s", e)
        return None
    forecast = result.get("forecast")
    if forecast is None or forecast.empty:
        return None
    pred_values = forecast["预测值"] if "预测值" in forecast.columns else forecast.iloc[:, 0]
    if len(pred_values) < FORECAST_DAYS:
        return None
    current = float(close_series.iloc[-1])
    end_price = float(pred_values.iloc[-1])
    magnitude = (end_price / current) - 1.0
    direction = 1 if magnitude > 0 else 0
    return {
        "direction": direction,
        "magnitude": magnitude,
        "prob_up": 0.6 if direction == 1 else 0.4,
        "source": "arima",
    }


def _predict_technical(close: pd.Series, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None, volume: Optional[pd.Series] = None) -> dict[str, Any]:
    """基于技术指标给出简单方向与幅度（备胎）。"""
    from analysis.technical import calc_rsi, calc_macd
    close = close.dropna()
    if len(close) < 20:
        return {"direction": 0, "magnitude": 0.0, "prob_up": 0.5, "source": "technical"}
    rsi = calc_rsi(close, period=14)
    macd = calc_macd(close, 12, 26, 9)
    rsi_last = float(rsi.iloc[-1]) if not rsi.empty and not np.isnan(rsi.iloc[-1]) else 50.0
    hist_last = float(macd["hist"].iloc[-1]) if not macd["hist"].empty and not np.isnan(macd["hist"].iloc[-1]) else 0.0
    # 简单规则：RSI 超卖(<30) 偏多，超买(>70) 偏空；MACD 柱为正偏多
    if rsi_last < 35:
        direction = 1
        prob_up = 0.6
    elif rsi_last > 65:
        direction = 0
        prob_up = 0.35
    else:
        direction = 1 if hist_last > 0 else 0
        prob_up = 0.55 if direction == 1 else 0.45
    # 幅度用近期波动近似
    ret = close.pct_change().dropna()
    magnitude = float(ret.tail(20).std() * np.sqrt(FORECAST_DAYS)) if len(ret) >= 20 else 0.02
    magnitude = max(-0.15, min(0.15, magnitude))
    return {"direction": direction, "magnitude": magnitude, "prob_up": prob_up, "source": "technical"}


def predict_with_fallback(
    symbol: str,
    df: pd.DataFrame,
    load_model_fn: Callable[..., Any],
    build_features_fn: Callable[..., Any],
    predict_lstm_fn: Callable[[Any, Any], tuple[int, float, float]],
    *,
    save_dir: Optional[Any] = None,
) -> dict[str, Any]:
    """
    带回退的预测：先尝试 LSTM，失败则 ARIMA，再失败则技术指标。
    返回 { "symbol", "direction", "direction_label", "magnitude", "prob_up", "prob_down", "source" }。
    """
    close_col = "收盘" if "收盘" in df.columns else "close"
    close = df[close_col].astype(float)
    high = df["最高"] if "最高" in df.columns else (df["high"] if "high" in df.columns else None)
    low = df["最低"] if "最低" in df.columns else (df["low"] if "low" in df.columns else None)
    volume = df["成交量"] if "成交量" in df.columns else (df["volume"] if "volume" in df.columns else None)

    # 1. 尝试 LSTM
    try:
        model, metadata = load_model_fn(save_dir=save_dir)
        X, _, _, _, _ = build_features_fn(df)
        if len(X) > 0:
            direction, magnitude_val, prob_up = predict_lstm_fn(model, X)
            return {
                "symbol": symbol,
                "direction": direction,
                "direction_label": "涨" if direction == 1 else "跌",
                "magnitude": round(magnitude_val, 6),
                "prob_up": round(prob_up, 4),
                "prob_down": round(1 - prob_up, 4),
                "source": "lstm",
            }
    except Exception as e:
        logger.warning("LSTM 预测失败，尝试回退: %s", e)

    # 2. 尝试 ARIMA
    out = _predict_arima(close)
    if out is not None:
        out["symbol"] = symbol
        out["direction_label"] = "涨" if out["direction"] == 1 else "跌"
        out["prob_down"] = round(1 - out["prob_up"], 4)
        out["prob_up"] = round(out["prob_up"], 4)
        out["magnitude"] = round(out["magnitude"], 6)
        return out

    # 3. 技术指标
    out = _predict_technical(close, high, low, volume)
    out["symbol"] = symbol
    out["direction_label"] = "涨" if out["direction"] == 1 else "跌"
    out["prob_down"] = round(1 - out["prob_up"], 4)
    out["prob_up"] = round(out["prob_up"], 4)
    out["magnitude"] = round(out["magnitude"], 6)
    return out
