#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算
从 analysis.technical 导入并封装
"""

import numpy as np
import pandas as pd

from analysis.technical import (
    calc_aroon,
    calc_bollinger_bands,
    calc_macd,
    calc_mfi,
    calc_obv,
    calc_rsi,
    calc_rolling_volatility,
)


def calculate_ma(close: pd.Series, period: int) -> float:
    """计算移动平均线"""
    return float(close.rolling(window=period).mean().iloc[-1])


def calculate_ema(close: pd.Series, period: int) -> float:
    """计算指数移动平均线"""
    return float(close.ewm(span=period, adjust=False).mean().iloc[-1])


def calculate_macd_values(close: pd.Series) -> dict[str, float]:
    """计算 MACD 指标"""
    macd_dict = calc_macd(close, fast=12, slow=26, signal=9)
    return {
        "macd": float(macd_dict["macd"].iloc[-1]),
        "signal": float(macd_dict["signal"].iloc[-1]),
        "hist": float(macd_dict["hist"].iloc[-1]),
    }


def calculate_rsi_values(close: pd.Series, period: int = 14) -> float:
    """计算 RSI 指标"""
    rsi = calc_rsi(close, period=period)
    return float(rsi.iloc[-1])


def calculate_bollinger_bands_values(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> dict[str, float]:
    """计算布林带"""
    bb = calc_bollinger_bands(close, period=period, num_std=num_std)
    return {
        "mid": float(bb["mid"].iloc[-1]),
        "upper": float(bb["upper"].iloc[-1]),
        "lower": float(bb["lower"].iloc[-1]),
    }


def calculate_volatility(close: pd.Series, window: int = 20) -> float:
    """计算波动率"""
    returns = close.pct_change()
    vol = calc_rolling_volatility(returns, window=window, annualize=True)
    return float(vol.iloc[-1])


def calculate_obv_values(close: pd.Series, volume: pd.Series) -> float:
    """计算能量潮 OBV"""
    obv = calc_obv(close, volume)
    return float(obv.iloc[-1])


def calculate_mfi_values(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> float:
    """计算资金流量指数 MFI"""
    mfi = calc_mfi(high, low, close, volume, period=period)
    return float(mfi.iloc[-1])


def calculate_aroon_values(
    high: pd.Series, low: pd.Series, period: int = 20
) -> dict[str, float]:
    """计算阿隆指标"""
    aroon = calc_aroon(high, low, period=period)
    return {
        "aroon_up": float(aroon["aroon_up"].iloc[-1]),
        "aroon_down": float(aroon["aroon_down"].iloc[-1]),
    }
