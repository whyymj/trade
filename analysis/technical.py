#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标与风险分析

- RSI（相对强弱指数）
- MACD（指数平滑异同移动平均）
- 布林带（Bollinger Bands）
- 滚动波动率（Rolling Volatility）
- VaR（在险价值，历史法）
"""

from typing import Any

import numpy as np
import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 RSI（相对强弱指数）。

    RSI = 100 - 100 / (1 + RS)，RS = 平均涨幅 / 平均跌幅（通常用 Wilder 平滑）。

    Args:
        close: 收盘价序列
        period: 周期（默认 14）

    Returns:
        RSI 序列，前 period 个为 NaN
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """
    计算 MACD 线、信号线、柱状图。

    MACD = EMA(fast) - EMA(slow), Signal = EMA(MACD, signal), Hist = MACD - Signal.

    Returns:
        dict: macd, signal, hist 三个 Series
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist_line}


def calc_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    """
    计算布林带：中轨=均线，上轨=中轨+num_std*标准差，下轨=中轨-num_std*标准差。

    Returns:
        dict: mid, upper, lower 三个 Series
    """
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return {"mid": mid, "upper": upper, "lower": lower}


def calc_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
) -> pd.Series:
    """
    滚动波动率（收益率标准差）。可选年化（乘以 sqrt(252)）。

    Args:
        returns: 日收益率序列
        window: 滚动窗口
        annualize: 是否年化

    Returns:
        滚动波动率序列
    """
    vol = returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def calc_var_historical(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    历史法 VaR：在险价值，即收益分布的第 alpha 分位数（负值表示亏损）。

    例如 alpha=0.05 表示 95% 置信下日最大亏损约为 |VaR|。

    Args:
        returns: 日收益率序列
        alpha: 显著性水平（默认 0.05 即 95% 置信）

    Returns:
        VaR 值（负数，表示亏损）
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float(np.percentile(r, alpha * 100))


def calc_cvar_historical(
    returns: pd.Series,
    alpha: float = 0.05,
) -> float:
    """
    历史法 CVaR（Expected Shortfall）：给定 VaR 条件下，超过 VaR 的损失的平均值。

    Args:
        returns: 日收益率序列
        alpha: 与 VaR 一致

    Returns:
        CVaR（负数）
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    var = np.percentile(r, alpha * 100)
    tail = r[r <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def analyze_technical(
    prices: pd.Series,
    returns: pd.Series,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    vol_window: int = 20,
    var_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    汇总技术指标与风险指标，便于报告和前端绘图。

    Returns:
        dict: 含 rsi, macd, bollinger, rolling_volatility, var, cvar，以及当前值（last）等
    """
    rsi = calc_rsi(prices, period=rsi_period)
    macd = calc_macd(prices, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    bb = calc_bollinger_bands(prices, period=bb_period, num_std=bb_std)
    vol = calc_rolling_volatility(returns, window=vol_window, annualize=True)
    var = calc_var_historical(returns, alpha=var_alpha)
    cvar = calc_cvar_historical(returns, alpha=var_alpha)

    return {
        "rsi": rsi,
        "macd": macd,
        "bollinger": bb,
        "rolling_volatility": vol,
        "var_95": var,
        "cvar_95": cvar,
        "last": {
            "rsi": float(rsi.iloc[-1]) if not rsi.dropna().empty else None,
            "macd": float(macd["macd"].iloc[-1]) if not macd["macd"].dropna().empty else None,
            "macd_signal": float(macd["signal"].iloc[-1]) if not macd["signal"].dropna().empty else None,
            "macd_hist": float(macd["hist"].iloc[-1]) if not macd["hist"].dropna().empty else None,
            "bb_upper": float(bb["upper"].iloc[-1]) if not bb["upper"].dropna().empty else None,
            "bb_mid": float(bb["mid"].iloc[-1]) if not bb["mid"].dropna().empty else None,
            "bb_lower": float(bb["lower"].iloc[-1]) if not bb["lower"].dropna().empty else None,
            "volatility_annual": float(vol.iloc[-1]) if not vol.dropna().empty else None,
            "var_95": var,
            "cvar_95": cvar,
        },
    }
