# -*- coding: utf-8 -*-
"""
多因子库：200+ 量化因子（动量、波动率、价值、质量、技术、流动性、情绪代理等）

用于集成学习多因子预测：XGBoost / LightGBM / RF 的输入特征。
因子按类别组织，支持批量计算与自动特征选择（RFE）前的特征名列表。
"""
from __future__ import annotations

from typing import Any, Optional

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


# ---------- 列名兼容 ----------
def _col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"未找到列: {candidates}")


def _get_series(df: pd.DataFrame, *candidates: str) -> pd.Series:
    c = _col(df, *candidates)
    return df[c].astype(float)


# ---------- 动量类因子 ----------
def _momentum_roc(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


def _momentum_returns(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period)) - 1.0


def _momentum_ma_ratio(close: pd.Series, period: int) -> pd.Series:
    ma = close.rolling(period).mean()
    return (close - ma) / (ma + 1e-10)


# ---------- 波动率类因子 ----------
def _volatility_std(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------- 价值/质量代理（无基本面时用价量比、收益稳定性等） ----------
def _price_volume_ratio(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """价量比：价格/成交量 的滚动均值，标准化为相对近期比例。"""
    pv = close / (volume + 1)
    return pv / (pv.rolling(window).mean() + 1e-10)


def _earnings_volatility_proxy(returns: pd.Series, window: int) -> pd.Series:
    """收益波动率（质量代理：波动越低越稳定）。"""
    return returns.rolling(window).std()


# ---------- 流动性/成交量 ----------
def _volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    return volume / (volume.rolling(window).mean() + 1e-10)


def _volume_std_ratio(volume: pd.Series, window: int) -> pd.Series:
    return volume / (volume.rolling(window).std() + 1e-10)


def _obv_ratio(obv: pd.Series, window: int) -> pd.Series:
    return obv / (obv.rolling(window).mean().abs() + 1e-10)


# ---------- 情绪代理（无舆情时用价格/成交量异常） ----------
def _sentiment_proxy_return(returns: pd.Series, window: int) -> pd.Series:
    """近期收益符号强度（情绪代理）。"""
    return returns.rolling(window).mean()


def _sentiment_proxy_volume(volume: pd.Series, window: int) -> pd.Series:
    """成交量相对均值的偏离（情绪热度代理）。"""
    mu = volume.rolling(window).mean()
    return (volume - mu) / (mu + 1e-10)


# ---------- 主入口：从 DataFrame 构建全部因子 ----------
def build_factor_library(
    df: pd.DataFrame,
    *,
    include_rsi_periods: tuple[int, ...] = (5, 6, 8, 10, 12, 14, 16, 20, 24),
    include_roc_periods: tuple[int, ...] = (2, 3, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 90, 120),
    include_vol_windows: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25, 30, 40, 60),
    include_bb_periods: tuple[int, ...] = (8, 10, 12, 15, 20, 25, 30),
    include_ma_periods: tuple[int, ...] = (5, 8, 10, 12, 15, 20, 25, 30, 40, 60),
    include_atr_periods: tuple[int, ...] = (7, 14, 21, 28),
    include_volume_windows: tuple[int, ...] = (5, 10, 15, 20, 30),
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    从日线 OHLCV DataFrame 构建因子面板。

    要求列：收盘(close/收盘)、成交量(volume/成交量)、最高(high/最高)、最低(low/最低)、开盘(open/开盘)；
    可选：成交额(amount/成交额)、换手率(turnover_rate)、涨跌幅(change_pct)。

    Returns:
        DataFrame，索引与 df 一致，每列为一个因子；列名为因子名（英文）。
    """
    close = _get_series(df, "收盘", "close", "Close")
    volume = _get_series(df, "成交量", "volume", "Volume")
    high = _get_series(df, "最高", "high", "High")
    low = _get_series(df, "最低", "low", "Low")
    open_ = _get_series(df, "开盘", "open", "Open")

    returns = close.pct_change()
    out = pd.DataFrame(index=df.index)

    # ----- 动量 -----
    for p in include_roc_periods:
        out[f"momentum_roc_{p}"] = _momentum_roc(close, p)
        out[f"momentum_ret_{p}"] = _momentum_returns(close, p)
    for p in include_ma_periods:
        out[f"momentum_ma_ratio_{p}"] = _momentum_ma_ratio(close, p)

    for p in include_rsi_periods:
        rsi = calc_rsi(close, period=p)
        out[f"rsi_{p}"] = rsi

    macd = calc_macd(close, 12, 26, 9)
    out["macd_line"] = macd["macd"]
    out["macd_signal"] = macd["signal"]
    out["macd_hist"] = macd["hist"]

    for fast, slow in [(6, 19), (8, 21), (10, 22), (12, 26)]:
        m = calc_macd(close, fast, slow, 9)
        out[f"macd_hist_{fast}_{slow}"] = m["hist"]

    for aroon_p in (14, 20, 25):
        a = calc_aroon(high, low, period=aroon_p)
        out[f"aroon_up_{aroon_p}"] = a["aroon_up"]
        out[f"aroon_down_{aroon_p}"] = a["aroon_down"]
    aroon = calc_aroon(high, low, period=20)
    out["aroon_up"] = aroon["aroon_up"]
    out["aroon_down"] = aroon["aroon_down"]
    out["aroon_osc"] = aroon["aroon_up"] - aroon["aroon_down"]

    # ----- 波动率 -----
    for w in include_vol_windows:
        vol = _volatility_std(returns, w)
        out[f"volatility_{w}"] = vol
    for p in include_atr_periods:
        out[f"atr_{p}"] = _atr(high, low, close, p)
    atr14 = _atr(high, low, close, 14)
    out["atr_pct"] = atr14 / (close + 1e-10)

    for p in include_bb_periods:
        bb = calc_bollinger_bands(close, period=p, num_std=2.0)
        out[f"bb_upper_{p}"] = bb["upper"]
        out[f"bb_lower_{p}"] = bb["lower"]
        out[f"bb_mid_{p}"] = bb["mid"]
        out[f"bb_width_{p}"] = (bb["upper"] - bb["lower"]) / (bb["mid"] + 1e-10)
        out[f"bb_position_{p}"] = (close - bb["mid"]) / (bb["upper"] - bb["lower"] + 1e-10)

    # ----- 成交量 / OBV / MFI -----
    obv = calc_obv(close, volume)
    out["obv"] = obv
    for w in include_volume_windows:
        out[f"obv_ratio_{w}"] = _obv_ratio(obv, w)
        out[f"volume_ratio_{w}"] = _volume_ratio(volume, w)
        out[f"volume_std_ratio_{w}"] = _volume_std_ratio(volume, w)

    mfi = calc_mfi(high, low, close, volume, period=14)
    out["mfi_14"] = mfi
    for p in (8, 10, 12, 20, 24):
        out[f"mfi_{p}"] = calc_mfi(high, low, close, volume, period=p)

    # ----- 价值/质量代理 -----
    for w in (10, 15, 20, 25, 30, 40, 60):
        out[f"pv_ratio_{w}"] = _price_volume_ratio(close, volume, w)
        out[f"return_vol_{w}"] = _earnings_volatility_proxy(returns, w)

    # ----- 情绪代理 -----
    for w in (3, 5, 6, 8, 10, 12, 15, 20):
        out[f"sentiment_ret_{w}"] = _sentiment_proxy_return(returns, w)
        out[f"sentiment_vol_{w}"] = _sentiment_proxy_volume(volume, w)

    # ----- 价格形态 -----
    for p in include_ma_periods:
        ma = close.rolling(p).mean()
        out[f"close_ma_{p}"] = ma
        out[f"close_to_ma_{p}"] = (close - ma) / (ma + 1e-10)
    for wh in (10, 20, 30, 60):
        out[f"close_high_{wh}"] = close / high.rolling(wh).max()
        out[f"close_low_{wh}"] = close / low.rolling(wh).min()
    out["range_hl"] = (high - low) / (close + 1e-10)
    out["range_oc"] = (close - open_) / (open_ + 1e-10)
    out["body_ratio"] = (close - open_).abs() / (high - low + 1e-10)

    # ----- 换手率（若有） -----
    try:
        tr = _get_series(df, "换手率", "turnover_rate")
        for w in (5, 10, 15, 20):
            out[f"turnover_ma_{w}"] = tr.rolling(w).mean()
            out[f"turnover_std_{w}"] = tr.rolling(w).std()
        out["turnover"] = tr
    except KeyError:
        pass

    # ----- 涨跌幅（若有） -----
    try:
        chg = _get_series(df, "涨跌幅", "change_pct")
        for w in (3, 5, 10, 15, 20):
            out[f"change_pct_cum_{w}"] = chg.rolling(w).sum()
        out["change_pct"] = chg
    except KeyError:
        out["change_pct"] = returns * 100  # 近似
        for w in (3, 15):
            out[f"change_pct_cum_{w}"] = (returns * 100).rolling(w).sum()

    # 填充缺失：先前向再后向
    if fill_method == "ffill":
        out = out.ffill().bfill()
    out = out.fillna(0)

    return out


def get_factor_names_by_category() -> dict[str, list[str]]:
    """返回按类别分组的因子名（与 build_factor_library 默认参数生成的列一致）。"""
    return {
        "momentum": (
            [f"momentum_roc_{p}" for p in (5, 10, 20, 40, 60)]
            + [f"momentum_ret_{p}" for p in (5, 10, 20, 40, 60)]
            + [f"momentum_ma_ratio_{p}" for p in (5, 10, 20, 60)]
            + [f"rsi_{p}" for p in (6, 12, 14, 24)]
            + ["macd_line", "macd_signal", "macd_hist", "macd_hist_6_19", "macd_hist_8_21", "macd_hist_12_26"]
            + ["aroon_up", "aroon_down", "aroon_osc", "aroon_up_14", "aroon_down_14", "aroon_up_20", "aroon_down_20"]
        ),
        "volatility": (
            [f"volatility_{w}" for w in (5, 10, 20, 40, 60)]
            + [f"atr_{p}" for p in (7, 14, 28)]
            + ["atr_pct"]
            + [f"bb_upper_{p}" for p in (10, 20, 30)]
            + [f"bb_lower_{p}" for p in (10, 20, 30)]
            + [f"bb_mid_{p}" for p in (10, 20, 30)]
            + [f"bb_width_{p}" for p in (10, 20, 30)]
            + [f"bb_position_{p}" for p in (10, 20, 30)]
        ),
        "volume_liquidity": (
            ["obv"]
            + [f"obv_ratio_{w}" for w in (5, 10, 20)]
            + [f"volume_ratio_{w}" for w in (5, 10, 20)]
            + [f"volume_std_ratio_{w}" for w in (5, 10, 20)]
            + ["mfi_14", "mfi_10", "mfi_20"]
        ),
        "value_quality": [f"pv_ratio_{w}" for w in (20, 40, 60)] + [f"return_vol_{w}" for w in (20, 40, 60)],
        "sentiment": [f"sentiment_ret_{w}" for w in (5, 10, 20)] + [f"sentiment_vol_{w}" for w in (5, 10, 20)],
        "price_pattern": (
            [f"close_ma_{p}" for p in (5, 10, 20, 60)]
            + [f"close_to_ma_{p}" for p in (5, 10, 20, 60)]
            + ["close_high_20", "close_low_20", "range_hl", "range_oc"]
        ),
        "other": ["change_pct", "change_pct_cum_5", "change_pct_cum_10", "change_pct_cum_20"],
    }


def get_all_factor_names(factor_df: pd.DataFrame) -> list[str]:
    """从已构建的因子 DataFrame 得到全部因子名列表。"""
    return list(factor_df.columns)
