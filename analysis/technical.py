#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标与风险分析

- RSI（相对强弱指数）
- MACD（指数平滑异同移动平均）
- 布林带（Bollinger Bands）
- 滚动波动率（Rolling Volatility）
- VaR（在险价值，历史法）
- OBV（能量潮）
- MFI（资金流量指数）
- Aroon（阿隆指标）
- 资金流向（大单/中单/小单净流入，基于成交额分档近似）
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


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV（能量潮）：收盘价上涨加成交量，下跌减成交量，平盘不变。

    Args:
        close: 收盘价序列
        volume: 成交量序列

    Returns:
        OBV 序列（累计）
    """
    delta = close.diff()
    direction = np.sign(delta)
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    return obv


def calc_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    MFI（资金流量指数）：典型价格 = (H+L+C)/3，原始资金流 = 典型价格×成交量，
    14 日内正负资金流比 → 0～100。>80 超买，<20 超卖。

    Returns:
        MFI 序列，前 period 个为 NaN
    """
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume
    delta = close.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0.0)
    negative_flow = raw_money_flow.where(delta < 0, 0.0)
    pos_sum = positive_flow.rolling(window=period).sum()
    neg_sum = negative_flow.rolling(window=period).sum()
    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mf_ratio))
    return mfi


def calc_aroon(high: pd.Series, low: pd.Series, period: int = 20) -> dict[str, pd.Series]:
    """
    阿隆指标（Aroon）：period 内最高价/最低价距今的天数，换算为 0～100。
    Aroon Up = (period - 距 period 内最高价的天数) / period * 100
    Aroon Down = (period - 距 period 内最低价的天数) / period * 100

    Returns:
        dict: aroon_up, aroon_down
    """
    n = len(high)
    aroon_up = pd.Series(index=high.index, dtype=float)
    aroon_down = pd.Series(index=low.index, dtype=float)
    for i in range(period - 1, n):
        window_high = high.iloc[i - period + 1 : i + 1]
        window_low = low.iloc[i - period + 1 : i + 1]
        # 窗口内最高/最低点的位置（0=窗口首日，period-1=当日）
        pos_high = int(window_high.values.argmax())
        pos_low = int(window_low.values.argmin())
        days_since_high = period - 1 - pos_high
        days_since_low = period - 1 - pos_low
        aroon_up.iloc[i] = (period - days_since_high) / period * 100
        aroon_down.iloc[i] = (period - days_since_low) / period * 100
    aroon_up.iloc[: period - 1] = np.nan
    aroon_down.iloc[: period - 1] = np.nan
    return {"aroon_up": aroon_up, "aroon_down": aroon_down}


def calc_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    amount: pd.Series | None = None,
) -> dict[str, Any]:
    """
    资金流向：基于典型价格×成交量计算正/负资金流；若无逐笔数据则按成交额分档近似大单/中单/小单。

    - 每日净流入 = 当日典型价格×成交量，收盘涨为正、跌为负。
    - 大单/中单/小单：按当日成交额在区间内的分位数分档（如 0～30% 小单，30%～70% 中单，70%～100% 大单），
      分别汇总各档的带符号资金流作为净流入近似（需 level-2 数据才能得到真实大单/中单/小单）。

    Returns:
        dict: net_flow (每日净流入序列), cumulative_net (累计净流入),
              tier_net (大单/中单/小单净流入标量), tier_series (各档每日净流入序列，供绘图)
    """
    typical_price = (high + low + close) / 3.0
    raw_mf = typical_price * volume
    delta = close.diff()
    signed = np.where(delta > 0, raw_mf, np.where(delta < 0, -raw_mf, 0.0))
    net_flow = pd.Series(signed, index=close.index)
    cumulative_net = net_flow.cumsum()

    tier_net = {"大单": None, "中单": None, "小单": None}
    tier_series = {"大单": None, "中单": None, "小单": None}

    if amount is not None and len(amount.dropna()) >= 3:
        amt = amount.fillna(0)
        q30 = amt.quantile(0.30)
        q70 = amt.quantile(0.70)
        small_mask = amt <= q30
        mid_mask = (amt > q30) & (amt <= q70)
        large_mask = amt > q70
        tier_series["小单"] = net_flow.where(small_mask, 0.0)
        tier_series["中单"] = net_flow.where(mid_mask, 0.0)
        tier_series["大单"] = net_flow.where(large_mask, 0.0)
        tier_net["小单"] = float(tier_series["小单"].sum())
        tier_net["中单"] = float(tier_series["中单"].sum())
        tier_net["大单"] = float(tier_series["大单"].sum())

    return {
        "net_flow": net_flow,
        "cumulative_net": cumulative_net,
        "tier_net": tier_net,
        "tier_series": tier_series,
    }


def _signals_overbought_oversold(
    rsi: pd.Series | None,
    mfi: pd.Series | None,
    aroon_up: pd.Series | None,
    aroon_down: pd.Series | None,
) -> dict[str, Any]:
    """
    根据 RSI、MFI、Aroon 当前值生成超买超卖信号。

    Returns:
        dict: rsi_signal, mfi_signal, aroon_signal, combined (简要文字)
    """
    def last(s: pd.Series | None):
        if s is None or s.dropna().empty:
            return None
        return float(s.iloc[-1])

    rsi_val = last(rsi)
    mfi_val = last(mfi)
    aup = last(aroon_up)
    adown = last(aroon_down)

    rsi_signal = None
    if rsi_val is not None:
        if rsi_val >= 70:
            rsi_signal = "超买"
        elif rsi_val <= 30:
            rsi_signal = "超卖"
        else:
            rsi_signal = "中性"

    mfi_signal = None
    if mfi_val is not None:
        if mfi_val >= 80:
            mfi_signal = "超买"
        elif mfi_val <= 20:
            mfi_signal = "超卖"
        else:
            mfi_signal = "中性"

    aroon_signal = None
    if aup is not None and adown is not None:
        if aup >= 70 and adown <= 30:
            aroon_signal = "强势"
        elif adown >= 70 and aup <= 30:
            aroon_signal = "弱势"
        elif aup > adown:
            aroon_signal = "偏多"
        else:
            aroon_signal = "偏空"

    combined = []
    if rsi_signal and rsi_signal != "中性":
        combined.append(f"RSI{rsi_signal}")
    if mfi_signal and mfi_signal != "中性":
        combined.append(f"MFI{mfi_signal}")
    if aroon_signal:
        combined.append(f"Aroon{aroon_signal}")

    return {
        "rsi_signal": rsi_signal,
        "mfi_signal": mfi_signal,
        "aroon_signal": aroon_signal,
        "combined": ", ".join(combined) if combined else "中性",
    }


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


def _resample_ohlcav(df: pd.DataFrame, rule: str = "W-MON") -> pd.DataFrame:
    """
    将日线 OHLCV 重采样为周线（或其它周期）。
    rule: 'W-MON' 周线（周一为终点）, 'M' 月线等。
    """
    agg = {
        "开盘": "first",
        "最高": "max",
        "最低": "min",
        "收盘": "last",
        "成交量": "sum",
    }
    if "成交额" in df.columns:
        agg["成交额"] = "sum"
    cols = [c for c in agg if c in df.columns]
    resampled = df[cols].resample(rule).agg({c: agg[c] for c in cols})
    return resampled.dropna(how="all")


def analyze_technical(
    prices: pd.Series,
    returns: pd.Series,
    df: pd.DataFrame | None = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    vol_window: int = 20,
    var_alpha: float = 0.05,
    mfi_period: int = 14,
    aroon_period: int = 20,
    include_weekly: bool = True,
) -> dict[str, Any]:
    """
    汇总技术指标与风险指标，便于报告和前端绘图。
    若传入 df（含 收盘、成交量、最高、最低，可选 成交额），则同时计算 OBV、MFI、Aroon、资金流向；
    并可选择包含周线周期指标（include_weekly=True）。

    Returns:
        dict: 含 rsi, macd, bollinger, rolling_volatility, var, cvar；
              若有 df 则还有 obv, mfi, aroon, money_flow；
              by_timeframe.daily / by_timeframe.weekly（周线仅含价量相关指标）；
              last、signals（超买超卖）等
    """
    rsi = calc_rsi(prices, period=rsi_period)
    macd = calc_macd(prices, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    bb = calc_bollinger_bands(prices, period=bb_period, num_std=bb_std)
    vol = calc_rolling_volatility(returns, window=vol_window, annualize=True)
    var = calc_var_historical(returns, alpha=var_alpha)
    cvar = calc_cvar_historical(returns, alpha=var_alpha)

    last = {
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
    }

    obv = mfi = aroon = money_flow = None
    weekly_result = None

    if df is not None and not df.empty:
        # 列名兼容：中文或英文
        close_col = "收盘" if "收盘" in df.columns else "close"
        high_col = "最高" if "最高" in df.columns else "high"
        low_col = "最低" if "最低" in df.columns else "low"
        vol_col = "成交量" if "成交量" in df.columns else "volume"
        amount_col = "成交额" if "成交额" in df.columns else "amount"
        close_s = df[close_col].astype(float)
        high_s = df[high_col].astype(float)
        low_s = df[low_col].astype(float)
        volume_s = df[vol_col].astype(float)
        amount_s = df[amount_col].astype(float) if amount_col in df.columns else None

        obv = calc_obv(close_s, volume_s)
        mfi = calc_mfi(high_s, low_s, close_s, volume_s, period=mfi_period)
        aroon = calc_aroon(high_s, low_s, period=aroon_period)
        money_flow = calc_money_flow(high_s, low_s, close_s, volume_s, amount_s)

        last["obv"] = float(obv.iloc[-1]) if not obv.dropna().empty else None
        last["mfi"] = float(mfi.iloc[-1]) if not mfi.dropna().empty else None
        last["aroon_up"] = float(aroon["aroon_up"].iloc[-1]) if not aroon["aroon_up"].dropna().empty else None
        last["aroon_down"] = float(aroon["aroon_down"].iloc[-1]) if not aroon["aroon_down"].dropna().empty else None
        last["money_flow_cumulative"] = float(money_flow["cumulative_net"].iloc[-1]) if not money_flow["cumulative_net"].dropna().empty else None
        last["money_flow_tier"] = money_flow["tier_net"]

        signals = _signals_overbought_oversold(
            rsi, mfi, aroon["aroon_up"], aroon["aroon_down"]
        )
        last["signals"] = signals

        # 周线周期
        if include_weekly and len(df) >= 10:
            try:
                df_week = _resample_ohlcav(df, rule="W-MON")
                if len(df_week) >= 5:
                    cw = df_week[close_col].astype(float)
                    hw = df_week[high_col].astype(float)
                    lw = df_week[low_col].astype(float)
                    vw = df_week[vol_col].astype(float)
                    aw = df_week[amount_col].astype(float) if amount_col in df_week.columns else None
                    ret_week = cw.pct_change()
                    obv_w = calc_obv(cw, vw)
                    mfi_w = calc_mfi(hw, lw, cw, vw, period=min(mfi_period, len(df_week) - 1) or 14)
                    aroon_w = calc_aroon(hw, lw, period=min(aroon_period, len(df_week) - 1) or 20)
                    mf_w = calc_money_flow(hw, lw, cw, vw, aw)
                    weekly_result = {
                        "obv": obv_w,
                        "mfi": mfi_w,
                        "aroon": aroon_w,
                        "money_flow": mf_w,
                        "last": {
                            "obv": float(obv_w.iloc[-1]) if not obv_w.dropna().empty else None,
                            "mfi": float(mfi_w.iloc[-1]) if not mfi_w.dropna().empty else None,
                            "aroon_up": float(aroon_w["aroon_up"].iloc[-1]) if not aroon_w["aroon_up"].dropna().empty else None,
                            "aroon_down": float(aroon_w["aroon_down"].iloc[-1]) if not aroon_w["aroon_down"].dropna().empty else None,
                            "money_flow_tier": mf_w["tier_net"],
                        },
                    }
            except Exception:
                weekly_result = None

    out = {
        "rsi": rsi,
        "macd": macd,
        "bollinger": bb,
        "rolling_volatility": vol,
        "var_95": var,
        "cvar_95": cvar,
        "last": last,
    }
    if obv is not None:
        out["obv"] = obv
    if mfi is not None:
        out["mfi"] = mfi
    if aroon is not None:
        out["aroon"] = aroon
    if money_flow is not None:
        out["money_flow"] = money_flow
    if weekly_result is not None:
        out["by_timeframe"] = {"daily": "see root keys", "weekly": weekly_result}
    return out
