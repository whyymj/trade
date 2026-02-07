# -*- coding: utf-8 -*-
"""
LSTM 波动增强特征：缓解预测过度平滑。

- 保留原始波动：日收益率、日内振幅、异常波动标记
- 波动率相关：短期/长期历史波动率、波动率比
- 量价关系：成交量变化率、量价背离

避免过度平滑特征（如过长均线、过度平滑 RSI），提升预测曲线波动性。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def 计算量价背离(
    close: pd.Series,
    volume: pd.Series,
    window: int = 10,
) -> pd.Series:
    """
    量价背离强度：价格变动与成交量变动的滚动相关系数取反。
    当价涨量缩或价跌量增时相关系数为负，取反后为正，表示存在背离。

    Returns:
        与 close 同长的 Series，负相关时为正（背离），正相关时为负，NaN 填 0。
    """
    ret = close.pct_change().fillna(0)
    vol_ret = volume.pct_change().replace(0, np.nan).fillna(0)
    if ret.empty or vol_ret.empty:
        return pd.Series(0.0, index=close.index)
    corr = ret.rolling(window=window, min_periods=2).corr(vol_ret)
    # 背离 = 负相关，取反便于“背离强”为正值
    out = -corr.fillna(0)
    return out.clip(-1, 1)


def 构建波动增强特征(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_series: pd.Series,
    volume: pd.Series,
) -> pd.DataFrame:
    """
    构建可增强预测波动性的特征，与 build_features_from_df 对齐使用。

    参数为与日线对齐的 Series（支持中文列名对应的 Series）。
    返回与 close 等长的 DataFrame，列名为特征名。
    """
    out = pd.DataFrame(index=close.index)
    ret = close.pct_change().fillna(0)

    # 1. 收益率特征（保持原始波动）
    out["日收益率"] = ret
    out["收益率绝对值"] = np.abs(ret)
    out["收益率符号"] = np.sign(ret).replace(0, 0)

    # 2. 日内波动特征
    out["日内振幅"] = (high - low) / (close + 1e-8)
    out["开盘缺口"] = (open_series - close.shift(1)) / (close.shift(1).replace(0, np.nan).fillna(close.iloc[0]) + 1e-8)
    out["开盘缺口"] = out["开盘缺口"].fillna(0)

    # 3. 波动率相关特征
    hv5 = ret.rolling(5, min_periods=1).std().fillna(0)
    hv20 = ret.rolling(20, min_periods=1).std().fillna(0)
    out["历史波动率_5日"] = hv5
    out["历史波动率_20日"] = hv20
    out["波动率比"] = hv5 / (hv20 + 1e-8)

    # 4. 异常波动标记
    vol_thresh = ret.std()
    vol_thresh = vol_thresh if vol_thresh > 1e-12 else 0.01
    out["大涨标记"] = (ret > 2 * vol_thresh).astype(np.float64)
    out["大跌标记"] = (ret < -2 * vol_thresh).astype(np.float64)

    # 5. 成交量波动与量价背离
    out["成交量变化率"] = volume.pct_change().fillna(0)
    out["量价背离"] = 计算量价背离(close, volume, window=10)

    return out


# 与 构建波动增强特征 列顺序一致，供 lstm_model 空样本返回与元数据使用
VOLATILITY_FEATURE_NAMES = [
    "日收益率",
    "收益率绝对值",
    "收益率符号",
    "日内振幅",
    "开盘缺口",
    "历史波动率_5日",
    "历史波动率_20日",
    "波动率比",
    "大涨标记",
    "大跌标记",
    "成交量变化率",
    "量价背离",
]


def 波动增强特征到数组(
    vol_df: pd.DataFrame,
    seq_len: int,
    clip_scale: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    将 构建波动增强特征 返回的 DataFrame 转为与 LSTM 输入对齐的 (T, n_features) 数组及特征名。
    对每列做简单缩放/裁剪，便于与现有特征拼接。
    """
    names = list(vol_df.columns)
    n = len(vol_df)
    F = np.zeros((n, len(names)), dtype=np.float64)
    for j, col in enumerate(names):
        s = vol_df[col].values.astype(np.float64)
        if clip_scale:
            if "收益率" in col or "缺口" in col or "变化率" in col or "量价背离" in col or col == "收益率符号":
                s = np.clip(s, -1, 1)
            elif "振幅" in col or "波动率" in col or "波动率比" in col:
                s = np.clip(s, 0, 2)
                s = s / 2.0
            elif "标记" in col:
                pass  # 0/1
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        F[:, j] = s
    return F, names
