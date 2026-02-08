# -*- coding: utf-8 -*-
"""
因子绩效分析报告：IC、Rank IC、换手率、分组收益、稳定性等

用于多因子预测模型的特征筛选与诊断。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def calc_ic(factor: np.ndarray, forward_return: np.ndarray) -> float:
    """信息系数：因子与未来收益的 Pearson 相关系数。"""
    mask = np.isfinite(factor) & np.isfinite(forward_return)
    if mask.sum() < 10:
        return np.nan
    return float(np.corrcoef(factor[mask], forward_return[mask])[0, 1])


def calc_rank_ic(factor: np.ndarray, forward_return: np.ndarray) -> float:
    """Rank IC：因子排名与未来收益排名的 Spearman 相关系数。"""
    mask = np.isfinite(factor) & np.isfinite(forward_return)
    if mask.sum() < 10:
        return np.nan
    from scipy.stats import spearmanr
    r, _ = spearmanr(factor[mask], forward_return[mask])
    return float(r) if np.isfinite(r) else np.nan


def calc_ic_series(
    factor_df: pd.DataFrame,
    forward_return: pd.Series,
    *,
    rolling_window: int = 20,
) -> pd.DataFrame:
    """
    滚动计算各因子的 IC 与 Rank IC 序列。
    factor_df 与 forward_return 索引需对齐。
    """
    common = factor_df.index.intersection(forward_return.index)
    F = factor_df.loc[common]
    y = forward_return.reindex(common).dropna()
    common = F.index.intersection(y.index)
    F = F.loc[common]
    y = y.loc[common].values

    ic_list = []
    rank_ic_list = []
    for w_start in range(0, len(common) - rolling_window):
        w_end = w_start + rolling_window
        f_slice = F.iloc[w_start:w_end].values
        y_slice = y[w_start:w_end]
        ic_row = {}
        rank_ic_row = {}
        for j, col in enumerate(F.columns):
            ic_row[col] = calc_ic(f_slice[:, j], y_slice)
            rank_ic_row[col] = calc_rank_ic(f_slice[:, j], y_slice)
        ic_list.append(ic_row)
        rank_ic_list.append(rank_ic_row)

    if not ic_list:
        return pd.DataFrame(columns=["ic_mean", "ic_std", "rank_ic_mean", "rank_ic_std"])

    ic_df = pd.DataFrame(ic_list, index=common[rolling_window:])
    rank_ic_df = pd.DataFrame(rank_ic_list, index=common[rolling_window:])
    return pd.DataFrame({
        "ic_mean": ic_df.mean(),
        "ic_std": ic_df.std(),
        "rank_ic_mean": rank_ic_df.mean(),
        "rank_ic_std": rank_ic_df.std(),
        "ic_ir": ic_df.mean() / (ic_df.std() + 1e-10),
        "rank_ic_ir": rank_ic_df.mean() / (rank_ic_df.std() + 1e-10),
    })


def calc_turnover(
    factor_df: pd.DataFrame,
    *,
    top_pct: float = 0.2,
    rolling_window: int = 1,
) -> pd.Series:
    """
    因子换手率：每日（或滚动窗口）前 top_pct 分位组合相对前一日（或前窗口）的变动比例。
    返回每个窗口的换手率序列。
    """
    common = factor_df.index
    n = len(common)
    if n < rolling_window + 1:
        return pd.Series(dtype=float)
    out = []
    for i in range(rolling_window, n):
        curr = factor_df.iloc[i]
        prev = factor_df.iloc[i - rolling_window]
        thresh_curr = np.nanpercentile(curr.values, 100 * (1 - top_pct))
        thresh_prev = np.nanpercentile(prev.values, 100 * (1 - top_pct))
        set_curr = set(curr.index[curr >= thresh_curr].tolist())
        set_prev = set(prev.index[prev >= thresh_prev].tolist())
        union = len(set_curr | set_prev)
        inter = len(set_curr & set_prev)
        to = 1 - (inter / (union + 1e-10))
        out.append(to)
    return pd.Series(out, index=common[rolling_window:])


def calc_group_returns(
    factor: np.ndarray,
    forward_return: np.ndarray,
    n_groups: int = 5,
) -> list[float]:
    """按因子分组，计算各组平均未来收益。组 0 为因子最小，组 n_groups-1 为因子最大。"""
    mask = np.isfinite(factor) & np.isfinite(forward_return)
    if mask.sum() < n_groups * 2:
        return [np.nan] * n_groups
    f = factor[mask]
    y = forward_return[mask]
    q = np.percentile(f, np.linspace(0, 100, n_groups + 1))
    q[-1] += 1
    group_ret = []
    for i in range(n_groups):
        g = (f >= q[i]) & (f < q[i + 1])
        group_ret.append(float(np.mean(y[g])) if g.sum() > 0 else np.nan)
    return group_ret


def generate_factor_performance_report(
    factor_df: pd.DataFrame,
    forward_return: pd.Series,
    *,
    rolling_window: int = 20,
    top_pct: float = 0.2,
    n_groups: int = 5,
) -> dict[str, Any]:
    """
    生成因子绩效分析报告：全样本 IC/Rank IC、滚动 IC 统计、换手率、分组收益、按类别汇总。
    """
    common = factor_df.index.intersection(forward_return.index)
    F = factor_df.loc[common].replace([np.inf, -np.inf], np.nan)
    y = forward_return.reindex(common).dropna()
    common = F.index.intersection(y.index)
    F = F.loc[common]
    y_arr = y.loc[common].values

    # 全样本 IC / Rank IC
    ic_full = {}
    rank_ic_full = {}
    group_returns = {}
    for col in F.columns:
        f = F[col].values
        ic_full[col] = calc_ic(f, y_arr)
        rank_ic_full[col] = calc_rank_ic(f, y_arr)
        group_returns[col] = calc_group_returns(f, y_arr, n_groups=n_groups)

    # 滚动 IC 统计
    ic_stats = calc_ic_series(factor_df, forward_return, rolling_window=rolling_window)

    # 换手率（取前 5 个因子示例，避免计算量过大）
    sample_cols = list(F.columns)[:5]
    turnover_sample = {}
    for col in sample_cols:
        to_ser = calc_turnover(F[[col]], top_pct=top_pct, rolling_window=1)
        turnover_sample[col] = float(to_ser.mean()) if len(to_ser) > 0 else np.nan

    # 多空收益（最高组 - 最低组）
    long_short = {}
    for col in F.columns:
        gr = group_returns[col]
        if len(gr) == n_groups and all(np.isfinite(gr)):
            long_short[col] = gr[-1] - gr[0]
        else:
            long_short[col] = np.nan

    report = {
        "summary": {
            "n_observations": int(len(common)),
            "n_factors": int(F.shape[1]),
            "rolling_window": rolling_window,
            "n_groups": n_groups,
        },
        "ic_full": ic_full,
        "rank_ic_full": rank_ic_full,
        "ic_rolling_stats": ic_stats.to_dict(orient="index") if not ic_stats.empty else {},
        "group_returns": group_returns,
        "long_short_return": long_short,
        "turnover_sample_mean": turnover_sample,
        "top_ic_factors": sorted(ic_full.items(), key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0, reverse=True)[:20],
        "top_rank_ic_factors": sorted(rank_ic_full.items(), key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0, reverse=True)[:20],
    }
    return report


def report_to_markdown(report: dict[str, Any]) -> str:
    """将因子绩效报告转为 Markdown 文本。"""
    lines = ["# 因子绩效分析报告\n"]
    s = report.get("summary", {})
    lines.append(f"- 观测数: {s.get('n_observations', 0)}")
    lines.append(f"- 因子数: {s.get('n_factors', 0)}")
    lines.append(f"- 滚动窗口: {s.get('rolling_window', 20)}")
    lines.append("")

    lines.append("## 全样本 IC / Rank IC 前 20\n")
    lines.append("| 因子 | IC | Rank IC |")
    lines.append("|------|-----|--------|")
    top_ic = report.get("top_ic_factors", [])[:20]
    rank_ic = dict(report.get("rank_ic_full", {}))
    for name, ic in top_ic:
        ric = rank_ic.get(name, np.nan)
        lines.append(f"| {name} | {ic:.4f} | {ric:.4f} |")
    lines.append("")

    lines.append("## 多空收益（最高组-最低组）前 15\n")
    ls = report.get("long_short_return", {})
    for name, ret in sorted(ls.items(), key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0, reverse=True)[:15]:
        lines.append(f"- **{name}**: {ret:.4f}")
    lines.append("")

    lines.append("## 换手率示例（前 5 个因子）\n")
    for name, to in report.get("turnover_sample_mean", {}).items():
        lines.append(f"- **{name}**: {to:.4f}")
    return "\n".join(lines)
