#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金特有指标计算模块

包含：
- 年化收益率
- 夏普比率 (Sharpe Ratio)
- 卡玛比率 (Calmar Ratio)
- 索提诺比率 (Sortino Ratio)
- 最大回撤 (Maximum Drawdown)
- 阿尔法 (Alpha) / 贝塔 (Beta)
- R² (决定系数)
- 信息比率 (Information Ratio)

使用示例:
    import pandas as pd
    from analysis.fund_metrics import analyze_fund_metrics

    nav = pd.read_csv('fund_nav.csv', index_col=0, parse_dates=True)['净值']
    result = analyze_fund_metrics(nav)
    print(f"年化收益率: {result['annual_return']:.2%}")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
"""

import numpy as np
import pandas as pd
from typing import Any, Optional

from data.fund_repo import get_fund_nav


TRADING_DAYS_PER_YEAR = 252


def calc_returns(nav: pd.Series) -> pd.Series:
    """计算日收益率序列。"""
    return nav.pct_change().dropna()


def calc_annual_return(
    nav: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    计算年化收益率。

    Args:
        nav: 净值序列
        periods_per_year: 每年交易日数，默认252

    Returns:
        年化收益率（小数形式）
    """
    if len(nav) < 2:
        return 0.0

    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(nav) / periods_per_year
    if years <= 0:
        return 0.0
    annual_return = (1 + total_return) ** (1 / years) - 1
    return annual_return


def calc_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    计算年化波动率。

    Args:
        returns: 日收益率序列
        periods_per_year: 每年交易日数

    Returns:
        年化波动率（小数形式）
    """
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def calc_sharpe_ratio(
    nav: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    计算夏普比率 (Sharpe Ratio)。

    夏普比率 = (年化收益率 - 无风险利率) / 年化波动率

    Args:
        nav: 净值序列
        risk_free_rate: 年化无风险利率，默认3%
        periods_per_year: 每年交易日数

    Returns:
        夏普比率
    """
    returns = calc_returns(nav)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    annual_return = calc_annual_return(nav, periods_per_year)
    volatility = calc_volatility(returns, periods_per_year)

    if volatility == 0:
        return 0.0
    return (annual_return - risk_free_rate) / volatility


def calc_max_drawdown(nav: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    计算最大回撤及对应区间。

    Args:
        nav: 净值序列

    Returns:
        tuple: (最大回撤, 峰值日期, 谷值日期)
    """
    if len(nav) < 2:
        return 0.0, nav.index[0], nav.index[0]

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    max_dd = drawdown.min()
    peak_idx = drawdown.idxmin()

    peak_date = nav.loc[:peak_idx].idxmax()
    trough_date = peak_idx

    return max_dd, peak_date, trough_date


def calc_calmar_ratio(
    nav: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    计算卡玛比率 (Calmar Ratio)。

    卡玛比率 = 年化收益率 / 最大回撤（绝对值）

    Args:
        nav: 净值序列
        periods_per_year: 每年交易日数

    Returns:
        卡玛比率
    """
    max_dd, _, _ = calc_max_drawdown(nav)
    if max_dd == 0:
        return 0.0

    annual_return = calc_annual_return(nav, periods_per_year)
    return annual_return / abs(max_dd)


def calc_sortino_ratio(
    nav: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    target_return: float = 0.0,
) -> float:
    """
    计算索提诺比率 (Sortino Ratio)。

    索提诺比率 = (年化收益率 - 目标收益率) / 下行波动率

    Args:
        nav: 净值序列
        risk_free_rate: 年化无风险利率
        periods_per_year: 每年交易日数
        target_return: 目标收益率，默认0

    Returns:
        索提诺比率
    """
    returns = calc_returns(nav)
    if len(returns) < 2:
        return 0.0

    annual_return = calc_annual_return(nav, periods_per_year)

    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
    if downside_vol == 0:
        return 0.0

    return (annual_return - risk_free_rate) / downside_vol


def calc_alpha_beta(
    nav: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.03 / TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """
    计算阿尔法 (Alpha) 和贝塔 (Beta)。

    CAPM模型: Rp - Rf = Alpha + Beta * (Rm - Rf)

    Args:
        nav: 基金净值序列
        benchmark_returns: 基准收益率序列
        risk_free_rate: 日无风险利率

    Returns:
        tuple: (Alpha, Beta)
    """
    fund_returns = calc_returns(nav)

    if len(fund_returns) != len(benchmark_returns):
        benchmark_returns = benchmark_returns.reindex(fund_returns.index)

    fund_returns = fund_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    common_idx = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    if len(fund_returns) < 10:
        return 0.0, 1.0

    excess_returns = fund_returns - risk_free_rate
    excess_benchmark = benchmark_returns - risk_free_rate

    covariance = excess_returns.cov(excess_benchmark)
    benchmark_variance = excess_benchmark.var()

    if benchmark_variance == 0:
        return 0.0, 1.0

    beta = covariance / benchmark_variance

    alpha_daily = excess_returns.mean() - beta * excess_benchmark.mean()
    alpha = alpha_daily * TRADING_DAYS_PER_YEAR

    return alpha, beta


def calc_r_squared(
    nav: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    计算 R² (决定系数)。

    R² 衡量基金收益率与基准收益率的相关程度。

    Args:
        nav: 基金净值序列
        benchmark_returns: 基准收益率序列

    Returns:
        R² 值 (0-1)
    """
    fund_returns = calc_returns(nav)

    if len(fund_returns) != len(benchmark_returns):
        benchmark_returns = benchmark_returns.reindex(fund_returns.index)

    fund_returns = fund_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    common_idx = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    if len(fund_returns) < 10:
        return 0.0

    return fund_returns.corr(benchmark_returns) ** 2


def calc_information_ratio(
    nav: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    计算信息比率 (Information Ratio)。

    信息比率 = 超额收益 / 跟踪误差

    Args:
        nav: 基金净值序列
        benchmark_returns: 基准收益率序列
        periods_per_year: 每年交易日数

    Returns:
        信息比率
    """
    fund_returns = calc_returns(nav)

    if len(fund_returns) != len(benchmark_returns):
        benchmark_returns = benchmark_returns.reindex(fund_returns.index)

    fund_returns = fund_returns.dropna()
    benchmark_returns = benchmark_returns.dropna()
    common_idx = fund_returns.index.intersection(benchmark_returns.index)
    fund_returns = fund_returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]

    if len(fund_returns) < 10:
        return 0.0

    excess_returns = fund_returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0

    annual_excess_return = excess_returns.mean() * periods_per_year
    return annual_excess_return / tracking_error


def calc_win_rate(returns: pd.Series) -> float:
    """计算胜率（正收益天数占比）。"""
    if len(returns) == 0:
        return 0.0
    return (returns > 0).sum() / len(returns)


def calc_profit_loss_ratio(returns: pd.Series) -> float:
    """计算盈亏比（平均盈利 / 平均亏损）。"""
    gains = returns[returns > 0]
    losses = returns[returns < 0]

    if len(gains) == 0 or len(losses) == 0:
        return 0.0

    avg_gain = gains.mean()
    avg_loss = abs(losses.mean())
    if avg_loss == 0:
        return 0.0
    return avg_gain / avg_loss


def analyze_fund_metrics(
    nav: pd.Series,
    benchmark_nav: Optional[pd.Series] = None,
    risk_free_rate: float = 0.03,
) -> dict[str, Any]:
    """
    基金综合指标分析。

    Args:
        nav: 基金净值序列
        benchmark_nav: 基准净值序列（可选，用于计算Alpha/Beta/R²）
        risk_free_rate: 年化无风险利率

    Returns:
        包含所有指标的字典
    """
    returns = calc_returns(nav)

    max_dd, peak_date, trough_date = calc_max_drawdown(nav)

    result = {
        "nav_start": nav.iloc[0],
        "nav_end": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
        "annual_return": calc_annual_return(nav),
        "volatility": calc_volatility(returns),
        "sharpe_ratio": calc_sharpe_ratio(nav, risk_free_rate),
        "max_drawdown": max_dd,
        "max_drawdown_peak": peak_date,
        "max_drawdown_trough": trough_date,
        "calmar_ratio": calc_calmar_ratio(nav),
        "sortino_ratio": calc_sortino_ratio(nav, risk_free_rate),
        "win_rate": calc_win_rate(returns),
        "profit_loss_ratio": calc_profit_loss_ratio(returns),
        "data_days": len(nav),
    }

    if benchmark_nav is not None:
        benchmark_returns = calc_returns(benchmark_nav)
        result["alpha"], result["beta"] = calc_alpha_beta(
            nav, benchmark_returns, risk_free_rate / TRADING_DAYS_PER_YEAR
        )
        result["r_squared"] = calc_r_squared(nav, benchmark_returns)
        result["information_ratio"] = calc_information_ratio(nav, benchmark_returns)

        fund_annual = calc_annual_return(nav)
        bench_annual = calc_annual_return(benchmark_nav)
        result["excess_return"] = fund_annual - bench_annual

    return result


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """将指标字典格式化为可读的报告字符串。"""
    lines = [
        "=" * 50,
        "基金指标分析报告",
        "=" * 50,
        "",
        f"数据天数: {metrics.get('data_days', 0)} 天",
        "",
        "【收益指标】",
        f"净值起始: {metrics.get('nav_start', 0):.4f}",
        f"净值结束: {metrics.get('nav_end', 0):.4f}",
        f"总收益率: {metrics.get('total_return', 0):.2%}",
        f"年化收益率: {metrics.get('annual_return', 0):.2%}",
        "",
        "【风险指标】",
        f"年化波动率: {metrics.get('volatility', 0):.2%}",
        f"最大回撤: {metrics.get('max_drawdown', 0):.2%}",
        "",
        "【风险调整收益】",
        f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}",
        f"卡玛比率: {metrics.get('calmar_ratio', 0):.2f}",
        f"索提诺比率: {metrics.get('sortino_ratio', 0):.2f}",
        "",
        "【交易指标】",
        f"胜率: {metrics.get('win_rate', 0):.2%}",
        f"盈亏比: {metrics.get('profit_loss_ratio', 0):.2f}",
    ]

    if "alpha" in metrics:
        lines.extend(
            [
                "",
                "【相对基准】",
                f"阿尔法 (Alpha): {metrics.get('alpha', 0):.2%}",
                f"贝塔 (Beta): {metrics.get('beta', 0):.2f}",
                f"R²: {metrics.get('r_squared', 0):.4f}",
                f"信息比率: {metrics.get('information_ratio', 0):.2f}",
                f"超额收益: {metrics.get('excess_return', 0):.2%}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def calculate_return(nav_series: pd.Series, period_days: int) -> float:
    """
    计算指定周期收益率。

    Args:
        nav_series: 净值序列
        period_days: 周期天数

    Returns:
        周期收益率（小数形式）
    """
    if len(nav_series) < 2:
        return 0.0
    period_days = min(period_days, len(nav_series))
    return (
        (nav_series.iloc[-1] / nav_series.iloc[-period_days] - 1)
        if period_days > 0
        else 0.0
    )


def calculate_annual_return(nav_series: pd.Series) -> float:
    """计算年化收益率。"""
    return calc_annual_return(nav_series)


def calculate_volatility(nav_series: pd.Series) -> float:
    """计算年化波动率。"""
    returns = calc_returns(nav_series)
    return calc_volatility(returns)


def calculate_sharpe_ratio(nav_series: pd.Series, rf: float = 0.03) -> float:
    """计算夏普比率。"""
    return calc_sharpe_ratio(nav_series, rf)


def calculate_max_drawdown(nav_series: pd.Series) -> float:
    """计算最大回撤。"""
    max_dd, _, _ = calc_max_drawdown(nav_series)
    return max_dd


def calculate_win_rate(nav_series: pd.Series) -> float:
    """计算胜率。"""
    returns = calc_returns(nav_series)
    return calc_win_rate(returns)


def get_fund_indicators(fund_code: str, days: int = 365) -> Optional[dict[str, Any]]:
    """
    获取基金完整业绩指标。

    Args:
        fund_code: 基金代码
        days: 计算天数

    Returns:
        指标字典，包含收益率、夏普比率、最大回撤等
    """
    from datetime import date, timedelta

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    df = get_fund_nav(fund_code, start_date=str(start_date), end_date=str(end_date))
    if df is None or df.empty:
        return None

    df = df.sort_values("nav_date")
    nav_series = df.set_index("nav_date")["unit_nav"]

    if len(nav_series) < 2:
        return None

    returns = calc_returns(nav_series)
    max_dd, peak_date, trough_date = calc_max_drawdown(nav_series)

    return {
        "return_1m": calculate_return(nav_series, 30),
        "return_3m": calculate_return(nav_series, 90),
        "return_6m": calculate_return(nav_series, 180),
        "return_1y": calculate_return(nav_series, min(365, days)),
        "annual_return": calc_annual_return(nav_series),
        "volatility": calc_volatility(returns),
        "sharpe_ratio": calc_sharpe_ratio(nav_series),
        "max_drawdown": max_dd,
        "max_drawdown_peak": str(peak_date) if peak_date else None,
        "max_drawdown_trough": str(trough_date) if trough_date else None,
        "win_rate": calc_win_rate(returns),
        "calmar_ratio": calc_calmar_ratio(nav_series),
        "sortino_ratio": calc_sortino_ratio(nav_series),
    }
