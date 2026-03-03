#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金基准对比分析模块

支持常用股票指数作为基准：
- 沪深300 (000300)
- 中证500 (000905)
- 中证1000 (000852)
- 上证指数 (000001)
- 创业板指 (399006)

使用示例:
    from analysis.fund_benchmark import get_benchmark_data, compare_with_benchmark

    benchmark = get_benchmark_data('000300', '2023-01-01', '2024-01-01')
    result = compare_with_benchmark(fund_nav, benchmark)
"""

import pandas as pd
from typing import Optional
from datetime import datetime

from data.index_repo import get_index_data


BENCHMARK_CODES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000001": "上证指数",
    "399006": "创业板指",
    "399300": "创业板50",
    "000016": "上证50",
}


def get_benchmark_nav(
    benchmark_code: str,
    start_date: str | datetime,
    end_date: str | datetime,
) -> pd.Series | None:
    """
    获取基准指数净值数据。

    由于 akshare 对场外基金支持有限，此函数尝试从本地存储或计算获取基准。
    实际使用时建议手动导入基准 CSV 数据。

    Args:
        benchmark_code: 指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        基准净值序列，失败返回 None
    """
    from pathlib import Path

    data_dir = Path(__file__).parent.parent / "data"
    benchmark_file = data_dir / f"benchmark_{benchmark_code}.csv"

    if benchmark_file.exists():
        df = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
        if "close" in df.columns:
            return df["close"]
        elif "收盘" in df.columns:
            return df["收盘"]

    return None


def load_benchmark_csv(csv_path: str) -> pd.Series:
    """
    从 CSV 文件加载基准指数数据。

    Args:
        csv_path: CSV 文件路径

    Returns:
        基准净值序列
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    date_col = None
    for col in ["日期", "date", "Date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("未找到日期列")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    nav_col = None
    for col in ["close", "Close", "收盘", "净值", "close_"]:
        if col in df.columns:
            nav_col = col
            break

    if nav_col is None:
        raise ValueError("未找到收盘/净值列")

    return df[nav_col].astype(float)


def compare_fund_with_benchmark(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series,
    fund_name: str = "基金",
    benchmark_name: str = "基准",
) -> dict:
    """
    基金与基准对比分析。

    Args:
        fund_nav: 基金净值序列
        benchmark_nav: 基准净值序列
        fund_name: 基金名称
        benchmark_name: 基准名称

    Returns:
        对比结果字典
    """
    from analysis.fund_metrics import (
        analyze_fund_metrics,
        calc_annual_return,
        calc_returns,
    )

    common_idx = fund_nav.index.intersection(benchmark_nav.index)
    if len(common_idx) < 10:
        return {"error": "基金与基准数据交集不足10天"}

    fund_aligned = fund_nav.loc[common_idx]
    bench_aligned = benchmark_nav.loc[common_idx]

    metrics = analyze_fund_metrics(fund_aligned, bench_aligned)

    result = {
        "fund_name": fund_name,
        "benchmark_name": benchmark_name,
        "comparison_days": len(common_idx),
        "fund_total_return": metrics.get("total_return", 0),
        "benchmark_total_return": bench_aligned.iloc[-1] / bench_aligned.iloc[0] - 1,
        "fund_annual_return": metrics.get("annual_return", 0),
        "benchmark_annual_return": calc_annual_return(bench_aligned),
        "fund_volatility": metrics.get("volatility", 0),
        "benchmark_volatility": calc_returns(bench_aligned).std() * 252,
        "fund_sharpe": metrics.get("sharpe_ratio", 0),
        "fund_max_drawdown": metrics.get("max_drawdown", 0),
        "alpha": metrics.get("alpha", 0),
        "beta": metrics.get("beta", 1),
        "r_squared": metrics.get("r_squared", 0),
        "information_ratio": metrics.get("information_ratio", 0),
        "excess_return": metrics.get("excess_return", 0),
    }

    return result


def format_comparison_report(comparison: dict) -> str:
    """格式化对比报告。"""
    if "error" in comparison:
        return f"错误: {comparison['error']}"

    lines = [
        "=" * 60,
        f"基金 vs {comparison['benchmark_name']} 对比报告",
        "=" * 60,
        f"对比天数: {comparison['comparison_days']} 天",
        "",
        "【收益率对比】",
        f"  基金总收益率:    {comparison['fund_total_return']:>10.2%}",
        f"  基准总收益率:    {comparison['benchmark_total_return']:>10.2%}",
        f"  超额收益:        {comparison['excess_return']:>10.2%}",
        "",
        "  基金年化收益率:  {comparison['fund_annual_return']:>10.2%}",
        f"  基准年化收益率:  {comparison['benchmark_annual_return']:>10.2%}",
        "",
        "【风险对比】",
        f"  基金年化波动率:  {comparison['fund_volatility']:>10.2%}",
        f"  基准年化波动率:  {comparison['benchmark_volatility']:>10.2%}",
        f"  基金最大回撤:    {comparison['fund_max_drawdown']:>10.2%}",
        "",
        "【风险调整收益】",
        f"  基金夏普比率:    {comparison['fund_sharpe']:>10.2f}",
        f"  阿尔法 (Alpha):  {comparison['alpha']:>10.2%}",
        f"  贝塔 (Beta):     {comparison['beta']:>10.2f}",
        f"  R²:              {comparison['r_squared']:>10.4f}",
        f"  信息比率:        {comparison['information_ratio']:>10.2f}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


def get_default_benchmarks() -> dict:
    """获取默认支持的基准指数列表。"""
    return BENCHMARK_CODES.copy()


def get_benchmark_data(
    index_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.Series]:
    """
    获取基准指数净值数据。

    Args:
        index_code: 指数代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        基准净值序列
    """
    df = get_index_data(index_code, start_date, end_date)
    if df is None or df.empty:
        return None
    df = df.sort_values("trade_date")
    return df.set_index("trade_date")["close_price"]


def compare_with_benchmark(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series,
) -> dict:
    """
    基金与基准对比分析。

    Args:
        fund_nav: 基金净值序列
        benchmark_nav: 基准净值序列

    Returns:
        对比结果字典
    """
    return compare_fund_with_benchmark(fund_nav, benchmark_nav)


def calculate_alpha_beta(
    fund_nav: pd.Series,
    benchmark_nav: pd.Series,
    rf: float = 0.03,
) -> tuple[float, float]:
    """
    计算 Alpha 和 Beta。

    Args:
        fund_nav: 基金净值序列
        benchmark_nav: 基准净值序列
        rf: 无风险利率

    Returns:
        tuple: (alpha, beta)
    """
    from analysis.fund_metrics import calc_returns, calc_annual_return

    fund_returns = calc_returns(fund_nav)
    bench_returns = calc_returns(benchmark_nav)

    if len(fund_returns) != len(bench_returns):
        bench_returns = bench_returns.reindex(fund_returns.index)

    fund_returns = fund_returns.dropna()
    bench_returns = bench_returns.dropna()
    common_idx = fund_returns.index.intersection(bench_returns.index)
    fund_returns = fund_returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    if len(fund_returns) < 10:
        return 0.0, 1.0

    rf_daily = rf / 252
    excess_returns = fund_returns - rf_daily
    excess_benchmark = bench_returns - rf_daily

    covariance = excess_returns.cov(excess_benchmark)
    benchmark_variance = excess_benchmark.var()

    if benchmark_variance == 0:
        return 0.0, 1.0

    beta = covariance / benchmark_variance

    alpha_daily = excess_returns.mean() - beta * excess_benchmark.mean()
    alpha = alpha_daily * 252

    return alpha, beta
