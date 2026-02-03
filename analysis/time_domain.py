#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块1：时域与统计特征分析

对股价时间序列进行时域分析，包括：
- 基础统计量（均值、标准差、偏度、峰度、最大回撤）
- 移动平均线（短期 20 日、长期 60 日）
- STL 季节性分解（趋势、季节、残差）
- 自相关图（ACF）

使用示例:
    import pandas as pd
    from analysis.time_domain import analyze_time_domain

    df = pd.read_csv("data/xxx.csv", parse_dates=["日期"], index_col="日期")
    result = analyze_time_domain(df)
    print(result)
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def _get_close_col(df: pd.DataFrame) -> str:
    """获取收盘价列名，兼容中英文。"""
    for col in ["close", "Close", "收盘", "收盘价"]:
        if col in df.columns:
            return col
    raise ValueError("DataFrame 中未找到收盘价列（'close' 或 '收盘'）")


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 DataFrame 索引为 DatetimeIndex。"""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        # 尝试从列中找日期
        for col in ["日期", "date", "Date"]:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col])
                out = out.set_index(col)
                break
        else:
            # 尝试直接转换索引
            out.index = pd.to_datetime(out.index)
    return out.sort_index()


def calc_max_drawdown(prices: pd.Series) -> float:
    """
    计算最大回撤（近期）。

    最大回撤 = (峰值 - 谷值) / 峰值
    返回 0~1 之间的比例值。
    """
    cummax = prices.cummax()
    drawdown = (cummax - prices) / cummax
    return float(drawdown.max())


def calc_basic_stats(prices: pd.Series) -> dict[str, float]:
    """
    计算基础统计量。

    Returns:
        dict: 包含 mean, std, skewness, kurtosis, max_drawdown
    """
    return {
        "mean": float(prices.mean()),
        "std": float(prices.std()),
        "skewness": float(stats.skew(prices.dropna())),
        "kurtosis": float(stats.kurtosis(prices.dropna())),
        "max_drawdown": calc_max_drawdown(prices),
    }


def calc_moving_averages(prices: pd.Series, short: int = 20, long: int = 60) -> pd.DataFrame:
    """
    计算短期和长期移动平均线。

    Args:
        prices: 收盘价序列
        short: 短期窗口（默认 20 日）
        long: 长期窗口（默认 60 日）

    Returns:
        DataFrame: 包含 close, MA_short, MA_long 列
    """
    ma_df = pd.DataFrame({"close": prices})
    ma_df[f"MA_{short}"] = prices.rolling(window=short, min_periods=1).mean()
    ma_df[f"MA_{long}"] = prices.rolling(window=long, min_periods=1).mean()
    return ma_df


def decompose_stl(prices: pd.Series, period: int | None = None) -> dict[str, pd.Series]:
    """
    使用 STL 方法分解时间序列为趋势、季节、残差。

    Args:
        prices: 收盘价序列（需为等间隔时间序列）
        period: 季节周期；若 None 则自动设为 5（周交易日）或数据长度的 1/4

    Returns:
        dict: 包含 trend, seasonal, resid 三个 Series
    """
    # 填充缺失值（STL 不支持 NaN）
    series = prices.dropna()
    if len(series) < 14:
        raise ValueError("数据长度不足以进行 STL 分解（至少需要 14 个点）")

    # 自动确定周期：股票日线一般以 5 日（周）为最小周期
    if period is None:
        period = min(5, len(series) // 4)
        period = max(2, period)

    stl = STL(series, period=period, robust=True)
    result = stl.fit()

    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
    }


def plot_price_with_ma(
    ma_df: pd.DataFrame,
    title: str = "收盘价与移动平均线",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制收盘价与移动平均线叠加图。

    Args:
        ma_df: 由 calc_moving_averages 返回的 DataFrame
        title: 图表标题
        save_path: 保存路径；为 None 则不保存
        show: 是否显示图表
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ma_df.index, ma_df["close"], label="收盘价", linewidth=1.2, alpha=0.9)

    for col in ma_df.columns:
        if col.startswith("MA_"):
            ax.plot(ma_df.index, ma_df[col], label=col, linewidth=1, alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("价格")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_stl_decomposition(
    decomp: dict[str, pd.Series],
    title: str = "STL 季节性分解",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制 STL 分解结果（趋势、季节、残差）。

    Args:
        decomp: 由 decompose_stl 返回的字典
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(decomp["trend"], label="趋势 (Trend)", color="tab:blue")
    axes[0].set_ylabel("趋势")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(decomp["seasonal"], label="季节 (Seasonal)", color="tab:orange")
    axes[1].set_ylabel("季节")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(decomp["resid"], label="残差 (Residual)", color="tab:green")
    axes[2].set_ylabel("残差")
    axes[2].set_xlabel("日期")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_acf_chart(
    prices: pd.Series,
    lags: int = 40,
    title: str = "收盘价自相关图 (ACF)",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制自相关图。

    Args:
        prices: 收盘价序列
        lags: 滞后阶数
        title: 图表标题
        save_path: 保存路径
        show: 是否显示
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_acf(prices.dropna(), lags=min(lags, len(prices) - 1), ax=ax, alpha=0.05)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("自相关系数")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def analyze_time_domain(
    data: pd.DataFrame,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    short_ma: int = 20,
    long_ma: int = 60,
    acf_lags: int = 40,
) -> dict[str, Any]:
    """
    时域与统计特征分析主函数。

    对输入的股价时间序列 DataFrame 进行完整时域分析：
    1. 计算基础统计量：均值、标准差、偏度、峰度、最大回撤
    2. 计算并绘制短期/长期移动平均线与收盘价叠加图
    3. STL 季节性分解（趋势、季节、残差）并绘图
    4. 绘制收盘价序列的自相关图（ACF）

    Args:
        data: DataFrame，索引为日期时间，至少包含 'close'（或 '收盘'）列
        save_dir: 图表保存目录；为 None 则不保存
        show_plots: 是否显示图表（交互模式）
        short_ma: 短期移动平均窗口（默认 20 日）
        long_ma: 长期移动平均窗口（默认 60 日）
        acf_lags: 自相关图滞后阶数（默认 40）

    Returns:
        dict: 包含所有统计结果：
            - basic_stats: 基础统计量字典
            - moving_averages: 移动平均 DataFrame
            - stl_decomposition: STL 分解结果字典（trend, seasonal, resid）
            - figures: 生成的图表对象列表（如需后续处理）

    Raises:
        ValueError: 输入数据不符合要求时抛出

    Example:
        >>> import pandas as pd
        >>> from analysis.time_domain import analyze_time_domain
        >>> df = pd.read_csv("data/xxx.csv", parse_dates=["日期"], index_col="日期")
        >>> result = analyze_time_domain(df, save_dir="output/", show_plots=False)
        >>> print(result["basic_stats"])
    """
    # 1. 数据预处理
    df = _ensure_datetime_index(data)
    close_col = _get_close_col(df)
    prices = df[close_col].astype(float)

    if prices.dropna().empty:
        raise ValueError("收盘价数据为空")

    # 准备保存目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    # 2. 计算基础统计量
    basic_stats = calc_basic_stats(prices)
    print("=" * 50)
    print("【基础统计量】")
    print(f"  均值 (Mean):       {basic_stats['mean']:.4f}")
    print(f"  标准差 (Std):      {basic_stats['std']:.4f}")
    print(f"  偏度 (Skewness):   {basic_stats['skewness']:.4f}")
    print(f"  峰度 (Kurtosis):   {basic_stats['kurtosis']:.4f}")
    print(f"  最大回撤 (Max DD): {basic_stats['max_drawdown']:.2%}")
    print("=" * 50)

    # 3. 计算并绘制移动平均线
    ma_df = calc_moving_averages(prices, short=short_ma, long=long_ma)
    fig_ma = plot_price_with_ma(
        ma_df,
        title=f"收盘价与移动平均线 (MA{short_ma}, MA{long_ma})",
        save_path=save_dir / "ma_plot.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_ma)

    # 4. STL 季节性分解
    stl_result = None
    try:
        stl_result = decompose_stl(prices)
        fig_stl = plot_stl_decomposition(
            stl_result,
            title="STL 季节性分解（趋势 / 季节 / 残差）",
            save_path=save_dir / "stl_decomposition.png" if save_dir else None,
            show=show_plots,
        )
        figures.append(fig_stl)
    except ValueError as e:
        print(f"[警告] STL 分解失败: {e}")

    # 5. 自相关图
    fig_acf = plot_acf_chart(
        prices,
        lags=acf_lags,
        title="收盘价自相关图 (ACF)",
        save_path=save_dir / "acf_plot.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_acf)

    # 6. 汇总结果
    result = {
        "basic_stats": basic_stats,
        "moving_averages": ma_df,
        "stl_decomposition": stl_result,
        "figures": figures,
    }

    if save_dir:
        print(f"\n图表已保存至: {save_dir.resolve()}")

    return result


# -----------------------------------------------------------------------------
# 命令行入口：可直接运行分析指定 CSV
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m analysis.time_domain <csv_path> [output_dir]")
        print("示例: python -m analysis.time_domain data/白银有色（20250129-20260129）.csv output/")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"读取数据: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    result = analyze_time_domain(df, save_dir=output_dir, show_plots=True)

    print("\n分析完成。")
