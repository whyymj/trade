#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块2：频域（周期性）分析

对收益率序列进行频域分析以探测周期，包括：
- 傅里叶变换与功率谱密度
- 识别并标注主要周期成分
- 小波变换时频谱分析（可选）

使用示例:
    import pandas as pd
    from analysis.frequency_domain import analyze_frequency_domain

    # 从价格序列计算收益率
    prices = pd.read_csv("data/xxx.csv", index_col="日期", parse_dates=True)["收盘"]
    returns = prices.pct_change().dropna()
    
    result = analyze_frequency_domain(returns)
    print(result)
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def calc_power_spectrum(
    return_series: pd.Series,
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算收益率序列的功率谱密度（Power Spectral Density, PSD）。

    使用 Welch 方法估计功率谱密度，相比直接 FFT 更加平滑稳定。

    Args:
        return_series: 收益率序列（Pandas Series）
        sampling_rate: 采样率（默认 1.0，表示每天 1 个数据点）

    Returns:
        tuple: (frequencies, psd)
            - frequencies: 频率数组（单位：1/天）
            - psd: 功率谱密度数组
    """
    data = return_series.dropna().values
    if len(data) < 10:
        raise ValueError("数据长度不足以进行频谱分析（至少需要 10 个点）")

    # 使用 Welch 方法计算功率谱密度
    # nperseg 设为数据长度的 1/4 或 256 中的较小值
    nperseg = min(len(data) // 4, 256)
    nperseg = max(nperseg, 8)  # 至少 8 个点

    frequencies, psd = signal.welch(
        data,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="linear",
    )

    return frequencies, psd


def calc_fft_spectrum(
    return_series: pd.Series,
    sampling_rate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 FFT 计算收益率序列的频谱。

    Args:
        return_series: 收益率序列
        sampling_rate: 采样率（默认 1.0）

    Returns:
        tuple: (frequencies, amplitudes, phases)
            - frequencies: 频率数组（仅正频率部分）
            - amplitudes: 振幅数组
            - phases: 相位数组
    """
    data = return_series.dropna().values
    n = len(data)
    if n < 10:
        raise ValueError("数据长度不足以进行 FFT 分析（至少需要 10 个点）")

    # 去均值
    data = data - np.mean(data)

    # 应用汉宁窗减少频谱泄漏
    window = np.hanning(n)
    data_windowed = data * window

    # FFT
    fft_result = np.fft.fft(data_windowed)
    frequencies = np.fft.fftfreq(n, d=1.0 / sampling_rate)

    # 只取正频率部分
    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    fft_positive = fft_result[positive_mask]

    amplitudes = np.abs(fft_positive) * 2 / n  # 归一化
    phases = np.angle(fft_positive)

    return frequencies, amplitudes, phases


def find_dominant_periods(
    frequencies: np.ndarray,
    power: np.ndarray,
    top_n: int = 3,
    min_period: float = 2.0,
) -> list[dict[str, float]]:
    """
    找出功率谱中能量最高的前 N 个周期成分。

    Args:
        frequencies: 频率数组
        power: 功率/振幅数组
        top_n: 返回前 N 个最强周期
        min_period: 最小有效周期（天数），过滤掉过短的周期

    Returns:
        list[dict]: 每个元素包含 frequency, period, power
    """
    # 过滤掉零频率和过短周期
    # 避免除零：先过滤正频率，再计算周期
    positive_mask = frequencies > 0
    periods = np.zeros_like(frequencies)
    periods[positive_mask] = 1.0 / frequencies[positive_mask]
    valid_mask = positive_mask & (periods >= min_period)
    valid_freqs = frequencies[valid_mask]
    valid_power = power[valid_mask]

    if len(valid_freqs) == 0:
        return []

    # 找出能量最高的 top_n 个
    top_indices = np.argsort(valid_power)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        freq = valid_freqs[idx]
        period = 1.0 / freq if freq > 0 else float("inf")
        results.append({
            "frequency": float(freq),
            "period_days": float(period),
            "power": float(valid_power[idx]),
        })

    return results


def plot_power_spectrum(
    frequencies: np.ndarray,
    psd: np.ndarray,
    dominant_periods: list[dict[str, float]],
    title: str = "收益率功率谱密度",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制功率谱图，并标注主要周期成分。

    Args:
        frequencies: 频率数组
        psd: 功率谱密度数组
        dominant_periods: 主要周期列表（由 find_dominant_periods 返回）
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 子图1：功率谱密度（频率域）
    ax1.semilogy(frequencies, psd, color="tab:blue", linewidth=1)
    ax1.set_xlabel("频率 (1/天)")
    ax1.set_ylabel("功率谱密度 (对数)")
    ax1.set_title(f"{title} - 频率域", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 标注主要周期
    colors = ["red", "orange", "green"]
    for i, dp in enumerate(dominant_periods[:3]):
        freq = dp["frequency"]
        power = dp["power"]
        period = dp["period_days"]
        color = colors[i] if i < len(colors) else "gray"

        # 在功率谱上标注
        ax1.axvline(x=freq, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.annotate(
            f"#{i+1}: {period:.1f}天",
            xy=(freq, power),
            xytext=(freq + 0.01, power * 2),
            fontsize=10,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.7),
        )

    # 子图2：功率谱密度（周期域）
    # 转换为周期显示
    valid_mask = frequencies > 0
    periods = 1.0 / frequencies[valid_mask]
    psd_valid = psd[valid_mask]

    # 按周期排序
    sort_idx = np.argsort(periods)
    periods_sorted = periods[sort_idx]
    psd_sorted = psd_valid[sort_idx]

    ax2.semilogy(periods_sorted, psd_sorted, color="tab:green", linewidth=1)
    ax2.set_xlabel("周期 (天)")
    ax2.set_ylabel("功率谱密度 (对数)")
    ax2.set_title(f"{title} - 周期域", fontsize=12)
    ax2.set_xlim(0, min(200, periods_sorted.max()))  # 限制显示范围
    ax2.grid(True, alpha=0.3)

    # 在周期域上标注主要周期
    for i, dp in enumerate(dominant_periods[:3]):
        period = dp["period_days"]
        power = dp["power"]
        color = colors[i] if i < len(colors) else "gray"

        ax2.axvline(x=period, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        ax2.annotate(
            f"#{i+1}: {period:.1f}天",
            xy=(period, power),
            xytext=(period + 5, power * 2),
            fontsize=10,
            color=color,
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def calc_wavelet_transform(
    return_series: pd.Series,
    wavelet: str = "morl",
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对收益率序列进行连续小波变换（CWT）。

    使用 PyWavelets 库进行小波变换。

    Args:
        return_series: 收益率序列
        wavelet: 小波类型（默认 'morl' 莫莱特小波）
        scales: 小波尺度数组；若 None 则自动生成

    Returns:
        tuple: (times, scales, coefficients)
            - times: 时间索引数组
            - scales: 尺度数组（对应不同周期）
            - coefficients: 小波系数矩阵（复数）

    Raises:
        ImportError: 若未安装 PyWavelets 库
    """
    try:
        import pywt
    except ImportError:
        raise ImportError(
            "小波变换需要安装 PyWavelets 库。请运行: pip install PyWavelets"
        )

    data = return_series.dropna().values
    n = len(data)

    if n < 20:
        raise ValueError("数据长度不足以进行小波变换（至少需要 20 个点）")

    # 自动生成尺度（对应 2~100 天的周期）
    if scales is None:
        # 尺度与周期的关系取决于小波类型
        # 对于 Morlet 小波，周期 ≈ 尺度 * 1.03
        min_scale = 2
        max_scale = min(100, n // 2)
        scales = np.arange(min_scale, max_scale + 1, 1)

    # 使用 PyWavelets 的连续小波变换
    coefficients, frequencies = pywt.cwt(data, scales, wavelet)

    # 时间索引
    if isinstance(return_series.index, pd.DatetimeIndex):
        times = return_series.dropna().index
    else:
        times = np.arange(n)

    return times, scales, coefficients


def plot_wavelet_spectrogram(
    times: np.ndarray | pd.DatetimeIndex,
    scales: np.ndarray,
    coefficients: np.ndarray,
    title: str = "小波时频谱图",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制小波变换的时频谱图。

    Args:
        times: 时间索引
        scales: 尺度数组
        coefficients: 小波系数矩阵
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 计算功率（小波系数的模的平方）
    power = np.abs(coefficients) ** 2

    # 绘制时频谱图
    if isinstance(times, pd.DatetimeIndex):
        # 转换为数值用于绘图
        time_numeric = np.arange(len(times))
        im = ax.pcolormesh(
            time_numeric,
            scales,
            power,
            cmap="jet",
            shading="auto",
        )
        # 设置 x 轴刻度为日期
        tick_positions = np.linspace(0, len(times) - 1, min(10, len(times))).astype(int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([times[i].strftime("%Y-%m") for i in tick_positions], rotation=45)
    else:
        im = ax.pcolormesh(times, scales, power, cmap="jet", shading="auto")

    ax.set_xlabel("时间")
    ax.set_ylabel("周期 (天)")
    ax.set_title(title, fontsize=14)

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, label="功率")
    cbar.ax.set_ylabel("功率", rotation=270, labelpad=15)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def analyze_frequency_domain(
    return_series: pd.Series,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    top_periods: int = 3,
    enable_wavelet: bool = True,
) -> dict[str, Any]:
    """
    频域（周期性）分析主函数。

    对收益率序列进行频域分析以探测周期：
    1. 傅里叶变换计算功率谱密度
    2. 识别并标注能量最高的前 N 个周期成分（换算回天数）
    3. 可选：小波变换绘制时频谱图

    Args:
        return_series: 价格序列计算出的收益率序列（Pandas Series）
        save_dir: 图表保存目录；为 None 则不保存
        show_plots: 是否显示图表（交互模式）
        top_periods: 标注前 N 个最强周期（默认 3）
        enable_wavelet: 是否启用小波变换分析（默认 True）

    Returns:
        dict: 包含分析结果：
            - frequencies: 频率数组
            - psd: 功率谱密度数组
            - dominant_periods: 主要周期列表
            - wavelet_result: 小波变换结果（若启用）
            - figures: 生成的图表对象列表

    Raises:
        ValueError: 输入数据不符合要求时抛出

    Example:
        >>> import pandas as pd
        >>> from analysis.frequency_domain import analyze_frequency_domain
        >>> prices = pd.read_csv("data/xxx.csv", index_col="日期", parse_dates=True)["收盘"]
        >>> returns = prices.pct_change().dropna()
        >>> result = analyze_frequency_domain(returns)
    """
    # 数据验证
    if not isinstance(return_series, pd.Series):
        raise TypeError("输入必须为 Pandas Series")

    series = return_series.dropna()
    if len(series) < 20:
        raise ValueError("数据长度不足以进行频域分析（至少需要 20 个点）")

    # 准备保存目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    # 1. 计算功率谱密度
    print("=" * 60)
    print("【频域（周期性）分析】")
    print("=" * 60)

    frequencies, psd = calc_power_spectrum(series)

    # 2. 找出主要周期
    dominant_periods = find_dominant_periods(frequencies, psd, top_n=top_periods)

    print(f"\n数据长度: {len(series)} 个交易日")
    print(f"\n【主要周期成分（能量最强的前 {top_periods} 个）】")
    print("-" * 40)

    for i, dp in enumerate(dominant_periods):
        print(f"  #{i+1}: 周期 = {dp['period_days']:.1f} 天, "
              f"频率 = {dp['frequency']:.4f} (1/天), "
              f"功率 = {dp['power']:.2e}")

    print("-" * 40)

    # 周期解读
    if dominant_periods:
        main_period = dominant_periods[0]["period_days"]
        if main_period <= 7:
            interpretation = "主周期接近一周，可能反映周内交易模式"
        elif 15 <= main_period <= 25:
            interpretation = "主周期接近一个月，可能反映月度效应"
        elif 55 <= main_period <= 70:
            interpretation = "主周期接近一个季度，可能反映季度财报效应"
        elif main_period >= 200:
            interpretation = "主周期较长，可能反映年度或宏观经济周期"
        else:
            interpretation = "主周期可能与特定市场事件或行业因素相关"
        print(f"\n【周期解读】: {interpretation}")

    print("=" * 60)

    # 3. 绘制功率谱图
    fig_psd = plot_power_spectrum(
        frequencies,
        psd,
        dominant_periods,
        title="收益率功率谱密度",
        save_path=save_dir / "power_spectrum.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_psd)

    # 4. 小波变换（可选）
    wavelet_result = None
    if enable_wavelet:
        try:
            print("\n正在进行小波变换分析...")
            times, scales, coefficients = calc_wavelet_transform(series)
            wavelet_result = {
                "times": times,
                "scales": scales,
                "coefficients": coefficients,
            }

            fig_wavelet = plot_wavelet_spectrogram(
                times,
                scales,
                coefficients,
                title="小波时频谱图 - 周期成分随时间变化",
                save_path=save_dir / "wavelet_spectrogram.png" if save_dir else None,
                show=show_plots,
            )
            figures.append(fig_wavelet)
            print("小波变换分析完成。")
        except Exception as e:
            print(f"[警告] 小波变换失败: {e}")

    # 5. 汇总结果
    result = {
        "frequencies": frequencies,
        "psd": psd,
        "dominant_periods": dominant_periods,
        "wavelet_result": wavelet_result,
        "figures": figures,
        "data_length": len(series),
    }

    if save_dir:
        print(f"\n图表已保存至: {save_dir.resolve()}")

    return result


def calc_returns_from_prices(prices: pd.Series, method: str = "simple") -> pd.Series:
    """
    从价格序列计算收益率。

    辅助函数，方便用户直接从价格数据计算收益率。

    Args:
        prices: 价格序列
        method: 计算方法
            - 'simple': 简单收益率 (P_t - P_{t-1}) / P_{t-1}
            - 'log': 对数收益率 log(P_t / P_{t-1})

    Returns:
        收益率序列
    """
    if method == "simple":
        return prices.pct_change().dropna()
    elif method == "log":
        return np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"不支持的收益率计算方法: {method}")


# -----------------------------------------------------------------------------
# 命令行入口：可直接运行分析指定 CSV
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m analysis.frequency_domain <csv_path> [output_dir]")
        print("示例: python -m analysis.frequency_domain data/xxx.csv output/")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"读取数据: {csv_path}")

    # 读取数据并计算收益率
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 尝试找到价格列
    price_col = None
    for col in ["收盘", "close", "Close", "收盘价"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        print("错误: 未找到收盘价列")
        sys.exit(1)

    prices = df[price_col].astype(float)
    returns = calc_returns_from_prices(prices, method="simple")

    print(f"数据长度: {len(prices)} 条, 收益率长度: {len(returns)} 条")

    result = analyze_frequency_domain(returns, save_dir=output_dir, show_plots=True)

    print("\n分析完成。")
