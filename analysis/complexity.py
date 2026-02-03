#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块5：非线性与复杂度分析

分析收益率序列的非线性与复杂度特征，包括：
- 赫斯特指数（Hurst Exponent）：判断趋势/随机/均值回复
- 样本熵（Sample Entropy）：量化序列复杂度
- 滞后相空间图：可视化非线性结构

使用示例:
    import pandas as pd
    from analysis.complexity import analyze_complexity

    returns = prices.pct_change().dropna()
    result = analyze_complexity(returns)
    print(f"赫斯特指数: {result['hurst_exponent']}")
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False


def calc_hurst_exponent(series: np.ndarray, max_lag: int | None = None) -> tuple[float, dict]:
    """
    计算赫斯特指数（Hurst Exponent）。

    使用R/S分析方法（Rescaled Range Analysis）。
    - H < 0.5: 均值回复（反持续性）
    - H ≈ 0.5: 随机游走（无记忆）
    - H > 0.5: 趋势增强（持续性）

    Args:
        series: 时间序列数据（numpy数组）
        max_lag: 最大滞后期数，默认为序列长度的1/4

    Returns:
        tuple: (hurst_exponent, details_dict)
    """
    n = len(series)
    if n < 20:
        raise ValueError("序列长度至少需要20个点来计算赫斯特指数")

    if max_lag is None:
        max_lag = n // 4

    # 不同时间尺度的R/S值
    lags = []
    rs_values = []

    for lag in range(10, max_lag + 1):
        # 将序列分割为多个子序列
        n_subseries = n // lag
        if n_subseries < 1:
            continue

        rs_list = []
        for i in range(n_subseries):
            subseries = series[i * lag:(i + 1) * lag]

            # 计算均值调整后的累积离差
            mean_val = np.mean(subseries)
            cumdev = np.cumsum(subseries - mean_val)

            # 计算极差R
            r = np.max(cumdev) - np.min(cumdev)

            # 计算标准差S
            s = np.std(subseries, ddof=1)

            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            lags.append(lag)
            rs_values.append(np.mean(rs_list))

    # 对数线性回归估计H
    if len(lags) < 3:
        raise ValueError("数据不足以进行赫斯特指数估计")

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)

    # 最小二乘拟合
    coeffs = np.polyfit(log_lags, log_rs, 1)
    hurst = coeffs[0]

    # 计算拟合优度R²
    y_pred = np.polyval(coeffs, log_lags)
    ss_res = np.sum((log_rs - y_pred) ** 2)
    ss_tot = np.sum((log_rs - np.mean(log_rs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    details = {
        "lags": lags,
        "rs_values": rs_values,
        "log_lags": log_lags,
        "log_rs": log_rs,
        "r_squared": r_squared,
        "coefficients": coeffs,
    }

    return hurst, details


def calc_sample_entropy(
    series: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float:
    """
    计算样本熵（Sample Entropy）。

    样本熵衡量序列的复杂度和不可预测性：
    - 值越小：序列越规则、可预测性越高
    - 值越大：序列越复杂、随机性越强

    Args:
        series: 时间序列数据
        m: 嵌入维度（模式长度），默认2
        r: 容差阈值，默认为0.2倍标准差

    Returns:
        样本熵值
    """
    n = len(series)
    if n < 10:
        raise ValueError("序列长度至少需要10个点")

    # 默认容差为0.2倍标准差
    if r is None:
        r = 0.2 * np.std(series)

    def _count_matches(templates: np.ndarray, tolerance: float) -> int:
        """计算模板匹配数量（切比雪夫距离）"""
        count = 0
        n_templates = len(templates)
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                # 切比雪夫距离（最大绝对差）
                if np.max(np.abs(templates[i] - templates[j])) < tolerance:
                    count += 1
        return count

    # 构建m维和(m+1)维模板
    templates_m = np.array([series[i:i + m] for i in range(n - m)])
    templates_m1 = np.array([series[i:i + m + 1] for i in range(n - m - 1)])

    # 计算匹配数
    count_m = _count_matches(templates_m, r)
    count_m1 = _count_matches(templates_m1, r)

    # 避免除零
    if count_m == 0 or count_m1 == 0:
        return float("inf")

    # 计算样本熵
    sample_entropy = -np.log(count_m1 / count_m)

    return sample_entropy


def calc_approximate_entropy(
    series: np.ndarray,
    m: int = 2,
    r: float | None = None,
) -> float:
    """
    计算近似熵（Approximate Entropy）。

    与样本熵类似，但包含自匹配，适用于较短序列。

    Args:
        series: 时间序列数据
        m: 嵌入维度
        r: 容差阈值

    Returns:
        近似熵值
    """
    n = len(series)
    if n < 10:
        raise ValueError("序列长度至少需要10个点")

    if r is None:
        r = 0.2 * np.std(series)

    def _phi(m_dim: int) -> float:
        """计算phi函数"""
        templates = np.array([series[i:i + m_dim] for i in range(n - m_dim + 1)])
        n_templates = len(templates)

        counts = np.zeros(n_templates)
        for i in range(n_templates):
            for j in range(n_templates):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    counts[i] += 1

        # 计算平均对数概率
        counts = counts / n_templates
        return np.mean(np.log(counts))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    return phi_m - phi_m1


def create_phase_space(
    series: np.ndarray,
    lag: int = 1,
    dim: int = 2,
) -> np.ndarray:
    """
    构建滞后相空间（延迟嵌入）。

    Args:
        series: 时间序列
        lag: 时间延迟
        dim: 嵌入维度

    Returns:
        相空间坐标矩阵，形状为 (n_points, dim)
    """
    n = len(series)
    n_points = n - (dim - 1) * lag

    if n_points < 1:
        raise ValueError("序列长度不足以构建相空间")

    phase_space = np.zeros((n_points, dim))
    for d in range(dim):
        phase_space[:, d] = series[d * lag:d * lag + n_points]

    return phase_space


def plot_hurst_analysis(
    hurst: float,
    details: dict,
    title: str = "赫斯特指数R/S分析",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制赫斯特指数R/S分析图。

    Args:
        hurst: 赫斯特指数值
        details: calc_hurst_exponent返回的详细信息
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    log_lags = details["log_lags"]
    log_rs = details["log_rs"]
    coeffs = details["coefficients"]
    r_squared = details["r_squared"]

    # 散点图
    ax.scatter(log_lags, log_rs, c="tab:blue", s=50, alpha=0.7, label="R/S数据点")

    # 拟合线
    x_fit = np.linspace(log_lags.min(), log_lags.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "r-", linewidth=2, label=f"拟合线 (H={hurst:.4f})")

    # 参考线
    y_05 = 0.5 * x_fit + (y_fit[0] - 0.5 * x_fit[0])
    ax.plot(x_fit, y_05, "g--", alpha=0.5, linewidth=1, label="H=0.5 (随机游走)")

    ax.set_xlabel("log(滞后期)", fontsize=12)
    ax.set_ylabel("log(R/S)", fontsize=12)
    ax.set_title(f"{title}\nH = {hurst:.4f}, R² = {r_squared:.4f}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_phase_space(
    phase_space: np.ndarray,
    lag: int,
    title: str = "滞后相空间图",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制2D/3D滞后相空间图。

    Args:
        phase_space: 相空间坐标矩阵
        lag: 时间延迟
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    dim = phase_space.shape[1]

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            phase_space[:, 0],
            phase_space[:, 1],
            c=np.arange(len(phase_space)),
            cmap="viridis",
            s=10,
            alpha=0.6,
        )
        ax.plot(phase_space[:, 0], phase_space[:, 1], "b-", alpha=0.1, linewidth=0.5)
        ax.set_xlabel(f"r(t)", fontsize=12)
        ax.set_ylabel(f"r(t + {lag})", fontsize=12)
        plt.colorbar(scatter, ax=ax, label="时间顺序")

    elif dim >= 3:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            phase_space[:, 0],
            phase_space[:, 1],
            phase_space[:, 2],
            c=np.arange(len(phase_space)),
            cmap="viridis",
            s=10,
            alpha=0.6,
        )
        ax.plot(
            phase_space[:, 0],
            phase_space[:, 1],
            phase_space[:, 2],
            "b-",
            alpha=0.1,
            linewidth=0.5,
        )
        ax.set_xlabel(f"r(t)")
        ax.set_ylabel(f"r(t + {lag})")
        ax.set_zlabel(f"r(t + {2 * lag})")
        plt.colorbar(scatter, ax=ax, label="时间顺序", shrink=0.6)

    else:
        raise ValueError("相空间维度必须至少为2")

    ax.set_title(f"{title} (延迟τ={lag})", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_return_distribution(
    returns: np.ndarray,
    title: str = "收益率分布分析",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制收益率分布图（含正态分布对比）。

    Args:
        returns: 收益率序列
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：直方图与正态分布对比
    ax1 = axes[0]
    ax1.hist(returns, bins=50, density=True, alpha=0.7, color="tab:blue", edgecolor="white")

    # 拟合正态分布
    mu, std = np.mean(returns), np.std(returns)
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), "r-", linewidth=2, label="正态分布")

    ax1.set_xlabel("收益率")
    ax1.set_ylabel("概率密度")
    ax1.set_title("收益率分布直方图")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图2：Q-Q图
    ax2 = axes[1]
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Q-Q图（正态性检验）")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def interpret_hurst(hurst: float) -> str:
    """解读赫斯特指数的市场含义。"""
    if hurst < 0.4:
        return "强均值回复 - 价格倾向于向均值回归，可能适合反转策略"
    elif hurst < 0.5:
        return "弱均值回复 - 存在一定的反持续性特征"
    elif 0.5 <= hurst <= 0.55:
        return "随机游走 - 序列接近布朗运动，难以预测"
    elif hurst < 0.65:
        return "弱趋势增强 - 存在一定的持续性，可能适合趋势跟踪"
    else:
        return "强趋势增强 - 明显的持续性特征，趋势策略可能有效"


def interpret_sample_entropy(se: float) -> str:
    """解读样本熵的市场含义。"""
    if se < 0.5:
        return "低复杂度 - 序列较规则，可预测性较高"
    elif se < 1.0:
        return "中低复杂度 - 存在一定的结构性特征"
    elif se < 1.5:
        return "中等复杂度 - 序列具有适度的随机性"
    elif se < 2.0:
        return "中高复杂度 - 序列较为复杂，预测难度增加"
    else:
        return "高复杂度 - 序列高度随机，难以预测"


def analyze_complexity(
    return_series: pd.Series,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    hurst_max_lag: int | None = None,
    entropy_m: int = 2,
    phase_lag: int = 1,
) -> dict[str, Any]:
    """
    非线性与复杂度分析主函数。

    分析收益率序列的非线性与复杂度特征：
    1. 计算赫斯特指数，判断趋势/随机/均值回复
    2. 计算样本熵，量化复杂度和不可预测性
    3. 绘制滞后相空间图，观察非线性结构

    Args:
        return_series: 收益率序列（Pandas Series）
        save_dir: 图表保存目录
        show_plots: 是否显示图表
        hurst_max_lag: 赫斯特指数计算的最大滞后期
        entropy_m: 样本熵的嵌入维度
        phase_lag: 相空间图的时间延迟

    Returns:
        dict: 包含以下字段：
            - hurst_exponent: 赫斯特指数
            - hurst_interpretation: 赫斯特指数解读
            - sample_entropy: 样本熵
            - entropy_interpretation: 样本熵解读
            - approximate_entropy: 近似熵
            - figures: 生成的图表列表

    Example:
        >>> import pandas as pd
        >>> from analysis.complexity import analyze_complexity
        >>> returns = prices.pct_change().dropna()
        >>> result = analyze_complexity(returns)
        >>> print(f"赫斯特指数: {result['hurst_exponent']}")
    """
    # 数据验证
    if not isinstance(return_series, pd.Series):
        raise TypeError("输入必须为 Pandas Series")

    returns = return_series.dropna().values.astype(float)
    if len(returns) < 50:
        raise ValueError("序列长度至少需要50个点以获得可靠结果")

    # 准备保存目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    print("=" * 60)
    print("【非线性与复杂度分析】")
    print("=" * 60)
    print(f"序列长度: {len(returns)} 个数据点")

    # 1. 计算赫斯特指数
    print("\n【1. 赫斯特指数 (Hurst Exponent)】")
    print("-" * 40)

    try:
        hurst, hurst_details = calc_hurst_exponent(returns, max_lag=hurst_max_lag)
        hurst_interp = interpret_hurst(hurst)

        print(f"  H = {hurst:.4f}")
        print(f"  拟合R² = {hurst_details['r_squared']:.4f}")
        print(f"  解读: {hurst_interp}")

        # 绘制赫斯特分析图
        fig_hurst = plot_hurst_analysis(
            hurst,
            hurst_details,
            title="赫斯特指数R/S分析",
            save_path=save_dir / "hurst_analysis.png" if save_dir else None,
            show=show_plots,
        )
        figures.append(fig_hurst)

    except Exception as e:
        print(f"  计算失败: {e}")
        hurst = None
        hurst_interp = "计算失败"
        hurst_details = {}

    # 2. 计算样本熵
    print("\n【2. 样本熵 (Sample Entropy)】")
    print("-" * 40)

    try:
        sample_ent = calc_sample_entropy(returns, m=entropy_m)
        entropy_interp = interpret_sample_entropy(sample_ent)

        print(f"  SampEn = {sample_ent:.4f}")
        print(f"  解读: {entropy_interp}")

    except Exception as e:
        print(f"  计算失败: {e}")
        sample_ent = None
        entropy_interp = "计算失败"

    # 3. 计算近似熵
    print("\n【3. 近似熵 (Approximate Entropy)】")
    print("-" * 40)

    try:
        approx_ent = calc_approximate_entropy(returns, m=entropy_m)
        print(f"  ApEn = {approx_ent:.4f}")
    except Exception as e:
        print(f"  计算失败: {e}")
        approx_ent = None

    # 4. 相空间分析
    print("\n【4. 滞后相空间分析】")
    print("-" * 40)

    try:
        # 2D相空间
        phase_2d = create_phase_space(returns, lag=phase_lag, dim=2)
        print(f"  2D相空间点数: {len(phase_2d)}")

        fig_phase_2d = plot_phase_space(
            phase_2d,
            lag=phase_lag,
            title="2D滞后相空间图",
            save_path=save_dir / "phase_space_2d.png" if save_dir else None,
            show=show_plots,
        )
        figures.append(fig_phase_2d)

        # 3D相空间
        if len(returns) > 3 * phase_lag:
            phase_3d = create_phase_space(returns, lag=phase_lag, dim=3)
            print(f"  3D相空间点数: {len(phase_3d)}")

            fig_phase_3d = plot_phase_space(
                phase_3d,
                lag=phase_lag,
                title="3D滞后相空间图",
                save_path=save_dir / "phase_space_3d.png" if save_dir else None,
                show=show_plots,
            )
            figures.append(fig_phase_3d)

    except Exception as e:
        print(f"  相空间构建失败: {e}")

    # 5. 收益率分布分析
    print("\n【5. 收益率分布特征】")
    print("-" * 40)

    from scipy import stats as sp_stats

    skewness = sp_stats.skew(returns)
    kurtosis = sp_stats.kurtosis(returns)
    jb_stat, jb_pvalue = sp_stats.jarque_bera(returns)

    print(f"  偏度 (Skewness): {skewness:.4f}")
    print(f"  峰度 (Kurtosis): {kurtosis:.4f}")
    print(f"  JB检验统计量: {jb_stat:.4f}, p值: {jb_pvalue:.4f}")

    if jb_pvalue < 0.05:
        print("  => 拒绝正态分布假设（存在非线性特征）")
    else:
        print("  => 无法拒绝正态分布假设")

    fig_dist = plot_return_distribution(
        returns,
        title="收益率分布分析",
        save_path=save_dir / "return_distribution.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_dist)

    print("=" * 60)

    # 汇总结果
    result = {
        "hurst_exponent": hurst,
        "hurst_interpretation": hurst_interp,
        "hurst_details": hurst_details,
        "sample_entropy": sample_ent,
        "entropy_interpretation": entropy_interp,
        "approximate_entropy": approx_ent,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue,
        "data_length": len(returns),
        "figures": figures,
    }

    if save_dir:
        print(f"\n图表已保存至: {save_dir.resolve()}")

    return result


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m analysis.complexity <csv_path> [output_dir]")
        print("示例: python -m analysis.complexity data/xxx.csv output/")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"读取数据: {csv_path}")

    # 读取数据
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 找收盘价列
    price_col = None
    for col in ["收盘", "close", "Close", "收盘价"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        print("错误: 未找到收盘价列")
        sys.exit(1)

    prices = df[price_col].astype(float)
    returns = prices.pct_change().dropna()

    print(f"价格数据: {len(prices)} 条, 收益率: {len(returns)} 条")

    result = analyze_complexity(returns, save_dir=output_dir, show_plots=True)

    print("\n分析完成。")
