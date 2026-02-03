#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块3：预测模型构建（ARIMA示例）

自动构建ARIMA模型并进行价格预测，包括：
- ADF检验判断平稳性
- 自动差分处理非平稳序列
- AIC准则自动搜索最优(p,d,q)参数
- 模型拟合与未来价格预测
- 可视化（历史数据、拟合曲线、预测区间）

使用示例:
    import pandas as pd
    from analysis.arima_model import build_arima_model

    prices = pd.read_csv("data/xxx.csv", index_col="日期", parse_dates=True)["收盘"]
    result = build_arima_model(prices, forecast_days=5)
    print(result["forecast"])
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "Heiti TC", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

# 忽略收敛警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def adf_test(series: pd.Series, significance: float = 0.05) -> dict[str, Any]:
    """
    ADF单位根检验，判断序列平稳性。

    Args:
        series: 时间序列
        significance: 显著性水平（默认0.05）

    Returns:
        dict: 包含检验统计量、p值、是否平稳等信息
    """
    series_clean = series.dropna()
    result = adfuller(series_clean, autolag="AIC")

    adf_stat = result[0]
    p_value = result[1]
    used_lag = result[2]
    n_obs = result[3]
    critical_values = result[4]

    is_stationary = p_value < significance

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "n_observations": n_obs,
        "critical_values": critical_values,
        "is_stationary": is_stationary,
        "significance": significance,
    }


def auto_diff(
    series: pd.Series,
    max_diff: int = 2,
    significance: float = 0.05,
) -> tuple[pd.Series, int]:
    """
    自动差分直到序列平稳。

    Args:
        series: 原始时间序列
        max_diff: 最大差分阶数
        significance: ADF检验显著性水平

    Returns:
        tuple: (差分后的序列, 差分阶数d)
    """
    d = 0
    diff_series = series.copy()

    for i in range(max_diff + 1):
        test_result = adf_test(diff_series, significance)
        if test_result["is_stationary"]:
            break
        if i < max_diff:
            diff_series = diff_series.diff().dropna()
            d += 1

    return diff_series, d


def search_best_arima_params(
    series: pd.Series,
    d: int,
    p_range: range = range(0, 5),
    q_range: range = range(0, 5),
    verbose: bool = True,
) -> tuple[int, int, int, float]:
    """
    使用AIC准则搜索最优ARIMA(p,d,q)参数。

    Args:
        series: 时间序列（原始序列，非差分）
        d: 差分阶数（已确定）
        p_range: AR阶数搜索范围
        q_range: MA阶数搜索范围
        verbose: 是否打印搜索过程

    Returns:
        tuple: (best_p, best_d, best_q, best_aic)
    """
    best_aic = float("inf")
    best_order = (0, d, 0)

    if verbose:
        print(f"\n搜索最优ARIMA参数 (d={d} 已确定)...")
        print("-" * 50)

    total_combinations = len(p_range) * len(q_range)
    tested = 0

    for p in p_range:
        for q in q_range:
            tested += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    aic = fitted.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)

                    if verbose and tested % 5 == 0:
                        print(f"  进度: {tested}/{total_combinations}, "
                              f"当前最优: ARIMA{best_order}, AIC={best_aic:.2f}")

            except Exception:
                # 某些参数组合可能导致模型无法收敛
                continue

    if verbose:
        print("-" * 50)
        print(f"最优参数: ARIMA{best_order}, AIC={best_aic:.2f}")

    return best_order[0], best_order[1], best_order[2], best_aic


def fit_arima_model(
    series: pd.Series,
    order: tuple[int, int, int],
) -> Any:
    """
    使用指定参数拟合ARIMA模型。

    Args:
        series: 时间序列
        order: (p, d, q) 参数元组

    Returns:
        拟合后的ARIMA模型结果对象
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def calc_model_metrics(
    actual: pd.Series,
    fitted_values: pd.Series,
) -> dict[str, float]:
    """
    计算模型评价指标。

    Args:
        actual: 实际值序列
        fitted_values: 拟合值序列

    Returns:
        dict: 包含RMSE、MAE、MAPE等指标
    """
    # 对齐索引
    common_idx = actual.index.intersection(fitted_values.index)
    actual_aligned = actual.loc[common_idx]
    fitted_aligned = fitted_values.loc[common_idx]

    residuals = actual_aligned - fitted_aligned

    # RMSE
    rmse = np.sqrt(np.mean(residuals ** 2))

    # MAE
    mae = np.mean(np.abs(residuals))

    # MAPE (避免除零)
    nonzero_mask = actual_aligned != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(residuals[nonzero_mask] / actual_aligned[nonzero_mask])) * 100
    else:
        mape = np.nan

    # R² (决定系数)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((actual_aligned - actual_aligned.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "r_squared": float(r_squared),
        "n_observations": len(common_idx),
    }


def forecast_arima(
    fitted_model: Any,
    steps: int,
    original_series: pd.Series,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    使用拟合好的ARIMA模型进行预测。

    Args:
        fitted_model: 拟合后的模型对象
        steps: 预测步数
        original_series: 原始序列（用于生成预测日期索引）
        confidence: 置信区间（默认95%）

    Returns:
        DataFrame: 包含预测值、置信区间上下界
    """
    # 获取预测结果
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=1 - confidence)

    # 生成预测日期索引
    last_date = original_series.index[-1]
    if isinstance(last_date, pd.Timestamp):
        # 生成未来交易日（简单按日历日递增）
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq="B",  # 工作日
        )
    else:
        forecast_dates = range(len(original_series), len(original_series) + steps)

    # 构建预测DataFrame
    forecast_df = pd.DataFrame({
        "预测值": forecast_mean.values,
        "下界": conf_int.iloc[:, 0].values,
        "上界": conf_int.iloc[:, 1].values,
    }, index=forecast_dates)

    forecast_df.index.name = "日期"

    return forecast_df


def plot_arima_results(
    original_series: pd.Series,
    fitted_values: pd.Series,
    forecast_df: pd.DataFrame,
    order: tuple[int, int, int],
    title: str = "ARIMA模型拟合与预测",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制ARIMA模型结果图。

    包含：历史数据、拟合曲线、未来预测及置信区间。

    Args:
        original_series: 原始价格序列
        fitted_values: 模型拟合值
        forecast_df: 预测结果DataFrame
        order: ARIMA参数(p,d,q)
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # --- 子图1：全景视图（历史 + 预测）---
    ax1 = axes[0]

    # 历史数据
    ax1.plot(
        original_series.index,
        original_series.values,
        label="历史价格",
        color="tab:blue",
        linewidth=1.2,
    )

    # 拟合曲线
    ax1.plot(
        fitted_values.index,
        fitted_values.values,
        label="模型拟合",
        color="tab:orange",
        linewidth=1,
        alpha=0.8,
    )

    # 预测曲线
    ax1.plot(
        forecast_df.index,
        forecast_df["预测值"],
        label="未来预测",
        color="tab:red",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # 置信区间
    ax1.fill_between(
        forecast_df.index,
        forecast_df["下界"],
        forecast_df["上界"],
        alpha=0.3,
        color="tab:red",
        label="95% 置信区间",
    )

    ax1.set_title(f"{title} - ARIMA{order}", fontsize=14)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("价格")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    # --- 子图2：近期细节视图（最近60天 + 预测）---
    ax2 = axes[1]

    # 取最近60天数据
    recent_days = 60
    recent_series = original_series.iloc[-recent_days:]
    recent_fitted = fitted_values.iloc[-recent_days:] if len(fitted_values) >= recent_days else fitted_values

    ax2.plot(
        recent_series.index,
        recent_series.values,
        label="近期价格",
        color="tab:blue",
        linewidth=1.5,
        marker=".",
        markersize=3,
    )

    ax2.plot(
        recent_fitted.index,
        recent_fitted.values,
        label="模型拟合",
        color="tab:orange",
        linewidth=1,
        alpha=0.8,
    )

    ax2.plot(
        forecast_df.index,
        forecast_df["预测值"],
        label="未来预测",
        color="tab:red",
        linewidth=2,
        marker="o",
        markersize=6,
    )

    ax2.fill_between(
        forecast_df.index,
        forecast_df["下界"],
        forecast_df["上界"],
        alpha=0.3,
        color="tab:red",
        label="95% 置信区间",
    )

    ax2.set_title(f"近期细节视图（最近{recent_days}天 + 预测）", fontsize=12)
    ax2.set_xlabel("日期")
    ax2.set_ylabel("价格")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_residual_diagnostics(
    fitted_model: Any,
    title: str = "残差诊断图",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    绘制模型残差诊断图。

    包含：残差时序图、残差直方图、Q-Q图、自相关图。

    Args:
        fitted_model: 拟合后的ARIMA模型
        title: 图表标题
        save_path: 保存路径
        show: 是否显示

    Returns:
        matplotlib Figure 对象
    """
    from scipy import stats
    from statsmodels.graphics.tsaplots import plot_acf

    residuals = fitted_model.resid.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 残差时序图
    ax1 = axes[0, 0]
    ax1.plot(residuals.index, residuals.values, color="tab:blue", linewidth=0.8)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.7)
    ax1.set_title("残差时序图")
    ax1.set_xlabel("日期")
    ax1.set_ylabel("残差")
    ax1.grid(True, alpha=0.3)

    # 2. 残差直方图
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=30, density=True, alpha=0.7, color="tab:blue", edgecolor="white")
    # 叠加正态分布曲线
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, stats.norm.pdf(x, mu, std), "r-", linewidth=2, label="正态分布")
    ax2.set_title("残差分布直方图")
    ax2.set_xlabel("残差")
    ax2.set_ylabel("密度")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Q-Q图
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q图（正态性检验）")
    ax3.grid(True, alpha=0.3)

    # 4. 残差自相关图
    ax4 = axes[1, 1]
    plot_acf(residuals, lags=min(30, len(residuals) - 1), ax=ax4, alpha=0.05)
    ax4.set_title("残差自相关图 (ACF)")
    ax4.set_xlabel("滞后阶数")
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def build_arima_model(
    price_series: pd.Series,
    forecast_days: int = 5,
    p_range: range = range(0, 5),
    q_range: range = range(0, 5),
    max_diff: int = 2,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    自动构建ARIMA模型并进行预测（主函数）。

    自动完成以下步骤：
    1. ADF检验判断序列平稳性，若不平稳则自动差分确定d
    2. 根据AIC准则搜索最优(p,d,q)参数
    3. 拟合ARIMA模型并预测未来价格
    4. 绘制历史数据、拟合曲线和预测区间图

    Args:
        price_series: 价格时间序列（Pandas Series，索引为日期）
        forecast_days: 预测天数（默认5天）
        p_range: AR阶数搜索范围（默认0-4）
        q_range: MA阶数搜索范围（默认0-4）
        max_diff: 最大差分阶数（默认2）
        save_dir: 图表保存目录；为 None 则不保存
        show_plots: 是否显示图表
        verbose: 是否打印详细信息

    Returns:
        dict: 包含以下字段：
            - model: 拟合好的ARIMA模型对象
            - forecast: 预测结果DataFrame（日期、预测值、置信区间）
            - metrics: 模型评价指标（RMSE、MAE、MAPE、R²）
            - order: 最优(p,d,q)参数
            - aic: 模型AIC值
            - adf_test: ADF检验结果
            - figures: 生成的图表列表

    Raises:
        ValueError: 输入数据不符合要求时抛出

    Example:
        >>> import pandas as pd
        >>> from analysis.arima_model import build_arima_model
        >>> df = pd.read_csv("data/xxx.csv", parse_dates=["日期"], index_col="日期")
        >>> result = build_arima_model(df["收盘"], forecast_days=5)
        >>> print(result["forecast"])
        >>> print(result["metrics"])
    """
    # 数据验证
    if not isinstance(price_series, pd.Series):
        raise TypeError("输入必须为 Pandas Series")

    series = price_series.dropna()
    if len(series) < 30:
        raise ValueError("数据长度不足以构建ARIMA模型（至少需要30个点）")

    # 确保有日期索引
    if not isinstance(series.index, pd.DatetimeIndex):
        # 尝试转换索引为日期
        try:
            series.index = pd.to_datetime(series.index)
        except Exception:
            pass

    series = series.sort_index()

    # 准备保存目录
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figures = []

    if verbose:
        print("=" * 60)
        print("【ARIMA模型构建与预测】")
        print("=" * 60)
        print(f"数据长度: {len(series)} 个交易日")
        print(f"预测天数: {forecast_days} 天")

    # Step 1: ADF检验
    if verbose:
        print("\n【Step 1: ADF平稳性检验】")

    adf_result = adf_test(series)

    if verbose:
        print(f"  ADF统计量: {adf_result['adf_statistic']:.4f}")
        print(f"  p值: {adf_result['p_value']:.4f}")
        print(f"  使用滞后阶数: {adf_result['used_lag']}")
        print(f"  临界值 (5%): {adf_result['critical_values']['5%']:.4f}")
        print(f"  序列平稳: {'是 ✓' if adf_result['is_stationary'] else '否 ✗'}")

    # Step 2: 自动差分
    if verbose:
        print("\n【Step 2: 自动差分确定d】")

    _, d = auto_diff(series, max_diff=max_diff)

    if verbose:
        if d == 0:
            print(f"  序列已平稳，无需差分 (d=0)")
        else:
            print(f"  需要 {d} 阶差分使序列平稳 (d={d})")

    # Step 3: 搜索最优参数
    if verbose:
        print("\n【Step 3: 搜索最优ARIMA参数】")

    best_p, best_d, best_q, best_aic = search_best_arima_params(
        series, d, p_range, q_range, verbose=verbose
    )
    order = (best_p, best_d, best_q)

    # Step 4: 拟合模型
    if verbose:
        print(f"\n【Step 4: 拟合ARIMA{order}模型】")

    fitted_model = fit_arima_model(series, order)
    fitted_values = fitted_model.fittedvalues

    if verbose:
        print(f"  模型拟合完成")
        print(f"  AIC: {fitted_model.aic:.2f}")
        print(f"  BIC: {fitted_model.bic:.2f}")

    # Step 5: 计算评价指标
    metrics = calc_model_metrics(series, fitted_values)

    if verbose:
        print("\n【Step 5: 模型评价指标】")
        print(f"  RMSE (均方根误差): {metrics['rmse']:.4f}")
        print(f"  MAE (平均绝对误差): {metrics['mae']:.4f}")
        print(f"  MAPE (平均绝对百分比误差): {metrics['mape']:.2f}%")
        print(f"  R² (决定系数): {metrics['r_squared']:.4f}")

    # Step 6: 预测
    if verbose:
        print(f"\n【Step 6: 预测未来{forecast_days}天】")

    forecast_df = forecast_arima(fitted_model, forecast_days, series)

    if verbose:
        print(forecast_df.to_string())

    # Step 7: 绘图
    if verbose:
        print("\n【Step 7: 生成可视化图表】")

    # 主图：拟合与预测
    fig_main = plot_arima_results(
        series,
        fitted_values,
        forecast_df,
        order,
        title="ARIMA模型拟合与预测",
        save_path=save_dir / "arima_forecast.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_main)

    # 残差诊断图
    fig_resid = plot_residual_diagnostics(
        fitted_model,
        title=f"ARIMA{order} 残差诊断",
        save_path=save_dir / "arima_residuals.png" if save_dir else None,
        show=show_plots,
    )
    figures.append(fig_resid)

    print("=" * 60)

    # 汇总结果
    result = {
        "model": fitted_model,
        "forecast": forecast_df,
        "metrics": metrics,
        "order": order,
        "aic": best_aic,
        "adf_test": adf_result,
        "fitted_values": fitted_values,
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
        print("用法: python -m analysis.arima_model <csv_path> [forecast_days] [output_dir]")
        print("      python -m analysis.arima_model <csv_path> [output_dir]")
        print("示例: python -m analysis.arima_model data/xxx.csv 5 output/")
        print("      python -m analysis.arima_model data/xxx.csv output/")
        sys.exit(1)

    csv_path = sys.argv[1]
    forecast_days = 5  # 默认值
    output_dir = None

    # 智能解析参数：判断第二个参数是数字还是路径
    if len(sys.argv) > 2:
        try:
            forecast_days = int(sys.argv[2])
            output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        except ValueError:
            # 第二个参数不是数字，当作 output_dir
            output_dir = sys.argv[2]

    print(f"读取数据: {csv_path}")

    # 读取数据
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 尝试找到日期和价格列
    date_col = None
    for col in ["日期", "date", "Date"]:
        if col in df.columns:
            date_col = col
            break

    price_col = None
    for col in ["收盘", "close", "Close", "收盘价"]:
        if col in df.columns:
            price_col = col
            break

    if date_col is None or price_col is None:
        print("错误: 未找到日期列或收盘价列")
        sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    prices = df[price_col].astype(float)

    result = build_arima_model(prices, forecast_days=forecast_days, save_dir=output_dir)

    print("\n预测完成。")
