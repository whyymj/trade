#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分析报告生成器

对股票CSV数据进行全部分析模块处理，生成结构化的Markdown报告，
便于AI阅读和进一步分析。

包含模块：
1. 时域与统计特征分析
2. 频域（周期性）分析
3. ARIMA预测模型
4. 几何形态相似度（需要对比数据时）
5. 非线性与复杂度分析

使用示例:
    python -m analysis.full_report data/xxx.csv output/report/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# 导入各分析模块
from analysis.time_domain import analyze_time_domain
from analysis.frequency_domain import analyze_frequency_domain, calc_returns_from_prices
from analysis.arima_model import build_arima_model
from analysis.complexity import analyze_complexity


def load_stock_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    加载股票CSV数据。

    Args:
        csv_path: CSV文件路径

    Returns:
        tuple: (完整DataFrame, 价格Series, 股票名称)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 找日期列并设为索引
    date_col = None
    for col in ["日期", "date", "Date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

    # 找收盘价列
    price_col = None
    for col in ["收盘", "close", "Close", "收盘价"]:
        if col in df.columns:
            price_col = col
            break

    if price_col is None:
        raise ValueError("未找到收盘价列")

    prices = df[price_col].astype(float)

    # 提取股票名称
    stock_name = Path(csv_path).stem
    if "股票代码" in df.columns:
        stock_name = f"{df['股票代码'].iloc[0]}_{stock_name}"

    return df, prices, stock_name


def format_number(value: Any, decimals: int = 4) -> str:
    """格式化数值为字符串。"""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.{decimals}f}"
    return str(value)


def generate_data_overview(df: pd.DataFrame, prices: pd.Series, stock_name: str) -> str:
    """生成数据概览部分。"""
    sections = []
    sections.append("## 1. 数据概览\n")

    sections.append(f"- **股票标识**: {stock_name}")
    sections.append(f"- **数据时间范围**: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}")
    sections.append(f"- **数据点数量**: {len(df)} 个交易日")
    sections.append(f"- **数据列**: {', '.join(df.columns.tolist())}")

    # 价格统计
    sections.append("\n### 价格基础统计\n")
    sections.append(f"| 指标 | 数值 |")
    sections.append(f"|------|------|")
    sections.append(f"| 起始价格 | {format_number(prices.iloc[0], 2)} |")
    sections.append(f"| 最新价格 | {format_number(prices.iloc[-1], 2)} |")
    sections.append(f"| 最高价格 | {format_number(prices.max(), 2)} |")
    sections.append(f"| 最低价格 | {format_number(prices.min(), 2)} |")
    sections.append(f"| 平均价格 | {format_number(prices.mean(), 2)} |")
    sections.append(f"| 价格标准差 | {format_number(prices.std(), 2)} |")
    sections.append(f"| 区间涨跌幅 | {format_number((prices.iloc[-1] / prices.iloc[0] - 1) * 100, 2)}% |")

    return "\n".join(sections)


def generate_time_domain_section(result: dict) -> str:
    """生成时域分析部分。"""
    sections = []
    sections.append("\n## 2. 时域与统计特征分析\n")

    stats = result.get("basic_stats", {})

    sections.append("### 基础统计量\n")
    sections.append("| 指标 | 数值 | 说明 |")
    sections.append("|------|------|------|")
    sections.append(f"| 均值 (Mean) | {format_number(stats.get('mean'))} | 平均价格水平 |")
    sections.append(f"| 标准差 (Std) | {format_number(stats.get('std'))} | 价格波动程度 |")
    sections.append(f"| 偏度 (Skewness) | {format_number(stats.get('skewness'))} | >0右偏, <0左偏 |")
    sections.append(f"| 峰度 (Kurtosis) | {format_number(stats.get('kurtosis'))} | >0尖峰, <0平坦 |")
    sections.append(f"| 最大回撤 | {format_number(stats.get('max_drawdown', 0) * 100, 2)}% | 最大跌幅 |")

    # 偏度峰度解读
    skew = stats.get('skewness', 0)
    kurt = stats.get('kurtosis', 0)

    sections.append("\n### 统计特征解读\n")
    if skew > 0.5:
        sections.append("- **偏度分析**: 右偏分布，存在较大正向波动的可能")
    elif skew < -0.5:
        sections.append("- **偏度分析**: 左偏分布，存在较大负向波动的可能")
    else:
        sections.append("- **偏度分析**: 接近对称分布")

    if kurt > 1:
        sections.append("- **峰度分析**: 尖峰厚尾，极端波动概率高于正态分布")
    elif kurt < -1:
        sections.append("- **峰度分析**: 平峰分布，波动相对温和")
    else:
        sections.append("- **峰度分析**: 接近正态分布")

    sections.append("\n### 生成图表\n")
    sections.append("- `ma_plot.png`: 收盘价与移动平均线叠加图")
    sections.append("- `stl_decomposition.png`: STL季节性分解（趋势/季节/残差）")
    sections.append("- `acf_plot.png`: 自相关函数图")

    return "\n".join(sections)


def generate_frequency_domain_section(result: dict) -> str:
    """生成频域分析部分。"""
    sections = []
    sections.append("\n## 3. 频域（周期性）分析\n")

    dominant = result.get("dominant_periods", [])

    sections.append("### 主要周期成分\n")
    sections.append("| 排名 | 周期（天） | 频率 (1/天) | 功率 |")
    sections.append("|------|-----------|-------------|------|")

    for i, dp in enumerate(dominant[:5]):
        sections.append(
            f"| #{i+1} | {format_number(dp['period_days'], 1)} | "
            f"{format_number(dp['frequency'])} | {dp['power']:.2e} |"
        )

    # 周期解读
    sections.append("\n### 周期特征解读\n")
    if dominant:
        main_period = dominant[0]['period_days']
        if main_period <= 7:
            sections.append(f"- **主周期 {main_period:.1f} 天**: 接近一周，可能反映周内交易模式或短期情绪波动")
        elif 15 <= main_period <= 25:
            sections.append(f"- **主周期 {main_period:.1f} 天**: 接近一个月，可能反映月度资金流动或期权到期效应")
        elif 55 <= main_period <= 70:
            sections.append(f"- **主周期 {main_period:.1f} 天**: 接近一个季度，可能与财报发布周期相关")
        elif main_period >= 200:
            sections.append(f"- **主周期 {main_period:.1f} 天**: 长周期，可能反映宏观经济或年度季节性因素")
        else:
            sections.append(f"- **主周期 {main_period:.1f} 天**: 可能与行业特定事件或市场结构相关")

    sections.append("\n### 生成图表\n")
    sections.append("- `power_spectrum.png`: 功率谱密度图（频率域+周期域）")
    if result.get("wavelet_result"):
        sections.append("- `wavelet_spectrogram.png`: 小波时频谱图")

    return "\n".join(sections)


def generate_arima_section(result: dict) -> str:
    """生成ARIMA预测部分。"""
    sections = []
    sections.append("\n## 4. ARIMA预测模型\n")

    order = result.get("order", (0, 0, 0))
    metrics = result.get("metrics", {})
    adf = result.get("adf_test", {})

    sections.append(f"### 模型参数: ARIMA{order}\n")

    # 平稳性检验
    sections.append("### 平稳性检验 (ADF)\n")
    sections.append("| 指标 | 数值 |")
    sections.append("|------|------|")
    sections.append(f"| ADF统计量 | {format_number(adf.get('adf_statistic'))} |")
    sections.append(f"| p值 | {format_number(adf.get('p_value'))} |")
    sections.append(f"| 原序列平稳 | {'是' if adf.get('is_stationary') else '否'} |")
    sections.append(f"| 差分阶数 (d) | {order[1]} |")

    # 模型评价
    sections.append("\n### 模型评价指标\n")
    sections.append("| 指标 | 数值 | 说明 |")
    sections.append("|------|------|------|")
    sections.append(f"| RMSE | {format_number(metrics.get('rmse'))} | 均方根误差，越小越好 |")
    sections.append(f"| MAE | {format_number(metrics.get('mae'))} | 平均绝对误差 |")
    sections.append(f"| MAPE | {format_number(metrics.get('mape'), 2)}% | 平均绝对百分比误差 |")
    sections.append(f"| R² | {format_number(metrics.get('r_squared'))} | 拟合优度，越接近1越好 |")
    sections.append(f"| AIC | {format_number(result.get('aic'))} | 信息准则，越小越好 |")

    # 预测结果
    forecast = result.get("forecast")
    if forecast is not None and len(forecast) > 0:
        sections.append("\n### 未来价格预测\n")
        sections.append("| 日期 | 预测值 | 95%置信下界 | 95%置信上界 |")
        sections.append("|------|--------|-------------|-------------|")
        for idx, row in forecast.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            sections.append(
                f"| {date_str} | {format_number(row['预测值'], 2)} | "
                f"{format_number(row['下界'], 2)} | {format_number(row['上界'], 2)} |"
            )

    sections.append("\n### 模型解读\n")
    r2 = metrics.get('r_squared', 0)
    if r2 > 0.95:
        sections.append("- **拟合质量**: 优秀 (R² > 0.95)，模型对历史数据解释力强")
    elif r2 > 0.85:
        sections.append("- **拟合质量**: 良好 (R² > 0.85)，模型较好捕捉价格走势")
    elif r2 > 0.7:
        sections.append("- **拟合质量**: 中等 (R² > 0.70)，存在未捕捉的波动")
    else:
        sections.append("- **拟合质量**: 较差 (R² < 0.70)，价格可能存在复杂非线性特征")

    sections.append("\n### 生成图表\n")
    sections.append("- `arima_forecast.png`: 历史数据、拟合曲线与预测区间")
    sections.append("- `arima_residuals.png`: 残差诊断图")

    return "\n".join(sections)


def generate_complexity_section(result: dict) -> str:
    """生成复杂度分析部分。"""
    sections = []
    sections.append("\n## 5. 非线性与复杂度分析\n")

    # 赫斯特指数
    sections.append("### 赫斯特指数 (Hurst Exponent)\n")
    hurst = result.get("hurst_exponent")
    sections.append(f"- **H = {format_number(hurst)}**")
    sections.append(f"- **解读**: {result.get('hurst_interpretation', 'N/A')}")

    sections.append("\n#### 赫斯特指数参考\n")
    sections.append("| H值范围 | 市场特征 | 策略建议 |")
    sections.append("|---------|----------|----------|")
    sections.append("| H < 0.5 | 均值回复 | 适合反转策略 |")
    sections.append("| H ≈ 0.5 | 随机游走 | 难以预测 |")
    sections.append("| H > 0.5 | 趋势增强 | 适合趋势跟踪 |")

    # 样本熵
    sections.append("\n### 样本熵 (Sample Entropy)\n")
    se = result.get("sample_entropy")
    sections.append(f"- **SampEn = {format_number(se)}**")
    sections.append(f"- **解读**: {result.get('entropy_interpretation', 'N/A')}")

    # 近似熵
    ae = result.get("approximate_entropy")
    sections.append(f"- **ApEn = {format_number(ae)}**")

    # 分布特征
    sections.append("\n### 收益率分布特征\n")
    sections.append("| 指标 | 数值 | 说明 |")
    sections.append("|------|------|------|")
    sections.append(f"| 偏度 | {format_number(result.get('skewness'))} | 分布对称性 |")
    sections.append(f"| 峰度 | {format_number(result.get('kurtosis'))} | 尾部厚度 |")
    sections.append(f"| JB统计量 | {format_number(result.get('jarque_bera_stat'))} | 正态性检验 |")
    sections.append(f"| JB p值 | {format_number(result.get('jarque_bera_pvalue'))} | <0.05拒绝正态 |")

    jb_p = result.get('jarque_bera_pvalue', 1)
    if jb_p < 0.05:
        sections.append("\n- **正态性检验**: 拒绝正态分布假设，收益率存在非线性特征")
    else:
        sections.append("\n- **正态性检验**: 无法拒绝正态分布假设")

    sections.append("\n### 生成图表\n")
    sections.append("- `hurst_analysis.png`: 赫斯特指数R/S分析")
    sections.append("- `phase_space_2d.png`: 2D滞后相空间图")
    sections.append("- `phase_space_3d.png`: 3D滞后相空间图")
    sections.append("- `return_distribution.png`: 收益率分布与Q-Q图")

    return "\n".join(sections)


def generate_summary_section(
    time_result: dict,
    freq_result: dict,
    arima_result: dict,
    complexity_result: dict,
) -> str:
    """生成综合结论部分。"""
    sections = []
    sections.append("\n## 6. 综合分析结论\n")

    # 收集关键指标
    hurst = complexity_result.get("hurst_exponent", 0.5)
    sample_entropy = complexity_result.get("sample_entropy", 1.0)
    r2 = arima_result.get("metrics", {}).get("r_squared", 0)
    max_dd = time_result.get("basic_stats", {}).get("max_drawdown", 0)
    dominant_periods = freq_result.get("dominant_periods", [])

    sections.append("### 关键发现\n")

    # 趋势特征
    if hurst > 0.6:
        sections.append("1. **趋势特征**: 序列呈现明显的趋势增强特征 (H={:.3f})，价格走势具有持续性".format(hurst))
    elif hurst < 0.4:
        sections.append("1. **趋势特征**: 序列呈现均值回复特征 (H={:.3f})，价格倾向于向均值回归".format(hurst))
    else:
        sections.append("1. **趋势特征**: 序列接近随机游走 (H={:.3f})，短期预测难度较大".format(hurst))

    # 复杂度
    if sample_entropy < 1.0:
        sections.append("2. **复杂度**: 序列复杂度较低 (SampEn={:.3f})，存在一定的规律性和可预测性".format(sample_entropy))
    elif sample_entropy > 1.5:
        sections.append("2. **复杂度**: 序列复杂度较高 (SampEn={:.3f})，随机性强，预测难度大".format(sample_entropy))
    else:
        sections.append("2. **复杂度**: 序列复杂度中等 (SampEn={:.3f})".format(sample_entropy))

    # 周期性
    if dominant_periods:
        main_period = dominant_periods[0]['period_days']
        sections.append(f"3. **周期特征**: 主周期约 {main_period:.1f} 天，可作为交易周期参考")

    # 模型拟合
    if r2 > 0.9:
        sections.append(f"4. **可预测性**: ARIMA模型拟合优度高 (R²={r2:.3f})，短期预测相对可靠")
    else:
        sections.append(f"4. **可预测性**: ARIMA模型拟合一般 (R²={r2:.3f})，预测需谨慎参考")

    # 风险
    sections.append(f"5. **风险指标**: 最大回撤 {max_dd*100:.1f}%")

    # 策略建议
    sections.append("\n### 策略建议\n")

    if hurst > 0.55 and r2 > 0.85:
        sections.append("- 趋势特征明显且模型拟合良好，**趋势跟踪策略**可能有效")
        sections.append("- 可参考ARIMA预测方向辅助决策")
    elif hurst < 0.45:
        sections.append("- 均值回复特征，**网格交易或均值回归策略**可能有效")
        sections.append("- 关注价格偏离均线的程度")
    else:
        sections.append("- 序列接近随机，建议**短线谨慎，中长期观察**")
        sections.append("- 可结合其他基本面/技术面因素综合判断")

    if sample_entropy > 1.5:
        sections.append("- 高复杂度表明市场效率较高，**量化模型预测优势有限**")

    return "\n".join(sections)


def generate_json_summary(
    stock_name: str,
    prices: pd.Series,
    time_result: dict,
    freq_result: dict,
    arima_result: dict,
    complexity_result: dict,
) -> dict:
    """生成JSON格式的结构化摘要，便于程序化处理。"""
    return {
        "stock_info": {
            "name": stock_name,
            "data_points": len(prices),
            "start_date": str(prices.index.min()),
            "end_date": str(prices.index.max()),
            "start_price": float(prices.iloc[0]),
            "end_price": float(prices.iloc[-1]),
            "total_return": float((prices.iloc[-1] / prices.iloc[0] - 1) * 100),
        },
        "time_domain": {
            "mean": time_result.get("basic_stats", {}).get("mean"),
            "std": time_result.get("basic_stats", {}).get("std"),
            "skewness": time_result.get("basic_stats", {}).get("skewness"),
            "kurtosis": time_result.get("basic_stats", {}).get("kurtosis"),
            "max_drawdown": time_result.get("basic_stats", {}).get("max_drawdown"),
        },
        "frequency_domain": {
            "dominant_periods": [
                {"period_days": dp["period_days"], "power": dp["power"]}
                for dp in freq_result.get("dominant_periods", [])[:3]
            ],
        },
        "arima": {
            "order": arima_result.get("order"),
            "aic": arima_result.get("aic"),
            "rmse": arima_result.get("metrics", {}).get("rmse"),
            "r_squared": arima_result.get("metrics", {}).get("r_squared"),
            "forecast": [
                {
                    "date": str(idx),
                    "predicted": float(row["预测值"]),
                    "lower": float(row["下界"]),
                    "upper": float(row["上界"]),
                }
                for idx, row in arima_result.get("forecast", pd.DataFrame()).iterrows()
            ],
        },
        "complexity": {
            "hurst_exponent": complexity_result.get("hurst_exponent"),
            "hurst_interpretation": complexity_result.get("hurst_interpretation"),
            "sample_entropy": complexity_result.get("sample_entropy"),
            "entropy_interpretation": complexity_result.get("entropy_interpretation"),
            "approximate_entropy": complexity_result.get("approximate_entropy"),
            "is_normal_distribution": bool(complexity_result.get("jarque_bera_pvalue", 0) >= 0.05),
        },
    }


def generate_full_report(
    csv_path: str | Path,
    output_dir: str | Path,
    forecast_days: int = 5,
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    生成完整的股票分析报告。

    Args:
        csv_path: CSV数据文件路径
        output_dir: 输出目录
        forecast_days: ARIMA预测天数
        show_plots: 是否显示图表

    Returns:
        dict: 包含所有分析结果的字典
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("【综合分析报告生成器】")
    print("=" * 70)
    print(f"输入文件: {csv_path}")
    print(f"输出目录: {output_dir}")

    # 加载数据
    print("\n[1/6] 加载数据...")
    df, prices, stock_name = load_stock_data(csv_path)
    returns = calc_returns_from_prices(prices)
    print(f"  股票: {stock_name}, 数据点: {len(prices)}")

    # 模块1: 时域分析
    print("\n[2/6] 运行时域分析...")
    time_dir = output_dir / "time_domain"
    time_result = analyze_time_domain(
        df, save_dir=time_dir, show_plots=show_plots
    )

    # 模块2: 频域分析
    print("\n[3/6] 运行频域分析...")
    freq_dir = output_dir / "frequency_domain"
    freq_result = analyze_frequency_domain(
        returns, save_dir=freq_dir, show_plots=show_plots
    )

    # 模块3: ARIMA预测
    print("\n[4/6] 运行ARIMA预测...")
    arima_dir = output_dir / "arima"
    arima_result = build_arima_model(
        prices,
        forecast_days=forecast_days,
        save_dir=arima_dir,
        show_plots=show_plots,
        verbose=False,
    )

    # 模块5: 复杂度分析
    print("\n[5/6] 运行复杂度分析...")
    complexity_dir = output_dir / "complexity"
    complexity_result = analyze_complexity(
        returns, save_dir=complexity_dir, show_plots=show_plots
    )

    # 生成报告
    print("\n[6/6] 生成分析报告...")

    # Markdown报告
    report_parts = []
    report_parts.append(f"# {stock_name} 综合分析报告\n")
    report_parts.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_parts.append(f"*数据来源: {csv_path.name}*\n")

    report_parts.append(generate_data_overview(df, prices, stock_name))
    report_parts.append(generate_time_domain_section(time_result))
    report_parts.append(generate_frequency_domain_section(freq_result))
    report_parts.append(generate_arima_section(arima_result))
    report_parts.append(generate_complexity_section(complexity_result))
    report_parts.append(generate_summary_section(
        time_result, freq_result, arima_result, complexity_result
    ))

    # 附录：图表列表
    report_parts.append("\n## 附录：生成的图表文件\n")
    report_parts.append("```")
    report_parts.append(f"{output_dir.name}/")
    report_parts.append("├── time_domain/")
    report_parts.append("│   ├── ma_plot.png")
    report_parts.append("│   ├── stl_decomposition.png")
    report_parts.append("│   └── acf_plot.png")
    report_parts.append("├── frequency_domain/")
    report_parts.append("│   ├── power_spectrum.png")
    report_parts.append("│   └── wavelet_spectrogram.png")
    report_parts.append("├── arima/")
    report_parts.append("│   ├── arima_forecast.png")
    report_parts.append("│   └── arima_residuals.png")
    report_parts.append("├── complexity/")
    report_parts.append("│   ├── hurst_analysis.png")
    report_parts.append("│   ├── phase_space_2d.png")
    report_parts.append("│   ├── phase_space_3d.png")
    report_parts.append("│   └── return_distribution.png")
    report_parts.append("├── report.md")
    report_parts.append("└── summary.json")
    report_parts.append("```")

    markdown_report = "\n".join(report_parts)

    # 保存Markdown报告
    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"  Markdown报告: {report_path}")

    # JSON摘要
    json_summary = generate_json_summary(
        stock_name, prices, time_result, freq_result, arima_result, complexity_result
    )
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, ensure_ascii=False, indent=2)
    print(f"  JSON摘要: {json_path}")

    print("\n" + "=" * 70)
    print("报告生成完成!")
    print(f"输出目录: {output_dir.resolve()}")
    print("=" * 70)

    return {
        "stock_name": stock_name,
        "time_domain": time_result,
        "frequency_domain": freq_result,
        "arima": arima_result,
        "complexity": complexity_result,
        "report_path": report_path,
        "json_path": json_path,
    }


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python -m analysis.full_report <csv_path> [output_dir] [forecast_days]")
        print("示例: python -m analysis.full_report data/xxx.csv output/report/ 5")
        sys.exit(1)

    csv_path = sys.argv[1]

    # 智能解析参数
    output_dir = None
    forecast_days = 5

    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        try:
            forecast_days = int(arg2)
            output_dir = Path(csv_path).stem + "_report"
        except ValueError:
            output_dir = arg2

    if len(sys.argv) > 3:
        try:
            forecast_days = int(sys.argv[3])
        except ValueError:
            pass

    if output_dir is None:
        output_dir = Path(csv_path).stem + "_report"

    result = generate_full_report(
        csv_path,
        output_dir,
        forecast_days=forecast_days,
        show_plots=False,
    )

    print(f"\n报告已保存: {result['report_path']}")
