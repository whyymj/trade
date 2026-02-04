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
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis.cleanup_temp import ANALYSIS_TEMP_PREFIX, cleanup_analysis_temp_dirs

import pandas as pd

# 非交互后端，避免在 Flask 等非主线程中报错（如 MacOS backend）
import matplotlib
matplotlib.use("agg")

# 导入各分析模块
from analysis.time_domain import analyze_time_domain
from analysis.frequency_domain import analyze_frequency_domain, calc_returns_from_prices
from analysis.arima_model import build_arima_model
from analysis.complexity import analyze_complexity
from analysis.technical import analyze_technical


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


def _forecast_to_list(forecast: Any) -> list[dict]:
    """将 ARIMA 的 forecast（DataFrame 或 None）转为列表，避免对 DataFrame 做布尔判断。"""
    if forecast is None:
        return []
    if isinstance(forecast, pd.DataFrame) and forecast.empty:
        return []
    if not isinstance(forecast, pd.DataFrame):
        return []
    return [
        {
            "date": str(idx),
            "predicted": float(row["预测值"]),
            "lower": float(row["下界"]),
            "upper": float(row["上界"]),
        }
        for idx, row in forecast.iterrows()
    ]


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
    """生成ARIMA预测部分。若因数据不足等原因未执行，则仅输出说明。"""
    sections = []
    sections.append("\n## 4. ARIMA预测模型\n")

    if result.get("skip_reason"):
        sections.append(f"**未执行**: {result['skip_reason']}")
        sections.append("\n请扩大时间范围（如选择至少约 1.5 个月以上交易日数据）后重新分析。")
        return "\n".join(sections)

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
    """生成复杂度分析部分。若因数据不足等原因未执行，则仅输出说明。"""
    sections = []
    sections.append("\n## 5. 非线性与复杂度分析\n")

    if result.get("skip_reason"):
        sections.append(f"**未执行**: {result['skip_reason']}")
        sections.append("\n请扩大时间范围（如选择至少约 3 个月以上交易日数据）后重新分析。")
        return "\n".join(sections)

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


def generate_technical_section(tech_result: dict) -> str:
    """生成技术指标与风险部分。"""
    sections = []
    sections.append("\n## 技术指标与风险\n")
    last = tech_result.get("last") or {}
    sections.append("| 指标 | 当前值 | 说明 |")
    sections.append("|------|--------|------|")
    rsi = last.get("rsi")
    sections.append(f"| RSI(14) | {format_number(rsi)} | >70 超买，<30 超卖 |")
    sections.append(f"| MACD | {format_number(last.get('macd'))} | 快慢线差 |")
    sections.append(f"| MACD柱 | {format_number(last.get('macd_hist'))} | 柱由负转正可视为金叉 |")
    sections.append(f"| 布林上/中/下 | {format_number(last.get('bb_upper'))} / {format_number(last.get('bb_mid'))} / {format_number(last.get('bb_lower'))} | 价格触及上轨偏超买、下轨偏超卖 |")
    vol = last.get("volatility_annual")
    sections.append(f"| 年化波动率(20日) | {format_number(vol, 4)} | 收益率滚动标准差年化 |")
    var = tech_result.get("var_95")
    cvar = tech_result.get("cvar_95")
    sections.append(f"| VaR(95%) | {format_number(var, 4)} | 日收益 95% 置信下界（负为亏损） |")
    sections.append(f"| CVaR(95%) | {format_number(cvar, 4)} | 超过 VaR 时的平均亏损 |")
    # 价量指标
    sections.append(f"| OBV（能量潮） | {format_number(last.get('obv'))} | 价涨加量、价跌减量累计 |")
    sections.append(f"| MFI(14) | {format_number(last.get('mfi'))} | 资金流量指数，>80 超买 <20 超卖 |")
    sections.append(f"| Aroon 上/下(20) | {format_number(last.get('aroon_up'))} / {format_number(last.get('aroon_down'))} | 趋势强度，上>下偏多 |")
    sections.append(f"| 累计资金净流入 | {format_number(last.get('money_flow_cumulative'))} | 典型价×量按涨跌方向累计 |")
    tier = last.get("money_flow_tier") or {}
    if tier:
        sections.append(f"| 大单/中单/小单净流入 | 大单 {format_number(tier.get('大单'))} / 中单 {format_number(tier.get('中单'))} / 小单 {format_number(tier.get('小单'))} | 按成交额分档近似 |")
    sig = last.get("signals") or {}
    if sig.get("combined"):
        sections.append(f"| 超买超卖信号 | {sig.get('combined', '')} | RSI/MFI/Aroon 综合 |")
    # 周线摘要
    weekly = (tech_result.get("by_timeframe") or {}).get("weekly")
    if weekly and weekly.get("last"):
        wl = weekly["last"]
        sections.append("\n### 周线周期（摘要）\n")
        sections.append(f"- OBV(周): {format_number(wl.get('obv'))} | MFI(周): {format_number(wl.get('mfi'))} | Aroon上/下: {format_number(wl.get('aroon_up'))} / {format_number(wl.get('aroon_down'))}")
    return "\n".join(sections)


def generate_json_summary(
    stock_name: str,
    prices: pd.Series,
    time_result: dict,
    freq_result: dict,
    arima_result: dict,
    complexity_result: dict,
    tech_result: dict | None = None,
) -> dict:
    """生成JSON格式的结构化摘要，便于程序化处理。"""
    out = {
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
            "forecast": _forecast_to_list(arima_result.get("forecast")),
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
    if tech_result:
        last = tech_result.get("last") or {}
        out["technical"] = {
            "rsi": last.get("rsi"),
            "macd": last.get("macd"),
            "macd_hist": last.get("macd_hist"),
            "bb_upper": last.get("bb_upper"),
            "bb_mid": last.get("bb_mid"),
            "bb_lower": last.get("bb_lower"),
            "volatility_annual": last.get("volatility_annual"),
            "var_95": tech_result.get("var_95"),
            "cvar_95": tech_result.get("cvar_95"),
            "obv": last.get("obv"),
            "mfi": last.get("mfi"),
            "aroon_up": last.get("aroon_up"),
            "aroon_down": last.get("aroon_down"),
            "money_flow_cumulative": last.get("money_flow_cumulative"),
            "money_flow_tier": last.get("money_flow_tier"),
            "signals": last.get("signals"),
        }
        weekly = (tech_result.get("by_timeframe") or {}).get("weekly")
        if weekly and weekly.get("last"):
            out["technical"]["weekly"] = weekly["last"]
    return out


def _build_charts_data(
    time_result: dict,
    freq_result: dict,
    complexity_result: dict,
    prices: pd.Series,
    tech_result: dict | None = None,
) -> dict[str, Any]:
    """从各分析结果提取可序列化的绘图数据，供前端 ECharts 渲染。"""
    import numpy as np
    charts: dict[str, Any] = {}

    # 频域：功率谱
    freq = freq_result.get("frequencies")
    psd = freq_result.get("psd")
    if freq is not None and psd is not None:
        try:
            charts["frequency_domain"] = {
                "frequencies": np.asarray(freq).tolist(),
                "psd": np.asarray(psd).tolist(),
            }
        except Exception:
            pass

    # 时域：STL 分解
    stl = time_result.get("stl_decomposition")
    if stl is not None and isinstance(stl, dict):
        try:
            trend = stl.get("trend")
            seasonal = stl.get("seasonal")
            resid = stl.get("resid")
            if trend is not None and hasattr(trend, "index"):
                dates = [str(t) for t in trend.index]
                charts["time_domain"] = charts.get("time_domain", {})
                charts["time_domain"]["stl"] = {
                    "dates": dates,
                    "trend": np.asarray(trend).tolist(),
                    "seasonal": np.asarray(seasonal).tolist() if seasonal is not None else [],
                    "resid": np.asarray(resid).tolist() if resid is not None else [],
                }
        except Exception:
            pass

    # 时域：ACF（自相关）
    try:
        from statsmodels.tsa.stattools import acf
        p = prices.dropna()
        nlags = min(40, len(p) - 1)
        if nlags > 0:
            acf_vals = acf(p, nlags=nlags)
            charts["time_domain"] = charts.get("time_domain", {})
            charts["time_domain"]["acf"] = {
                "lags": list(range(len(acf_vals))),
                "values": np.asarray(acf_vals).tolist(),
            }
    except Exception:
        pass

    # 复杂度：赫斯特 R/S 曲线
    hd = complexity_result.get("hurst_details") or {}
    if hd.get("log_lags") is not None and hd.get("log_rs") is not None:
        try:
            log_lags = np.asarray(hd["log_lags"]).tolist()
            log_rs = np.asarray(hd["log_rs"]).tolist()
            coeffs = hd.get("coefficients")
            if coeffs is not None:
                coeffs = np.asarray(coeffs).tolist()
            charts["complexity"] = {
                "hurst": {
                    "log_lags": log_lags,
                    "log_rs": log_rs,
                    "hurst": complexity_result.get("hurst_exponent"),
                    "coefficients": coeffs,
                }
            }
        except Exception:
            pass

    # 技术指标与风险（NaN 转为 None 以便 JSON 序列化）
    def _safe_list(s, default=None):
        if s is None:
            return []
        a = np.asarray(s)
        out = a.tolist()
        return [None if isinstance(x, float) and np.isnan(x) else x for x in out]

    if tech_result:
        try:
            dates = [str(t) for t in prices.index]
            rsi = tech_result.get("rsi")
            macd = tech_result.get("macd") or {}
            bb = tech_result.get("bollinger") or {}
            vol = tech_result.get("rolling_volatility")
            charts["technical"] = {
                "dates": dates,
                "rsi": _safe_list(rsi),
                "macd": _safe_list(macd.get("macd")),
                "macd_signal": _safe_list(macd.get("signal")),
                "macd_hist": _safe_list(macd.get("hist")),
                "bb_upper": _safe_list(bb.get("upper")),
                "bb_mid": _safe_list(bb.get("mid")),
                "bb_lower": _safe_list(bb.get("lower")),
                "rolling_volatility": _safe_list(vol),
            }
            # 价量指标
            obv = tech_result.get("obv")
            mfi = tech_result.get("mfi")
            aroon = tech_result.get("aroon") or {}
            mf = tech_result.get("money_flow") or {}
            if obv is not None:
                charts["technical"]["obv"] = _safe_list(obv)
            if mfi is not None:
                charts["technical"]["mfi"] = _safe_list(mfi)
            if aroon:
                charts["technical"]["aroon_up"] = _safe_list(aroon.get("aroon_up"))
                charts["technical"]["aroon_down"] = _safe_list(aroon.get("aroon_down"))
            if mf:
                charts["technical"]["money_flow_net"] = _safe_list(mf.get("net_flow"))
                charts["technical"]["money_flow_cumulative"] = _safe_list(mf.get("cumulative_net"))
                tier_series = mf.get("tier_series") or {}
                if tier_series.get("大单") is not None:
                    charts["technical"]["money_flow_large"] = _safe_list(tier_series["大单"].cumsum())
                    charts["technical"]["money_flow_mid"] = _safe_list(tier_series["中单"].cumsum())
                    charts["technical"]["money_flow_small"] = _safe_list(tier_series["小单"].cumsum())
            # 周线图表数据
            weekly = (tech_result.get("by_timeframe") or {}).get("weekly")
            if weekly:
                try:
                    w_dates = [str(t) for t in weekly["obv"].index] if weekly.get("obv") is not None else []
                    charts["technical_weekly"] = {
                        "dates": w_dates,
                        "obv": _safe_list(weekly.get("obv")),
                        "mfi": _safe_list(weekly.get("mfi")),
                        "aroon_up": _safe_list(weekly.get("aroon", {}).get("aroon_up")),
                        "aroon_down": _safe_list(weekly.get("aroon", {}).get("aroon_down")),
                        "money_flow_cumulative": _safe_list(weekly.get("money_flow", {}).get("cumulative_net")),
                    }
                except Exception:
                    pass
        except Exception:
            pass

    return charts


def _prepare_df_from_raw(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    将 akshare/API 返回的日线 DataFrame（含「日期」「收盘」列）转为 (df, prices)。
    df 索引为日期，prices 为收盘价序列。
    """
    df = df_raw.copy()
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    price_col = "收盘" if "收盘" in df.columns else None
    for c in ("close", "Close", "收盘价"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        raise ValueError("DataFrame 中未找到收盘价列")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    prices = df[price_col].astype(float)
    return df, prices


def run_analysis_from_dataframe(
    df_raw: pd.DataFrame,
    stock_name: str,
    output_dir: str | Path | None = None,
    forecast_days: int = 5,
    show_plots: bool = False,
) -> dict[str, Any]:
    """
    基于内存中的日线 DataFrame 运行完整分析，返回 JSON 摘要与 Markdown 报告文本。
    供 API 等调用，无需 CSV 文件。

    Args:
        df_raw: 日线 DataFrame，需含「日期」「收盘」列（如 akshare / fetch_hist 返回格式）
        stock_name: 股票标识（如代码或名称）
        output_dir: 图表保存目录；为 None 时使用临时目录（不持久化）
        forecast_days: ARIMA 预测天数
        show_plots: 是否显示图表

    Returns:
        dict: 含 summary（JSON 摘要）、report_md（Markdown 报告全文）
    """
    df, prices = _prepare_df_from_raw(df_raw)
    returns = calc_returns_from_prices(prices)
    use_temp = output_dir is None
    out = Path(tempfile.mkdtemp(prefix=ANALYSIS_TEMP_PREFIX)) if use_temp else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    time_dir = out / "time_domain"
    time_result = analyze_time_domain(df, save_dir=time_dir, show_plots=show_plots)

    freq_dir = out / "frequency_domain"
    freq_result = analyze_frequency_domain(returns, save_dir=freq_dir, show_plots=show_plots)

    arima_dir = out / "arima"
    try:
        arima_result = build_arima_model(
            prices,
            forecast_days=forecast_days,
            save_dir=arima_dir,
            show_plots=show_plots,
            verbose=False,
        )
    except (ValueError, TypeError) as e:
        arima_result = {
            "order": None,
            "forecast": None,
            "metrics": {},
            "adf_test": {},
            "aic": None,
            "skip_reason": str(e),
        }

    complexity_dir = out / "complexity"
    try:
        complexity_result = analyze_complexity(
            returns, save_dir=complexity_dir, show_plots=show_plots
        )
    except (ValueError, TypeError) as e:
        complexity_result = {"skip_reason": str(e)}

    tech_result = analyze_technical(prices, returns, df=df)

    json_summary = generate_json_summary(
        stock_name, prices, time_result, freq_result, arima_result, complexity_result, tech_result
    )

    report_parts = []
    report_parts.append(f"# {stock_name} 综合分析报告\n")
    report_parts.append(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_parts.append("*数据范围: 选定时间区间*\n")
    report_parts.append(generate_data_overview(df, prices, stock_name))
    report_parts.append(generate_time_domain_section(time_result))
    report_parts.append(generate_frequency_domain_section(freq_result))
    report_parts.append(generate_arima_section(arima_result))
    report_parts.append(generate_complexity_section(complexity_result))
    report_parts.append(generate_technical_section(tech_result))
    report_parts.append(
        generate_summary_section(
            time_result, freq_result, arima_result, complexity_result
        )
    )
    report_md = "\n".join(report_parts)

    charts = _build_charts_data(time_result, freq_result, complexity_result, prices, tech_result)

    if use_temp and out.exists():
        try:
            shutil.rmtree(out, ignore_errors=True)
        except Exception:
            pass

    return {
        "summary": json_summary,
        "report_md": report_md,
        "charts": charts,
    }


def build_export_document(
    result: dict[str, Any],
    export_time: str | None = None,
) -> str:
    """
    将分析结果整理为可下载的 Markdown 文档，便于存档与 AI 解析。

    文档结构：
    - YAML front matter：标题、标的、日期范围、导出时间
    - 结构化摘要：JSON 格式的 summary，供程序/AI 解析
    - 完整报告正文：report_md 全文

    Args:
        result: run_analysis_from_dataframe 的返回值（含 summary、report_md）
        export_time: 导出时间字符串，默认当前时间

    Returns:
        完整 Markdown 字符串，UTF-8
    """
    if export_time is None:
        export_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = result.get("summary") or {}
    report_md = result.get("report_md") or ""
    symbol = summary.get("stock_info", {}).get("name", "")
    start_date = summary.get("stock_info", {}).get("start_date", "")
    end_date = summary.get("stock_info", {}).get("end_date", "")

    # 确保 summary 可 JSON 序列化（去除 NaN 等）
    def _json_safe(obj):
        if obj is None:
            return None
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(x) for x in obj]
        return obj

    summary_clean = _json_safe(summary)
    summary_json = json.dumps(summary_clean, ensure_ascii=False, indent=2)

    lines = [
        "---",
        "title: \"股票综合分析报告\"",
        f"symbol: \"{symbol}\"",
        f"start_date: \"{start_date}\"",
        f"end_date: \"{end_date}\"",
        f"export_time: \"{export_time}\"",
        "---",
        "",
        "# 结构化摘要（供程序/AI解析）",
        "",
        "以下为 JSON 格式的关键指标，便于程序或大模型解析。",
        "",
        "```json",
        summary_json,
        "```",
        "",
        "---",
        "",
        "# 完整报告正文",
        "",
        report_md,
    ]
    return "\n".join(lines)


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

    # 任意位置出现 --cleanup/-c 均只做清理并退出
    if any(a in ("--cleanup", "-c") for a in sys.argv[1:]):
        n = cleanup_analysis_temp_dirs()
        print(f"已清理 {n} 个分析临时目录")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("用法: python -m analysis.full_report <csv_path> [output_dir] [forecast_days]")
        print("      python -m analysis.full_report --cleanup   # 清理遗留临时目录")
        print("示例: python -m analysis.full_report data/xxx.csv output/report/ 5")
        sys.exit(1)

    csv_path = sys.argv[1]
    if csv_path.startswith("-"):
        print("错误: 第一个参数不能为选项。请提供 CSV 文件路径。")
        print("若需清理临时目录，请使用: python -m analysis.full_report --cleanup")
        sys.exit(1)

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
