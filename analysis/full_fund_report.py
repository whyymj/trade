#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金综合分析报告生成器

整合基金特有指标、技术分析、复杂度分析，生成全面的基金分析报告。

使用示例:
    python -m analysis.full_fund_report data/fund_nav.csv output/report/
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from analysis.fund_data import load_fund_nav_from_csv
from analysis.fund_metrics import analyze_fund_metrics, format_metrics_report
from analysis.fund_benchmark import (
    compare_fund_with_benchmark,
    format_comparison_report,
    load_benchmark_csv,
)
from analysis.time_domain import analyze_time_domain
from analysis.frequency_domain import analyze_frequency_domain
from analysis.complexity import analyze_complexity
from analysis.technical import analyze_technical


def generate_fund_report(
    csv_path: str | Path,
    output_dir: Optional[str | Path] = None,
    benchmark_csv: Optional[str | Path] = None,
    risk_free_rate: float = 0.03,
) -> str:
    """
    生成基金综合分析报告。

    Args:
        csv_path: 基金净值 CSV 文件路径
        output_dir: 输出目录（可选）
        benchmark_csv: 基准指数 CSV 文件路径（可选）
        risk_free_rate: 无风险利率

    Returns:
        Markdown 格式的报告字符串
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else None

    df, nav, fund_name = load_fund_nav_from_csv(csv_path)

    lines = [
        f"# {fund_name} 综合分析报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"数据来源: {csv_path.name}",
        f"数据区间: {nav.index[0].strftime('%Y-%m-%d')} ~ {nav.index[-1].strftime('%Y-%m-%d')}",
        f"数据天数: {len(nav)} 天",
        "",
        "---",
        "",
    ]

    metrics = analyze_fund_metrics(nav, risk_free_rate=risk_free_rate)
    lines.append("## 一、基金特有指标分析")
    lines.append("")
    lines.append(format_metrics_report(metrics))
    lines.append("")

    if benchmark_csv:
        lines.append("---")
        lines.append("")
        lines.append("## 二、基准对比分析")
        lines.append("")
        try:
            benchmark_nav = load_benchmark_csv(benchmark_csv)
            comparison = compare_fund_with_benchmark(
                nav,
                benchmark_nav,
                fund_name=fund_name,
                benchmark_name=Path(benchmark_csv).stem,
            )
            lines.append(format_comparison_report(comparison))
        except Exception as e:
            lines.append(f"基准对比分析失败: {e}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 三、时域与统计特征分析")
    lines.append("")
    try:
        time_result = analyze_time_domain(nav)
        lines.append(f"均值: {time_result.get('mean', 0):.6f}")
        lines.append(f"标准差: {time_result.get('std', 0):.6f}")
        lines.append(f"偏度: {time_result.get('skewness', 0):.4f}")
        lines.append(f"峰度: {time_result.get('kurtosis', 0):.4f}")
        lines.append(f"变异系数: {time_result.get('coefficient_of_variation', 0):.4f}")
    except Exception as e:
        lines.append(f"时域分析失败: {e}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 四、频域与周期性分析")
    lines.append("")
    try:
        freq_result = analyze_frequency_domain(nav)
        if "dominant_period" in freq_result:
            lines.append(f"主周期: {freq_result.get('dominant_period', 'N/A'):.2f} 天")
        if "spectral_entropy" in freq_result:
            lines.append(f"谱熵: {freq_result.get('spectral_entropy', 0):.4f}")
    except Exception as e:
        lines.append(f"频域分析失败: {e}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 五、非线性与复杂度分析")
    lines.append("")
    try:
        complexity_result = analyze_complexity(nav.pct_change().dropna())
        if "hurst_exponent" in complexity_result:
            h = complexity_result.get("hurst_exponent", 0)
            if h < 0.5:
                interpretation = "均值回复"
            elif h > 0.5:
                interpretation = "趋势增强"
            else:
                interpretation = "随机游走"
            lines.append(f"赫斯特指数: {h:.4f} ({interpretation})")
        if "sample_entropy" in complexity_result:
            lines.append(f"样本熵: {complexity_result.get('sample_entropy', 0):.4f}")
    except Exception as e:
        lines.append(f"复杂度分析失败: {e}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 六、技术指标分析")
    lines.append("")
    try:
        tech_result = analyze_technical(nav)
        if "rsi" in tech_result:
            lines.append(f"当前 RSI(14): {tech_result.get('rsi', 0):.2f}")
        if "macd" in tech_result:
            macd_data = tech_result["macd"]
            if isinstance(macd_data, dict):
                lines.append(f"MACD: {macd_data.get('macd', 0):.4f}")
                lines.append(f"Signal: {macd_data.get('signal', 0):.4f}")
                lines.append(f"Hist: {macd_data.get('hist', 0):.4f}")
        if "bollinger_bands" in tech_result:
            bb = tech_result["bollinger_bands"]
            if isinstance(bb, dict):
                lines.append(f"布林带上轨: {bb.get('upper', 0):.4f}")
                lines.append(f"布林带中轨: {bb.get('mid', 0):.4f}")
                lines.append(f"布林带下轨: {bb.get('lower', 0):.4f}")
    except Exception as e:
        lines.append(f"技术指标分析失败: {e}")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 七、分析结论")
    lines.append("")
    lines.append("### 7.1 收益评价")
    annual_ret = metrics.get("annual_return", 0)
    if annual_ret > 0.15:
        lines.append("- 年化收益率较高（>15%），收益表现优秀")
    elif annual_ret > 0.08:
        lines.append("- 年化收益率良好（8%-15%），跑赢无风险利率")
    elif annual_ret > 0:
        lines.append("- 年化收益率一般（0-8%），建议关注")
    else:
        lines.append("- 年化收益率为负，需要关注风险")
    lines.append("")

    lines.append("### 7.2 风险评价")
    max_dd = metrics.get("max_drawdown", 0)
    if max_dd < -0.1:
        lines.append(f"- 最大回撤较大（{-max_dd:.1%}），风险较高")
    elif max_dd < -0.2:
        lines.append(f"- 最大回撤中等（{-max_dd:.1%}），风险适中")
    else:
        lines.append(f"- 最大回撤控制良好（{-max_dd:.1%}）")
    lines.append("")

    lines.append("### 7.3 风险调整收益")
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe > 1.5:
        lines.append(f"- 夏普比率优秀（{sharpe:.2f}），风险收益比高")
    elif sharpe > 1:
        lines.append(f"- 夏普比率良好（{sharpe:.2f}）")
    elif sharpe > 0:
        lines.append(f"- 夏普比率一般（{sharpe:.2f}）")
    else:
        lines.append(f"- 夏普比率为负（{sharpe:.2f}），不建议持有")
    lines.append("")

    if benchmark_csv and "alpha" in metrics:
        lines.append("### 7.4 相对基准评价")
        alpha = metrics.get("alpha", 0)
        beta = metrics.get("beta", 1)
        if alpha > 0:
            lines.append(f"- 阿尔法为正（{alpha:.2%}），跑赢基准")
        else:
            lines.append(f"- 阿尔法为负（{alpha:.2%}），跑输基准")
        lines.append(f"- 贝塔为 {beta:.2f}，波动{'高于' if beta > 1 else '低于'}基准")
        lines.append("")

    report = "\n".join(lines)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{fund_name}_报告.md"
        report_file.write_text(report, encoding="utf-8")
        print(f"报告已保存至: {report_file}")

    return report


def main():
    """命令行入口。"""
    if len(sys.argv) < 2:
        print(
            "用法: python -m analysis.full_fund_report <基金CSV> [输出目录] [基准CSV]"
        )
        print("示例: python -m analysis.full_fund_report data/fund_nav.csv output/")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    benchmark_csv = sys.argv[3] if len(sys.argv) > 3 else None

    report = generate_fund_report(csv_path, output_dir, benchmark_csv)
    print(report)


if __name__ == "__main__":
    main()
