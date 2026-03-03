#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金数据获取与处理

支持:
1. CSV 文件导入（推荐）
2. 天天基金网页爬取（需要处理反爬虫）
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def load_fund_nav_from_csv(
    csv_path: str | Path,
    date_col: str = "日期",
    nav_col: str = "净值",
    acc_nav_col: Optional[str] = "累计净值",
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    从 CSV 文件加载基金净值数据。

    Args:
        csv_path: CSV 文件路径
        date_col: 日期列名（支持：日期、date、Date）
        nav_col: 净值列名（支持：净值、nav、NAV、单位净值）
        acc_nav_col: 累计净值列名（可选）

    Returns:
        tuple: (完整DataFrame, 净值Series, 基金名称)
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    date_col_found = None
    for col in [date_col, "日期", "date", "Date"]:
        if col in df.columns:
            date_col_found = col
            break

    if date_col_found is None:
        raise ValueError(f"未找到日期列，可用列: {df.columns.tolist()}")

    df[date_col_found] = pd.to_datetime(df[date_col_found])
    df = df.set_index(date_col_found).sort_index()

    nav_col_found = None
    for col in [nav_col, "净值", "nav", "NAV", "单位净值"]:
        if col in df.columns:
            nav_col_found = col
            break

    if nav_col_found is None:
        raise ValueError(f"未找到净值列，可用列: {df.columns.tolist()}")

    nav = df[nav_col_found].astype(float)

    acc_nav = None
    if acc_nav_col and acc_nav_col in df.columns:
        acc_nav = df[acc_nav_col].astype(float)

    fund_name = Path(csv_path).stem

    result_df = df.copy()
    if acc_nav is not None:
        result_df["净值"] = nav
        result_df["累计净值"] = acc_nav
    else:
        result_df["净值"] = nav

    return result_df, nav, fund_name


def load_fund_from_csv(csv_path: str | Path) -> pd.Series:
    """
    简化版：从 CSV 加载基金净值 Series。

    Returns:
        净值 Series，索引为日期
    """
    _, nav, _ = load_fund_nav_from_csv(csv_path)
    return nav
