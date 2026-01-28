#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 akshare 获取股票数据并生成数据文件。
可自由配置股票编号与时间范围，生成的文件以「股票名称（日期范围）」命名。
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """加载 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_stock_name(symbol: str) -> str:
    """
    根据股票代码获取股票名称。
    使用 akshare stock_individual_info_em 接口，返回 DataFrame 中「股票简称」对应的值。
    """
    try:
        df = ak.stock_individual_info_em(symbol=symbol)
        # 常见为两列：项/值 或 item/value，取「股票简称」对应的值
        if df is None or df.empty:
            return symbol
        # 兼容列名为 item/value 或 第一列/第二列
        cols = df.columns.tolist()
        if len(cols) >= 2:
            key_col, val_col = cols[0], cols[1]
            row = df[df[key_col].astype(str).str.strip() == "股票简称"]
            if not row.empty:
                return str(row[val_col].iloc[0]).strip()
        return symbol
    except Exception as e:
        print(f"  警告: 获取股票名称失败 {symbol}, 使用代码作为名称. 错误: {e}", file=sys.stderr)
        return symbol


def fetch_hist(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame | None:
    """获取单只股票日线历史数据。"""
    try:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception as e:
        print(f"  错误: 获取历史数据失败 {symbol}: {e}", file=sys.stderr)
        return None


def main():
    config_path = "config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    start_date = (config.get("start_date") or "").strip().replace("-", "")
    end_date = (config.get("end_date") or "").strip().replace("-", "")
    adjust = config.get("adjust", "qfq") or ""
    stocks = config.get("stocks", [])
    output_dir = (config.get("output_dir") or "").strip()

    if not stocks:
        print("配置中未指定 stocks 列表，请编辑 config.yaml 添加股票代码。", file=sys.stderr)
        sys.exit(1)

    # 未配置或空字符串时，默认近一年数据
    today = datetime.now().date()
    if not end_date:
        end_date = today.strftime("%Y%m%d")
    if not start_date:
        one_year_ago = today - timedelta(days=365)
        start_date = one_year_ago.strftime("%Y%m%d")

    out_path = Path(output_dir) if output_dir else Path(".")
    out_path.mkdir(parents=True, exist_ok=True)

    # 清空输出目录中的旧数据文件（.csv）
    for f in out_path.glob("*.csv"):
        try:
            f.unlink()
        except OSError as e:
            print(f"  警告: 删除旧文件失败 {f}: {e}", file=sys.stderr)

    date_range_str = f"{start_date}-{end_date}"

    for symbol in stocks:
        symbol = str(symbol).strip()
        if not symbol:
            continue
        print(f"处理: {symbol} ...")
        name = get_stock_name(symbol)
        df = fetch_hist(symbol, start_date, end_date, adjust)
        if df is None or df.empty:
            print(f"  跳过 {symbol}（无数据）")
            continue
        # 文件名：股票名称（日期范围）.csv，去除名称中空格与非法文件名字符
        name_clean = name.replace(" ", "")
        safe_name = "".join(c for c in name_clean if c not in r'\/:*?"<>|')
        filename = f"{safe_name}（{date_range_str}）.csv"
        filepath = out_path / filename
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"  已保存: {filepath} ({len(df)} 条)")

    print("完成。")


if __name__ == "__main__":
    main()
