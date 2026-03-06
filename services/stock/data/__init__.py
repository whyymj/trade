#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock 数据仓储 - 股票服务专用
"""

from shared.db import fetch_all, fetch_one


def get_stock_data(symbol: str, days: int = 60):
    """获取股票日线数据"""
    import pandas as pd

    from data.stock_repo import get_stock_daily_df

    df = get_stock_daily_df(symbol)
    if df is None:
        return None

    if len(df) > days:
        df = df.tail(days)

    df = df.sort_values("日期").reset_index(drop=True)

    return df


def get_stock_list(page: int = 1, size: int = 20):
    """获取股票列表"""
    offset = (page - 1) * size

    sql = """
    SELECT 
        symbol, name, market,
        last_trade_date, first_trade_date,
        created_at, updated_at
    FROM stock_meta
    ORDER BY symbol
    LIMIT %s OFFSET %s
    """
    rows = fetch_all(sql, (size, offset))

    count_sql = "SELECT COUNT(*) as total FROM stock_meta"
    count_row = fetch_one(count_sql)
    total = count_row["total"] if count_row else 0

    return {
        "items": rows,
        "total": total,
        "page": page,
        "size": size,
        "pages": (total + size - 1) // size,
    }
