# -*- coding: utf-8 -*-
"""
股票数据仓储：日线写入/读取、元信息维护、列表与删除。
依赖 data.mysql、data.schema（列映射）。
"""
from datetime import date
from typing import Any, Optional

import pandas as pd

from data.mysql import execute, execute_many, fetch_all, fetch_one
from data.schema import AKSHARE_DAILY_COLUMNS


def _parse_date(v: Any) -> Optional[date]:
    """将 DataFrame 中的日期格转为 date。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return None


def upsert_stock_meta(symbol: str, name: Optional[str] = None, market: str = "a") -> None:
    """插入或更新股票元信息（存在则更新 name/updated_at）。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return
    # INSERT ... ON DUPLICATE KEY UPDATE
    sql = """
    INSERT INTO stock_meta (symbol, name, market)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE name = COALESCE(VALUES(name), name), updated_at = CURRENT_TIMESTAMP
    """
    execute(sql, (symbol, (name or "").strip() or None, (market or "a").strip()[:8]))


def save_stock_daily_batch(symbol: str, df: pd.DataFrame) -> int:
    """
    将 akshare 日线 DataFrame 批量写入 stock_daily，按 (symbol, trade_date) 覆盖。
    返回写入/更新的行数。
    """
    symbol = (symbol or "").strip()
    if not symbol or df is None or df.empty:
        return 0
    # 构建列映射：只取存在的列
    row_list = []
    date_col = "日期" if "日期" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        trade_date = _parse_date(row.get(date_col))
        if trade_date is None:
            continue
        values = [symbol, trade_date]
        for cn_name, col in AKSHARE_DAILY_COLUMNS:
            if cn_name == "日期":
                continue
            if cn_name not in df.columns:
                values.append(None)
                continue
            v = row.get(cn_name)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                values.append(None)
            else:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    values.append(None)
        row_list.append(tuple(values))
    if not row_list:
        return 0
    cols = ["symbol", "trade_date"] + [c[1] for c in AKSHARE_DAILY_COLUMNS if c[0] != "日期"]
    placeholders = ", ".join(["%s"] * len(cols))
    sql = f"""
    INSERT INTO stock_daily ({", ".join(cols)})
    VALUES ({placeholders})
    ON DUPLICATE KEY UPDATE
        open = VALUES(open), high = VALUES(high), low = VALUES(low), close = VALUES(close),
        volume = VALUES(volume), amount = VALUES(amount), amplitude = VALUES(amplitude),
        change_pct = VALUES(change_pct), change_amt = VALUES(change_amt),
        turnover_rate = VALUES(turnover_rate)
    """
    n = execute_many(sql, row_list)
    # 更新该股已入库的最新交易日，供每日增量更新使用
    max_date = max(r[1] for r in row_list)  # row_list 每项 (symbol, trade_date, ...)
    execute(
        "UPDATE stock_meta SET last_trade_date = %s WHERE symbol = %s",
        (max_date, symbol),
    )
    return n


def get_stock_daily_df(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    从数据库读取日线，返回与 akshare 列名一致的 DataFrame（日期、开盘、收盘等）。
    若 start_date/end_date 为 None 则不过滤日期。
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return None
    conditions = ["symbol = %s"]
    args: list[Any] = [symbol]
    if start_date:
        conditions.append("trade_date >= %s")
        args.append(start_date.replace("-", "")[:8])
    if end_date:
        conditions.append("trade_date <= %s")
        args.append(end_date.replace("-", "")[:8])
    where = " AND ".join(conditions)
    sql = f"""
    SELECT trade_date, open, high, low, close, volume, amount,
           amplitude, change_pct, change_amt, turnover_rate
    FROM stock_daily
    WHERE {where}
    ORDER BY trade_date
    """
    rows = fetch_all(sql, tuple(args))
    if not rows:
        return None
    # 转为中文列名，与 akshare/df_to_chart_result 一致
    col_map = {
        "trade_date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
        "amplitude": "振幅",
        "change_pct": "涨跌幅",
        "change_amt": "涨跌额",
        "turnover_rate": "换手率",
    }
    df = pd.DataFrame(rows)
    df = df.rename(columns=col_map)
    df["日期"] = df["日期"].astype(str)
    return df


def has_stock_data(symbol: str) -> bool:
    """判断该股票在库中是否已有日线数据。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return False
    row = fetch_one("SELECT 1 FROM stock_daily WHERE symbol = %s LIMIT 1", (symbol,))
    return row is not None


def get_symbols_for_daily_update() -> list[dict[str, Any]]:
    """
    返回需要做每日增量更新的股票列表，每项含 symbol、last_trade_date（已入库最新交易日，可能为 None）。
    日更任务可据此拉取：从 (last_trade_date + 1 天) 到今天的区间；若 last_trade_date 为 None 则按配置全量拉取。
    """
    sql = """
    SELECT symbol, last_trade_date FROM stock_meta ORDER BY symbol
    """
    return fetch_all(sql)


def list_stocks_from_db() -> list[dict[str, str]]:
    """返回库中已有数据的股票列表，每项 { symbol, displayName }，displayName 优先来自 stock_meta.name。"""
    sql = """
    SELECT d.symbol, COALESCE(m.name, d.symbol) AS display_name
    FROM (SELECT DISTINCT symbol FROM stock_daily) d
    LEFT JOIN stock_meta m ON m.symbol = d.symbol
    ORDER BY d.symbol
    """
    rows = fetch_all(sql)
    return [{"symbol": r["symbol"], "displayName": (r.get("display_name") or r["symbol"]) or ""} for r in rows]


def delete_stock_data(symbol: str) -> int:
    """删除该股票全部日线数据及元信息，返回删除的日线行数。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return 0
    n = execute("DELETE FROM stock_daily WHERE symbol = %s", (symbol,))
    execute("DELETE FROM stock_meta WHERE symbol = %s", (symbol,))
    return n


def clear_all_stock_data() -> None:
    """清空全部日线与元信息（用于全量同步前重置）。"""
    execute("TRUNCATE TABLE stock_daily")
    execute("DELETE FROM stock_meta")
