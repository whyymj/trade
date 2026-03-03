# -*- coding: utf-8 -*-
"""
指数数据仓储：指数 CRUD、指数数据操作。
依赖 data.mysql、data.cache。
"""

from datetime import date
from typing import Any, Optional

import pandas as pd

from data.cache import get_cache
from data.mysql import execute, fetch_all, fetch_one, run_connection


def add_index(index_code: str, index_name: str) -> bool:
    """
    添加指数基本信息。
    指数信息存储在 index_data 表中，index_code 为主键。
    """
    index_code = (index_code or "").strip()
    if not index_code:
        return False
    index_name = (index_name or "").strip() or index_code
    test_sql = "SELECT 1 FROM index_data WHERE index_code = %s LIMIT 1"
    if fetch_one(test_sql, (index_code,)):
        execute(
            "UPDATE index_data SET trade_date = NULL WHERE index_code = %s",
            (index_code,),
        )
        return True
    sql = """
    INSERT INTO index_data (index_code, trade_date, close_price, daily_return)
    VALUES (%s, NULL, NULL, NULL)
    """
    try:
        execute(sql, (index_code,))
    except Exception:
        pass
    get_cache().delete("index_list")
    return True


def get_index_list() -> list[dict[str, Any]]:
    """获取全部已有数据的指数列表。"""
    sql = """
    SELECT DISTINCT index_code, MIN(trade_date) AS start_date, MAX(trade_date) AS end_date
    FROM index_data
    WHERE trade_date IS NOT NULL
    GROUP BY index_code
    ORDER BY index_code
    """
    rows = fetch_all(sql)
    for r in rows:
        if r.get("start_date"):
            r["start_date"] = str(r["start_date"])
        if r.get("end_date"):
            r["end_date"] = str(r["end_date"])
    return rows


def upsert_index_data(index_code: str, df: pd.DataFrame) -> int:
    """
    批量插入或更新指数数据，按 (index_code, trade_date) 覆盖。
    df 需包含列：trade_date, close_price, daily_return（可选）。
    返回写入/更新的行数。
    """
    index_code = (index_code or "").strip()
    if not index_code or df is None or df.empty:
        return 0
    row_list = []
    date_col = "trade_date" if "trade_date" in df.columns else "日期"
    for _, row in df.iterrows():
        trade_date_raw = row.get(date_col)
        if trade_date_raw is None:
            continue
        try:
            trade_date = pd.to_datetime(trade_date_raw).date()
        except Exception:
            continue
        close_price = row.get("close_price") or row.get("收盘")
        daily_return = row.get("daily_return") or row.get("涨跌幅")
        try:
            close_price = (
                float(close_price) if close_price and pd.notna(close_price) else None
            )
        except (TypeError, ValueError):
            close_price = None
        try:
            daily_return = (
                float(daily_return) if daily_return and pd.notna(daily_return) else None
            )
        except (TypeError, ValueError):
            daily_return = None
        row_list.append((index_code, trade_date, close_price, daily_return))
    if not row_list:
        return 0
    sql = """
    INSERT INTO index_data (index_code, trade_date, close_price, daily_return)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        close_price = VALUES(close_price),
        daily_return = VALUES(daily_return)
    """

    def _batch_insert(conn: Any) -> int:
        with conn.cursor() as cur:
            cur.executemany(sql, row_list)
            return cur.rowcount

    return run_connection(_batch_insert)


def get_index_data(
    index_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    获取指数历史数据，返回 DataFrame。
    列：index_code, trade_date, close_price, daily_return
    """
    index_code = (index_code or "").strip()
    if not index_code:
        return None
    conditions = ["index_code = %s"]
    args: list[Any] = [index_code]
    if start_date:
        conditions.append("trade_date >= %s")
        args.append(start_date.replace("-", "")[:8])
    if end_date:
        conditions.append("trade_date <= %s")
        args.append(end_date.replace("-", "")[:8])
    where = " AND ".join(conditions)
    sql = f"""
    SELECT trade_date, close_price, daily_return
    FROM index_data
    WHERE {where}
    ORDER BY trade_date
    """
    rows = fetch_all(sql, tuple(args))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["trade_date"] = df["trade_date"].astype(str)
    df["close_price"] = df["close_price"].astype(float)
    df["daily_return"] = df["daily_return"].astype(float)
    return df


def get_latest_index_data(index_code: str) -> Optional[dict[str, Any]]:
    """获取指数最新数据。"""
    index_code = (index_code or "").strip()
    if not index_code:
        return None
    sql = """
    SELECT index_code, trade_date, close_price, daily_return
    FROM index_data
    WHERE index_code = %s AND trade_date IS NOT NULL
    ORDER BY trade_date DESC
    LIMIT 1
    """
    row = fetch_one(sql, (index_code,))
    if row:
        if row.get("trade_date"):
            row["trade_date"] = str(row["trade_date"])
        if row.get("close_price"):
            row["close_price"] = float(row["close_price"])
        if row.get("daily_return"):
            row["daily_return"] = float(row["daily_return"])
    return row


def delete_index_data(index_code: str) -> int:
    """删除指数全部数据。"""
    index_code = (index_code or "").strip()
    if not index_code:
        return 0
    n = execute("DELETE FROM index_data WHERE index_code = %s", (index_code,))
    get_cache().delete("index_list")
    return n
