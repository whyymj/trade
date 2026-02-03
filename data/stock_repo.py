# -*- coding: utf-8 -*-
"""
股票数据仓储：日线写入/读取、元信息维护、列表与删除。
依赖 data.mysql、data.schema（列映射）。
"""
from datetime import date
from typing import Any, Optional

import pandas as pd

from data.mysql import execute, execute_many, fetch_all, fetch_one, run_connection
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
    min_date = min(r[1] for r in row_list)
    max_date = max(r[1] for r in row_list)

    def _batch_and_update(conn: Any) -> int:
        with conn.cursor() as cur:
            cur.executemany(sql, row_list)
            n = cur.rowcount
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE stock_meta SET
                   last_trade_date = %s,
                   first_trade_date = CASE
                     WHEN first_trade_date IS NULL THEN %s
                     WHEN %s < first_trade_date THEN %s
                     ELSE first_trade_date
                   END
                   WHERE symbol = %s""",
                (max_date, min_date, min_date, min_date, symbol),
            )
        return n

    return run_connection(_batch_and_update)


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


def get_stock_daily_by_range(
    symbols: list[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> dict[str, Any]:
    """
    按日期范围查询多只股票日线，分页返回。
    返回 { "total": int, "page": int, "page_size": int, "data": [ { symbol, trade_date, open, high, low, close, volume, ... }, ... ] }。
    """
    symbols = [str(s).strip() for s in symbols if str(s).strip()]
    if not symbols:
        return {"total": 0, "page": 1, "page_size": page_size, "data": []}
    page = max(1, page)
    page_size = max(1, min(500, page_size))
    conditions = ["symbol IN ({})".format(", ".join(["%s"] * len(symbols)))]
    args: list[Any] = list(symbols)
    if start_date:
        conditions.append("trade_date >= %s")
        args.append(start_date.replace("-", "")[:8])
    if end_date:
        conditions.append("trade_date <= %s")
        args.append(end_date.replace("-", "")[:8])
    where = " AND ".join(conditions)
    count_sql = f"SELECT COUNT(*) AS cnt FROM stock_daily WHERE {where}"
    row = fetch_one(count_sql, tuple(args))
    total = int(row["cnt"]) if row else 0
    offset = (page - 1) * page_size
    data_sql = f"""
    SELECT symbol, trade_date, open, high, low, close, volume, amount,
           amplitude, change_pct, change_amt, turnover_rate
    FROM stock_daily
    WHERE {where}
    ORDER BY trade_date DESC, symbol
    LIMIT %s OFFSET %s
    """
    args.extend([page_size, offset])
    rows = fetch_all(data_sql, tuple(args))
    for r in rows:
        if r.get("trade_date") is not None:
            r["trade_date"] = str(r["trade_date"])
    return {"total": total, "page": page, "page_size": page_size, "data": rows}


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
    return fetch_all("SELECT symbol, last_trade_date FROM stock_meta ORDER BY symbol")


def list_stocks_from_db() -> list[dict[str, Any]]:
    """返回库中已有数据的股票列表，每项 { symbol, displayName, remark }。"""
    sql = """
    SELECT d.symbol, COALESCE(m.name, d.symbol) AS display_name, m.remark
    FROM (SELECT DISTINCT symbol FROM stock_daily) d
    LEFT JOIN stock_meta m ON m.symbol = d.symbol
    ORDER BY d.symbol
    """
    rows = fetch_all(sql)
    return [
        {
            "symbol": r["symbol"],
            "displayName": (r.get("display_name") or r["symbol"]) or "",
            "remark": r.get("remark") or "",
        }
        for r in rows
    ]


def update_stock_remark(symbol: str, remark: Optional[str] = None) -> bool:
    """更新股票说明（用户手动输入）。返回是否更新成功。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return False
    remark_val = (remark or "").strip() or None
    execute("UPDATE stock_meta SET remark = %s, updated_at = CURRENT_TIMESTAMP WHERE symbol = %s", (remark_val, symbol))
    return True


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
