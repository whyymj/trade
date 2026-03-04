# -*- coding: utf-8 -*-
"""
基金数据仓储：基金 CRUD、净值操作、关注列表。
依赖 data.mysql、data.cache、data.schema。
"""

from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from data.cache import get_cache
from data.mysql import execute, execute_many, fetch_all, fetch_one, run_connection


def add_fund(
    fund_code: str,
    fund_name: str,
    fund_type: Optional[str] = None,
    manager: Optional[str] = None,
    establishment_date: Optional[str] = None,
    fund_scale: Optional[float] = None,
) -> bool:
    """添加基金基本信息，返回是否成功。"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return False
    fund_name = (fund_name or "").strip() or None
    fund_type = (fund_type or "").strip() or None
    manager = (manager or "").strip() or None
    est_date = None
    if establishment_date:
        try:
            est_date = pd.to_datetime(establishment_date).date()
        except Exception:
            est_date = None
    sql = """
    INSERT INTO fund_meta (fund_code, fund_name, fund_type, manager, establishment_date, fund_scale)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        fund_name = COALESCE(VALUES(fund_name), fund_name),
        fund_type = COALESCE(VALUES(fund_type), fund_type),
        manager = COALESCE(VALUES(manager), manager),
        establishment_date = COALESCE(VALUES(establishment_date), establishment_date),
        fund_scale = COALESCE(VALUES(fund_scale), fund_scale),
        updated_at = CURRENT_TIMESTAMP
    """
    execute(sql, (fund_code, fund_name, fund_type, manager, est_date, fund_scale))
    get_cache().delete("fund_list")
    return True


def get_fund_list(
    page: int = 1,
    size: int = 20,
    fund_type: Optional[str] = None,
    watchlist_only: bool = False,
) -> dict[str, Any]:
    """
    获取基金列表，分页返回。
    支持按基金类型筛选、仅返回关注列表。
    """
    page = max(1, page)
    size = max(1, min(100, size))
    conditions = []
    args: list[Any] = []
    if fund_type:
        conditions.append("fund_type = %s")
        args.append(fund_type)
    if watchlist_only:
        conditions.append("watchlist = 1")
    where = " AND ".join(conditions) if conditions else "1=1"
    count_sql = f"SELECT COUNT(*) AS cnt FROM fund_meta WHERE {where}"
    row = fetch_one(count_sql, tuple(args))
    total = int(row["cnt"]) if row else 0
    offset = (page - 1) * size
    data_sql = f"""
    SELECT fund_code, fund_name, fund_type, manager, establishment_date, fund_scale, watchlist
    FROM fund_meta
    WHERE {where}
    ORDER BY fund_code
    LIMIT %s OFFSET %s
    """
    args.extend([size, offset])
    rows = fetch_all(data_sql, tuple(args))
    for r in rows:
        if r.get("establishment_date"):
            r["establishment_date"] = str(r["establishment_date"])
        if r.get("fund_scale"):
            r["fund_scale"] = float(r["fund_scale"])
    return {"total": total, "page": page, "page_size": size, "data": rows}


def get_fund_info(fund_code: str) -> Optional[dict[str, Any]]:
    """获取基金详细信息。"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return None
    sql = """
    SELECT fund_code, fund_name, fund_type, manager, establishment_date, fund_scale, watchlist
    FROM fund_meta
    WHERE fund_code = %s
    """
    row = fetch_one(sql, (fund_code,))
    if row:
        if row.get("establishment_date"):
            row["establishment_date"] = str(row["establishment_date"])
        if row.get("fund_scale"):
            row["fund_scale"] = float(row["fund_scale"])
    return row


def get_fund_nav(
    fund_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    获取基金净值历史，返回 DataFrame。
    列：fund_code, nav_date, unit_nav, accum_nav, daily_return
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return None
    conditions = ["fund_code = %s"]
    args: list[Any] = [fund_code]
    if start_date:
        conditions.append("nav_date >= %s")
        args.append(start_date.replace("-", "")[:8])
    if end_date:
        conditions.append("nav_date <= %s")
        args.append(end_date.replace("-", "")[:8])
    where = " AND ".join(conditions)
    sql = f"""
    SELECT nav_date, unit_nav, accum_nav, daily_return
    FROM fund_nav
    WHERE {where}
    ORDER BY nav_date
    """
    rows = fetch_all(sql, tuple(args))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["nav_date"] = df["nav_date"].astype(str)
    df["unit_nav"] = df["unit_nav"].astype(float)
    df["accum_nav"] = df["accum_nav"].astype(float)
    df["daily_return"] = df["daily_return"].astype(float)
    # 将 NaN 转换为 None（JSON 序列化需要）
    import numpy as np

    df = df.replace({np.nan: None})
    return df


def get_latest_nav(fund_code: str) -> Optional[dict[str, Any]]:
    """获取基金最新净值。"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return None
    sql = """
    SELECT fund_code, nav_date, unit_nav, accum_nav, daily_return
    FROM fund_nav
    WHERE fund_code = %s
    ORDER BY nav_date DESC
    LIMIT 1
    """
    row = fetch_one(sql, (fund_code,))
    if row:
        if row.get("nav_date"):
            row["nav_date"] = str(row["nav_date"])
        if row.get("unit_nav"):
            row["unit_nav"] = float(row["unit_nav"])
        if row.get("accum_nav"):
            row["accum_nav"] = float(row["accum_nav"])
        if row.get("daily_return"):
            row["daily_return"] = float(row["daily_return"])
    return row


def update_fund_watchlist(fund_code: str, watch: bool) -> bool:
    """更新基金关注状态，返回是否成功。"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return False
    watch_val = 1 if watch else 0
    execute(
        "UPDATE fund_meta SET watchlist = %s, updated_at = CURRENT_TIMESTAMP WHERE fund_code = %s",
        (watch_val, fund_code),
    )
    get_cache().delete("fund_list")
    return True


def upsert_fund_nav(fund_code: str, nav_df: pd.DataFrame) -> int:
    """
    批量插入或更新基金净值，按 (fund_code, nav_date) 覆盖。
    nav_df 需包含列：nav_date, unit_nav, accum_nav, daily_return（可选）。
    返回写入/更新的行数。
    """
    fund_code = (fund_code or "").strip()
    if not fund_code or nav_df is None or nav_df.empty:
        return 0
    row_list = []
    date_col = "nav_date" if "nav_date" in nav_df.columns else "日期"
    for _, row in nav_df.iterrows():
        nav_date_raw = row.get(date_col)
        if nav_date_raw is None:
            continue
        try:
            nav_date = pd.to_datetime(nav_date_raw).date()
        except Exception:
            continue
        unit_nav = row.get("unit_nav") or row.get("单位净值")
        accum_nav = row.get("accum_nav") or row.get("累计净值")
        daily_return = row.get("daily_return") or row.get("日增长率")
        try:
            unit_nav = float(unit_nav) if unit_nav else None
        except (TypeError, ValueError):
            unit_nav = None
        try:
            accum_nav = float(accum_nav) if accum_nav else None
        except (TypeError, ValueError):
            accum_nav = None
        try:
            daily_return = float(daily_return) if daily_return else None
        except (TypeError, ValueError):
            daily_return = None
        row_list.append((fund_code, nav_date, unit_nav, accum_nav, daily_return))
    if not row_list:
        return 0
    sql = """
    INSERT INTO fund_nav (fund_code, nav_date, unit_nav, accum_nav, daily_return)
    VALUES (%s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        unit_nav = VALUES(unit_nav),
        accum_nav = VALUES(accum_nav),
        daily_return = VALUES(daily_return)
    """

    def _batch_insert(conn: Any) -> int:
        with conn.cursor() as cur:
            cur.executemany(sql, row_list)
            return cur.rowcount

    return run_connection(_batch_insert)


def delete_fund(fund_code: str) -> int:
    """删除基金及净值数据，返回删除的记录数。"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return 0
    n = execute("DELETE FROM fund_nav WHERE fund_code = %s", (fund_code,))
    execute("DELETE FROM fund_meta WHERE fund_code = %s", (fund_code,))
    get_cache().delete("fund_list")
    return n + 1


def get_fund_types() -> list[str]:
    """获取所有基金类型。"""
    rows = fetch_all(
        "SELECT DISTINCT fund_type FROM fund_meta WHERE fund_type IS NOT NULL ORDER BY fund_type"
    )
    return [r["fund_type"] for r in rows]


def search_funds(keyword: str, limit: int = 10) -> list[dict[str, Any]]:
    """按基金代码或名称搜索。"""
    keyword = (keyword or "").strip()
    if not keyword:
        return []
    limit = max(1, min(50, limit))
    sql = """
    SELECT fund_code, fund_name, fund_type, watchlist
    FROM fund_meta
    WHERE fund_code LIKE %s OR fund_name LIKE %s
    ORDER BY fund_code
    LIMIT %s
    """
    pattern = f"%{keyword}%"
    rows = fetch_all(sql, (pattern, pattern, limit))
    return rows
