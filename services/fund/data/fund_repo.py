# -*- coding: utf-8 -*-
"""
基金数据仓储 - 使用共享数据库连接池
"""

from datetime import date, datetime
from typing import Any, Optional
import pandas as pd
import numpy as np
import json

from shared.db import fetch_all, fetch_one, execute, execute_many


class FundRepo:
    def __init__(self):
        pass

    def get_fund_list(
        self,
        page: int = 1,
        size: int = 20,
        fund_type: Optional[str] = None,
        watchlist_only: bool = False,
        industry_tag: Optional[str] = None,
    ) -> dict[str, Any]:
        """获取基金列表，分页返回"""
        page = max(1, page)
        size = max(1, min(100, size))
        conditions = []
        args: list[Any] = []
        if fund_type:
            conditions.append("fund_type = %s")
            args.append(fund_type)
        if watchlist_only:
            conditions.append("watchlist = 1")
        if industry_tag:
            conditions.append("JSON_CONTAINS(industry_tags, %s)")
            args.append(f'"{industry_tag}"')
        where = " AND ".join(conditions) if conditions else "1=1"
        count_sql = f"SELECT COUNT(*) AS cnt FROM fund_meta WHERE {where}"
        row = fetch_one(count_sql, tuple(args))
        total = int(row["cnt"]) if row else 0
        offset = (page - 1) * size
        data_sql = f"""
        SELECT fund_code, fund_name, fund_type, manager, establishment_date, fund_scale, watchlist, industry_tags, analysis_status
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
            if r.get("industry_tags"):
                try:
                    r["industry_tags"] = (
                        json.loads(r["industry_tags"])
                        if isinstance(r["industry_tags"], str)
                        else r["industry_tags"]
                    )
                except:
                    r["industry_tags"] = []
            else:
                r["industry_tags"] = []
        return {"total": total, "page": page, "page_size": size, "data": rows}

    def get_fund_info(self, fund_code: str) -> Optional[dict[str, Any]]:
        """获取基金详细信息"""
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
        self,
        fund_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """获取基金净值历史"""
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
        if days and days > 0:
            limit = f"LIMIT %s"
            args.append(days)
        else:
            limit = ""
        where = " AND ".join(conditions)
        sql = f"""
        SELECT nav_date, unit_nav, accum_nav, daily_return
        FROM fund_nav
        WHERE {where}
        ORDER BY nav_date DESC
        {limit}
        """
        rows = fetch_all(sql, tuple(args))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df = df.sort_values("nav_date")
        df["nav_date"] = df["nav_date"].astype(str)
        df["unit_nav"] = df["unit_nav"].astype(float)
        df["accum_nav"] = df["accum_nav"].astype(float)
        df["daily_return"] = df["daily_return"].astype(float)
        df = df.replace({np.nan: None})
        return df

    def upsert_fund_nav(self, fund_code: str, nav_df: pd.DataFrame) -> int:
        """批量插入或更新基金净值"""
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
        from shared.db import run_connection

        def _batch_insert(conn: Any) -> int:
            with conn.cursor() as cur:
                cur.executemany(sql, row_list)
                return cur.rowcount

        return run_connection(_batch_insert)
