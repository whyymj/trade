import pandas as pd
from typing import Optional, List
from datetime import datetime

from shared.db import fetch_all, fetch_one, run_connection


class MarketRepo:
    """市场数据仓储"""

    def save_macro(self, df: pd.DataFrame) -> int:
        """保存宏观经济数据"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO macro_data 
        (indicator, period, value, unit, source, publish_date, trade_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            value = VALUES(value),
            unit = VALUES(unit),
            source = VALUES(source),
            publish_date = VALUES(publish_date),
            trade_date = VALUES(trade_date)
        """

        params_list = []
        for _, row in df.iterrows():
            params_list.append(
                (
                    row.get("indicator"),
                    row.get("period"),
                    row.get("value"),
                    row.get("unit"),
                    row.get("source"),
                    row.get("publish_date")
                    if pd.notna(row.get("publish_date"))
                    else None,
                    row.get("trade_date") if pd.notna(row.get("trade_date")) else None,
                )
            )

        def _insert(conn):
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            print(f"[MarketRepo] save_macro error: {e}")
            return 0

    def get_macro(self, indicator: str = None, days: int = 30) -> list:
        """获取宏观经济数据"""
        sql = """
        SELECT id, indicator, period, value, unit, source, publish_date, trade_date
        FROM macro_data
        WHERE publish_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        params = [days]

        if indicator:
            sql += " AND indicator = %s"
            params.append(indicator)

        sql += " ORDER BY publish_date DESC, period DESC"

        return fetch_all(sql, tuple(params))

    def get_latest_macro(self, indicator: str = None) -> Optional[dict]:
        """获取最新宏观经济数据"""
        if indicator:
            sql = """
            SELECT id, indicator, period, value, unit, source, publish_date
            FROM macro_data
            WHERE indicator = %s
            ORDER BY publish_date DESC
            LIMIT 1
            """
            row = fetch_one(sql, (indicator,))
        else:
            sql = """
            SELECT id, indicator, period, value, unit, source, publish_date
            FROM macro_data
            ORDER BY publish_date DESC
            LIMIT 1
            """
            row = fetch_one(sql)

        if not row:
            return None

        return {
            "indicator": row.get("indicator"),
            "period": row.get("period"),
            "value": float(row.get("value")) if row.get("value") else None,
            "unit": row.get("unit"),
            "source": row.get("source"),
            "publish_date": str(row.get("publish_date"))
            if row.get("publish_date")
            else None,
        }

    def save_money_flow(self, df: pd.DataFrame) -> int:
        """保存资金流向"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO money_flow 
        (trade_date, north_money, north_buy, north_sell, main_money, margin_balance)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            north_money = VALUES(north_money),
            north_buy = VALUES(north_buy),
            north_sell = VALUES(north_sell),
            main_money = VALUES(main_money),
            margin_balance = VALUES(margin_balance)
        """

        params_list = []
        for _, row in df.iterrows():
            params_list.append(
                (
                    row.get("trade_date").date().isoformat()
                    if hasattr(row.get("trade_date"), "date")
                    else str(row.get("trade_date"))[:10],
                    row.get("north_money"),
                    row.get("north_buy"),
                    row.get("north_sell"),
                    row.get("main_money"),
                    row.get("margin_balance"),
                )
            )

        def _insert(conn):
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            print(f"[MarketRepo] save_money_flow error: {e}")
            return 0

    def get_money_flow(self, days: int = 30) -> list:
        """获取资金流向"""
        sql = """
        SELECT trade_date, north_money, north_buy, north_sell, main_money, margin_balance
        FROM money_flow
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY trade_date DESC
        """
        return fetch_all(sql, (days,))

    def get_latest_money_flow(self) -> Optional[dict]:
        """获取最新资金流向"""
        sql = """
        SELECT trade_date, north_money, north_buy, north_sell, main_money, margin_balance
        FROM money_flow
        ORDER BY trade_date DESC
        LIMIT 1
        """
        row = fetch_one(sql)
        if not row:
            return None

        return {
            "trade_date": str(row.get("trade_date")) if row.get("trade_date") else None,
            "north_money": float(row.get("north_money"))
            if row.get("north_money")
            else None,
            "north_buy": float(row.get("north_buy")) if row.get("north_buy") else None,
            "north_sell": float(row.get("north_sell"))
            if row.get("north_sell")
            else None,
            "main_money": float(row.get("main_money"))
            if row.get("main_money")
            else None,
            "margin_balance": float(row.get("margin_balance"))
            if row.get("margin_balance")
            else None,
        }

    def save_sentiment(self, df: pd.DataFrame) -> int:
        """保存市场情绪"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO market_sentiment 
        (trade_date, volume, up_count, down_count, turnover_rate, advance_count, decline_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            volume = VALUES(volume),
            up_count = VALUES(up_count),
            down_count = VALUES(down_count),
            turnover_rate = VALUES(turnover_rate),
            advance_count = VALUES(advance_count),
            decline_count = VALUES(decline_count)
        """

        params_list = []
        for _, row in df.iterrows():
            trade_date = row.get("trade_date")
            if hasattr(trade_date, "date"):
                trade_date_str = trade_date.date().isoformat()
            else:
                trade_date_str = str(trade_date)[:10]

            params_list.append(
                (
                    trade_date_str,
                    row.get("volume"),
                    row.get("up_count", 0),
                    row.get("down_count", 0),
                    row.get("turnover_rate"),
                    row.get("advance_count", 0),
                    row.get("decline_count", 0),
                )
            )

        def _insert(conn):
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            print(f"[MarketRepo] save_sentiment error: {e}")
            return 0

    def get_sentiment(self, days: int = 30) -> list:
        """获取市场情绪"""
        sql = """
        SELECT trade_date, volume, up_count, down_count, turnover_rate, advance_count, decline_count
        FROM market_sentiment
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY trade_date DESC
        """
        return fetch_all(sql, (days,))

    def get_latest_sentiment(self) -> Optional[dict]:
        """获取最新市场情绪"""
        sql = """
        SELECT trade_date, volume, up_count, down_count, turnover_rate, advance_count, decline_count
        FROM market_sentiment
        ORDER BY trade_date DESC
        LIMIT 1
        """
        row = fetch_one(sql)
        if not row:
            return None

        return {
            "trade_date": str(row.get("trade_date")) if row.get("trade_date") else None,
            "volume": float(row.get("volume")) if row.get("volume") else None,
            "up_count": int(row.get("up_count")) if row.get("up_count") else 0,
            "down_count": int(row.get("down_count")) if row.get("down_count") else 0,
            "turnover_rate": float(row.get("turnover_rate"))
            if row.get("turnover_rate")
            else None,
            "advance_count": int(row.get("advance_count"))
            if row.get("advance_count")
            else 0,
            "decline_count": int(row.get("decline_count"))
            if row.get("decline_count")
            else 0,
        }

    def save_global_macro(self, df: pd.DataFrame) -> int:
        """保存全球宏观数据"""
        if df is None or df.empty:
            return 0

        sql = """
        INSERT INTO global_macro 
        (trade_date, symbol, close_price, change_pct)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            close_price = VALUES(close_price),
            change_pct = VALUES(change_pct)
        """

        params_list = []
        for _, row in df.iterrows():
            trade_date = row.get("trade_date")
            if hasattr(trade_date, "date"):
                trade_date_str = trade_date.date().isoformat()
            elif isinstance(trade_date, str):
                trade_date_str = trade_date[:10]
            else:
                trade_date_str = str(trade_date)

            params_list.append(
                (
                    trade_date_str,
                    row.get("symbol"),
                    row.get("close_price"),
                    row.get("change_pct"),
                )
            )

        def _insert(conn):
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            print(f"[MarketRepo] save_global_macro error: {e}")
            return 0

    def get_global_macro(self, symbol: str = None, days: int = 30) -> list:
        """获取全球宏观数据"""
        if symbol:
            sql = """
            SELECT trade_date, symbol, close_price, change_pct
            FROM global_macro
            WHERE symbol = %s
            ORDER BY trade_date DESC
            """
            return fetch_all(sql, (symbol,))
        else:
            sql = """
            SELECT trade_date, symbol, close_price, change_pct
            FROM global_macro
            WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY trade_date DESC
            """
            return fetch_all(sql, (days,))

    def get_latest_global(self) -> Optional[List[dict]]:
        """获取最新全球宏观数据"""
        sql = """
        SELECT trade_date, symbol, close_price, change_pct
        FROM global_macro
        ORDER BY trade_date DESC
        LIMIT 20
        """
        rows = fetch_all(sql)
        if not rows:
            return None

        result = []
        for row in rows:
            result.append(
                {
                    "trade_date": str(row.get("trade_date"))
                    if row.get("trade_date")
                    else None,
                    "symbol": row.get("symbol"),
                    "close_price": float(row.get("close_price"))
                    if row.get("close_price")
                    else None,
                    "change_pct": float(row.get("change_pct"))
                    if row.get("change_pct")
                    else None,
                }
            )

        return result

    def get_market_features(self, days: int = 30) -> dict:
        """获取合并后的市场特征（包含所有市场数据）"""
        return {
            "macro": self.get_macro(days=days),
            "money_flow": self.get_money_flow(days=days),
            "sentiment": self.get_sentiment(days=days),
            "global": self.get_global_macro(days=days),
        }

    def get_sentiment_summary(self) -> dict:
        """获取情绪摘要"""
        latest = self.get_latest_sentiment()
        if not latest:
            return {
                "volume": None,
                "up_count": 0,
                "down_count": 0,
                "turnover_rate": None,
                "advance_count": 0,
                "decline_count": 0,
            }
        return latest


def get_repo() -> MarketRepo:
    """获取仓储实例"""
    return MarketRepo()
