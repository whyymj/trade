#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻仓储模块 - 支持去重、按日存储、自动清理
"""

import logging
from datetime import date, datetime
from typing import List, Optional

from shared.db import execute, fetch_all, fetch_one, run_connection

logger = logging.getLogger(__name__)


class NewsRepo:
    """新闻仓储 - 去重、清理"""

    KEEP_DAYS = 30

    def save_news(self, news_list: List) -> int:
        """保存新闻（自动去重，基于URL）"""
        if not news_list:
            return 0

        sql = """
        INSERT IGNORE INTO news_data 
        (news_date, title, content, source, url, published_at, category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        params_list = []
        for news in news_list:
            params_list.append(
                (
                    news.news_date.isoformat()
                    if news.news_date
                    else date.today().isoformat(),
                    news.title,
                    news.content[:2000] if news.content else "",
                    news.source,
                    news.url,
                    news.published_at.isoformat() if news.published_at else None,
                    news.category,
                )
            )

        def _insert(conn):
            from sqlalchemy import text

            with conn.cursor() as cur:
                cur.executemany(text(sql), params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            logger.error("save_news error: %s", e)
            return 0

    def get_news(
        self, days: int = 1, category: str = None, limit: int = 100
    ) -> List[dict]:
        """获取新闻"""
        conditions = ["news_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)"]
        params = [days]

        if category and category != "all":
            conditions.append("category = %s")
            params.append(category)

        where = " AND ".join(conditions)
        sql = f"""
        SELECT id, news_date, title, content, source, url, published_at, category, importance
        FROM news_data
        WHERE {where}
        ORDER BY published_at DESC
        LIMIT %s
        """
        params.append(limit)

        return fetch_all(sql, tuple(params))

    def get_latest_news(self, limit: int = 20) -> List[dict]:
        """获取最新新闻"""
        return self.get_news(days=1, limit=limit)

    def get_news_by_id(self, news_id: int) -> Optional[dict]:
        """根据ID获取单条新闻"""
        sql = "SELECT * FROM news_data WHERE id = %s"
        return fetch_one(sql, (news_id,))

    def get_news_by_url(self, url: str) -> Optional[dict]:
        """根据URL获取单条新闻"""
        sql = "SELECT * FROM news_data WHERE url = %s LIMIT 1"
        return fetch_one(sql, (url,))

    def get_news_count(self, days: int = 1) -> int:
        """获取新闻数量"""
        sql = """
        SELECT COUNT(*) as cnt
        FROM news_data
        WHERE news_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """
        row = fetch_one(sql, (days,))
        return row.get("cnt", 0) if row else 0

    def cleanup_old_news(self, keep_days: int = None) -> int:
        """清理过期新闻"""
        keep_days = keep_days or self.KEEP_DAYS

        sql = """
        DELETE FROM news_data 
        WHERE news_date < DATE_SUB(CURDATE(), INTERVAL %s DAY)
        """

        try:
            n = execute(sql, (keep_days,))
            logger.info("Cleaned %d old news", n)
            return n
        except Exception as e:
            logger.error("cleanup error: %s", e)
            return 0

    def get_categories(self) -> List[dict]:
        """获取分类统计"""
        sql = """
        SELECT category, COUNT(*) as cnt
        FROM news_data
        WHERE news_date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
        GROUP BY category
        ORDER BY cnt DESC
        """

        rows = fetch_all(sql)
        return [{"category": r.get("category"), "count": r.get("cnt")} for r in rows]


def get_repo() -> NewsRepo:
    """获取仓储实例"""
    return NewsRepo()
