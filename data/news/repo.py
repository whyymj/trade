# data/news/repo.py
"""
新闻仓储模块 - 支持去重、按日存储、自动清理
"""

import json
import logging
from datetime import date, datetime
from typing import List, Optional
from dataclasses import asdict

from data.mysql import execute, fetch_all, fetch_one, run_connection
from .interfaces import NewsItem, AnalysisResult

logger = logging.getLogger(__name__)


class NewsRepo:
    """新闻仓储 - 去重、清理"""

    KEEP_DAYS = 30

    def save_news(self, news_list: List[NewsItem]) -> int:
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
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
                return cur.rowcount

        try:
            return run_connection(_insert)
        except Exception as e:
            logger.error("save_news error: %s", e)
            return 0

    def get_news(
        self, days: int = 1, category: str = None, limit: int = 100
    ) -> List[NewsItem]:
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

        rows = fetch_all(sql, tuple(params))

        news_list = []
        for row in rows:
            news = NewsItem(
                title=row.get("title", ""),
                content=row.get("content", ""),
                source=row.get("source", ""),
                url=row.get("url", ""),
                published_at=row.get("published_at"),
                category=row.get("category", "general"),
                news_date=row.get("news_date"),
                id=row.get("id"),
            )
            news_list.append(news)

        return news_list

    def get_today_news(self) -> List[NewsItem]:
        """获取今天的新闻"""
        sql = """
        SELECT id, news_date, title, content, source, url, published_at, category, importance
        FROM news_data
        WHERE news_date = CURDATE()
        ORDER BY published_at DESC
        """

        rows = fetch_all(sql)

        news_list = []
        for row in rows:
            news = NewsItem(
                title=row.get("title", ""),
                content=row.get("content", ""),
                source=row.get("source", ""),
                url=row.get("url", ""),
                published_at=row.get("published_at"),
                category=row.get("category", "general"),
                news_date=row.get("news_date"),
                id=row.get("id"),
            )
            news_list.append(news)

        return news_list

    def get_news_by_url(self, url: str) -> Optional[NewsItem]:
        """根据URL获取单条新闻"""
        sql = "SELECT * FROM news_data WHERE url = %s LIMIT 1"
        row = fetch_one(sql, (url,))

        if not row:
            return None

        pub_at = row.get("published_at")
        if pub_at:
            if isinstance(pub_at, datetime):
                published_at = pub_at
            else:
                published_at = datetime.fromisoformat(str(pub_at))
        else:
            published_at = None

        return NewsItem(
            title=row.get("title", ""),
            content=row.get("content", ""),
            source=row.get("source", ""),
            url=row.get("url", ""),
            published_at=published_at,
            category=row.get("category", "general"),
            id=row.get("id"),
        )

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

    def save_analysis(self, result: AnalysisResult) -> bool:
        """保存分析结果"""
        sql = """
        INSERT INTO news_analysis 
        (analysis_date, news_count, summary, deep_analysis, market_impact, key_events, investment_advice)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            news_count = VALUES(news_count),
            summary = VALUES(summary),
            deep_analysis = VALUES(deep_analysis),
            market_impact = VALUES(market_impact),
            key_events = VALUES(key_events),
            investment_advice = VALUES(investment_advice)
        """

        try:
            execute(
                sql,
                (
                    result.analyzed_at.date().isoformat(),
                    result.news_count,
                    result.summary,
                    result.deep_analysis,
                    result.market_impact,
                    json.dumps(result.key_events, ensure_ascii=False),
                    result.investment_advice,
                ),
            )
            return True
        except Exception as e:
            logger.error("save_analysis error: %s", e)
            return False

    def get_latest_analysis(self) -> Optional[AnalysisResult]:
        """获取最新分析结果"""
        sql = """
        SELECT id, analysis_date, news_count, summary, deep_analysis, 
               market_impact, key_events, investment_advice
        FROM news_analysis
        ORDER BY analysis_date DESC
        LIMIT 1
        """

        row = fetch_one(sql)
        if not row:
            return None

        return AnalysisResult(
            news_count=row.get("news_count", 0),
            summary=row.get("summary", ""),
            deep_analysis=row.get("deep_analysis", ""),
            market_impact=row.get("market_impact", "neutral"),
            key_events=json.loads(row.get("key_events", "[]"))
            if row.get("key_events")
            else [],
            investment_advice=row.get("investment_advice", ""),
            analyzed_at=row.get("analysis_date", datetime.now()),
        )

    def update_news_industry_tags(self, news_ids: list, industry_tags: list) -> bool:
        """更新新闻的行业标签"""
        if not news_ids or not industry_tags:
            return False

        tags_json = json.dumps(industry_tags, ensure_ascii=False)

        placeholders = ",".join(["%s"] * len(news_ids))
        sql = f"""
        UPDATE news_data 
        SET industry_tags = %s 
        WHERE id IN ({placeholders})
        """

        try:
            execute(sql, (tags_json, *news_ids))
            return True
        except Exception as e:
            logger.error("update_news_industry_tags error: %s", e)
            return False

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
