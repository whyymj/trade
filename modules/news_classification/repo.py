from datetime import date, datetime
from typing import List, Optional
import json

from data.mysql import execute, fetch_all, fetch_one
from modules.news_classification.interfaces import ClassifiedNews


class ClassificationRepo:
    """新闻行业分类仓储"""

    def save_classification(self, classified: ClassifiedNews) -> bool:
        """保存单条分类结果"""
        sql = """
        INSERT INTO news_industry_classification 
        (news_id, title, source, url, published_at, news_date, 
         original_category, industry, industry_code, confidence, classified_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            industry = VALUES(industry),
            industry_code = VALUES(industry_code),
            confidence = VALUES(confidence),
            classified_at = VALUES(classified_at)
        """

        try:
            pub_at = (
                classified.published_at.isoformat() if classified.published_at else None
            )
            n_date = classified.news_date.isoformat() if classified.news_date else None

            execute(
                sql,
                (
                    classified.news_id,
                    classified.title[:500],
                    classified.source,
                    classified.url,
                    pub_at,
                    n_date,
                    classified.original_category,
                    classified.industry,
                    classified.industry_code,
                    classified.confidence,
                    classified.classified_at.isoformat(),
                ),
            )
            return True
        except Exception as e:
            print(f"[ClassificationRepo] save error: {e}")
            return False

    def save_batch(self, classified_list: List[ClassifiedNews]) -> int:
        """批量保存分类结果"""
        if not classified_list:
            return 0

        sql = """
        INSERT INTO news_industry_classification 
        (news_id, title, source, url, published_at, news_date, 
         original_category, industry, industry_code, confidence, classified_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            industry = VALUES(industry),
            industry_code = VALUES(industry_code),
            confidence = VALUES(confidence),
            classified_at = VALUES(classified_at)
        """

        params_list = []
        for c in classified_list:
            pub_at = c.published_at.isoformat() if c.published_at else None
            n_date = c.news_date.isoformat() if c.news_date else None

            params_list.append(
                (
                    c.news_id,
                    c.title[:500],
                    c.source,
                    c.url,
                    pub_at,
                    n_date,
                    c.original_category,
                    c.industry,
                    c.industry_code,
                    c.confidence,
                    c.classified_at.isoformat(),
                )
            )

        try:
            from data.mysql import run_connection

            def _insert(conn):
                with conn.cursor() as cur:
                    cur.executemany(sql, params_list)
                    return cur.rowcount

            return run_connection(_insert)
        except Exception as e:
            print(f"[ClassificationRepo] save_batch error: {e}")
            return 0

    def get_by_industry(
        self, industry_code: str, days: int = 7
    ) -> List[ClassifiedNews]:
        """按行业获取分类结果"""
        sql = """
        SELECT * FROM news_industry_classification
        WHERE industry_code = %s
          AND classified_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY classified_at DESC
        """

        rows = fetch_all(sql, (industry_code, days))
        return [self._row_to_classified(r) for r in rows]

    def get_industry_stats(self, days: int = 7) -> List[dict]:
        """获取行业统计"""
        sql = """
        SELECT industry, industry_code, COUNT(*) as cnt, AVG(confidence) as avg_conf
        FROM news_industry_classification
        WHERE classified_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        GROUP BY industry, industry_code
        ORDER BY cnt DESC
        """

        rows = fetch_all(sql, (days,))
        return [
            {
                "industry": r.get("industry"),
                "industry_code": r.get("industry_code"),
                "count": r.get("cnt"),
                "avg_confidence": round(r.get("avg_conf", 0), 2),
            }
            for r in rows
        ]

    def get_today_classified(self) -> List[ClassifiedNews]:
        """获取今日已分类的新闻"""
        sql = """
        SELECT * FROM news_industry_classification
        WHERE DATE(classified_at) = CURDATE()
        ORDER BY classified_at DESC
        """

        rows = fetch_all(sql)
        return [self._row_to_classified(r) for r in rows]

    def _row_to_classified(self, row: dict) -> ClassifiedNews:
        """行转对象"""
        pub_at = row.get("published_at")
        if pub_at:
            if isinstance(pub_at, datetime):
                published_at = pub_at
            else:
                published_at = datetime.fromisoformat(str(pub_at))
        else:
            published_at = None

        n_date = row.get("news_date")
        if n_date:
            if isinstance(n_date, date):
                news_date = n_date
            else:
                news_date = date.fromisoformat(str(n_date))
        else:
            news_date = None

        classified_at = row.get("classified_at")
        if classified_at:
            if isinstance(classified_at, datetime):
                classified_dt = classified_at
            else:
                classified_dt = datetime.fromisoformat(str(classified_at))
        else:
            classified_dt = datetime.now()

        return ClassifiedNews(
            news_id=row.get("news_id", 0),
            title=row.get("title", ""),
            content=row.get("content", ""),
            source=row.get("source", ""),
            url=row.get("url", ""),
            published_at=published_at,
            news_date=news_date,
            original_category=row.get("original_category", "general"),
            industry=row.get("industry", "其他"),
            industry_code=row.get("industry_code", "I000"),
            confidence=row.get("confidence", 0.0),
            classified_at=classified_dt,
        )


def get_repo() -> ClassificationRepo:
    return ClassificationRepo()
