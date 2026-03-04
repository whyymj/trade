from datetime import datetime
from typing import List, Optional

from data.mysql import execute, fetch_all, fetch_one
from modules.fund_news_association.interfaces import FundNewsAssociation


class AssociationRepo:
    """基金-新闻关联仓储"""

    def save_association(self, assoc: FundNewsAssociation) -> bool:
        """保存关联"""
        sql = """
        INSERT INTO fund_news_association 
        (fund_code, fund_name, news_id, news_title, news_source, news_url,
         industry, industry_code, match_type, match_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            match_score = VALUES(match_score),
            match_type = VALUES(match_type)
        """

        try:
            execute(
                sql,
                (
                    assoc.fund_code,
                    assoc.fund_name,
                    assoc.news_id,
                    assoc.news_title[:500],
                    assoc.news_source,
                    assoc.news_url,
                    assoc.industry,
                    assoc.industry_code,
                    assoc.match_type,
                    assoc.match_score,
                    assoc.created_at.isoformat(),
                ),
            )
            return True
        except Exception as e:
            print(f"[AssociationRepo] save error: {e}")
            return False

    def save_batch(self, associations: List[FundNewsAssociation]) -> int:
        """批量保存关联"""
        if not associations:
            return 0

        sql = """
        INSERT INTO fund_news_association 
        (fund_code, fund_name, news_id, news_title, news_source, news_url,
         industry, industry_code, match_type, match_score, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            match_score = VALUES(match_score),
            match_type = VALUES(match_type)
        """

        params_list = []
        for a in associations:
            params_list.append(
                (
                    a.fund_code,
                    a.fund_name,
                    a.news_id,
                    a.news_title[:500],
                    a.news_source,
                    a.news_url,
                    a.industry,
                    a.industry_code,
                    a.match_type,
                    a.match_score,
                    a.created_at.isoformat(),
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
            print(f"[AssociationRepo] save_batch error: {e}")
            return 0

    def get_by_fund(self, fund_code: str, days: int = 7) -> List[FundNewsAssociation]:
        """获取基金的关联新闻"""
        sql = """
        SELECT * FROM fund_news_association
        WHERE fund_code = %s
          AND created_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY match_score DESC
        """

        rows = fetch_all(sql, (fund_code, days))
        return [self._row_to_association(r) for r in rows]

    def get_funds_with_news(self, days: int = 7) -> List[dict]:
        """获取有关联新闻的基金列表"""
        sql = """
        SELECT fund_code, fund_name, COUNT(*) as news_count
        FROM fund_news_association
        WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        GROUP BY fund_code, fund_name
        ORDER BY news_count DESC
        """

        rows = fetch_all(sql, (days,))
        return [
            {
                "fund_code": r.get("fund_code"),
                "fund_name": r.get("fund_name"),
                "news_count": r.get("news_count"),
            }
            for r in rows
        ]

    def _row_to_association(self, row: dict) -> FundNewsAssociation:
        """行转对象"""
        created_at = row.get("created_at")
        if isinstance(created_at, datetime):
            dt = created_at
        else:
            dt = (
                datetime.fromisoformat(str(created_at))
                if created_at
                else datetime.now()
            )

        return FundNewsAssociation(
            id=row.get("id", 0),
            fund_code=row.get("fund_code", ""),
            fund_name=row.get("fund_name", ""),
            news_id=row.get("news_id", 0),
            news_title=row.get("news_title", ""),
            news_source=row.get("news_source", ""),
            news_url=row.get("news_url", ""),
            industry=row.get("industry", ""),
            industry_code=row.get("industry_code", ""),
            match_type=row.get("match_type", ""),
            match_score=row.get("match_score", 0.0),
            created_at=dt,
        )


def get_repo() -> AssociationRepo:
    return AssociationRepo()
