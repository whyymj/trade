# modules/fund_industry/repo.py
"""
基金行业仓储层
"""

from typing import List, Optional

from data.mysql import execute, fetch_all, fetch_one
from .interfaces import FundIndustry


class FundIndustryRepo:
    """基金行业仓储"""

    def save_industries(self, fund_code: str, industries: List[dict]) -> bool:
        """
        保存基金行业
        
        Args:
            fund_code: 基金代码
            industries: 行业列表 [{"industry": "新能源", "confidence": 95.0, "source": "llm"}, ...]
        
        Returns:
            bool: 是否保存成功
        """
        if not fund_code or not industries:
            return False

        # 先删除原有行业
        self.delete_industries(fund_code)

        # 插入新行业
        sql = """
        INSERT INTO fund_industry (fund_code, industry, confidence, source)
        VALUES (%s, %s, %s, %s)
        """
        
        params_list = []
        for item in industries:
            params_list.append((
                fund_code,
                item.get("industry", ""),
                float(item.get("confidence", 0)),
                item.get("source", "llm")
            ))

        try:
            for params in params_list:
                execute(sql, params)
            return True
        except Exception as e:
            print(f"[FundIndustryRepo] save_industries error: {e}")
            return False

    def get_industries(self, fund_code: str) -> List[dict]:
        """
        获取基金行业
        
        Args:
            fund_code: 基金代码
        
        Returns:
            List[dict]: 行业列表
        """
        if not fund_code:
            return []

        sql = """
        SELECT id, fund_code, industry, confidence, source, updated_at
        FROM fund_industry
        WHERE fund_code = %s
        ORDER BY confidence DESC
        """

        rows = fetch_all(sql, (fund_code,))
        
        result = []
        for row in rows:
            result.append({
                "fund_code": row.get("fund_code"),
                "industry": row.get("industry"),
                "confidence": float(row.get("confidence", 0)),
                "source": row.get("source", "llm"),
                "updated_at": str(row.get("updated_at")) if row.get("updated_at") else None
            })
        
        return result

    def delete_industries(self, fund_code: str) -> int:
        """
        删除基金行业
        
        Args:
            fund_code: 基金代码
        
        Returns:
            int: 删除的记录数
        """
        if not fund_code:
            return 0

        sql = "DELETE FROM fund_industry WHERE fund_code = %s"
        
        try:
            return execute(sql, (fund_code,))
        except Exception as e:
            print(f"[FundIndustryRepo] delete_industries error: {e}")
            return 0

    def get_industry_by_fund(self, fund_code: str) -> Optional[dict]:
        """
        获取基金主要行业（置信度最高的）
        
        Args:
            fund_code: 基金代码
        
        Returns:
            Optional[dict]: 主要行业信息
        """
        if not fund_code:
            return None

        sql = """
        SELECT id, fund_code, industry, confidence, source, updated_at
        FROM fund_industry
        WHERE fund_code = %s
        ORDER BY confidence DESC
        LIMIT 1
        """

        row = fetch_one(sql, (fund_code,))
        
        if not row:
            return None
        
        return {
            "fund_code": row.get("fund_code"),
            "industry": row.get("industry"),
            "confidence": float(row.get("confidence", 0)),
            "source": row.get("source", "llm"),
            "updated_at": str(row.get("updated_at")) if row.get("updated_at") else None
        }


def get_repo() -> FundIndustryRepo:
    """获取仓储实例"""
    return FundIndustryRepo()
