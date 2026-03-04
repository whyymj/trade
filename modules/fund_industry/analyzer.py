# modules/fund_industry/analyzer.py
"""
基金行业分析器 - 基于LLM进行行业分类
"""

import os
from typing import List, Optional

from data import fund_repo
from .repo import FundIndustryRepo

# 行业分类标准
INDUSTRY_KEYWORDS = {
    "新能源": ["光伏", "锂电", "储能", "电动车", "新能源汽车", "锂电池", "太阳能", "风电", "核电", "氢能"],
    "半导体": ["芯片", "集成电路", "AI", "算力", "半导体", "晶圆", "芯片", "GPU", "CPU", "传感器"],
    "医药": ["创新药", "医疗器械", "中药", "生物医药", "疫苗", "医疗服务", "医药", "制药", "CRO", "CXO"],
    "消费": ["白酒", "食品饮料", "零售", "家电", "纺织服装", "餐饮", "旅游", "酒店", "免税"],
    "金融": ["银行", "保险", "券商", "信托", "租赁", "担保", "金融科技", "FinTech"],
    "军工": ["航空航天", "船舶", "军工", "国防", "导弹", "无人机", "卫星", "雷达"],
    "TMT": ["互联网", "软件", "云服务", "游戏", "传媒", "通信", "5G", "数据中心"],
    "基建": ["建筑", "建材", "工程机械", "房地产", "园林", "装饰", "建筑设计", "装配式建筑"],
    "农业": ["种植", "养殖", "农林牧渔", "种子", "饲料", "农药", "化肥", "农产品"],
    "化工": ["石化", "化工新材料", "精细化工", "化学制品", "塑料", "橡胶", "纤维"],
}


class FundIndustryAnalyzer:
    """基金行业分析器"""

    def __init__(self, repo: FundIndustryRepo = None):
        self.repo = repo or FundIndustryRepo()
        self._llm_client = None

    def _get_llm_client(self):
        """获取LLM客户端"""
        if self._llm_client is not None:
            return self._llm_client
        
        # 尝试获取 DeepSeek 客户端
        try:
            from analysis.llm.deepseek import get_client
            self._llm_client = get_client()
            if self._llm_client and self._llm_client.is_available():
                return self._llm_client
        except:
            pass
        
        # 尝试获取 MiniMax 客户端
        try:
            from analysis.llm.minimax import get_client
            self._llm_client = get_client()
            if self._llm_client and self._llm_client.is_available():
                return self._llm_client
        except:
            pass
        
        return None

    def analyze(self, fund_code: str) -> List[dict]:
        """
        分析基金行业
        
        Args:
            fund_code: 基金代码
        
        Returns:
            List[dict]: 行业列表 [{"industry": "新能源", "confidence": 95.0, "source": "llm"}, ...]
        """
        if not fund_code:
            return []

        # 获取基金信息
        fund_info = fund_repo.get_fund_info(fund_code)
        if not fund_info:
            # 尝试从公开信息获取
            fund_info = self._fetch_fund_info(fund_code)
            if not fund_info:
                return []

        # 获取重仓股信息（如果有）
        holdings = self._get_fund_holdings(fund_code)

        # 使用LLM分析行业
        industries = self._analyze_with_llm(fund_info, holdings)
        
        if not industries:
            # 降级：使用关键词匹配
            industries = self._analyze_with_keywords(fund_info, holdings)

        # 保存到数据库
        if industries:
            self.repo.save_industries(fund_code, industries)

        return industries

    def _fetch_fund_info(self, fund_code: str) -> Optional[dict]:
        """从外部获取基金信息"""
        # 这里可以调用基金数据接口获取基金信息
        # 暂时返回空，后续可以扩展
        return {"fund_code": fund_code, "fund_name": "", "fund_type": ""}

    def _get_fund_holdings(self, fund_code: str) -> List[dict]:
        """获取基金重仓股"""
        # 这里可以调用基金数据接口获取重仓股信息
        # 暂时返回空，后续可以扩展
        return []

    def _analyze_with_llm(self, fund_info: dict, holdings: List[dict]) -> List[dict]:
        """使用LLM分析行业"""
        llm = self._get_llm_client()
        if not llm:
            return []

        fund_name = fund_info.get("fund_name", "")
        fund_type = fund_info.get("fund_type", "")
        
        # 构建持仓信息
        holdings_text = ""
        if holdings:
            stock_names = [h.get("stock_name", "") for h in holdings]
            holdings_text = f"重仓股：{', '.join(stock_names)}"

        # 行业列表
        industry_list = list(INDUSTRY_KEYWORDS.keys())

        prompt = f"""作为资深基金分析师，请分析以下基金的主要投资行业：

基金名称：{fund_name}
基金类型：{fund_type}
{holdings_text}

可选行业：{', '.join(industry_list)}

请按以下JSON格式输出分析结果（只输出JSON，不要其他内容）：
{{
    "industries": [
        {{"industry": "行业名称", "confidence": 置信度(0-100)}},
        ...
    ]
}}

只输出置信度最高的2-3个行业，置信度必须大于50才输出。
"""

        try:
            result = llm.chat([{"role": "user", "content": prompt}])
            
            # 解析JSON结果
            import json
            import re
            
            # 尝试提取JSON
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                data = json.loads(json_match.group())
                industries = data.get("industries", [])
                
                # 验证行业名称是否在标准列表中
                valid_industries = []
                for item in industries:
                    industry = item.get("industry", "")
                    if industry in INDUSTRY_KEYWORDS:
                        valid_industries.append({
                            "industry": industry,
                            "confidence": float(item.get("confidence", 0)),
                            "source": "llm"
                        })
                
                return valid_industries[:3]
        except Exception as e:
            print(f"[FundIndustryAnalyzer] LLM analysis error: {e}")
        
        return []

    def _analyze_with_keywords(self, fund_info: dict, holdings: List[dict]) -> List[dict]:
        """使用关键词匹配分析行业"""
        # 合并基金名称、类型和持仓信息
        text = " ".join([
            fund_info.get("fund_name", ""),
            fund_info.get("fund_type", ""),
            " ".join([h.get("stock_name", "") for h in holdings])
        ])

        # 统计每个行业的关键词匹配次数
        industry_scores = {}
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            if score > 0:
                industry_scores[industry] = score

        if not industry_scores:
            return []

        # 计算置信度
        max_score = max(industry_scores.values())
        total_score = sum(industry_scores.values())
        
        industries = []
        for industry, score in sorted(industry_scores.items(), key=lambda x: -x[1])[:3]:
            confidence = (score / max_score) * 50 + 50  # 基础50分，最高100分
            confidence = min(95, confidence)  # 最高95分
            industries.append({
                "industry": industry,
                "confidence": round(confidence, 2),
                "source": "keyword"
            })

        return industries


def get_analyzer() -> FundIndustryAnalyzer:
    """获取分析器实例"""
    return FundIndustryAnalyzer()
