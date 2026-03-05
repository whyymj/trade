# analysis/fund_analyzer.py
"""
基金自动分析器 - 自动提取行业标签和生成分析报告
"""

import json
import logging
from typing import Optional

from data import fund_repo

logger = logging.getLogger(__name__)


def auto_analyze_fund_industry_tags(fund_code: str) -> list[str]:
    """
    自动分析基金行业标签
    1. 获取基金基本信息
    2. 调用LLM分析行业/主题
    3. 更新数据库
    """
    fund_info = fund_repo.get_fund_info(fund_code)
    if not fund_info:
        logger.warning(f"Fund {fund_code} not found")
        return []

    fund_repo.update_fund_analysis_status(fund_code, "analyzing")

    fund_name = fund_info.get("fund_name", "")
    fund_type = fund_info.get("fund_type", "")
    manager = fund_info.get("manager", "")

    nav_df = fund_repo.get_fund_nav(fund_code, start_date=None, end_date=None)
    recent_performance = ""
    if nav_df is not None and len(nav_df) > 0:
        recent_30 = nav_df.tail(30)
        if len(recent_30) > 0:
            start_nav = recent_30.iloc[0]["unit_nav"]
            end_nav = recent_30.iloc[-1]["unit_nav"]
            if start_nav and end_nav:
                change = ((end_nav - start_nav) / start_nav) * 100
                recent_performance = f"近30日涨幅: {change:.2f}%"

    prompt = f"""请分析以下基金的核心投资行业和主题标签：

基金名称: {fund_name}
基金类型: {fund_type}
基金经理: {manager}
{recent_performance}

要求：
1. 根据基金名称、类型、基金经理投资风格推断其主要持仓行业
2. 返回3-5个行业/主题标签，如：["新能源", "半导体", "白酒", "医药", "人工智能"]
3. 使用中文标签，精确到二级行业（如"新能源汽车"而非"汽车"）
4. 只返回JSON数组格式，不要其他内容

输出格式：
["行业1", "行业2", "行业3"]
"""

    try:
        from analysis.llm.deepseek import DeepSeekClient

        client = DeepSeekClient()

        if client and client.is_available():
            result = client.chat([{"role": "user", "content": prompt}])

            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            tags = json.loads(result)

            if isinstance(tags, list):
                fund_repo.update_fund_industry_tags(fund_code, tags)
                fund_repo.update_fund_analysis_status(fund_code, "completed")
                logger.info(f"Fund {fund_code} industry tags updated: {tags}")
                return tags
    except Exception as e:
        logger.error(f"Failed to analyze fund {fund_code}: {e}")

    fund_repo.update_fund_analysis_status(fund_code, "failed")
    return []


def auto_analyze_fund_full(fund_code: str) -> str:
    """
    自动生成基金完整分析报告
    """
    fund_info = fund_repo.get_fund_info(fund_code)
    if not fund_info:
        return "基金不存在"

    fund_repo.update_fund_analysis_status(fund_code, "analyzing")

    fund_name = fund_info.get("fund_name", "")
    fund_type = fund_info.get("fund_type", "")

    nav_df = fund_repo.get_fund_nav(fund_code, start_date=None, end_date=None)

    performance_data = ""
    if nav_df is not None and len(nav_df) > 0:
        df = nav_df.sort_values("nav_date")
        recent_1y = df.tail(250)

        if len(recent_1y) >= 30:
            start_nav = recent_1y.iloc[0]["unit_nav"]
            end_nav = recent_1y.iloc[-1]["unit_nav"]
            if start_nav and end_nav:
                y1_return = ((end_nav - start_nav) / start_nav) * 100
                performance_data += f"近1年收益率: {y1_return:.2f}%\n"

        if len(recent_1y) >= 60:
            start_nav = recent_1y.iloc[0]["unit_nav"]
            max_nav = recent_1y["unit_nav"].max()
            if start_nav and max_nav:
                max_dd = ((max_nav - start_nav) / max_nav) * 100
                performance_data += f"历史最大回撤: {max_dd:.2f}%\n"

        recent_30 = df.tail(30)
        if len(recent_30) > 0:
            returns = recent_30["daily_return"].dropna()
            if len(returns) > 0:
                win_rate = (returns > 0).sum() / len(returns) * 100
                performance_data += f"近30日胜率: {win_rate:.1f}%\n"

    prompt = f"""作为资深基金分析师，请对以下基金进行全面分析：

基金名称: {fund_name}
基金类型: {fund_type}

近期表现数据：
{performance_data}

请生成包含以下内容的分析报告：
1. **基金概况** - 基金类型、投资策略
2. **业绩分析** - 短期和中长期表现评价
3. **行业配置** - 推断主要持仓行业及前景
4. **投资建议** - 短期、中期、长期操作建议
5. **风险提示** - 主要风险因素

请用专业但易懂的语言撰写分析报告。"""

    try:
        from analysis.llm.deepseek import DeepSeekClient

        client = DeepSeekClient()

        if client and client.is_available():
            result = client.chat([{"role": "user", "content": prompt}])

            fund_repo.update_fund_llm_analysis(fund_code, result)
            fund_repo.update_fund_analysis_status(fund_code, "completed")
            logger.info(f"Fund {fund_code} full analysis completed")
            return result
    except Exception as e:
        logger.error(f"Failed to analyze fund {fund_code}: {e}")

    fund_repo.update_fund_analysis_status(fund_code, "failed")
    return "分析生成失败，请稍后重试"


def get_fund_industry_tags(fund_code: str) -> list[str]:
    """获取基金行业标签"""
    fund_info = fund_repo.get_fund_info(fund_code)
    if not fund_info:
        return []

    tags = fund_info.get("industry_tags")
    if tags:
        if isinstance(tags, str):
            try:
                return json.loads(tags)
            except:
                return []
        elif isinstance(tags, list):
            return tags
    return []
