# analysis/llm/news_analyzer.py
"""
新闻分析器 - 双LLM流程
"""

import os
from typing import List, Optional
from datetime import datetime
from dataclasses import asdict

from .minimax import MiniMaxClient
from .deepseek import DeepSeekClient


class NewsAnalyzer:
    """新闻分析器 - 支持双LLM"""

    def __init__(
        self,
        minimax_client: MiniMaxClient = None,
        deepseek_client: DeepSeekClient = None,
    ):
        self.minimax = minimax_client or self._get_minimax()
        self.deepseek = deepseek_client or self._get_deepseek()

    def _get_minimax(self) -> Optional[MiniMaxClient]:
        try:
            from .minimax import get_client

            return get_client()
        except:
            return None

    def _get_deepseek(self) -> Optional[DeepSeekClient]:
        try:
            from .deepseek import get_client

            return get_client()
        except:
            return None

    def extract_key_info(self, news_list: List[dict]) -> str:
        """使用 MiniMax 提取关键信息"""
        if not self.minimax or not self.minimax.is_available():
            return self._mock_extract(news_list)

        formatted = self._format_news(news_list)

        prompt = f"""请从以下财经新闻中提取关键信息：
        
{formatted}

要求：
1. 提取每个新闻的：时间、事件、影响
2. 按重要性排序（重要的在前）
3. 过滤噪音信息
4. 合并同类事件

输出格式（每条不超过20字）：
- 政策：xxx
- 数据：xxx
- 事件：xxx
- 行业：xxx
"""

        try:
            result = self.minimax.chat([{"role": "user", "content": prompt}])
            return result
        except Exception as e:
            print(f"[NewsAnalyzer] MiniMax extract error: {e}")
            return self._mock_extract(news_list)

    def analyze(self, news_list: List[dict], use_deepseek: bool = True) -> dict:
        """
        综合分析

        Args:
            news_list: 新闻列表
            use_deepseek: 是否使用 DeepSeek (测试=False, 生产=True)
        """
        if not news_list:
            return {
                "news_count": 0,
                "summary": "无新闻数据",
                "deep_analysis": "无新闻数据",
                "market_impact": "neutral",
                "key_events": [],
                "investment_advice": "暂无建议",
            }

        summary = self.extract_key_info(news_list)

        deep_analysis = ""
        market_impact = "neutral"
        investment_advice = ""

        llm = self.deepseek if use_deepseek else self.minimax

        if llm and llm.is_available():
            prompt = f"""作为资深财经分析师，请根据以下新闻要点进行分析：

{summary}

请按以下格式输出：
## 市场判断
[看涨/看跌/中性] - 简要说明

## 原因分析
1. 宏观层面：xxx
2. 资金面：xxx
3. 情绪面：xxx

## 操作建议
[仓位建议]

## 风险提示
[需要注意的风险]
"""
            try:
                deep_analysis = llm.chat([{"role": "user", "content": prompt}])

                if "看涨" in deep_analysis or "利好" in deep_analysis:
                    market_impact = "bullish"
                elif "看跌" in deep_analysis or "利空" in deep_analysis:
                    market_impact = "bearish"
                else:
                    market_impact = "neutral"

                if "加仓" in deep_analysis or "买入" in deep_analysis:
                    investment_advice = "建议适度加仓"
                elif "减仓" in deep_analysis or "卖出" in deep_analysis:
                    investment_advice = "建议减仓避险"
                else:
                    investment_advice = "建议持有观望"

            except Exception as e:
                print(f"[NewsAnalyzer] Analysis error: {e}")
                deep_analysis = "分析生成失败"
        else:
            deep_analysis = summary
            market_impact = "neutral"
            investment_advice = "LLM不可用，请稍后再试"

        key_events = self._extract_key_events(news_list)

        return {
            "news_count": len(news_list),
            "summary": summary,
            "deep_analysis": deep_analysis,
            "market_impact": market_impact,
            "key_events": key_events,
            "investment_advice": investment_advice,
            "analyzed_at": datetime.now().isoformat(),
        }

    def get_available_provider(self) -> str:
        """获取当前可用的 LLM 提供商"""
        if self.deepseek and self.deepseek.is_available():
            return "deepseek"
        elif self.minimax and self.minimax.is_available():
            return "minimax"
        return "none"

    def _format_news(self, news_list: List[dict]) -> str:
        """格式化新闻列表"""
        lines = []
        for i, news in enumerate(news_list[:20], 1):
            title = news.get("title", "")
            source = news.get("source", "")
            published_at = news.get("published_at", "")
            lines.append(f"{i}. [{source}] {title} - {published_at}")
        return "\n".join(lines)

    def _extract_key_events(self, news_list: List[dict]) -> List[dict]:
        """提取关键事件"""
        events = []
        for news in news_list[:5]:
            events.append(
                {
                    "title": news.get("title", ""),
                    "source": news.get("source", ""),
                    "category": news.get("category", "general"),
                }
            )
        return events

    def _mock_extract(self, news_list: List[dict]) -> str:
        """Mock提取（当LLM不可用时）"""
        if not news_list:
            return "无新闻"

        titles = [n.get("title", "") for n in news_list[:5]]
        return "要点：" + "；".join(titles)


def get_analyzer() -> NewsAnalyzer:
    """获取分析器实例"""
    return NewsAnalyzer()
