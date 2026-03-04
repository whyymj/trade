from datetime import datetime
from typing import List, Optional

from analysis.llm.deepseek import DeepSeekClient
from analysis.llm.minimax import MiniMaxClient
from data.news.repo import NewsRepo
from modules.news_classification.interfaces import (
    ClassifiedNews,
    IndustryClassification,
    INDUSTRY_CATEGORIES,
)


class NewsClassifier:
    """新闻行业分类器 - 使用 LLM 进行行业分类"""

    def __init__(self, use_deepseek: bool = True):
        self.llm = DeepSeekClient() if use_deepseek else MiniMaxClient()
        self.news_repo = NewsRepo()

    def classify_news(
        self, title: str, content: str = "", source: str = ""
    ) -> IndustryClassification:
        """使用 LLM 对单条新闻进行行业分类"""
        industries_text = "\n".join(
            f"- {c['code']}: {c['name']} (关键词: {', '.join(c['keywords'])})"
            for c in INDUSTRY_CATEGORIES
        )

        prompt = f"""你是一个专业的财经新闻分析师。请根据以下新闻标题和内容，将其分类到最匹配的行业类别中。

可选行业类别：
{industries_text}

新闻标题：{title}
新闻内容：{content[:500] if content else "无"}
新闻来源：{source}

请返回 JSON 格式的分类结果：
{{
    "industry": "行业名称",
    "industry_code": "Ixxx",
    "confidence": 0.85,
    "reasoning": "分类理由（30字以内）"
}}

只返回 JSON，不要其他内容。"""

        try:
            response = self.llm.chat(prompt)
            result = self._parse_response(response)
            return result
        except Exception as e:
            print(f"[NewsClassifier] classify_news error: {e}")
            return self._fallback_classify(title, content)

    def classify_batch(
        self, news_list: List[dict], use_cache: bool = True
    ) -> List[ClassifiedNews]:
        """批量分类新闻"""
        results = []
        for news in news_list:
            classification = self.classify_news(
                news.get("title", ""),
                news.get("content", ""),
                news.get("source", ""),
            )

            classified = ClassifiedNews(
                news_id=news.get("id", 0),
                title=news.get("title", ""),
                content=news.get("content", ""),
                source=news.get("source", ""),
                url=news.get("url", ""),
                published_at=news.get("published_at"),
                news_date=news.get("news_date"),
                original_category=news.get("category", "general"),
                industry=classification.industry,
                industry_code=classification.industry_code,
                confidence=classification.confidence,
                classified_at=datetime.now(),
            )
            results.append(classified)

        return results

    def classify_today_news(self) -> List[ClassifiedNews]:
        """分类今日新闻"""
        today_news = self.news_repo.get_today_news()
        news_dicts = [
            {
                "id": i,
                "title": n.title,
                "content": n.content,
                "source": n.source,
                "url": n.url,
                "published_at": n.published_at,
                "news_date": n.news_date,
                "category": n.category,
            }
            for i, n in enumerate(today_news)
        ]
        return self.classify_batch(news_dicts)

    def _parse_response(self, response: str) -> IndustryClassification:
        """解析 LLM 响应"""
        import json
        import re

        json_match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if not json_match:
            return IndustryClassification(
                industry="其他",
                industry_code="I000",
                confidence=0.0,
                reasoning="解析失败",
            )

        try:
            data = json.loads(json_match.group())
            return IndustryClassification(
                industry=data.get("industry", "其他"),
                industry_code=data.get("industry_code", "I000"),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
        except:
            return IndustryClassification(
                industry="其他",
                industry_code="I000",
                confidence=0.0,
                reasoning="解析错误",
            )

    def _fallback_classify(self, title: str, content: str) -> IndustryClassification:
        """基于关键词的兜底分类"""
        text = (title + " " + content).lower()

        for cat in INDUSTRY_CATEGORIES:
            for keyword in cat["keywords"]:
                if keyword.lower() in text:
                    return IndustryClassification(
                        industry=cat["name"],
                        industry_code=cat["code"],
                        confidence=0.7,
                        reasoning=f"关键词匹配: {keyword}",
                    )

        return IndustryClassification(
            industry="其他",
            industry_code="I000",
            confidence=0.3,
            reasoning="无匹配关键词",
        )


def get_classifier(use_deepseek: bool = True) -> NewsClassifier:
    return NewsClassifier(use_deepseek)
