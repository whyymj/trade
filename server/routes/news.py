# server/routes/news.py
"""
新闻相关 API 路由
"""

import logging

from flask import Blueprint, jsonify, request
from datetime import datetime

from data.news import NewsCrawler, NewsRepo
from analysis.llm import get_analyzer

logger = logging.getLogger(__name__)

news_bp = Blueprint("news", __name__, url_prefix="/api/news")


@news_bp.route("/detail/<path:news_id>", methods=["GET"])
def get_news_detail(news_id):
    """获取新闻详情"""
    import urllib.parse

    url = urllib.parse.unquote(news_id)

    repo = NewsRepo()
    news = repo.get_news_by_url(url)

    if not news:
        return jsonify({"code": 404, "message": "新闻不存在"})

    return jsonify(
        {
            "code": 0,
            "data": {
                "id": news.url,
                "title": news.title,
                "content": news.content,
                "source": news.source,
                "url": news.url,
                "published_at": news.published_at.isoformat()
                if news.published_at
                else None,
                "category": news.category,
            },
        }
    )


@news_bp.route("/list", methods=["GET"])
def get_news_list():
    """获取新闻列表"""
    days = request.args.get("days", 1, type=int)
    category = request.args.get("category", None)
    limit = request.args.get("limit", 100, type=int)

    repo = NewsRepo()
    news_list = repo.get_news(days=days, category=category, limit=limit)

    data = []
    for news in news_list:
        data.append(
            {
                "id": news.url,
                "title": news.title,
                "content": news.content[:200] if news.content else "",
                "source": news.source,
                "url": news.url,
                "published_at": news.published_at.isoformat()
                if news.published_at
                else None,
                "category": news.category,
            }
        )

    return jsonify({"code": 0, "data": data})


@news_bp.route("/latest", methods=["GET"])
def get_news_latest():
    """获取最新新闻"""
    repo = NewsRepo()
    news_list = repo.get_today_news()

    data = []
    for news in news_list[:10]:
        data.append(
            {
                "id": news.url,
                "title": news.title,
                "content": news.content[:200] if news.content else "",
                "source": news.source,
                "url": news.url,
                "published_at": news.published_at.isoformat()
                if news.published_at
                else None,
                "category": news.category,
            }
        )

    return jsonify({"code": 0, "data": data})


@news_bp.route("/sync", methods=["POST"])
def sync_news():
    """手动同步新闻"""
    crawler = NewsCrawler()
    repo = NewsRepo()

    if not crawler.can_fetch():
        return jsonify(
            {
                "code": 429,
                "message": "频率限制，请稍后再试",
                "status": crawler.get_fetch_status(),
            }
        )

    news_list = crawler.fetch_today()
    saved = repo.save_news(news_list)

    return jsonify(
        {
            "code": 0,
            "data": {
                "fetched": len(news_list),
                "saved": saved,
                "duplicates": len(news_list) - saved,
                "status": crawler.get_fetch_status(),
            },
        }
    )


@news_bp.route("/analyze", methods=["POST"])
def analyze_news():
    """分析新闻"""
    data = request.json or {}
    days = data.get("days", 1)
    use_deepseek = data.get("use_deepseek", True)

    repo = NewsRepo()
    news_list = repo.get_news(days=days, limit=50)

    news_dicts = []
    news_ids = []
    for news in news_list:
        news_dicts.append(
            {
                "title": news.title,
                "content": news.content,
                "source": news.source,
                "published_at": news.published_at.isoformat()
                if news.published_at
                else None,
                "category": news.category,
            }
        )
        if hasattr(news, "id"):
            news_ids.append(news.id)

    analyzer = get_analyzer()
    result = analyzer.analyze(news_dicts, use_deepseek=use_deepseek)

    repo.save_analysis(
        type(
            "AnalysisResult",
            (),
            {
                "news_count": result["news_count"],
                "summary": result["summary"],
                "deep_analysis": result["deep_analysis"],
                "market_impact": result["market_impact"],
                "key_events": result["key_events"],
                "investment_advice": result["investment_advice"],
                "analyzed_at": datetime.now(),
            },
        )()
    )

    if result.get("industry_tags") and news_ids:
        repo.update_news_industry_tags(news_ids, result["industry_tags"])

    return jsonify({"code": 0, "data": result})


@news_bp.route("/analysis/latest", methods=["GET"])
def get_analysis_latest():
    """获取最新分析结果"""
    repo = NewsRepo()
    result = repo.get_latest_analysis()

    if not result:
        return jsonify({"code": 0, "data": None, "message": "暂无分析结果"})

    return jsonify(
        {
            "code": 0,
            "data": {
                "news_count": result.news_count,
                "summary": result.summary,
                "deep_analysis": result.deep_analysis,
                "market_impact": result.market_impact,
                "key_events": result.key_events,
                "investment_advice": result.investment_advice,
                "analyzed_at": result.analyzed_at.isoformat()
                if hasattr(result.analyzed_at, "isoformat")
                else str(result.analyzed_at),
            },
        }
    )


@news_bp.route("/status", methods=["GET"])
def get_status():
    """获取爬虫状态"""
    crawler = NewsCrawler()
    repo = NewsRepo()

    return jsonify(
        {
            "code": 0,
            "data": {
                "crawler": crawler.get_fetch_status(),
                "news_count": repo.get_news_count(7),
                "categories": repo.get_categories(),
            },
        }
    )


@news_bp.route("/cleanup", methods=["POST"])
def cleanup_news():
    """清理过期新闻"""
    repo = NewsRepo()
    keep_days = request.json.get("keep_days", 30) if request.json else 30

    deleted = repo.cleanup_old_news(keep_days)

    return jsonify({"code": 0, "data": {"deleted": deleted}})
