# server/routes/news_classification.py
"""
新闻行业分类 API 路由
"""

from flask import Blueprint, jsonify, request

from modules.news_classification import (
    NewsClassifier,
    ClassificationRepo,
    INDUSTRY_CATEGORIES,
)

news_classification_bp = Blueprint(
    "news_classification", __name__, url_prefix="/api/news-classification"
)


@news_classification_bp.route("/industries", methods=["GET"])
def get_industries():
    """
    获取所有行业分类

    GET /api/news-classification/industries

    Returns:
        {
            "code": 0,
            "data": [
                {"code": "I001", "name": "新能源汽车", "keywords": [...]},
                ...
            ]
        }
    """
    return jsonify({"code": 0, "data": INDUSTRY_CATEGORIES})


@news_classification_bp.route("/classify", methods=["POST"])
def classify_news():
    """
    分类单条新闻

    POST /api/news-classification/classify

    Body: {"title": "...", "content": "...", "source": "..."}

    Returns:
        {
            "code": 0,
            "data": {
                "industry": "新能源汽车",
                "industry_code": "I001",
                "confidence": 0.85,
                "reasoning": "包含比亚迪、锂电池等关键词"
            }
        }
    """
    data = request.get_json() or {}
    title = data.get("title", "")
    content = data.get("content", "")
    source = data.get("source", "")

    if not title:
        return jsonify({"code": 400, "message": "标题不能为空"})

    use_deepseek = data.get("use_deepseek", True)
    classifier = NewsClassifier(use_deepseek=use_deepseek)

    try:
        result = classifier.classify_news(title, content, source)
        return jsonify(
            {
                "code": 0,
                "data": {
                    "industry": result.industry,
                    "industry_code": result.industry_code,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "message": f"分类失败: {str(e)}"})


@news_classification_bp.route("/classify-today", methods=["POST"])
def classify_today_news():
    """
    分类今日所有新闻

    POST /api/news-classification/classify-today

    Returns:
        {
            "code": 0,
            "data": {
                "classified_count": 50,
                "stats": [
                    {"industry": "新能源汽车", "count": 10},
                    ...
                ]
            }
        }
    """
    use_deepseek = request.json.get("use_deepseek", True) if request.json else True
    classifier = NewsClassifier(use_deepseek=use_deepseek)
    repo = ClassificationRepo()

    try:
        classified = classifier.classify_today_news()
        repo.save_batch(classified)

        stats = {}
        for c in classified:
            stats[c.industry] = stats.get(c.industry, 0) + 1

        stats_list = [
            {"industry": k, "count": v}
            for k, v in sorted(stats.items(), key=lambda x: -x[1])
        ]

        return jsonify(
            {
                "code": 0,
                "data": {
                    "classified_count": len(classified),
                    "stats": stats_list,
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "message": f"分类失败: {str(e)}"})


@news_classification_bp.route("/industry/<industry_code>", methods=["GET"])
def get_news_by_industry(industry_code):
    """
    按行业获取新闻

    GET /api/news-classification/industry/:code

    Query: days=7

    Returns:
        {
            "code": 0,
            "data": [
                {"title": "...", "industry": "新能源汽车", "confidence": 0.85, ...},
                ...
            ]
        }
    """
    days = request.args.get("days", 7, type=int)
    repo = ClassificationRepo()

    try:
        news_list = repo.get_by_industry(industry_code, days)
        data = [
            {
                "news_id": n.news_id,
                "title": n.title,
                "source": n.source,
                "url": n.url,
                "published_at": n.published_at.isoformat() if n.published_at else None,
                "industry": n.industry,
                "industry_code": n.industry_code,
                "confidence": n.confidence,
            }
            for n in news_list
        ]
        return jsonify({"code": 0, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})


@news_classification_bp.route("/stats", methods=["GET"])
def get_industry_stats():
    """
    获取行业统计

    GET /api/news-classification/stats

    Query: days=7

    Returns:
        {
            "code": 0,
            "data": [
                {"industry": "新能源汽车", "industry_code": "I001", "count": 15, "avg_confidence": 0.82},
                ...
            ]
        }
    """
    days = request.args.get("days", 7, type=int)
    repo = ClassificationRepo()

    try:
        stats = repo.get_industry_stats(days)
        return jsonify({"code": 0, "data": stats})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})


@news_classification_bp.route("/today", methods=["GET"])
def get_today_classified():
    """
    获取今日已分类新闻

    GET /api/news-classification/today

    Returns:
        {
            "code": 0,
            "data": [...]
        }
    """
    repo = ClassificationRepo()

    try:
        news_list = repo.get_today_classified()
        data = [
            {
                "news_id": n.news_id,
                "title": n.title,
                "source": n.source,
                "url": n.url,
                "industry": n.industry,
                "industry_code": n.industry_code,
                "confidence": n.confidence,
                "classified_at": n.classified_at.isoformat()
                if n.classified_at
                else None,
            }
            for n in news_list
        ]
        return jsonify({"code": 0, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})
