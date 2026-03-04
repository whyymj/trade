# server/routes/fund_news_association.py
"""
基金-新闻关联 API 路由
"""

from flask import Blueprint, jsonify, request

from modules.fund_news_association import FundNewsMatcher, AssociationRepo

fund_news_association_bp = Blueprint(
    "fund_news_association", __name__, url_prefix="/api/fund-news"
)


@fund_news_association_bp.route("/match/<fund_code>", methods=["GET"])
def match_fund_news(fund_code):
    """
    为基金匹配相关新闻

    GET /api/fund-news/match/:code

    Query: days=7, min_confidence=0.5

    Returns:
        {
            "code": 0,
            "data": [
                {
                    "news_title": "...",
                    "news_source": "...",
                    "news_url": "...",
                    "industry": "新能源汽车",
                    "match_score": 0.85,
                    "match_type": "industry"
                },
                ...
            ]
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    days = request.args.get("days", 7, type=int)
    min_confidence = request.args.get("min_confidence", 0.5, type=float)

    matcher = FundNewsMatcher()

    try:
        associations = matcher.match_fund_news(fund_code, days, min_confidence)

        data = [
            {
                "news_id": a.news_id,
                "news_title": a.news_title,
                "news_source": a.news_source,
                "news_url": a.news_url,
                "industry": a.industry,
                "industry_code": a.industry_code,
                "match_score": a.match_score,
                "match_type": a.match_type,
            }
            for a in associations
        ]

        return jsonify({"code": 0, "data": data})
    except Exception as e:
        return jsonify({"code": 500, "message": f"匹配失败: {str(e)}"})


@fund_news_association_bp.route("/summary/<fund_code>", methods=["GET"])
def get_fund_news_summary(fund_code):
    """
    获取基金新闻摘要

    GET /api/fund-news/summary/:code

    Query: days=7

    Returns:
        {
            "code": 0,
            "data": {
                "fund_code": "000001",
                "fund_name": "平安新混合",
                "industries": [...],
                "news_count": 15,
                "latest_news": [...],
                "sentiment": "positive",
                "updated_at": "2025-01-01T10:00:00"
            }
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    days = request.args.get("days", 7, type=int)
    matcher = FundNewsMatcher()

    try:
        summary = matcher.get_fund_news_summary(fund_code, days)

        if not summary:
            return jsonify(
                {"code": 404, "message": "该基金暂无行业信息，请先分析基金行业"}
            )

        return jsonify(
            {
                "code": 0,
                "data": {
                    "fund_code": summary.fund_code,
                    "fund_name": summary.fund_name,
                    "industries": summary.industries,
                    "news_count": summary.news_count,
                    "latest_news": summary.latest_news,
                    "sentiment": summary.sentiment,
                    "updated_at": summary.updated_at.isoformat(),
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})


@fund_news_association_bp.route("/list", methods=["GET"])
def list_funds_with_news():
    """
    获取有关联新闻的基金列表

    GET /api/fund-news/list

    Query: days=7

    Returns:
        {
            "code": 0,
            "data": [
                {"fund_code": "000001", "fund_name": "平安新混合", "news_count": 15},
                ...
            ]
        }
    """
    days = request.args.get("days", 7, type=int)
    repo = AssociationRepo()

    try:
        funds = repo.get_funds_with_news(days)
        return jsonify({"code": 0, "data": funds})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})


@fund_news_association_bp.route("/match-all", methods=["POST"])
def match_all_funds():
    """
    为所有基金匹配新闻（批量）

    POST /api/fund-news/match-all

    Body: {"days": 7}

    Returns:
        {
            "code": 0,
            "data": {
                "matched_funds": 10,
                "total_associations": 150
            }
        }
    """
    days = request.json.get("days", 7) if request.json else 7
    matcher = FundNewsMatcher()

    try:
        results = matcher.match_all_funds(days)
        repo = AssociationRepo()

        total = 0
        for code, associations in results.items():
            repo.save_batch(associations)
            total += len(associations)

        return jsonify(
            {
                "code": 0,
                "data": {
                    "matched_funds": len(results),
                    "total_associations": total,
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "message": f"匹配失败: {str(e)}"})
