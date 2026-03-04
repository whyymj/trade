# server/routes/investment_advice.py
"""
投资建议 API 路由
"""

from flask import Blueprint, jsonify, request

from modules.investment_advice import InvestmentAdvisor

investment_advice_bp = Blueprint(
    "investment_advice", __name__, url_prefix="/api/investment-advice"
)


@investment_advice_bp.route("/<fund_code>", methods=["GET"])
def get_investment_advice(fund_code):
    """
    获取投资建议

    GET /api/investment-advice/:code

    Query: days=7

    Returns:
        {
            "code": 0,
            "data": {
                "short_term": "短期建议...",
                "medium_term": "中期建议...",
                "long_term": "长期建议...",
                "risk_level": "中",
                "confidence": 75,
                "key_factors": ["因素1", "因素2"],
                "generated_at": "2025-01-01T10:00:00"
            }
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    days = request.args.get("days", 7, type=int)
    advisor = InvestmentAdvisor()

    try:
        advice = advisor.get_advice(fund_code, days)

        if not advice:
            return jsonify(
                {"code": 404, "message": "该基金暂无行业信息，请先分析基金行业"}
            )

        return jsonify(
            {
                "code": 0,
                "data": {
                    "short_term": advice.short_term,
                    "medium_term": advice.medium_term,
                    "long_term": advice.long_term,
                    "risk_level": advice.risk_level,
                    "confidence": advice.confidence,
                    "key_factors": advice.key_factors,
                    "generated_at": advice.generated_at.isoformat(),
                },
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取建议失败: {str(e)}"})


@investment_advice_bp.route("/batch", methods=["POST"])
def get_batch_advice():
    """
    批量获取投资建议

    POST /api/investment-advice/batch

    Body: {"fund_codes": ["000001", "000002"], "days": 7}

    Returns:
        {
            "code": 0,
            "data": {
                "000001": {...},
                "000002": {...}
            }
        }
    """
    data = request.get_json() or {}
    fund_codes = data.get("fund_codes", [])
    days = data.get("days", 7)

    if not fund_codes:
        return jsonify({"code": 400, "message": "基金代码列表不能为空"})

    advisor = InvestmentAdvisor()
    results = {}

    try:
        for code in fund_codes:
            try:
                advice = advisor.get_advice(code, days)
                if advice:
                    results[code] = {
                        "short_term": advice.short_term,
                        "medium_term": advice.medium_term,
                        "long_term": advice.long_term,
                        "risk_level": advice.risk_level,
                        "confidence": advice.confidence,
                        "key_factors": advice.key_factors,
                        "generated_at": advice.generated_at.isoformat(),
                    }
            except Exception:
                continue

        return jsonify({"code": 0, "data": results})
    except Exception as e:
        return jsonify({"code": 500, "message": f"批量获取失败: {str(e)}"})
