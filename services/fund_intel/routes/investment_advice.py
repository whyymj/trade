from flask import Blueprint, request, jsonify
from datetime import datetime
from services.fund_intel.clients import FundClient, NewsClient
from services.fund_intel.modules.investment_advice import InvestmentAdviceGenerator

investment_advice_bp = Blueprint("investment_advice", __name__)
fund_client = FundClient()
news_client = NewsClient()
advice_generator = InvestmentAdviceGenerator()


@investment_advice_bp.route("/<fund_code>", methods=["GET"])
def get_investment_advice(fund_code):
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({"success": False, "message": "Fund not found"}), 404

    fund_nav = fund_client.get_fund_nav(fund_code, days=30)

    advice = advice_generator.generate(fund_code)

    if not advice:
        return jsonify({"success": False, "message": "Failed to generate advice"}), 500

    return jsonify(
        {
            "success": True,
            "data": {
                "fund_code": fund_code,
                "fund_name": fund_info.get("fund_name"),
                "advice": advice,
                "generated_at": datetime.now().isoformat(),
            },
        }
    )


@investment_advice_bp.route("/batch", methods=["POST"])
def get_batch_investment_advice():
    data = request.get_json()
    fund_codes = data.get("fund_codes", [])

    if not fund_codes:
        return jsonify({"success": False, "message": "Fund codes required"}), 400

    results = []
    for fund_code in fund_codes[:10]:
        try:
            advice = advice_generator.generate(fund_code)
            results.append({"fund_code": fund_code, "success": True, "advice": advice})
        except Exception as e:
            results.append({"fund_code": fund_code, "success": False, "error": str(e)})

    return jsonify({"success": True, "data": results})
