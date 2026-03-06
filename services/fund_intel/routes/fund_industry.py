from flask import Blueprint, jsonify
from services.fund_intel.clients import FundClient, LLMClient
from services.fund_intel.modules.fund_industry import FundIndustryAnalyzer

fund_industry_bp = Blueprint("fund_industry", __name__)
fund_client = FundClient()
llm_client = LLMClient()
analyzer = FundIndustryAnalyzer()


@fund_industry_bp.route("/analyze/<fund_code>", methods=["POST"])
def analyze_fund_industry(fund_code):
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({"success": False, "message": "Fund not found"}), 404

    industry_result = analyzer.analyze(fund_code)

    llm_messages = [
        {"role": "system", "content": "你是一个基金行业分析专家。"},
        {"role": "user", "content": f"请分析以下基金的行业配置：{industry_result}"},
    ]
    llm_analysis = llm_client.chat(llm_messages, provider="deepseek")

    return jsonify(
        {
            "success": True,
            "data": {
                "fund_code": fund_code,
                "industry_distribution": industry_result,
                "llm_analysis": llm_analysis,
            },
        }
    )


@fund_industry_bp.route("/<fund_code>", methods=["GET"])
def get_fund_industry(fund_code):
    from shared.cache import get_cache

    cache = get_cache()

    cache_key = f"fund_industry_{fund_code}"
    cached = cache.get(cache_key)

    if cached:
        return jsonify({"success": True, "data": cached})
    else:
        result = analyzer.analyze(fund_code)
        return jsonify({"success": True, "data": result})


@fund_industry_bp.route("/primary/<fund_code>", methods=["GET"])
def get_primary_industry(fund_code):
    result = analyzer.analyze(fund_code)
    if result:
        primary = result[:3]
        return jsonify({"success": True, "data": primary})
    return jsonify({"success": True, "data": []})
