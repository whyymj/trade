from flask import Blueprint, jsonify
from services.fund_intel.clients import FundClient, NewsClient
from services.fund_intel.modules.fund_news_association import FundNewsMatcher

fund_news_bp = Blueprint("fund_news", __name__)
fund_client = FundClient()
news_client = NewsClient()
matcher = FundNewsMatcher()


@fund_news_bp.route("/match/<fund_code>", methods=["GET"])
def match_fund_news(fund_code):
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({"success": False, "message": "Fund not found"}), 404

    industries = matcher.get_fund_industries(fund_code)

    related_news = matcher.match_fund_news(fund_code, days=7)

    return jsonify(
        {
            "success": True,
            "data": {
                "fund_code": fund_code,
                "industries": industries,
                "news_count": len(related_news),
                "news": related_news[:20],
            },
        }
    )


@fund_news_bp.route("/summary/<fund_code>", methods=["GET"])
def get_fund_news_summary(fund_code):
    fund_info = fund_client.get_fund_info(fund_code)
    if not fund_info:
        return jsonify({"success": False, "message": "Fund not found"}), 404

    industries = matcher.get_fund_industries(fund_code)
    news_list = matcher.match_fund_news(fund_code, days=7)

    positive = sum(1 for n in news_list if n.get("match_score", 0) >= 0.7)
    negative = sum(1 for n in news_list if n.get("match_score", 0) < 0.4)
    sentiment = (
        "positive"
        if positive > negative
        else "negative"
        if negative > positive
        else "neutral"
    )

    return jsonify(
        {
            "success": True,
            "data": {
                "fund_code": fund_code,
                "fund_name": fund_info.get("fund_name", ""),
                "industries": industries,
                "news_count": len(news_list),
                "latest_news": news_list[:5],
                "sentiment": sentiment,
            },
        }
    )


@fund_news_bp.route("/list", methods=["GET"])
def get_funds_with_news():
    fund_list = fund_client.get_fund_list(page=1, size=50)
    funds = fund_list.get("data", [])

    results = []
    for fund in funds[:20]:
        fund_code = fund.get("fund_code")
        if fund_code:
            news = matcher.match_fund_news(fund_code, days=7)
            if news:
                results.append(
                    {
                        "fund_code": fund_code,
                        "fund_name": fund.get("fund_name", ""),
                        "news_count": len(news),
                    }
                )

    return jsonify({"success": True, "data": results})
