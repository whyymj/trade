# server/routes/fund_industry.py
"""
基金行业相关 API 路由
"""

from flask import Blueprint, jsonify, request

from modules.fund_industry import FundIndustryRepo, FundIndustryAnalyzer

fund_industry_bp = Blueprint("fund_industry", __name__, url_prefix="/api/fund-industry")


@fund_industry_bp.route("/analyze/<fund_code>", methods=["POST"])
def analyze_fund_industry(fund_code):
    """
    分析基金行业
    
    POST /api/fund-industry/analyze/:code
    
    Returns:
        {
            "code": 0,
            "data": [
                {"industry": "新能源", "confidence": 95.0, "source": "llm"},
                {"industry": "半导体", "confidence": 60.0, "source": "llm"}
            ]
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    analyzer = FundIndustryAnalyzer()
    
    try:
        industries = analyzer.analyze(fund_code)
        
        if not industries:
            return jsonify({
                "code": 404, 
                "message": "无法分析该基金的行业，请检查基金代码是否正确"
            })
        
        return jsonify({"code": 0, "data": industries})
    except Exception as e:
        return jsonify({"code": 500, "message": f"分析失败: {str(e)}"})


@fund_industry_bp.route("/<fund_code>", methods=["GET"])
def get_fund_industry(fund_code):
    """
    获取基金行业
    
    GET /api/fund-industry/:code
    
    Returns:
        {
            "code": 0,
            "data": [
                {"industry": "新能源", "confidence": 95.0, "source": "llm", "updated_at": "2025-01-01 10:00:00"}
            ]
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    repo = FundIndustryRepo()
    
    try:
        industries = repo.get_industries(fund_code)
        
        return jsonify({"code": 0, "data": industries})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})


@fund_industry_bp.route("/primary/<fund_code>", methods=["GET"])
def get_primary_industry(fund_code):
    """
    获取基金主要行业（置信度最高的）
    
    GET /api/fund-industry/primary/:code
    
    Returns:
        {
            "code": 0,
            "data": {
                "industry": "新能源",
                "confidence": 95.0,
                "source": "llm",
                "updated_at": "2025-01-01 10:00:00"
            }
        }
    """
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return jsonify({"code": 400, "message": "基金代码不能为空"})

    repo = FundIndustryRepo()
    
    try:
        industry = repo.get_industry_by_fund(fund_code)
        
        if not industry:
            return jsonify({
                "code": 404, 
                "message": "未找到该基金的行业信息，请先调用分析接口"
            })
        
        return jsonify({"code": 0, "data": industry})
    except Exception as e:
        return jsonify({"code": 500, "message": f"获取失败: {str(e)}"})
