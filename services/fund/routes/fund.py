# -*- coding: utf-8 -*-
"""
基金路由 - 基金列表和详情
"""

from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.fund.data import FundRepo

fund_bp = Blueprint("fund", __name__)
cache = get_cache()
repo = FundRepo()


@fund_bp.route("/list", methods=["GET"])
def get_fund_list():
    """获取基金列表"""
    page = int(request.args.get("page", 1))
    size = int(request.args.get("size", 20))
    fund_type = request.args.get("fund_type")
    watchlist_only = request.args.get("watchlist_only", "false").lower() == "true"
    industry_tag = request.args.get("industry_tag")

    cache_key = f"fund_list_{page}_{size}_{fund_type}_{watchlist_only}_{industry_tag}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result})

    result = repo.get_fund_list(
        page=page,
        size=size,
        fund_type=fund_type,
        watchlist_only=watchlist_only,
        industry_tag=industry_tag,
    )

    cache.set(cache_key, result, ttl=1800)

    return jsonify({"success": True, "data": result})


@fund_bp.route("/<fund_code>", methods=["GET"])
def get_fund_detail(fund_code):
    """获取基金详情"""
    cache_key = f"fund_detail_{fund_code}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result})

    result = repo.get_fund_info(fund_code)

    if not result:
        return jsonify({"success": False, "message": "基金不存在"}), 404

    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result})


@fund_bp.route("/<fund_code>/nav", methods=["GET"])
def get_fund_nav(fund_code):
    """获取基金净值"""
    days = int(request.args.get("days", 30))
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    cache_key = f"fund_nav_{fund_code}_{days}_{start_date}_{end_date}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result})

    result_df = repo.get_fund_nav(
        fund_code=fund_code,
        start_date=start_date,
        end_date=end_date,
        days=days if not start_date and not end_date else None,
    )

    if result_df is None:
        return jsonify({"success": False, "message": "净值数据不存在"}), 404

    result = result_df.to_dict(orient="records")

    cache.set(cache_key, result, ttl=300)

    return jsonify({"success": True, "data": result})
