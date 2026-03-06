from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from services.market.data import MarketRepo, MarketCrawler

market_bp = Blueprint("market", __name__)
cache = get_cache()
repo = MarketRepo()


@market_bp.route("/macro", methods=["GET"])
def get_macro_data():
    """获取宏观经济数据（1天缓存）"""
    indicator = request.args.get("indicator")
    days = int(request.args.get("days", 30))

    cache_key = f"market_macro_{indicator}_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_macro(indicator=indicator, days=days)
    cache.set(cache_key, result, ttl=86400)

    return jsonify({"success": True, "data": result, "cached": False})


@market_bp.route("/money-flow", methods=["GET"])
def get_money_flow():
    """获取资金流向数据（1小时缓存）"""
    days = int(request.args.get("days", 30))

    cache_key = f"market_money_flow_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_money_flow(days=days)
    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result, "cached": False})


@market_bp.route("/sentiment", methods=["GET"])
def get_sentiment():
    """获取市场情绪数据（1小时缓存）"""
    days = int(request.args.get("days", 30))

    cache_key = f"market_sentiment_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_sentiment(days=days)
    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result, "cached": False})


@market_bp.route("/global", methods=["GET"])
def get_global_macro():
    """获取全球宏观数据（1小时缓存）"""
    symbol = request.args.get("symbol")
    days = int(request.args.get("days", 30))

    cache_key = f"market_global_{symbol}_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_global_macro(symbol=symbol, days=days)
    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result, "cached": False})


@market_bp.route("/features", methods=["GET"])
def get_market_features():
    """获取市场特征（合并所有数据，1小时缓存）"""
    days = int(request.args.get("days", 30))

    cache_key = f"market_features_{days}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_market_features(days=days)
    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result, "cached": False})


@market_bp.route("/sync", methods=["POST"])
def sync_market_data():
    """同步市场数据"""
    crawler = MarketCrawler()

    data = crawler.sync_all()

    saved = 0
    saved += repo.save_macro(data["macro"])
    saved += repo.save_money_flow(data["money_flow"])
    saved += repo.save_sentiment(data["sentiment"])
    saved += repo.save_global_macro(data["global"])

    return jsonify(
        {
            "success": True,
            "message": "Market data synced successfully",
            "data": {
                "saved": saved,
                "macro_records": len(data["macro"]) if data["macro"] is not None else 0,
                "money_flow_records": len(data["money_flow"])
                if data["money_flow"] is not None
                else 0,
                "sentiment_records": len(data["sentiment"])
                if data["sentiment"] is not None
                else 0,
                "global_records": len(data["global"])
                if data["global"] is not None
                else 0,
            },
        }
    )
