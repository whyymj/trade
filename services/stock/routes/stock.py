#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock API 路由
提供股票列表、技术指标接口
"""

from flask import Blueprint, request, jsonify

from services.stock.analysis.indicators import (
    calculate_aroon_values,
    calculate_bollinger_bands_values,
    calculate_macd_values,
    calculate_ma,
    calculate_mfi_values,
    calculate_obv_values,
    calculate_rsi_values,
    calculate_volatility,
)
from services.stock.data import get_stock_data, get_stock_list
from shared.cache import get_cache

stock_bp = Blueprint("stock", __name__)
cache = get_cache()


@stock_bp.route("/indicators", methods=["GET"])
def get_indicators():
    """获取技术指标"""
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"success": False, "message": "Symbol required"}), 400

    cache_key = f"stock:indicators:{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "data": cached, "cached": True})

    try:
        data = get_stock_data(symbol, days=60)
        if data is None or len(data) < 20:
            return (
                jsonify({"success": False, "message": "Insufficient data"}),
                400,
            )

        close = data["收盘"]
        volume = data["成交量"]
        high = data["最高"]
        low = data["最低"]

        result = {
            "symbol": symbol,
            "ma5": calculate_ma(close, 5),
            "ma10": calculate_ma(close, 10),
            "ma20": calculate_ma(close, 20),
            "ma60": calculate_ma(close, 60),
            "macd": calculate_macd_values(close),
            "rsi": calculate_rsi_values(close, 14),
            "bollinger": calculate_bollinger_bands_values(close),
            "volatility": calculate_volatility(close, 20),
            "obv": calculate_obv_values(close, volume),
            "mfi": calculate_mfi_values(high, low, close, volume, 14),
            "aroon": calculate_aroon_values(high, low, 20),
        }

        cache.set(cache_key, result, ttl=3600)

        return jsonify({"success": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@stock_bp.route("/list", methods=["GET"])
def get_stock_list_route():
    """获取股票列表"""
    page = int(request.args.get("page", 1))
    size = int(request.args.get("size", 20))

    cache_key = f"stock_list_{page}_{size}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result})

    result = get_stock_list(page=page, size=size)
    cache.set(cache_key, result, ttl=1800)

    return jsonify({"success": True, "data": result})


@stock_bp.route("/health", methods=["GET"])
def health():
    """Stock 服务健康检查"""
    return jsonify({"service": "stock", "status": "healthy"})
