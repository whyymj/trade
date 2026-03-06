# -*- coding: utf-8 -*-
"""
预测路由 - LSTM 预测
"""

from flask import Blueprint, jsonify
from shared.cache import get_cache
from datetime import datetime

predict_bp = Blueprint("predict", __name__)
cache = get_cache()


@predict_bp.route("/<fund_code>/predict", methods=["GET"])
def predict_fund(fund_code):
    """LSTM 预测基金走势"""
    cache_key = f"fund_predict_{fund_code}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result})

    try:
        import sys
        import os

        sys.path.insert(
            0,
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
        )
        from analysis.fund_lstm import predict as lstm_predict

        result = lstm_predict(fund_code)
    except Exception as e:
        result = {
            "fund_code": fund_code,
            "direction": 1,
            "direction_label": "涨",
            "magnitude": 0.0,
            "prob_up": 0.5,
            "magnitude_5": [0.0] * 5,
            "predict_date": datetime.now().strftime("%Y-%m-%d"),
            "error": str(e),
        }

    cache.set(cache_key, result, ttl=300)

    return jsonify({"success": True, "data": result})
