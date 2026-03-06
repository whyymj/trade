#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM API 路由
提供 LSTM 训练和预测接口
"""

import os
import redis
from flask import Blueprint, request, jsonify

from services.stock.analysis import train_model
from shared.cache import get_cache
from shared.messaging import get_mq, LSTM_TRAIN

lstm_bp = Blueprint("lstm", __name__)
cache = get_cache()
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")


def _get_redis():
    """获取 Redis 连接"""
    return redis.from_url(redis_url, decode_responses=True)


def _acquire_lock(symbol: str, timeout: int = 7200) -> bool:
    """获取分布式锁"""
    r = _get_redis()
    lock_key = f"lstm:train_lock:{symbol}"
    return r.set(lock_key, "1", nx=True, ex=timeout)


def _release_lock(symbol: str):
    """释放锁"""
    r = _get_redis()
    lock_key = f"lstm:train_lock:{symbol}"
    r.delete(lock_key)


@lstm_bp.route("/train", methods=["POST"])
def train():
    """LSTM 训练 API"""
    data = request.get_json()
    symbol = data.get("symbol")

    if not symbol:
        return jsonify({"success": False, "message": "Symbol required"}), 400

    if not _acquire_lock(symbol):
        return (
            jsonify(
                {"success": False, "message": f"Training in progress for {symbol}"}
            ),
            409,
        )

    try:
        result = train_model(symbol)

        return jsonify(
            {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "status": "completed",
                    "metrics": result,
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        _release_lock(symbol)


@lstm_bp.route("/predict", methods=["GET"])
def predict():
    """LSTM 预测 API"""
    symbol = request.args.get("symbol")

    if not symbol:
        return jsonify({"success": False, "message": "Symbol required"}), 400

    cache_key = f"lstm:predict:{symbol}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "data": cached, "cached": True})

    try:
        from services.stock.analysis import predict

        result = predict(symbol)

        cache.set(cache_key, result, ttl=300)

        return jsonify({"success": True, "data": result, "cached": False})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@lstm_bp.route("/health", methods=["GET"])
def health():
    """LSTM 服务健康检查"""
    return jsonify({"service": "lstm", "status": "healthy"})
