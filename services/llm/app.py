#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 微服务 - Flask 应用入口
"""

from flask import Flask, jsonify, request
from functools import wraps
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

rate_limit_store = {}


def rate_limit(max_calls=100, per_minute=1):
    """速率限制装饰器"""

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            now = int(time.time())
            key = f"{client_ip}:{now // per_minute}"

            if rate_limit_store.get(key, 0) >= max_calls:
                return jsonify(
                    {"success": False, "message": "Rate limit exceeded"}
                ), 429

            rate_limit_store[key] = rate_limit_store.get(key, 0) + 1

            for k in list(rate_limit_store.keys()):
                if int(k.split(":")[1]) < now // per_minute:
                    del rate_limit_store[k]

            return f(*args, **kwargs)

        return wrapped

    return decorator


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "llm-service"})


@app.route("/metrics")
def metrics():
    """服务指标"""
    return jsonify(
        {"service": "llm-service", "rate_limit_store_size": len(rate_limit_store)}
    )


from services.llm.routes import llm_bp

app.register_blueprint(llm_bp, url_prefix="/api/llm")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8006))
    app.run(host="0.0.0.0", port=port)
