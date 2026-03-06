#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻服务 - News Service
负责新闻爬取、存储和 API 提供的微服务
"""

import os
import sys
from flask import Flask, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "news-service"})


@app.route("/metrics")
def metrics():
    """指标"""
    return jsonify({"service": "news-service", "uptime": "0"})


from services.news.routes import news_bp

app.register_blueprint(news_bp, url_prefix="/api/news")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    app.run(host="0.0.0.0", port=port)
