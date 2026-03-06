#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Service - 股票服务微服务
提供股票数据、LSTM 训练/预测、技术指标 API
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/health")
def health():
    """健康检查"""
    return jsonify({"status": "healthy", "service": "stock-service"})


@app.route("/metrics")
def metrics():
    """指标"""
    return jsonify({"service": "stock-service", "uptime": "0"})


# 导入路由
from services.stock.routes import stock_bp, lstm_bp

app.register_blueprint(stock_bp, url_prefix="/api/stock")
app.register_blueprint(lstm_bp, url_prefix="/api/lstm")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    app.run(host="0.0.0.0", port=port)
