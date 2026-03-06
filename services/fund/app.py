# -*- coding: utf-8 -*-
"""
Fund Service - 基金服务微服务
"""

from flask import Flask, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "fund-service"})


@app.route("/metrics")
def metrics():
    return jsonify({"service": "fund-service", "uptime": "0"})


from services.fund.routes import fund_bp, predict_bp

app.register_blueprint(fund_bp, url_prefix="/api/fund")
app.register_blueprint(predict_bp, url_prefix="/api/fund")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    app.run(host="0.0.0.0", port=port)
