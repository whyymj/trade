# -*- coding: utf-8 -*-
"""
Fund-Intel Service - 基金智能服务微服务
"""

from flask import Flask, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "fund-intel-service"})


@app.route("/metrics")
def metrics():
    return jsonify({"service": "fund-intel-service", "uptime": "0"})


from services.fund_intel.routes import (
    fund_industry_bp,
    news_classification_bp,
    fund_news_bp,
    investment_advice_bp,
)

app.register_blueprint(fund_industry_bp, url_prefix="/api/fund-industry")
app.register_blueprint(news_classification_bp, url_prefix="/api/news-classification")
app.register_blueprint(fund_news_bp, url_prefix="/api/fund-news")
app.register_blueprint(investment_advice_bp, url_prefix="/api/investment-advice")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8005))
    app.run(host="0.0.0.0", port=port)
