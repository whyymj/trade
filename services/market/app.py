from flask import Flask, jsonify
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "market-service"})


@app.route("/metrics")
def metrics():
    return jsonify({"service": "market-service", "uptime": "0"})


from services.market.routes import market_bp

app.register_blueprint(market_bp, url_prefix="/api/market")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8004))
    app.run(host="0.0.0.0", port=port)
