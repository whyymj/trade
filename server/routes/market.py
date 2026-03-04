# server/routes/market.py
"""
市场数据相关 API 路由
"""

from flask import Blueprint, jsonify, request

from data.market import MarketCrawler, MarketRepo

market_bp = Blueprint("market", __name__, url_prefix="/api/market")


# ---------- 宏观经济数据 ----------


@market_bp.route("/macro", methods=["GET"])
def get_macro_data():
    """获取宏观经济数据"""
    indicator = request.args.get("indicator")
    days = request.args.get("days", 30, type=int)

    repo = MarketRepo()
    df = repo.get_macro_data(indicator=indicator, days=days)

    if df is None or df.empty:
        return jsonify({"code": 0, "data": []})

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "id": int(row.get("id")) if row.get("id") else None,
                "indicator": row.get("indicator"),
                "period": row.get("period"),
                "value": float(row.get("value")) if row.get("value") else None,
                "unit": row.get("unit"),
                "source": row.get("source"),
                "publish_date": str(row.get("publish_date"))
                if row.get("publish_date")
                else None,
                "trade_date": str(row.get("trade_date"))[:10]
                if row.get("trade_date")
                else None,
            }
        )

    return jsonify({"code": 0, "data": data})


@market_bp.route("/macro/latest", methods=["GET"])
def get_latest_macro():
    """获取最新宏观经济数据"""
    indicator = request.args.get("indicator")

    repo = MarketRepo()
    result = repo.get_latest_macro(indicator=indicator)

    if not result:
        return jsonify({"code": 0, "data": None})

    return jsonify({"code": 0, "data": result})


# ---------- 资金流向 ----------


@market_bp.route("/money-flow", methods=["GET"])
def get_money_flow():
    """获取资金流向"""
    days = request.args.get("days", 7, type=int)

    repo = MarketRepo()
    df = repo.get_money_flow(days=days)

    if df is None or df.empty:
        return jsonify({"code": 0, "data": []})

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "trade_date": row["trade_date"].strftime("%Y-%m-%d")
                if hasattr(row["trade_date"], "strftime")
                else str(row["trade_date"])[:10],
                "north_money": float(row.get("north_money", 0))
                if row.get("north_money")
                else None,
                "north_buy": float(row.get("north_buy", 0))
                if row.get("north_buy")
                else None,
                "north_sell": float(row.get("north_sell", 0))
                if row.get("north_sell")
                else None,
                "main_money": float(row.get("main_money", 0))
                if row.get("main_money")
                else None,
                "margin_balance": float(row.get("margin_balance", 0))
                if row.get("margin_balance")
                else None,
            }
        )

    return jsonify({"code": 0, "data": data})


@market_bp.route("/money-flow/latest", methods=["GET"])
def get_latest_money_flow():
    """获取最新资金流向"""
    repo = MarketRepo()
    result = repo.get_latest_money_flow()

    if not result:
        return jsonify({"code": 0, "data": None})

    return jsonify({"code": 0, "data": result})


# ---------- 市场情绪 ----------


@market_bp.route("/sentiment", methods=["GET"])
def get_sentiment():
    """获取市场情绪"""
    days = request.args.get("days", 7, type=int)

    repo = MarketRepo()
    df = repo.get_sentiment(days=days)

    if df is None or df.empty:
        return jsonify({"code": 0, "data": []})

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "trade_date": row["trade_date"].strftime("%Y-%m-%d")
                if hasattr(row["trade_date"], "strftime")
                else str(row["trade_date"])[:10],
                "volume": float(row.get("volume", 0)) if row.get("volume") else None,
                "up_count": int(row.get("up_count", 0)) if row.get("up_count") else 0,
                "down_count": int(row.get("down_count", 0))
                if row.get("down_count")
                else 0,
                "turnover_rate": float(row.get("turnover_rate", 0))
                if row.get("turnover_rate")
                else None,
                "advance_count": int(row.get("advance_count", 0))
                if row.get("advance_count")
                else 0,
                "decline_count": int(row.get("decline_count", 0))
                if row.get("decline_count")
                else 0,
            }
        )

    return jsonify({"code": 0, "data": data})


@market_bp.route("/sentiment/latest", methods=["GET"])
def get_latest_sentiment():
    """获取最新市场情绪"""
    repo = MarketRepo()
    sentiment = repo.get_latest_sentiment()

    if not sentiment:
        return jsonify(
            {
                "code": 0,
                "data": {
                    "volume": None,
                    "up_count": 0,
                    "down_count": 0,
                    "turnover_rate": None,
                    "advance_count": 0,
                    "decline_count": 0,
                },
            }
        )

    return jsonify({"code": 0, "data": sentiment})


# ---------- 全球宏观数据 ----------


@market_bp.route("/global", methods=["GET"])
def get_global_macro():
    """获取全球宏观数据"""
    symbol = request.args.get("symbol")
    days = request.args.get("days", 7, type=int)

    repo = MarketRepo()
    df = repo.get_global_macro(symbol=symbol, days=days)

    if df is None or df.empty:
        return jsonify({"code": 0, "data": []})

    data = []
    for _, row in df.iterrows():
        data.append(
            {
                "trade_date": row["trade_date"].strftime("%Y-%m-%d")
                if hasattr(row["trade_date"], "strftime")
                else str(row["trade_date"])[:10],
                "symbol": row.get("symbol"),
                "close_price": float(row.get("close_price"))
                if row.get("close_price")
                else None,
                "change_pct": float(row.get("change_pct"))
                if row.get("change_pct")
                else None,
            }
        )

    return jsonify({"code": 0, "data": data})


@market_bp.route("/global/latest", methods=["GET"])
def get_latest_global():
    """获取最新全球宏观数据"""
    repo = MarketRepo()
    result = repo.get_latest_global()

    if not result:
        return jsonify({"code": 0, "data": []})

    return jsonify({"code": 0, "data": result})


# ---------- 市场特征 ----------


@market_bp.route("/features", methods=["GET"])
def get_market_features():
    """获取合并后的市场特征"""
    days = request.args.get("days", 30, type=int)

    repo = MarketRepo()
    df = repo.get_market_features(days=days)

    if df is None or df.empty:
        return jsonify({"code": 0, "data": []})

    from decimal import Decimal

    data = []
    for _, row in df.iterrows():
        record = {}
        for col, val in row.items():
            if col == "trade_date":
                if hasattr(val, "strftime"):
                    record[col] = val.strftime("%Y-%m-%d")
                else:
                    record[col] = str(val)[:10]
            elif pd.isna(val):
                record[col] = None
            elif isinstance(val, (int, float, Decimal)):
                record[col] = float(val)
            else:
                record[col] = val
        data.append(record)

    return jsonify({"code": 0, "data": data})


# ---------- 数据同步 ----------


@market_bp.route("/sync", methods=["POST"])
def sync_market():
    """同步市场数据"""
    crawler = MarketCrawler()
    repo = MarketRepo()

    results = crawler.sync_all()

    saved = {}

    if results.get("money_flow") is not None and not results["money_flow"].empty:
        saved["money_flow"] = repo.save_money_flow(results["money_flow"])

    if results.get("sentiment") is not None and not results["sentiment"].empty:
        saved["sentiment"] = repo.save_sentiment(results["sentiment"])

    # 同步宏观经济和全球宏观
    try:
        macro_df = crawler.fetch_macro()
        if macro_df is not None and not macro_df.empty:
            saved["macro_data"] = repo.save_macro_data(macro_df)
    except Exception as e:
        print(f"[Market API] sync macro error: {e}")
        saved["macro_data"] = 0

    try:
        global_df = crawler.fetch_global()
        if global_df is not None and not global_df.empty:
            saved["global_macro"] = repo.save_global_macro(global_df)
    except Exception as e:
        print(f"[Market API] sync global error: {e}")
        saved["global_macro"] = 0

    return jsonify(
        {
            "code": 0,
            "data": {
                "saved": saved,
                "message": "同步完成" if saved else "无新数据",
            },
        }
    )


# ---------- 市场摘要 ----------


@market_bp.route("/summary", methods=["GET"])
def get_market_summary():
    """获取市场摘要"""
    repo = MarketRepo()

    sentiment = repo.get_latest_sentiment()
    money_flow = repo.get_latest_money_flow()
    macro = repo.get_latest_macro()
    global_data = repo.get_latest_global()

    money_flow_data = None
    if money_flow is not None:
        money_flow_df = repo.get_money_flow(days=1)
        if money_flow_df is not None and not money_flow_df.empty:
            money_flow_data = money_flow_df.iloc[0].to_dict()

    return jsonify(
        {
            "code": 0,
            "data": {
                "sentiment": sentiment,
                "money_flow": money_flow_data,
                "macro": macro,
                "global": global_data[:5] if global_data else [],
            },
        }
    )


# 导入 pandas 用于类型检查
import pandas as pd
