#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地数据服务器：将 data 目录下的 CSV 以接口形式返回，供前端 ECharts 展示。
工具函数见 server 包。
"""

from server.utils import (
    add_stock_and_fetch,
    df_to_chart_result,
    fetch_hist,
    get_data_dir,
    get_date_range_from_config,
    update_all_stocks,
)

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/")
def index():
    """前端页面。"""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/static/<path:path>")
def static_file(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/fetch_data/<stock_code>", methods=["GET"])
def api_fetch_data(stock_code: str):
    """
    按股票代码实时拉取数据（使用 akshare），返回与 /api/data 相同的 JSON 格式。
    日期范围与复权方式从 config.yaml 读取，未配置则默认近一年、前复权。
    """
    stock_code = (stock_code or "").strip()
    if not stock_code or not stock_code.isdigit() or len(stock_code) != 6:
        return jsonify({"error": "股票代码需为 6 位数字"}), 400

    start_date, end_date, adjust = get_date_range_from_config()
    df = fetch_hist(stock_code, start_date, end_date, adjust)
    if df is None or df.empty:
        return jsonify({"error": "拉取数据失败或暂无数据"}), 502

    return jsonify(df_to_chart_result(df))


@app.route("/api/update_all", methods=["POST"])
def api_update_all():
    """一键更新全部：按 config.stocks 从网络拉取并覆盖保存。"""
    result = update_all_stocks()
    return jsonify({"ok": True, "results": result})


@app.route("/api/add_stock", methods=["POST"])
def api_add_stock():
    """输入股票代码：抓取数据保存到 data，并将代码加入 config.stocks。"""
    body = request.get_json(silent=True) or {}
    code = (body.get("code") or request.form.get("code") or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "缺少 code 参数"}), 400
    out = add_stock_and_fetch(code)
    if not out.get("ok"):
        return jsonify(out), 502
    return jsonify(out)


@app.route("/api/list", methods=["GET"])
def api_list():
    """
    返回可用的股票数据文件列表。
    每项包含 filename（用于请求详情）、displayName（用于展示）。
    """
    data_dir = get_data_dir()
    if not data_dir.exists():
        return jsonify({"files": []})

    files = []
    for f in sorted(data_dir.glob("*.csv")):
        name = f.stem
        display_name = name.split("（")[0].strip() if "（" in name else name
        files.append({"filename": f.name, "displayName": display_name})
    return jsonify({"files": files})


@app.route("/api/data", methods=["GET"])
def api_data():
    """
    按文件名返回单只股票的 CSV 数据，转为 JSON，便于 ECharts 使用。
    查询参数: file=文件名（如 贵州茅台（20250128-20260128）.csv）
    返回: { dates, 开盘, 收盘, 最高, 最低, 成交量, 成交额, ... }
    """
    filename = request.args.get("file", "").strip()
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "缺少或非法 file 参数"}), 400

    data_dir = get_data_dir()
    path = data_dir / filename
    if not path.exists() or not path.is_file():
        return jsonify({"error": "文件不存在"}), 404

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as e:
        return jsonify({"error": f"读取文件失败: {e}"}), 500

    return jsonify(df_to_chart_result(df))


if __name__ == "__main__":
    print(f"数据目录: {get_data_dir().resolve()}")
    app.run(host="0.0.0.0", port=5050, debug=True)
