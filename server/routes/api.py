# -*- coding: utf-8 -*-
"""
数据 API 路由。挂载于 /api。
接口说明见项目根目录 docs/API.md。
"""
from flask import Blueprint, Response, jsonify, request, send_from_directory
import pandas as pd

from server.utils import (
    add_stock_and_fetch,
    df_to_chart_result,
    fetch_hist,
    get_data_dir,
    get_date_range_from_config,
    is_valid_stock_code,
    load_config,
    remove_stock_from_config,
    save_config,
    sync_all_from_config,
    update_daily_stocks,
    update_daily_stocks_from_last,
)

try:
    from analysis.full_report import build_export_document, run_analysis_from_dataframe
except ImportError:
    run_analysis_from_dataframe = None
    build_export_document = None


def _list_stocks():
    """从数据库返回股票列表，每项含 filename（symbol）、displayName、remark、lastUpdateDate。"""
    from data import stock_repo
    items = stock_repo.list_stocks_from_db()
    return [
        {
            "filename": x["symbol"],
            "displayName": x["displayName"],
            "remark": x.get("remark") or "",
            "lastUpdateDate": x.get("lastUpdateDate"),
        }
        for x in items
    ]

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/fetch_data/<stock_code>", methods=["GET"])
def fetch_data(stock_code: str):
    """按股票代码获取日线数据。"""
    stock_code = (stock_code or "").strip()
    if not stock_code or not is_valid_stock_code(stock_code):
        return jsonify({"error": "股票代码需为 A股6位数字 或 港股5位数字/5位.HK"}), 400
    start_date, end_date, adjust = get_date_range_from_config()
    df = fetch_hist(stock_code, start_date, end_date, adjust)
    if df is None or df.empty:
        return jsonify({"error": "拉取数据失败或暂无数据"}), 502
    return jsonify(df_to_chart_result(df))


@api_bp.route("/update_all", methods=["POST"])
def update_all():
    """一键更新全部。Body: {"fromLastUpdate": true} 为最后更新日期至今；或 {"months": 1} / {"years": 3|5|10}。"""
    body = request.get_json(silent=True) or {}
    if body.get("fromLastUpdate"):
        result = update_daily_stocks_from_last()
        return jsonify({"ok": True, "results": result})
    months = body.get("months")
    years = body.get("years")
    if months is not None:
        try:
            months = int(months)
        except (TypeError, ValueError):
            months = 1
    elif years is not None:
        try:
            years = int(years)
        except (TypeError, ValueError):
            years = 5
        months = None
    else:
        months = 1
        years = None
    result = update_daily_stocks(months=months, years=years)
    return jsonify({"ok": True, "results": result})


@api_bp.route("/add_stock", methods=["POST"])
def add_stock():
    """抓取单只股票并加入配置。"""
    body = request.get_json(silent=True) or {}
    code = (body.get("code") or request.form.get("code") or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "缺少 code 参数"}), 400
    out = add_stock_and_fetch(code)
    if not out.get("ok"):
        return jsonify(out), 502
    return jsonify(out)


@api_bp.route("/list", methods=["GET"])
def list_files():
    """返回数据库中已有数据的股票列表（filename 为 symbol，供 /api/data?file= 使用）。"""
    try:
        files = _list_stocks()
    except Exception:
        files = []
    return jsonify({"files": files})


def _normalize_ymd(s: str) -> str:
    """将 YYYY-MM-DD 或 YYYYMMDD 转为 YYYYMMDD。"""
    if not s or not isinstance(s, str):
        return ""
    s = s.strip().replace("-", "")[:8]
    return s if len(s) == 8 else ""


@api_bp.route("/data", methods=["GET"])
def get_data():
    """按 file 参数（股票代码 symbol）从数据库读取日线并返回图表用 JSON。可选 start、end（YYYY-MM-DD）指定时间范围。"""
    file_or_code = request.args.get("file", "").strip()
    if not file_or_code or ".." in file_or_code or "/" in file_or_code or "\\" in file_or_code:
        return jsonify({"error": "缺少或非法 file 参数"}), 400
    symbol = file_or_code
    start_arg = _normalize_ymd(request.args.get("start", ""))
    end_arg = _normalize_ymd(request.args.get("end", ""))
    if start_arg and end_arg:
        start_date, end_date = start_arg, end_arg
    else:
        start_date, end_date, _ = get_date_range_from_config()
    df = fetch_hist(symbol, start_date, end_date)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    return jsonify(df_to_chart_result(df))


# ---------- 数据管理（原 fetch_stock_data 功能） ----------


@api_bp.route("/config", methods=["GET"])
def get_config():
    """返回当前 config.yaml 内容（供数据管理页展示与编辑）。"""
    cfg = load_config()
    return jsonify(cfg)


@api_bp.route("/config", methods=["PUT", "PATCH"])
def update_config():
    """更新 config：请求体可含 start_date, end_date, adjust, stocks, output_dir。"""
    body = request.get_json(silent=True) or {}
    out = save_config(body)
    if not out.get("ok"):
        return jsonify(out), 500
    return jsonify(out)


@api_bp.route("/sync_all", methods=["POST"])
def sync_all():
    """全量同步：清空数据目录后按 config.stocks 逐个拉取并保存（等同 fetch_stock_data 逻辑）。"""
    result = sync_all_from_config(clear_first=True)
    return jsonify({"ok": True, "results": result})


@api_bp.route("/data_range", methods=["GET"])
def get_data_range():
    """
    按日期范围查询多只股票日线，分页。
    Query: symbols=600519,000001（逗号分隔）, start=YYYY-MM-DD, end=YYYY-MM-DD, page=1, size=20
    """
    from data import stock_repo

    symbols_str = request.args.get("symbols", "").strip()
    if not symbols_str:
        return jsonify({"error": "缺少 symbols 参数（逗号分隔股票代码）"}), 400
    symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
    if not symbols:
        return jsonify({"error": "symbols 为空"}), 400
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
        size = max(1, min(500, int(request.args.get("size", 20))))
    except (TypeError, ValueError):
        page, size = 1, 20
    try:
        out = stock_repo.get_stock_daily_by_range(symbols, start or None, end or None, page=page, page_size=size)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(out)


@api_bp.route("/stock_remark", methods=["PUT", "PATCH"])
def update_stock_remark():
    """更新股票名称与说明。Body: { "symbol": "600519", "name": "股票名称（可选）", "remark": "说明文字（可选）" }。"""
    from data import stock_repo as repo

    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    if not symbol:
        return jsonify({"ok": False, "message": "缺少 symbol"}), 400
    name = body.get("name")
    if name is not None and not isinstance(name, str):
        name = str(name)
    remark = body.get("remark")
    if remark is not None and not isinstance(remark, str):
        remark = str(remark)
    repo.update_stock_meta_info(symbol, name=name, remark=remark)
    return jsonify({"ok": True, "message": "已保存"})


@api_bp.route("/remove_stock", methods=["POST"])
def remove_stock():
    """从 config 移除股票代码，并删除数据库中该股票数据。请求体 JSON: { "code": "600519" }。"""
    body = request.get_json(silent=True) or {}
    code = (body.get("code") or request.form.get("code") or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "缺少 code 参数"}), 400
    out = remove_stock_from_config(code, delete_data=True)
    return jsonify(out)


# ---------- 分析（analysis 模块） ----------


@api_bp.route("/analyze", methods=["GET"])
def analyze_stock():
    """
    对指定股票在时间范围内运行综合分析（时域、频域、ARIMA、复杂度）。
    Query: symbol=600519, start=YYYY-MM-DD, end=YYYY-MM-DD
    返回: { summary: {...}, report_md: "..." }
    """
    if run_analysis_from_dataframe is None:
        return jsonify({"error": "分析模块不可用"}), 503
    symbol = request.args.get("symbol", "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    start_arg = _normalize_ymd(request.args.get("start", ""))
    end_arg = _normalize_ymd(request.args.get("end", ""))
    if not start_arg or not end_arg:
        return jsonify({"error": "请提供 start 与 end 参数（YYYY-MM-DD）"}), 400
    df = fetch_hist(symbol, start_arg, end_arg)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    try:
        result = run_analysis_from_dataframe(
            df,
            stock_name=symbol,
            output_dir=None,
            forecast_days=5,
            show_plots=False,
        )
    except Exception as e:
        return jsonify({"error": f"分析执行失败: {e}"}), 500
    return jsonify(result)


@api_bp.route("/analyze/export", methods=["GET"])
def export_analysis():
    """
    对指定股票在时间范围内运行综合分析，并将结果整理为可下载的 Markdown 文档。
    Query: symbol=600519, start=YYYY-MM-DD, end=YYYY-MM-DD
    返回: Markdown 文件附件，含 YAML 元数据、结构化摘要(JSON)、完整报告正文。便于存档与 AI 解析。
    """
    if run_analysis_from_dataframe is None or build_export_document is None:
        return jsonify({"error": "分析/导出模块不可用"}), 503
    symbol = request.args.get("symbol", "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    start_arg = _normalize_ymd(request.args.get("start", ""))
    end_arg = _normalize_ymd(request.args.get("end", ""))
    if not start_arg or not end_arg:
        return jsonify({"error": "请提供 start 与 end 参数（YYYY-MM-DD）"}), 400
    df = fetch_hist(symbol, start_arg, end_arg)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    try:
        result = run_analysis_from_dataframe(
            df,
            stock_name=symbol,
            output_dir=None,
            forecast_days=5,
            show_plots=False,
        )
        doc = build_export_document(result)
    except Exception as e:
        return jsonify({"error": f"分析或导出失败: {e}"}), 500
    from datetime import datetime
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in symbol)
    filename = f"{safe_name}_分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    return Response(
        doc.encode("utf-8"),
        mimetype="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{__quote_filename(filename)}",
        },
    )


def __quote_filename(name: str) -> str:
    """对文件名做 RFC 5987 编码，用于 Content-Disposition。"""
    from urllib.parse import quote
    return quote(name, safe="")


