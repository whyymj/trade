# -*- coding: utf-8 -*-
"""
数据 API 路由。挂载于 /api。
接口说明见项目根目录 docs/API.md。
"""
import os
import threading
import time
from datetime import date, datetime, timedelta

# 训练进程：symbol -> 开始时间戳，避免同一股票并发训练；超时未结束的视为异常残留，自动解除
TRAINING_LOCK_MAX_SECONDS = 7200  # 2 小时，防止刷新/断线后一直报「正在训练中」
_training_symbols_in_progress = {}  # symbol -> start time
_training_stop_events = {}  # symbol -> threading.Event，用于停止正在进行的训练
_training_lock = threading.Lock()


def _training_acquire(symbol: str) -> bool:
    """尝试占用训练锁，成功返回 True，已被占用返回 False。会先清理超时占用。"""
    now = time.time()
    with _training_lock:
        expired = [s for s, t in _training_symbols_in_progress.items() if now - t > TRAINING_LOCK_MAX_SECONDS]
        for s in expired:
            _training_symbols_in_progress.pop(s, None)
        if symbol in _training_symbols_in_progress:
            return False
        _training_symbols_in_progress[symbol] = now
        return True


def _training_release(symbol: str) -> None:
    with _training_lock:
        _training_symbols_in_progress.pop(symbol, None)
        _training_stop_events.pop(symbol, None)


def _training_get_or_create_stop_event(symbol: str):
    """为某 symbol 创建或获取停止用 Event，调用方应在训练开始时 clear、训练结束后 release 时 pop。"""
    with _training_lock:
        if symbol not in _training_stop_events:
            _training_stop_events[symbol] = threading.Event()
        return _training_stop_events[symbol]


def _training_acquire_wait(symbol: str, timeout_seconds: float = None) -> bool:
    """排队等待直到该 symbol 可训练，然后占用锁。超时未轮到则返回 False。"""
    if timeout_seconds is None:
        timeout_seconds = TRAINING_LOCK_MAX_SECONDS
    deadline = time.time() + timeout_seconds
    while True:
        if _training_acquire(symbol):
            return True
        if time.time() >= deadline:
            return False
        time.sleep(5)

from flask import Blueprint, Response, jsonify, request, send_file, send_from_directory
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

try:
    from analysis.lstm_model import (
        DEFAULT_MODEL_DIR,
        build_features_from_df,
        incremental_train_and_save,
        load_model,
        run_lstm_pipeline,
    )
except ImportError:
    DEFAULT_MODEL_DIR = None
    run_lstm_pipeline = None
    load_model = None
    build_features_from_df = None
    incremental_train_and_save = None

try:
    from analysis.lstm_versioning import (
        get_current_version_id,
        list_versions,
        record_prediction,
        set_current_version,
        update_accuracy_for_symbol,
    )
except ImportError:
    get_current_version_id = None
    list_versions = None
    record_prediction = None
    set_current_version = None
    update_accuracy_for_symbol = None

try:
    from analysis.lstm_triggers import check_triggers, run_triggered_training
except ImportError:
    check_triggers = None
    run_triggered_training = None

try:
    from analysis.lstm_monitoring import (
        check_alerts,
        fire_alerts,
        get_alert_config_from_env_or_config,
        get_monitoring_status,
        record_training_failure,
        run_performance_decay_detection,
    )
except ImportError:
    check_alerts = None
    fire_alerts = None
    get_alert_config_from_env_or_config = None
    get_monitoring_status = None
    record_training_failure = None
    run_performance_decay_detection = None

try:
    from analysis.lstm_predict_flow import model_health_check, run_training_trigger_async
except ImportError:
    model_health_check = None
    run_training_trigger_async = None

try:
    from analysis.lstm_fallback import predict_with_fallback
except ImportError:
    predict_with_fallback = None

try:
    from analysis.lstm_spec import get_training_spec
except ImportError:
    get_training_spec = None

try:
    from data.lstm_repo import (
        dedupe_training_runs_keep_latest,
        delete_current_version_for_symbols,
        delete_model_version,
        delete_training_runs_by_symbols,
        get_all_last_predictions,
        get_current_version_from_db,
        get_last_prediction_for_symbol,
        get_last_prediction_for_symbol_years,
        get_model_version_symbol_years,
        get_trained_years_per_symbol,
        insert_training_run,
        list_model_versions_from_db,
        list_training_runs,
        set_current_version_db,
    )
except ImportError:
    dedupe_training_runs_keep_latest = None
    delete_current_version_for_symbols = None
    delete_model_version = None
    delete_training_runs_by_symbols = None
    insert_training_run = None
    list_model_versions_from_db = None
    list_training_runs = None
    set_current_version_db = None
    get_current_version_from_db = None
    get_last_prediction_for_symbol = None
    get_last_prediction_for_symbol_years = None
    get_model_version_symbol_years = None
    get_trained_years_per_symbol = None
    get_all_last_predictions = None

try:
    from analysis.factor_library import build_factor_library, get_all_factor_names
    from analysis.ensemble_models import run_ensemble_pipeline, save_ensemble_artifacts
    from analysis.ensemble_report import generate_factor_performance_report, report_to_markdown
    _ENSEMBLE_AVAILABLE = True
except ImportError:
    _ENSEMBLE_AVAILABLE = False
    build_factor_library = None  # type: ignore[assignment]
    get_all_factor_names = None  # type: ignore[assignment]
    run_ensemble_pipeline = None  # type: ignore[assignment]
    save_ensemble_artifacts = None  # type: ignore[assignment]
    generate_factor_performance_report = None  # type: ignore[assignment]
    report_to_markdown = None  # type: ignore[assignment]


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
    # 附加 LSTM 预测曲线供图表页使用：有则优先在价格图上画 LSTM 预测
    if get_last_prediction_for_symbol is not None and result.get("summary"):
        try:
            pred = get_last_prediction_for_symbol(symbol)
            mag5 = pred.get("magnitude_5") if pred else None
            if isinstance(mag5, list) and len(mag5) >= 5:
                close_col = "收盘" if "收盘" in df.columns else "close"
                if close_col in df.columns and len(df) > 0:
                    last_close = float(df[close_col].iloc[-1])
                    d = date.today()
                    forecast_dates = []
                    for _ in range(10):
                        d += timedelta(days=1)
                        if d.weekday() < 5:
                            forecast_dates.append(d.strftime("%Y-%m-%d"))
                            if len(forecast_dates) >= 5:
                                break
                    p = last_close
                    predicted = []
                    for m in mag5[:5]:
                        p = p * (1 + float(m))
                        predicted.append(p)
                    result["summary"]["lstm_forecast"] = [
                        {"date": dt, "predicted": pr} for dt, pr in zip(forecast_dates, predicted)
                    ]
        except Exception:
            pass
    return jsonify(result)


def _build_lstm_section_for_export(symbol: str) -> str | None:
    """拉取该股票最近一次 LSTM 预测，格式化为报告中的「LSTM 预测摘要」Markdown 片段。无数据时返回 None。"""
    if get_last_prediction_for_symbol is None:
        return None
    try:
        pred = get_last_prediction_for_symbol(symbol)
        if not pred:
            return None
        lines = ["## LSTM 预测摘要", ""]
        lines.append(f"- **预测日期**: {pred.get('predict_date') or '—'}")
        dir_label = "涨" if pred.get("direction") == 1 else "跌"
        lines.append(f"- **方向**: {dir_label}")
        mag = pred.get("magnitude")
        if mag is not None:
            lines.append(f"- **5 日涨跌幅**: {float(mag) * 100:.2f}%")
        prob = pred.get("prob_up")
        if prob is not None:
            lines.append(f"- **看涨概率**: {float(prob) * 100:.1f}%")
        mag5 = pred.get("magnitude_5")
        if isinstance(mag5, list) and len(mag5) >= 5:
            lines.append("- **未来 5 日逐日涨跌幅**: " + ", ".join(f"{float(m) * 100:.2f}%" for m in mag5[:5]))
        lines.append("")
        return "\n".join(lines)
    except Exception:
        return None


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
        lstm_section = _build_lstm_section_for_export(symbol)
        doc = build_export_document(result, lstm_section=lstm_section)
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


# ---------- 集成学习多因子预测 ----------


@api_bp.route("/ensemble/factor-report", methods=["GET"])
def ensemble_factor_report():
    """
    多因子绩效分析报告。
    Query: symbol=600519, start=YYYY-MM-DD, end=YYYY-MM-DD, forward_days=5, rolling_window=20
    返回: summary, report_md, top_ic_factors, top_rank_ic_factors, n_factors 等。
    """
    if not _ENSEMBLE_AVAILABLE or generate_factor_performance_report is None or report_to_markdown is None:
        return jsonify({
            "error": "集成学习模块不可用。请安装依赖：pip install xgboost lightgbm（或 pip install -r requirements.txt）",
        }), 503
    symbol = request.args.get("symbol", "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    start_arg = _normalize_ymd(request.args.get("start", ""))
    end_arg = _normalize_ymd(request.args.get("end", ""))
    if not start_arg or not end_arg:
        return jsonify({"error": "请提供 start 与 end 参数（YYYY-MM-DD）"}), 400
    forward_days = request.args.get("forward_days", "5", type=str)
    try:
        forward_days = int(forward_days) if forward_days else 5
    except ValueError:
        forward_days = 5
    rolling_window = request.args.get("rolling_window", "20", type=str)
    try:
        rolling_window = int(rolling_window) if rolling_window else 20
    except ValueError:
        rolling_window = 20
    df = fetch_hist(symbol, start_arg, end_arg)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    close_col = "收盘" if "收盘" in df.columns else "close"
    if close_col not in df.columns:
        return jsonify({"error": "数据缺少收盘价列"}), 400
    try:
        factor_df = build_factor_library(df)
        forward_return = df[close_col].astype(float).pct_change(forward_days).shift(-forward_days)
        report = generate_factor_performance_report(
            factor_df, forward_return.dropna(),
            rolling_window=rolling_window, top_pct=0.2, n_groups=5,
        )
        report_md = report_to_markdown(report)
    except Exception as e:
        return jsonify({"error": f"因子报告生成失败: {e}"}), 500
    return jsonify({
        "symbol": symbol,
        "start": start_arg,
        "end": end_arg,
        "forward_days": forward_days,
        "n_factors": report["summary"]["n_factors"],
        "n_observations": report["summary"]["n_observations"],
        "report_md": report_md,
        "top_ic_factors": report.get("top_ic_factors", [])[:15],
        "top_rank_ic_factors": report.get("top_rank_ic_factors", [])[:15],
        "long_short_return": {k: v for k, v in list(report.get("long_short_return", {}).items())[:15]},
    })


@api_bp.route("/ensemble/train", methods=["POST"])
def ensemble_train():
    """
    运行集成学习多因子训练流水线（XGBoost + LightGBM + RF，RFE 特征选择，权重优化）。
    Body (JSON): symbol, start, end, forward_days=5, train_ratio=0.7, use_rfe=true, rfe_min_features=15,
                 rfe_cv=3, early_stopping_rounds=30, optimize_weights=true
    返回: val_auc, val_accuracy, ensemble_weights, feature_names, rf_feature_importance, n_samples 等。
    """
    if not _ENSEMBLE_AVAILABLE or run_ensemble_pipeline is None:
        return jsonify({
            "error": "集成学习模块不可用。请安装依赖：pip install xgboost lightgbm（或 pip install -r requirements.txt）",
        }), 503
    data = request.get_json(silent=True) or {}
    symbol = (data.get("symbol") or "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    start_arg = _normalize_ymd(str(data.get("start", "")))
    end_arg = _normalize_ymd(str(data.get("end", "")))
    if not start_arg or not end_arg:
        return jsonify({"error": "请提供 start 与 end（YYYY-MM-DD）"}), 400
    forward_days = int(data.get("forward_days", 5)) or 5
    train_ratio = float(data.get("train_ratio", 0.7)) or 0.7
    use_rfe = bool(data.get("use_rfe", True))
    rfe_min_features = int(data.get("rfe_min_features", 15)) or 15
    rfe_cv = int(data.get("rfe_cv", 3)) or 3
    early_stopping_rounds = int(data.get("early_stopping_rounds", 30)) or 30
    optimize_weights = bool(data.get("optimize_weights", True))
    df = fetch_hist(symbol, start_arg, end_arg)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    close_col = "收盘" if "收盘" in df.columns else "close"
    if close_col not in df.columns:
        return jsonify({"error": "数据缺少收盘价列"}), 400
    try:
        factor_df = build_factor_library(df)
        forward_return = df[close_col].astype(float).pct_change(forward_days).shift(-forward_days)
        result = run_ensemble_pipeline(
            factor_df, forward_return,
            task="classification",
            train_ratio=train_ratio,
            use_rfe=use_rfe,
            rfe_min_features=rfe_min_features,
            rfe_cv=rfe_cv,
            early_stopping_rounds=early_stopping_rounds,
            optimize_weights=optimize_weights,
        )
    except Exception as e:
        return jsonify({"error": f"集成训练失败: {e}"}), 500
    if result.get("error"):
        return jsonify(result), 400
    out_dir = None
    try:
        from pathlib import Path
        _ens_dir = Path(__file__).resolve().parent.parent.parent / "analysis_temp" / "ensemble"
        save_ensemble_artifacts(result, _ens_dir)
        out_dir = str(_ens_dir)
    except Exception:
        pass
    return jsonify({
        "symbol": symbol,
        "start": start_arg,
        "end": end_arg,
        "forward_days": forward_days,
        "n_samples": result.get("n_samples"),
        "n_features_used": result.get("n_features_used"),
        "val_auc": result.get("val_auc"),
        "val_accuracy": result.get("val_accuracy"),
        "val_rmse": result.get("val_rmse"),
        "ensemble_weights": result.get("ensemble_weights"),
        "optimized_metric": result.get("optimized_metric"),
        "feature_names": result.get("feature_names", [])[:30],
        "rf_feature_importance": result.get("rf_feature_importance", [])[:20],
        "artifacts_saved_to": out_dir,
    })


# ---------- LSTM 深度学习预测 ----------


@api_bp.route("/lstm/recommended-range", methods=["GET"])
def lstm_recommended_range():
    """
    返回 LSTM 训练推荐的日期范围，便于调试页自动加载合适周期。
    Query: years=1|2（默认 1，表示最近 N 年）; use_config=1 时优先使用 config 的 start_date/end_date。
    返回: { start, end, hint }，日期格式 YYYY-MM-DD。
    """
    use_config = request.args.get("use_config", "").strip().lower() in ("1", "true", "yes")
    if use_config:
        start_date, end_date, _ = get_date_range_from_config()
        start = (start_date or "")[:10] if start_date else ""
        end = (end_date or "")[:10] if end_date else ""
        if len(start_date or "") >= 8:
            start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        if len(end_date or "") >= 8:
            end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        return jsonify({
            "start": start or None,
            "end": end or None,
            "hint": "与配置一致",
        })
    try:
        years = int(request.args.get("years", "1").strip() or "1")
        years = max(1, min(5, years))
    except ValueError:
        years = 1
    today = datetime.now().date()
    end = today.isoformat()
    start_date = today - timedelta(days=years * 365)
    start = start_date.isoformat()
    return jsonify({
        "start": start,
        "end": end,
        "hint": f"最近 {years} 年",
    })


def _date_range_for_years(years: int):
    """返回 (start_ymd, end_ymd) 用于最近 years 年的训练，格式 YYYYMMDD。"""
    today = datetime.now().date()
    end_arg = today.strftime("%Y%m%d")
    start_date = today - timedelta(days=max(1, years) * 365)
    start_arg = start_date.strftime("%Y%m%d")
    return start_arg, end_arg


@api_bp.route("/lstm/train", methods=["POST"])
def lstm_train():
    """
    训练 LSTM 模型：使用指定股票日线数据，特征为过去60日收盘/成交量/技术指标，预测未来5日方向与涨跌幅。
    Body: { "symbol": "600519", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD",
            "years": 1|2|3（单年）, "all_years": true（一次请求训练 1/2/3 年，推荐）,
            "do_cv_tune": true, "do_shap": true, "do_plot": true, "fast_training": false }
    返回: metrics、metadata 等；all_years 时返回最后一年的结果（3 年），便于前端展示。
    """
    if run_lstm_pipeline is None:
        return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn"}), 503
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    if not _training_acquire_wait(symbol):
        return jsonify({"error": "等待训练超时，请稍后再试"}), 408
    do_cv_tune = body.get("do_cv_tune", True)
    do_shap = body.get("do_shap", True)
    do_plot = body.get("do_plot", True)
    fast_training = body.get("fast_training", False)
    all_years = body.get("all_years", False)
    use_improved_training = body.get("use_improved_training", True)
    reg_loss_type = body.get("reg_loss_type") or "full"

    stop_event = _training_get_or_create_stop_event(symbol)
    stop_event.clear()
    try:
        if all_years:
            # 一次请求内依次训练 1、2、3 年，减少前端 3 次请求
            result = None
            for years in (1, 2, 3):
                if stop_event.is_set():
                    break
                start_arg, end_arg = _date_range_for_years(years)
                df = fetch_hist(symbol, start_arg, end_arg)
                if df is None or df.empty:
                    _training_release(symbol)
                    return jsonify({"error": f"暂无数据或拉取失败（{years} 年）"}), 404
                result = run_lstm_pipeline(
                    df,
                    symbol=symbol,
                    save_dir=None,
                    do_cv_tune=do_cv_tune,
                    do_shap=do_shap,
                    do_plot=do_plot,
                    param_grid=body.get("param_grid"),
                    fast_training=fast_training,
                    years=years,
                    use_improved_training=use_improved_training,
                    reg_loss_type=reg_loss_type,
                    stop_event=stop_event,
                )
                if "error" in result:
                    if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
                        record_training_failure(symbol=symbol, error_message=result.get("error", "未知错误"), save_dir=DEFAULT_MODEL_DIR)
                    _training_release(symbol)
                    return jsonify(result), 400
                if insert_training_run is not None:
                    try:
                        validation = result.get("validation") or {}
                        meta = result.get("metadata") or {}
                        insert_training_run(
                            version_id=meta.get("version_id"),
                            symbol=result.get("symbol", symbol),
                            training_type="full",
                            trigger_type="manual",
                            data_start=result.get("data_start") or meta.get("data_start"),
                            data_end=result.get("data_end") or meta.get("data_end"),
                            params={
                                "lr": meta.get("lr"),
                                "hidden_size": meta.get("hidden_size"),
                                "epochs": meta.get("epochs"),
                                "reg_loss_type": meta.get("reg_loss_type"),
                                "use_improved_training": meta.get("use_improved_training"),
                            },
                            metrics=result.get("metrics"),
                            validation_deployed=validation.get("deployed", False),
                            validation_reason=validation.get("reason"),
                            holdout_metrics=validation.get("new_holdout_metrics") if validation else None,
                        )
                    except Exception:
                        pass
            _training_release(symbol)
            return jsonify(result)
        # 单年训练（原有逻辑）
        start_arg = _normalize_ymd(body.get("start", ""))
        end_arg = _normalize_ymd(body.get("end", ""))
        if not start_arg or not end_arg:
            start_date, end_date, _ = get_date_range_from_config()
            start_arg = start_date.replace("-", "")[:8] if start_date else ""
            end_arg = end_date.replace("-", "")[:8] if end_date else ""
        if not start_arg or not end_arg:
            _training_release(symbol)
            return jsonify({"error": "请提供 start 与 end（YYYY-MM-DD）或配置日期范围"}), 400
        df = fetch_hist(symbol, start_arg, end_arg)
        if df is None or df.empty:
            _training_release(symbol)
            return jsonify({"error": "暂无数据或拉取失败"}), 404
        years = max(1, min(3, int(body.get("years", 1) or 1)))
        result = run_lstm_pipeline(
            df,
            symbol=symbol,
            save_dir=None,
            do_cv_tune=do_cv_tune,
            do_shap=do_shap,
            do_plot=do_plot,
            param_grid=body.get("param_grid"),
            fast_training=fast_training,
            years=years,
            use_improved_training=use_improved_training,
            reg_loss_type=reg_loss_type,
            stop_event=stop_event,
        )
    except (ImportError, ModuleNotFoundError) as e:
        _training_release(symbol)
        return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn"}), 503
    except RuntimeError as e:
        _training_release(symbol)
        if "PyTorch" in str(e) or "torch" in str(e) or "scikit-learn" in str(e):
            return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn"}), 503
        if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
            record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)
        return jsonify({"error": f"LSTM 训练失败: {e}"}), 500
    except Exception as e:
        if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
            record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)
        _training_release(symbol)
        return jsonify({"error": f"LSTM 训练失败: {e}"}), 500
    _training_release(symbol)
    if "error" in result:
        if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
            record_training_failure(symbol=symbol, error_message=result.get("error", "未知错误"), save_dir=DEFAULT_MODEL_DIR)
        return jsonify(result), 400
    if insert_training_run is not None:
        try:
            validation = result.get("validation") or {}
            meta = result.get("metadata") or {}
            insert_training_run(
                version_id=meta.get("version_id"),
                symbol=result.get("symbol", symbol),
                training_type="full",
                trigger_type="manual",
                data_start=result.get("data_start") or meta.get("data_start"),
                data_end=result.get("data_end") or meta.get("data_end"),
                params={
                    "lr": meta.get("lr"),
                    "hidden_size": meta.get("hidden_size"),
                    "epochs": meta.get("epochs"),
                    "reg_loss_type": meta.get("reg_loss_type"),
                    "use_improved_training": meta.get("use_improved_training"),
                },
                metrics=result.get("metrics"),
                validation_deployed=validation.get("deployed", False),
                validation_reason=validation.get("reason"),
                holdout_metrics=validation.get("new_holdout_metrics") if validation else None,
            )
        except Exception:
            pass
    return jsonify(result)


@api_bp.route("/lstm/train/stop", methods=["POST"])
def lstm_train_stop():
    """
    请求停止指定股票正在进行的训练。
    Body: { "symbol": "600519" }
    返回: { "ok": true, "message": "已请求停止" } 或 { "ok": false, "message": "该股票当前无进行中的训练" }
    """
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"ok": False, "message": "缺少或非法 symbol 参数"}), 400
    with _training_lock:
        event = _training_stop_events.get(symbol)
    if event is None:
        return jsonify({"ok": False, "message": "该股票当前无进行中的训练"})
    event.set()
    return jsonify({"ok": True, "message": "已请求停止，训练将在当前 epoch 结束后退出"})


@api_bp.route("/lstm/train-all", methods=["POST"])
def lstm_train_all():
    """
    一键训练当前全部股票（以列表中已有数据的股票为准）。
    Body: start?, end?（YYYY-MM-DD）或 years?（默认 1）；do_cv_tune?, do_shap?, do_plot?, fast_training?（默认 true，批量建议开启快速训练）。
    返回: { results: [ { symbol, ok, version_id?, error? }, ... ], total, success_count, fail_count }。
    """
    if run_lstm_pipeline is None:
        return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn"}), 503
    body = request.get_json(silent=True) or {}
    try:
        files = _list_stocks()
    except Exception:
        files = []
    symbols = [x["filename"] for x in files if (x.get("filename") or "").strip()]
    if not symbols:
        return jsonify({"error": "当前无股票数据，请先在股票列表或数据管理中添加股票", "results": [], "total": 0, "success_count": 0, "fail_count": 0}), 400

    start_arg = _normalize_ymd(body.get("start", ""))
    end_arg = _normalize_ymd(body.get("end", ""))
    if not start_arg or not end_arg:
        try:
            years = int(body.get("years", "1") or "1")
            years = max(1, min(5, years))
        except (TypeError, ValueError):
            years = 1
        today = datetime.now().date()
        end_arg = today.strftime("%Y%m%d")
        start_arg = (today - timedelta(days=years * 365)).strftime("%Y%m%d")

    do_cv_tune = body.get("do_cv_tune", True)
    do_shap = body.get("do_shap", False)
    do_plot = body.get("do_plot", False)
    fast_training = body.get("fast_training", True)
    use_improved_training = body.get("use_improved_training", True)
    reg_loss_type = body.get("reg_loss_type") or "full"

    results = []
    for symbol in symbols:
        if ".." in symbol or "/" in symbol or "\\" in symbol:
            results.append({"symbol": symbol, "ok": False, "error": "非法代码"})
            continue
        if not _training_acquire_wait(symbol):
            results.append({"symbol": symbol, "ok": False, "error": "等待训练超时，已跳过"})
            continue
        stop_event = _training_get_or_create_stop_event(symbol)
        stop_event.clear()
        try:
            df = fetch_hist(symbol, start_arg, end_arg)
            if df is None or df.empty:
                results.append({"symbol": symbol, "ok": False, "error": "暂无数据或拉取失败"})
                continue
            years_param = max(1, min(3, years))
            result = run_lstm_pipeline(
                df,
                symbol=symbol,
                save_dir=None,
                do_cv_tune=do_cv_tune,
                do_shap=do_shap,
                do_plot=do_plot,
                param_grid=({"lr": [5e-4], "hidden_size": [32], "epochs": [25], "batch_size": 32} if fast_training else None),
                fast_training=fast_training,
                years=years_param,
                use_improved_training=use_improved_training,
                reg_loss_type=reg_loss_type,
                stop_event=stop_event,
            )
            if result.get("error"):
                results.append({"symbol": symbol, "ok": False, "error": result.get("error", "训练失败")})
                if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
                    record_training_failure(symbol=symbol, error_message=result.get("error", ""), save_dir=DEFAULT_MODEL_DIR)
            else:
                vid = (result.get("metadata") or {}).get("version_id")
                results.append({"symbol": symbol, "ok": True, "version_id": vid})
                if insert_training_run is not None:
                    try:
                        validation = result.get("validation") or {}
                        meta = result.get("metadata") or {}
                        insert_training_run(
                            version_id=meta.get("version_id"),
                            symbol=result.get("symbol", symbol),
                            training_type="full",
                            trigger_type="manual",
                            data_start=result.get("data_start") or meta.get("data_start"),
                            data_end=result.get("data_end") or meta.get("data_end"),
                            params={
                                "lr": meta.get("lr"),
                                "hidden_size": meta.get("hidden_size"),
                                "epochs": meta.get("epochs"),
                                "reg_loss_type": meta.get("reg_loss_type"),
                                "use_improved_training": meta.get("use_improved_training"),
                            },
                            metrics=result.get("metrics"),
                            validation_deployed=validation.get("deployed", False),
                            validation_reason=validation.get("reason"),
                            holdout_metrics=validation.get("new_holdout_metrics") if validation else None,
                        )
                    except Exception:
                        pass
        except (ImportError, ModuleNotFoundError) as e:
            _training_release(symbol)
            return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn", "results": results}), 503
        except RuntimeError as e:
            if "PyTorch" in str(e) or "torch" in str(e) or "scikit-learn" in str(e):
                _training_release(symbol)
                return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn", "results": results}), 503
            results.append({"symbol": symbol, "ok": False, "error": str(e)})
            if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
                record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)
        except Exception as e:
            results.append({"symbol": symbol, "ok": False, "error": str(e)})
            if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
                record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)
        finally:
            _training_release(symbol)

    success_count = sum(1 for r in results if r.get("ok"))
    return jsonify({
        "results": results,
        "total": len(results),
        "success_count": success_count,
        "fail_count": len(results) - success_count,
    })


def _lstm_predict_single(model, X, y_mag_mean=None, y_mag_std=None):
    """LSTM 单次预测：返回 (direction, magnitude, prob_up, magnitude_5)。
    magnitude_5 为未来 5 日逐日涨跌幅列表；若模型仅输出 1 个值则重复 5 次以兼容。
    若提供 y_mag_mean / y_mag_std（来自训练 metadata），则对模型输出做反标准化。
    """
    import torch
    import numpy as np
    x_last = torch.from_numpy(X[-1:]).float()
    model.eval()
    with torch.no_grad():
        logits, magnitude = model(x_last)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    direction = int(logits.argmax(dim=1).item())
    mag_np = magnitude.cpu().numpy().flatten()
    if y_mag_mean is not None and y_mag_std is not None:
        mag_np = (mag_np * y_mag_std + y_mag_mean).astype(np.float64)
    if mag_np.size >= 5:
        magnitude_5 = [float(x) for x in mag_np[:5]]
    else:
        single = float(mag_np[0]) if mag_np.size > 0 else 0.0
        magnitude_5 = [single] * 5
    cum = 1.0
    for r in magnitude_5:
        cum *= 1.0 + r
    magnitude_val = float(cum - 1.0)
    direction = 1 if magnitude_val > 0 else 0
    prob_up = float(prob[1])
    return direction, magnitude_val, prob_up, magnitude_5


@api_bp.route("/lstm/predict", methods=["GET"])
def lstm_predict():
    """
    每日预测流程：获取最新数据 -> 模型健康检查 -> 预测 -> 记录；可选回退与异步训练。
    Query: symbol=600519, use_fallback=0|1（1= LSTM 失败时回退 ARIMA/技术指标）, trigger_train_async=0|1（1= 预测后后台检查并执行训练）
    返回: direction, magnitude, prob_up, source(lstm|arima|technical), model_health(可选)
    """
    if load_model is None or build_features_from_df is None:
        return jsonify({"error": "LSTM 模块不可用"}), 503
    symbol = request.args.get("symbol", "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    try:
        years = max(1, min(3, int(request.args.get("years", "3") or "3")))
    except (TypeError, ValueError):
        years = 3
    use_fallback = request.args.get("use_fallback", "0").strip() in ("1", "true", "yes")
    trigger_train_async = request.args.get("trigger_train_async", "0").strip() in ("1", "true", "yes")
    start_date, end_date, _ = get_date_range_from_config()
    df = fetch_hist(symbol, start_date, end_date)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    # 模型健康度检查
    health = None
    if model_health_check is not None and DEFAULT_MODEL_DIR is not None:
        health = model_health_check(DEFAULT_MODEL_DIR)
    # 带回退的预测
    if use_fallback and predict_with_fallback is not None:
        try:
            _years = years
            _symbol = symbol
            def _predict_lstm_with_meta(model, X, metadata=None):
                m = metadata or {}
                return _lstm_predict_single(model, X, m.get("y_mag_mean"), m.get("y_mag_std"))
            out = predict_with_fallback(
                symbol=symbol,
                df=df,
                load_model_fn=lambda **kw: load_model(device=__import__("torch").device("cpu"), symbol=_symbol, years=_years, **kw),
                build_features_fn=build_features_from_df,
                predict_lstm_fn=_predict_lstm_with_meta,
                save_dir=DEFAULT_MODEL_DIR,
            )
            if health is not None:
                out["model_health"] = health
            if record_prediction is not None and DEFAULT_MODEL_DIR is not None:
                from datetime import date
                predict_date = date.today().strftime("%Y-%m-%d")
                version_id = get_current_version_id(DEFAULT_MODEL_DIR, symbol=symbol, years=years) if get_current_version_id and out.get("source") == "lstm" else None
                record_prediction(
                    symbol=symbol,
                    predict_date=predict_date,
                    direction=out["direction"],
                    magnitude=out["magnitude"],
                    prob_up=out["prob_up"],
                    model_version_id=version_id,
                    save_dir=DEFAULT_MODEL_DIR,
                    source=out.get("source", "lstm"),
                    years=years,
                    magnitude_5=out.get("magnitude_5"),
                )
            if trigger_train_async and run_training_trigger_async is not None and check_triggers is not None and run_triggered_training is not None:
                run_training_trigger_async(
                    symbol=symbol,
                    check_triggers_fn=check_triggers,
                    run_triggered_training_fn=run_triggered_training,
                    fetch_hist_fn=fetch_hist,
                    get_date_range_fn=get_date_range_from_config,
                    run_lstm_pipeline_fn=run_lstm_pipeline,
                    incremental_train_fn=incremental_train_and_save,
                    save_dir=DEFAULT_MODEL_DIR,
                )
            return jsonify(out)
        except Exception as e:
            return jsonify({"error": f"预测失败: {e}"}), 500
    # 仅 LSTM 预测（按当前股票 + 年份加载对应模型）
    try:
        import torch
        model, metadata = load_model(save_dir=DEFAULT_MODEL_DIR, device=torch.device("cpu"), symbol=symbol, years=years)
        X, feature_names, y_info, y_dir, y_mag = build_features_from_df(df)
        if len(X) == 0:
            return jsonify({"error": "数据不足 65 个交易日，无法构造输入"}), 400
        direction, magnitude_val, prob_up, magnitude_5 = _lstm_predict_single(
            model, X,
            y_mag_mean=metadata.get("y_mag_mean"),
            y_mag_std=metadata.get("y_mag_std"),
        )
        if record_prediction is not None and DEFAULT_MODEL_DIR is not None:
            from datetime import date
            predict_date = date.today().strftime("%Y-%m-%d")
            version_id = get_current_version_id(DEFAULT_MODEL_DIR, symbol=symbol, years=years) if get_current_version_id else None
            record_prediction(
                symbol=symbol,
                predict_date=predict_date,
                direction=direction,
                magnitude=magnitude_val,
                prob_up=prob_up,
                model_version_id=version_id,
                save_dir=DEFAULT_MODEL_DIR,
                years=years,
                magnitude_5=magnitude_5,
            )
        out = {
            "symbol": symbol,
            "direction": direction,
            "direction_label": "涨" if direction == 1 else "跌",
            "magnitude": round(magnitude_val, 6),
            "prob_up": round(prob_up, 4),
            "prob_down": round(1 - prob_up, 4),
            "source": "lstm",
            "magnitude_5": magnitude_5,
        }
        if health is not None:
            out["model_health"] = health
        if trigger_train_async and run_training_trigger_async is not None and check_triggers is not None and run_triggered_training is not None:
            run_training_trigger_async(
                symbol=symbol,
                check_triggers_fn=check_triggers,
                run_triggered_training_fn=run_triggered_training,
                fetch_hist_fn=fetch_hist,
                get_date_range_fn=get_date_range_from_config,
                run_lstm_pipeline_fn=run_lstm_pipeline,
                incremental_train_fn=incremental_train_and_save,
                save_dir=DEFAULT_MODEL_DIR,
            )
        return jsonify(out)
    except FileNotFoundError:
        return jsonify({"error": "未找到已保存的模型，请先调用 /api/lstm/train"}), 404
    except Exception as e:
        return jsonify({"error": f"预测失败: {e}"}), 500


@api_bp.route("/lstm/predict-all", methods=["POST"])
def lstm_predict_all():
    """
    对当前全部股票执行预测。
    Body: years?（1|2|3，默认 3）, all_years?（true 时依次预测 1/2/3 年并记录，推荐）,
          use_fallback?（默认 false）, trigger_train_async?（默认 false，批量建议关闭）
    返回: { results: [ { symbol, ok, direction?, ... }, ... ], success_count, fail_count }
    """
    if load_model is None or build_features_from_df is None:
        return jsonify({"error": "LSTM 模块不可用"}), 503
    try:
        files = _list_stocks()
    except Exception as e:
        return jsonify({"error": f"获取股票列表失败: {e}"}), 500
    symbols = [x["filename"] for x in files if (x.get("filename") or "").strip()]
    if not symbols:
        return jsonify({
            "error": "当前无股票数据，请先在股票列表或数据管理中添加股票",
            "results": [],
            "success_count": 0,
            "fail_count": 0,
        }), 400

    body = request.get_json(silent=True) or {}
    use_fallback = body.get("use_fallback", False)
    trigger_train_async = body.get("trigger_train_async", False)
    all_years = body.get("all_years", False)
    try:
        predict_all_years = max(1, min(3, int(body.get("years", 3) or 3)))
    except (TypeError, ValueError):
        predict_all_years = 3
    years_list = (1, 2, 3) if all_years else (predict_all_years,)
    try:
        start_date, end_date, _ = get_date_range_from_config()
    except Exception as e:
        return jsonify({"error": f"获取日期范围失败: {e}"}), 500
    health = None
    if model_health_check is not None and DEFAULT_MODEL_DIR is not None:
        try:
            health = model_health_check(DEFAULT_MODEL_DIR)
        except Exception:
            pass

    results = []
    for symbol in symbols:
        if ".." in symbol or "/" in symbol or "\\" in symbol:
            results.append({"symbol": symbol, "ok": False, "error": "非法代码"})
            continue
        df = fetch_hist(symbol, start_date, end_date)
        if df is None or df.empty:
            results.append({"symbol": symbol, "ok": False, "error": "暂无数据或拉取失败"})
            continue
        last_pred = None
        last_error = None
        for predict_all_years in years_list:
            try:
                if use_fallback and predict_with_fallback is not None:
                    _sym, _y = symbol, predict_all_years
                    out = predict_with_fallback(
                        symbol=symbol,
                        df=df,
                        load_model_fn=lambda **kw: load_model(device=__import__("torch").device("cpu"), symbol=_sym, years=_y, **kw),
                        build_features_fn=build_features_from_df,
                        predict_lstm_fn=lambda model, X: _lstm_predict_single(model, X),
                        save_dir=DEFAULT_MODEL_DIR,
                    )
                    pred = {
                        "symbol": symbol,
                        "ok": True,
                        "direction": out["direction"],
                        "direction_label": out.get("direction_label") or ("涨" if out["direction"] == 1 else "跌"),
                        "magnitude": round(out.get("magnitude", 0), 6),
                        "prob_up": round(out.get("prob_up", 0.5), 4),
                        "prob_down": round(out.get("prob_down", 0.5), 4),
                        "source": out.get("source", "lstm"),
                    }
                else:
                    import torch
                    try:
                        model, meta = load_model(save_dir=DEFAULT_MODEL_DIR, device=torch.device("cpu"), symbol=symbol, years=predict_all_years)
                    except FileNotFoundError:
                        last_error = "未找到该股票对应年份的模型，请先训练"
                        continue
                    X, _, _, _, _ = build_features_from_df(df)
                    if len(X) == 0:
                        last_error = "数据不足 65 个交易日"
                        continue
                    direction, magnitude_val, prob_up, magnitude_5 = _lstm_predict_single(
                        model, X,
                        y_mag_mean=meta.get("y_mag_mean"),
                        y_mag_std=meta.get("y_mag_std"),
                    )
                    pred = {
                        "symbol": symbol,
                        "ok": True,
                        "direction": direction,
                        "direction_label": "涨" if direction == 1 else "跌",
                        "magnitude": round(magnitude_val, 6),
                        "prob_up": round(prob_up, 4),
                        "prob_down": round(1 - prob_up, 4),
                        "source": "lstm",
                        "magnitude_5": magnitude_5,
                    }
                if health is not None:
                    pred["model_health"] = health
                if record_prediction is not None and DEFAULT_MODEL_DIR is not None:
                    predict_date = date.today().strftime("%Y-%m-%d")
                    version_id = get_current_version_id(DEFAULT_MODEL_DIR, symbol=symbol, years=predict_all_years) if get_current_version_id else None
                    record_prediction(
                        symbol=symbol,
                        predict_date=predict_date,
                        direction=pred["direction"],
                        magnitude=pred["magnitude"],
                        prob_up=pred["prob_up"],
                        model_version_id=version_id,
                        save_dir=DEFAULT_MODEL_DIR,
                        source=pred.get("source", "lstm"),
                        years=predict_all_years,
                        magnitude_5=pred.get("magnitude_5"),
                    )
                last_pred = pred
                last_error = None
            except Exception as e:
                last_error = str(e)
        if trigger_train_async and last_pred is not None and run_training_trigger_async is not None and check_triggers is not None and run_triggered_training is not None:
            try:
                run_training_trigger_async(
                    symbol=symbol,
                    check_triggers_fn=check_triggers,
                    run_triggered_training_fn=run_triggered_training,
                    fetch_hist_fn=fetch_hist,
                    get_date_range_fn=get_date_range_from_config,
                    run_lstm_pipeline_fn=run_lstm_pipeline,
                    incremental_train_fn=incremental_train_and_save,
                    save_dir=DEFAULT_MODEL_DIR,
                )
            except Exception:
                pass
        if last_pred is not None:
            results.append(last_pred)
        else:
            results.append({"symbol": symbol, "ok": False, "error": last_error or "预测失败"})

    success_count = sum(1 for r in results if r.get("ok"))
    return jsonify({
        "results": results,
        "success_count": success_count,
        "fail_count": len(results) - success_count,
    })


@api_bp.route("/lstm/last-prediction", methods=["GET"])
def lstm_last_prediction():
    """
    获取指定股票最近一次预测记录（从 lstm_prediction_log），用于刷新页面后恢复展示。
    Query: symbol=600519
    """
    if get_last_prediction_for_symbol is None:
        return jsonify({"error": "预测记录不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    symbol = request.args.get("symbol", "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    row = get_last_prediction_for_symbol(symbol)
    if not row:
        return jsonify({})
    return jsonify(row)


@api_bp.route("/lstm/last-predictions", methods=["GET"])
def lstm_last_predictions():
    """
    获取每只股票最近一次预测记录，用于按股票分别展示。
    返回: { "predictions": [ { symbol, predict_date, direction, magnitude, prob_up, prob_down, direction_label, source }, ... ] }
    """
    if get_all_last_predictions is None:
        return jsonify({"error": "预测记录不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    try:
        predictions = get_all_last_predictions()
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/plot-data", methods=["GET"])
def lstm_plot_data():
    """
    返回「预测 vs 实际」曲线图数据（JSON），供前端 ECharts 实时绘制。
    Query: symbol（必填）；years=1|2|3（必填）。
    返回: { dates: [], actual_dir: [], pred_dir: [], actual_mag: [], pred_mag: [] } 或 404。
    """
    if DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 模块不可用"}), 503
    symbol = request.args.get("symbol", "").strip()
    years_arg = request.args.get("years", "").strip()
    years = 1 if not years_arg or years_arg not in ("1", "2", "3") else int(years_arg)
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "请指定 symbol 与 years 参数"}), 400
    try:
        from analysis.lstm_model import get_fit_plot_data
    except ImportError:
        return jsonify({"error": "LSTM 模块不可用"}), 503
    data = get_fit_plot_data(
        symbol=symbol,
        save_dir=DEFAULT_MODEL_DIR,
        fetch_hist_fn=fetch_hist,
        years=years,
    )
    if not data:
        return jsonify({"error": f"暂无该股票（{symbol}）{years} 年模型的曲线数据；请先训练"}), 404
    # 附加「预测走势」：该 symbol+years 最近一次预测 + 未来 5 个交易日日期，供前端在曲线图上延伸绘制
    if get_last_prediction_for_symbol_years is not None:
        try:
            pred = get_last_prediction_for_symbol_years(symbol, years)
            if pred:
                today = date.today()
                forecast_dates = []
                d = today
                for _ in range(10):
                    d += timedelta(days=1)
                    if d.weekday() < 5:
                        forecast_dates.append(d.strftime("%Y-%m-%d"))
                        if len(forecast_dates) >= 5:
                            break
                mag5 = pred.get("magnitude_5")
                if isinstance(mag5, list) and len(mag5) >= 5:
                    magnitude_list = [float(m) for m in mag5[:5]]
                    direction_list = [1 if m > 0 else 0 for m in magnitude_list]
                else:
                    single = float(pred.get("magnitude", 0))
                    magnitude_list = [single] * 5
                    direction_list = [1 if single > 0 else 0] * 5
                data["forecast"] = {
                    "magnitude": magnitude_list,
                    "direction": direction_list,
                    "forecast_dates": forecast_dates,
                }
        except Exception:
            pass
    return jsonify(data)


def _training_span_days(data_start: str | None, data_end: str | None) -> int | None:
    """根据 data_start/data_end 计算跨度天数，用于区分 1 年/2 年训练。支持 YYYY-MM-DD 或 YYYYMMDD。"""
    if not data_start or not data_end:
        return None
    def to_ymd(x: str) -> str | None:
        x = (x or "").strip().replace("-", "")[:8]
        if len(x) == 8 and x.isdigit():
            return f"{x[:4]}-{x[4:6]}-{x[6:8]}"
        return None
    try:
        s = to_ymd(data_start) or (data_start or "")[:10].strip()
        e = to_ymd(data_end) or (data_end or "")[:10].strip()
        if len(s) < 10 or len(e) < 10:
            return None
        d1 = datetime.strptime(s[:10], "%Y-%m-%d")
        d2 = datetime.strptime(e[:10], "%Y-%m-%d")
        return abs((d2 - d1).days)
    except (ValueError, TypeError):
        return None


@api_bp.route("/lstm/stocks-training-status", methods=["GET"])
def lstm_stocks_training_status():
    """
    以表格数据返回全部股票、最后一次训练时间、以及已训练年份（1/2/3 年）。
    返回: { stocks: [ { symbol, displayName, last_train, trained_years: [1,2,3] }, ... ] }。
    前端用 trained_years 判断是否 1/2/3 年都训练完毕，避免部分年份失败仍显示“已训练”。
    """
    if list_training_runs is None:
        return jsonify({"error": "LSTM 训练流水不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    try:
        files = _list_stocks()
        runs = list_training_runs(symbol=None, limit=3000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 按 symbol 汇总：该 symbol 下最后一次训练时间（取 created_at 最大）
    by_symbol = {}  # symbol -> last_train: str
    for r in runs:
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        created = r.get("created_at")
        if created is None:
            continue
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created = str(created)[:19].replace("T", " ")
        if sym not in by_symbol or created > by_symbol[sym]:
            by_symbol[sym] = created

    symbols_from_files = [(f.get("filename") or f.get("symbol") or "").strip() for f in files if (f.get("filename") or f.get("symbol") or "").strip()]
    trained_years_map = get_trained_years_per_symbol(symbols_from_files) if get_trained_years_per_symbol else {}

    stocks = []
    for f in files:
        symbol = (f.get("filename") or f.get("symbol") or "").strip()
        if not symbol:
            continue
        display_name = (f.get("displayName") or symbol) or ""
        stocks.append({
            "symbol": symbol,
            "displayName": display_name,
            "last_train": by_symbol.get(symbol),
            "trained_years": trained_years_map.get(symbol, []),
        })
    return jsonify({"stocks": stocks})


@api_bp.route("/lstm/clear-training", methods=["POST"])
def lstm_clear_training():
    """
    清理指定股票的训练数据（训练流水、对应模型版本、拟合曲线图），使该股票可重新训练。
    Body: { symbols: string[] }，如 ["600519", "000001"]。
    返回: { cleared: int, symbols: list, message: str }。
    """
    if (
        list_training_runs is None
        or delete_training_runs_by_symbols is None
        or delete_model_version is None
        or delete_current_version_for_symbols is None
    ):
        return jsonify({"error": "LSTM 数据仓储不可用"}), 503
    body = request.get_json(silent=True) or {}
    raw = body.get("symbols")
    if isinstance(raw, list):
        symbols = [str(s).strip() for s in raw if (s or "").strip()]
    elif isinstance(raw, str):
        symbols = [raw.strip()] if raw.strip() else []
    else:
        symbols = []
    if not symbols:
        return jsonify({"error": "请指定要清理的股票代码", "cleared": 0, "symbols": []}), 400
    symbols = list(dict.fromkeys(symbols))
    version_ids_to_delete = set()
    for sym in symbols:
        if ".." in sym or "/" in sym or "\\" in sym:
            continue
        try:
            runs = list_training_runs(symbol=sym, limit=500)
            for r in runs:
                vid = (r.get("version_id") or "").strip()
                if vid:
                    version_ids_to_delete.add(vid)
        except Exception:
            pass
    try:
        deleted_runs = delete_training_runs_by_symbols(symbols)
        for vid in version_ids_to_delete:
            try:
                delete_model_version(vid)
            except Exception:
                pass
        try:
            delete_current_version_for_symbols(symbols)
        except Exception:
            pass
        return jsonify({
            "cleared": len(symbols),
            "symbols": symbols,
            "deleted_runs": deleted_runs,
            "message": f"已清理 {len(symbols)} 只股票的训练数据，可重新训练",
        })
    except Exception as e:
        return jsonify({"error": str(e), "cleared": 0, "symbols": symbols}), 500


@api_bp.route("/lstm/training-runs", methods=["GET"])
def lstm_training_runs():
    """
    从 MySQL 查询 LSTM 训练流水（参数、指标、验证结果等）。
    Query: symbol=600519（可选）, limit=50, dedupe=1（按股票去重，每只只返回最新一条）
    """
    if list_training_runs is None:
        return jsonify({"error": "LSTM 训练流水不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    symbol = request.args.get("symbol", "").strip() or None
    limit = min(int(request.args.get("limit", 50) or 50), 200)
    dedupe = request.args.get("dedupe", "").strip().lower() in ("1", "true", "yes")
    try:
        runs = list_training_runs(symbol=symbol, limit=limit, dedupe_by_symbol=dedupe)
        return jsonify({"runs": runs, "count": len(runs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/training-runs/dedupe", methods=["POST"])
def lstm_training_runs_dedupe():
    """
    数据库去重：每只股票只保留最新一条训练流水，删除同 symbol 的旧记录。
    返回删除的行数。用于清理历史重复数据。
    """
    if dedupe_training_runs_keep_latest is None:
        return jsonify({"error": "LSTM 训练流水不可用"}), 503
    try:
        deleted = dedupe_training_runs_keep_latest()
        return jsonify({"deleted": deleted, "message": f"已删除 {deleted} 条重复记录"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/versions", methods=["GET"])
def lstm_versions():
    """
    列出 LSTM 模型版本（最近 5 个），含训练时间、数据范围、验证分数。
    """
    if DEFAULT_MODEL_DIR is None or list_versions is None:
        return jsonify({"error": "LSTM 版本管理不可用"}), 503
    try:
        versions = list_versions(DEFAULT_MODEL_DIR)
        current = get_current_version_id(DEFAULT_MODEL_DIR) if get_current_version_id else None
        return jsonify({"current_version_id": current, "versions": versions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/training-spec", methods=["GET"])
def lstm_training_spec():
    """
    返回 LSTM 训练与架构规格（JSON），便于 AI 或脚本分析。
    包含：数据规格、特征列表、模型架构、训练参数、交叉验证、损失选项、验证与部署规则。
    """
    if get_training_spec is None:
        return jsonify({"error": "LSTM 规格模块不可用"}), 503
    try:
        spec = get_training_spec()
        return jsonify(spec)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/rollback", methods=["POST"])
def lstm_rollback():
    """
    回滚到指定版本。Body: { "version_id": "20250205_143022" }。按该版本所属的 (symbol, years) 更新当前版本。
    """
    if set_current_version is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 版本管理不可用"}), 503
    body = request.get_json(silent=True) or {}
    version_id = (body.get("version_id") or "").strip()
    if not version_id:
        return jsonify({"error": "缺少 version_id"}), 400
    try:
        sym_years = get_model_version_symbol_years(version_id) if get_model_version_symbol_years else None
        if sym_years:
            set_current_version(version_id, save_dir=DEFAULT_MODEL_DIR, symbol=sym_years[0], years=sym_years[1])
        else:
            set_current_version(version_id, save_dir=DEFAULT_MODEL_DIR)
        return jsonify({"ok": True, "current_version_id": version_id})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/check-triggers", methods=["POST"])
def lstm_check_triggers():
    """
    检查训练触发条件；若满足且传入 run=true，则执行对应训练。
    Body: { "symbol": "600519", "run": true }
    - weekly: 周五触发，周度增量训练
    - monthly: 当月最后交易日触发，完整重新训练
    - performance_decay: 最近20日平均预测误差 > 历史平均×1.5 时触发重新训练
    """
    if check_triggers is None or run_triggered_training is None or run_lstm_pipeline is None:
        return jsonify({"error": "LSTM 触发模块不可用"}), 503
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    do_run = body.get("run", False)
    if not symbol:
        start_date, end_date, _ = get_date_range_from_config()
        from data import stock_repo
        stocks = stock_repo.list_stocks_from_db()
        symbol = (stocks[0].get("symbol") or "").strip() if stocks else ""
    if not symbol or ".." in symbol or "/" in symbol:
        return jsonify({"error": "缺少或非法 symbol"}), 400
    try:
        triggers = check_triggers(save_dir=str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR else None)
        result = {"triggers": triggers}
        if not do_run:
            return jsonify(result)
        run_type = None
        if triggers.get("monthly"):
            run_type = "monthly"
        elif triggers.get("quarterly"):
            run_type = "quarterly"
        elif triggers.get("performance_decay"):
            run_type = "performance_decay"
        elif triggers.get("weekly"):
            run_type = "weekly"
        if run_type:
            training_result = run_triggered_training(
                symbol=symbol,
                trigger_type=run_type,
                fetch_hist_fn=fetch_hist,
                get_date_range_fn=get_date_range_from_config,
                run_lstm_pipeline_fn=run_lstm_pipeline,
                incremental_train_fn=incremental_train_and_save,
                save_dir=DEFAULT_MODEL_DIR,
            )
            result["training"] = training_result
            if "error" not in training_result and insert_training_run is not None:
                try:
                    insert_training_run(
                        version_id=training_result.get("version_id") or training_result.get("metadata", {}).get("version_id"),
                        symbol=symbol,
                        training_type="incremental" if run_type == "weekly" else "full",
                        trigger_type=run_type,
                        data_start=training_result.get("data_start"),
                        data_end=training_result.get("data_end"),
                        params=training_result.get("metadata", {}),
                        metrics=training_result.get("metrics"),
                        validation_deployed=training_result.get("validation", {}).get("deployed", True),
                        validation_reason=training_result.get("validation", {}).get("reason"),
                        holdout_metrics=training_result.get("validation", {}).get("new_holdout_metrics"),
                    )
                except Exception:
                    pass
            if "error" in training_result:
                return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/update-accuracy", methods=["POST"])
def lstm_update_accuracy():
    """
    根据实际行情回填预测准确性（预测日+5 交易日后可计算）。
    Body: { "symbol": "600519", "as_of_date": "2025-02-05" }，as_of_date 可选，默认今天。
    """
    if update_accuracy_for_symbol is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 准确性回填不可用"}), 503
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    if not symbol or ".." in symbol or "/" in symbol:
        return jsonify({"error": "缺少或非法 symbol"}), 400
    from datetime import date
    as_of = body.get("as_of_date") or date.today().strftime("%Y-%m-%d")
    try:
        n = update_accuracy_for_symbol(
            symbol=symbol,
            as_of_date=as_of,
            fetch_hist_fn=fetch_hist,
            save_dir=DEFAULT_MODEL_DIR,
        )
        return jsonify({"ok": True, "symbol": symbol, "updated_count": n})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/monitoring", methods=["GET"])
def lstm_monitoring():
    """
    返回 LSTM 监控状态：当前版本、最后训练时间、数据范围、验证分数、
    近期预测次数、准确性指标、最近一次性能衰减检测结果。
    """
    if get_monitoring_status is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 监控模块不可用"}), 503
    try:
        status = get_monitoring_status(save_dir=DEFAULT_MODEL_DIR)
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/performance-decay", methods=["GET"])
def lstm_performance_decay():
    """
    执行一次性能衰减检测并返回报告（可选写入检测历史）。
    Query: threshold=1.5, n_recent=20, log=1
    """
    if run_performance_decay_detection is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 监控模块不可用"}), 503
    threshold = float(request.args.get("threshold", 1.5))
    n_recent = int(request.args.get("n_recent", 20))
    log = request.args.get("log", "1").strip().lower() in ("1", "true", "yes")
    try:
        result = run_performance_decay_detection(
            save_dir=DEFAULT_MODEL_DIR,
            threshold_multiplier=threshold,
            n_recent_trading_days=n_recent,
            log_result=log,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/lstm/alerts", methods=["POST"])
def lstm_alerts():
    """
    检查告警条件并返回告警列表；若 Body 传 fire=true 且配置了 webhook，则发送通知。
    Body: { "fire": true }。告警配置来自 config.yaml 的 lstm.webhook_url、lstm.performance_decay_multiplier 等，
    或环境变量 LSTM_ALERT_WEBHOOK。
    """
    if check_alerts is None or fire_alerts is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 告警模块不可用"}), 503
    body = request.get_json(silent=True) or {}
    do_fire = body.get("fire", False)
    try:
        config = get_alert_config_from_env_or_config(load_config) if get_alert_config_from_env_or_config else {}
        alerts = check_alerts(
            save_dir=DEFAULT_MODEL_DIR,
            performance_decay_multiplier=config.get("performance_decay_multiplier", 1.5),
            max_days_without_training=config.get("max_days_without_training", 30),
            min_direction_accuracy=config.get("min_direction_accuracy"),
        )
        result = {"alerts": alerts, "count": len(alerts)}
        if do_fire and alerts:
            fired = fire_alerts(
                alerts,
                save_dir=DEFAULT_MODEL_DIR,
                webhook_url=config.get("webhook_url"),
                log_alerts=True,
            )
            result["fired"] = fired
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

