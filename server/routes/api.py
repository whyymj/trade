# -*- coding: utf-8 -*-
"""
数据 API 路由。挂载于 /api。
接口说明见项目根目录 docs/API.md。
"""
import os
from datetime import datetime, timedelta

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
    from data.lstm_repo import (
        get_current_version_from_db,
        get_last_prediction_for_symbol,
        insert_training_run,
        list_training_runs,
        set_current_version_db,
    )
except ImportError:
    insert_training_run = None
    list_training_runs = None
    set_current_version_db = None
    get_current_version_from_db = None
    get_last_prediction_for_symbol = None


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


@api_bp.route("/lstm/train", methods=["POST"])
def lstm_train():
    """
    训练 LSTM 模型：使用指定股票日线数据，特征为过去60日收盘/成交量/技术指标，预测未来5日方向与涨跌幅。
    Body: { "symbol": "600519", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD",
            "do_cv_tune": true, "do_shap": true }
    返回: metrics(accuracy/recall/f1/mse)、cross_validation、interpretability、plot_path 等。
    """
    if run_lstm_pipeline is None:
        return jsonify({"error": "LSTM 模块不可用，请安装 torch、scikit-learn"}), 503
    body = request.get_json(silent=True) or {}
    symbol = (body.get("symbol") or "").strip()
    if not symbol or ".." in symbol or "/" in symbol or "\\" in symbol:
        return jsonify({"error": "缺少或非法 symbol 参数"}), 400
    start_arg = _normalize_ymd(body.get("start", ""))
    end_arg = _normalize_ymd(body.get("end", ""))
    if not start_arg or not end_arg:
        start_date, end_date, _ = get_date_range_from_config()
        start_arg = start_date.replace("-", "")[:8] if start_date else ""
        end_arg = end_date.replace("-", "")[:8] if end_date else ""
    if not start_arg or not end_arg:
        return jsonify({"error": "请提供 start 与 end（YYYY-MM-DD）或配置日期范围"}), 400
    df = fetch_hist(symbol, start_arg, end_arg)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据或拉取失败"}), 404
    try:
        fast_training = body.get("fast_training", False)
        result = run_lstm_pipeline(
            df,
            symbol=symbol,
            save_dir=None,
            do_cv_tune=body.get("do_cv_tune", True),
            do_shap=body.get("do_shap", True),
            do_plot=body.get("do_plot", True),
            param_grid=body.get("param_grid"),
            fast_training=fast_training,
        )
    except Exception as e:
        if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
            record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)
        return jsonify({"error": f"LSTM 训练失败: {e}"}), 500
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
                data_start=meta.get("data_start"),
                data_end=meta.get("data_end"),
                params={"lr": meta.get("lr"), "hidden_size": meta.get("hidden_size"), "epochs": meta.get("epochs")},
                metrics=result.get("metrics"),
                validation_deployed=validation.get("deployed", False),
                validation_reason=validation.get("reason"),
                holdout_metrics=validation.get("new_holdout_metrics") if validation else None,
            )
        except Exception:
            pass
    return jsonify(result)


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

    results = []
    for symbol in symbols:
        if ".." in symbol or "/" in symbol or "\\" in symbol:
            results.append({"symbol": symbol, "ok": False, "error": "非法代码"})
            continue
        df = fetch_hist(symbol, start_arg, end_arg)
        if df is None or df.empty:
            results.append({"symbol": symbol, "ok": False, "error": "暂无数据或拉取失败"})
            continue
        try:
            result = run_lstm_pipeline(
                df,
                symbol=symbol,
                save_dir=None,
                do_cv_tune=do_cv_tune,
                do_shap=do_shap,
                do_plot=do_plot,
                param_grid=({"lr": [5e-4], "hidden_size": [32], "epochs": [25], "batch_size": 32} if fast_training else None),
                fast_training=fast_training,
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
                            data_start=meta.get("data_start"),
                            data_end=meta.get("data_end"),
                            params={"lr": meta.get("lr"), "hidden_size": meta.get("hidden_size"), "epochs": meta.get("epochs")},
                            metrics=result.get("metrics"),
                            validation_deployed=validation.get("deployed", False),
                            validation_reason=validation.get("reason"),
                            holdout_metrics=validation.get("new_holdout_metrics") if validation else None,
                        )
                    except Exception:
                        pass
        except Exception as e:
            results.append({"symbol": symbol, "ok": False, "error": str(e)})
            if record_training_failure is not None and DEFAULT_MODEL_DIR is not None:
                record_training_failure(symbol=symbol, error_message=str(e), save_dir=DEFAULT_MODEL_DIR)

    success_count = sum(1 for r in results if r.get("ok"))
    return jsonify({
        "results": results,
        "total": len(results),
        "success_count": success_count,
        "fail_count": len(results) - success_count,
    })


def _lstm_predict_single(model, X):
    """LSTM 单次预测：返回 (direction, magnitude, prob_up)。"""
    import torch
    x_last = torch.from_numpy(X[-1:]).float()
    model.eval()
    with torch.no_grad():
        logits, magnitude = model(x_last)
    prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    direction = int(logits.argmax(dim=1).item())
    magnitude_val = float(magnitude.item())
    prob_up = float(prob[1])
    return direction, magnitude_val, prob_up


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
            out = predict_with_fallback(
                symbol=symbol,
                df=df,
                load_model_fn=lambda **kw: load_model(device=__import__("torch").device("cpu"), **kw),
                build_features_fn=build_features_from_df,
                predict_lstm_fn=lambda model, X: _lstm_predict_single(model, X),
                save_dir=DEFAULT_MODEL_DIR,
            )
            if health is not None:
                out["model_health"] = health
            if record_prediction is not None and DEFAULT_MODEL_DIR is not None:
                from datetime import date
                predict_date = date.today().strftime("%Y-%m-%d")
                version_id = get_current_version_id(DEFAULT_MODEL_DIR) if get_current_version_id and out.get("source") == "lstm" else None
                record_prediction(
                    symbol=symbol,
                    predict_date=predict_date,
                    direction=out["direction"],
                    magnitude=out["magnitude"],
                    prob_up=out["prob_up"],
                    model_version_id=version_id,
                    save_dir=DEFAULT_MODEL_DIR,
                    source=out.get("source", "lstm"),
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
    # 仅 LSTM 预测
    try:
        import torch
        model, metadata = load_model(save_dir=DEFAULT_MODEL_DIR, device=torch.device("cpu"))
        X, feature_names, y_info, y_dir, y_mag = build_features_from_df(df)
        if len(X) == 0:
            return jsonify({"error": "数据不足 65 个交易日，无法构造输入"}), 400
        direction, magnitude_val, prob_up = _lstm_predict_single(model, X)
        if record_prediction is not None and DEFAULT_MODEL_DIR is not None:
            from datetime import date
            predict_date = date.today().strftime("%Y-%m-%d")
            version_id = get_current_version_id(DEFAULT_MODEL_DIR) if get_current_version_id else None
            record_prediction(
                symbol=symbol,
                predict_date=predict_date,
                direction=direction,
                magnitude=magnitude_val,
                prob_up=prob_up,
                model_version_id=version_id,
                save_dir=DEFAULT_MODEL_DIR,
            )
        out = {
            "symbol": symbol,
            "direction": direction,
            "direction_label": "涨" if direction == 1 else "跌",
            "magnitude": round(magnitude_val, 6),
            "prob_up": round(prob_up, 4),
            "prob_down": round(1 - prob_up, 4),
            "source": "lstm",
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


@api_bp.route("/lstm/plot", methods=["GET"])
def lstm_plot():
    """
    返回最近一次 LSTM 训练生成的「预测 vs 实际」曲线图（PNG）。
    优先从当前版本目录读取，否则从根目录读取。
    """
    if DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 模块不可用"}), 503
    plot_path = os.path.join(str(DEFAULT_MODEL_DIR), "lstm_pred_vs_actual.png")
    if get_current_version_id is not None:
        try:
            from analysis.lstm_versioning import get_current_version_path
            base = DEFAULT_MODEL_DIR
            version_dir = get_current_version_path(base)
            if version_dir is not None:
                v_plot = os.path.join(str(version_dir), "lstm_pred_vs_actual.png")
                if os.path.isfile(v_plot):
                    plot_path = v_plot
        except Exception:
            pass
    if not os.path.isfile(plot_path):
        return jsonify({"error": "暂无训练曲线图，请先完成一次 LSTM 训练"}), 404
    return send_file(plot_path, mimetype="image/png", download_name="lstm_pred_vs_actual.png")


def _training_span_days(data_start: str | None, data_end: str | None) -> int | None:
    """根据 data_start/data_end 计算跨度天数，用于区分 1 年/2 年训练。"""
    if not data_start or not data_end:
        return None
    try:
        s = (data_start or "")[:10].strip()
        e = (data_end or "")[:10].strip()
        if len(s) < 10 or len(e) < 10:
            return None
        d1 = datetime.strptime(s, "%Y-%m-%d")
        d2 = datetime.strptime(e, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except (ValueError, TypeError):
        return None


@api_bp.route("/lstm/stocks-training-status", methods=["GET"])
def lstm_stocks_training_status():
    """
    以表格数据返回全部股票及其一年/二年数据的最后训练时间。
    返回: { stocks: [ { symbol, displayName, last_train_1y, last_train_2y }, ... ] }。
    1 年：数据跨度约 200～450 天；2 年：约 450～850 天。
    """
    if list_training_runs is None:
        return jsonify({"error": "LSTM 训练流水不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    try:
        files = _list_stocks()
        runs = list_training_runs(symbol=None, limit=3000)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 按 symbol 汇总：该 symbol 下「1 年」与「2 年」各自最后一次训练时间
    span_1y = (200, 450)
    span_2y = (451, 850)
    by_symbol = {}  # symbol -> { last_train_1y: str, last_train_2y: str }
    for r in runs:
        sym = (r.get("symbol") or "").strip()
        if not sym:
            continue
        days = _training_span_days(r.get("data_start"), r.get("data_end"))
        created = r.get("created_at")
        if created is None:
            continue
        if isinstance(created, datetime):
            created = created.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created = str(created)[:19].replace("T", " ")
        if days is not None:
            if span_1y[0] <= days <= span_1y[1]:
                if sym not in by_symbol:
                    by_symbol[sym] = {}
                if "last_train_1y" not in by_symbol[sym] or created > by_symbol[sym]["last_train_1y"]:
                    by_symbol[sym]["last_train_1y"] = created
            elif span_2y[0] <= days <= span_2y[1]:
                if sym not in by_symbol:
                    by_symbol[sym] = {}
                if "last_train_2y" not in by_symbol[sym] or created > by_symbol[sym]["last_train_2y"]:
                    by_symbol[sym]["last_train_2y"] = created

    stocks = []
    for f in files:
        symbol = (f.get("filename") or f.get("symbol") or "").strip()
        if not symbol:
            continue
        display_name = (f.get("displayName") or symbol) or ""
        info = by_symbol.get(symbol) or {}
        stocks.append({
            "symbol": symbol,
            "displayName": display_name,
            "last_train_1y": info.get("last_train_1y"),
            "last_train_2y": info.get("last_train_2y"),
        })
    return jsonify({"stocks": stocks})


@api_bp.route("/lstm/training-runs", methods=["GET"])
def lstm_training_runs():
    """
    从 MySQL 查询 LSTM 训练流水（参数、指标、验证结果等）。
    Query: symbol=600519（可选）, limit=50
    """
    if list_training_runs is None:
        return jsonify({"error": "LSTM 训练流水不可用（请确保 MySQL 与 data.lstm_repo 可用）"}), 503
    symbol = request.args.get("symbol", "").strip() or None
    limit = min(int(request.args.get("limit", 50) or 50), 200)
    try:
        runs = list_training_runs(symbol=symbol, limit=limit)
        return jsonify({"runs": runs, "count": len(runs)})
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


@api_bp.route("/lstm/rollback", methods=["POST"])
def lstm_rollback():
    """
    回滚到指定版本。Body: { "version_id": "20250205_143022" }
    """
    if set_current_version is None or DEFAULT_MODEL_DIR is None:
        return jsonify({"error": "LSTM 版本管理不可用"}), 503
    body = request.get_json(silent=True) or {}
    version_id = (body.get("version_id") or "").strip()
    if not version_id:
        return jsonify({"error": "缺少 version_id"}), 400
    try:
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

