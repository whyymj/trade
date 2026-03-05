# -*- coding: utf-8 -*-
"""
数据 API 路由。挂载于 /api。
接口说明见项目根目录 docs/API.md。
"""

import logging
import os
import threading
import time
from datetime import date, datetime, timedelta

logger = logging.getLogger(__name__)

# 训练进程：symbol -> 开始时间戳，避免同一股票并发训练；超时未结束的视为异常残留，自动解除
TRAINING_LOCK_MAX_SECONDS = 7200  # 2 小时，防止刷新/断线后一直报「正在训练中」
_training_symbols_in_progress = {}  # symbol -> start time
_training_stop_events = {}  # symbol -> threading.Event，用于停止正在进行的训练
_training_lock = threading.Lock()


def _training_acquire(symbol: str) -> bool:
    """尝试占用训练锁，成功返回 True，已被占用返回 False。会先清理超时占用。"""
    now = time.time()
    with _training_lock:
        expired = [
            s
            for s, t in _training_symbols_in_progress.items()
            if now - t > TRAINING_LOCK_MAX_SECONDS
        ]
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
    from analysis.lstm_predict_flow import (
        model_health_check,
        run_training_trigger_async,
    )
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
    from analysis.ensemble_report import (
        generate_factor_performance_report,
        report_to_markdown,
    )

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


@api_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({"code": 0, "data": {"status": "healthy"}, "message": "ok"})


# =====================================================
# 基金 API 路由（/api/fund/*, /api/index/*, /api/sync/*）
# =====================================================

from functools import wraps

from data.cache import get_cache

_FUND_CACHE_TTL = {
    "list": 30 * 60,
    "nav": 30 * 60,
    "indicators": 60 * 60,
    "prediction": 5 * 60,
    "llm": 24 * 60 * 60,
}


def _cache_key(prefix: str, *args, **kwargs) -> str:
    parts = [str(a) for a in args if a]
    parts.extend(str(v) for v in kwargs.values() if v)
    return f"{prefix}:{':'.join(parts)}"


def _cached(ttl_key: str):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            # 从 request 对象获取查询参数，确保不同参数生成不同缓存 key
            from flask import request

            query_params = {k: v for k, v in request.args.items() if v}
            key = _cache_key(f.__name__, **query_params)
            cached_val = cache.get(key)
            if cached_val is not None:
                return cached_val
            result = f(*args, **kwargs)
            ttl = _FUND_CACHE_TTL.get(ttl_key, 300)
            cache.set(key, result, ttl)
            return result

        return wrapper

    return decorator


def _clear_fund_cache(*keys):
    cache = get_cache()
    for key in keys:
        # 清除带和不带冒号的两种key
        cache.delete(key)
        cache.delete(key + ":")


# ---------- 基金列表 ----------


@api_bp.route("/fund/list", methods=["GET"])
@_cached("list")
def fund_list():
    """基金列表，支持分页和类型筛选。Query: page, size, fund_type"""
    from data import fund_repo

    try:
        page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        page = 1
    try:
        size = int(request.args.get("size", 20))
    except (TypeError, ValueError):
        size = 20
    fund_type = request.args.get("fund_type", "").strip() or None
    result = fund_repo.get_fund_list(page=page, size=size, fund_type=fund_type)
    return jsonify(result)


@api_bp.route("/fund/watchlist", methods=["GET"])
@_cached("list")
def fund_watchlist():
    """关注列表。"""
    from data import fund_repo

    try:
        page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        page = 1
    try:
        size = int(request.args.get("size", 20))
    except (TypeError, ValueError):
        size = 20
    result = fund_repo.get_fund_list(page=page, size=size, watchlist_only=True)
    return jsonify(result)


@api_bp.route("/fund/add", methods=["POST"])
def fund_add():
    """添加基金。Body: fund_code, fund_name?, fund_type?, manager?, establishment_date?, fund_scale?
    如果不提供 fund_name，会自动从网上抓取
    添加成功后会自动抓取行业标签和持仓信息"""
    from data import fund_repo, fund_fetcher
    from data import fund_holdings
    import pandas as pd
    import threading

    body = request.get_json(silent=True) or {}
    fund_code = (body.get("fund_code") or "").strip()
    fund_name = (body.get("fund_name") or "").strip()
    if not fund_code:
        return jsonify({"ok": False, "message": "缺少 fund_code"}), 400

    # 如果没有提供 fund_name，自动抓取
    if not fund_name:
        try:
            fund_info = fund_fetcher.fetch_fund_info(fund_code)
            if fund_info and fund_info.get("fund_name"):
                fund_name = fund_info.get("fund_name", "")
        except Exception:
            pass

        # 如果仍然没有名称，使用代码作为名称
        if not fund_name:
            fund_name = f"基金{fund_code}"

    fund_type = body.get("fund_type")
    manager = body.get("manager")
    establishment_date = body.get("establishment_date")
    fund_scale = body.get("fund_scale")

    try:
        if establishment_date:
            establishment_date = pd.to_datetime(establishment_date).strftime("%Y-%m-%d")
        if fund_scale is not None:
            fund_scale = float(fund_scale)
    except Exception:
        return jsonify({"ok": False, "message": "参数格式错误"}), 400

    ok = fund_repo.add_fund(
        fund_code=fund_code,
        fund_name=fund_name,
        fund_type=fund_type,
        manager=manager,
        establishment_date=establishment_date,
        fund_scale=fund_scale,
    )

    if ok:
        _clear_fund_cache("fund_list")

        # 异步抓取行业标签和持仓信息
        def async_fetch():
            try:
                # 抓取持仓信息
                holdings_data = fund_holdings.get_fund_detail_info(fund_code)
                if holdings_data.get("holdings"):
                    # 更新基金经理
                    if holdings_data.get("manager"):
                        manager_name = holdings_data["manager"].get("name")
                        if manager_name:
                            fund_repo.add_fund(
                                fund_code=fund_code, fund_name="", manager=manager_name
                            )
                    # 设置行业标签（根据持仓推断）
                    tags = _infer_industry_tags(holdings_data.get("holdings", []))
                    if tags:
                        fund_repo.update_fund_industry_tags(fund_code, tags)
            except Exception as e:
                logger.error("异步抓取失败: %s", e)

        # 启动异步任务
        threading.Thread(target=async_fetch, daemon=True).start()

        return jsonify({"ok": True, "message": "添加成功，正在后台分析"})
    return jsonify({"ok": False, "message": "添加失败"}), 500


def _infer_industry_tags(holdings: list) -> list:
    """根据持仓股票推断行业标签"""
    if not holdings:
        return []

    # 行业关键词映射
    industry_keywords = {
        "新能源": [
            "宁德时代",
            "比亚迪",
            "隆基绿能",
            "光伏",
            "储能",
            "锂电池",
            "新能源车",
        ],
        "半导体": ["中芯国际", "北方华创", "立讯精密", "芯片", "半导体", "集成电路"],
        "医药": ["恒瑞医药", "药明康德", "迈瑞医疗", "医药", "医疗器械", "创新药"],
        "消费": ["贵州茅台", "五粮液", "伊利股份", "美的", "消费", "白酒", "食品"],
        "金融": ["中国平安", "招商银行", "兴业银行", "银行", "保险", "证券"],
        "互联网": ["腾讯", "阿里", "美团", "字节", "互联网", "软件"],
        "港股": ["腾讯控股", "阿里巴巴", "美团", "快手", "港股"],
        "银行": ["工商银行", "建设银行", "农业银行", "招商银行", "银行"],
        "军工": ["中国航发", "中航沈飞", "军工", "航天", "航空"],
    }

    stock_names = [h.get("stock_name", "") for h in holdings]
    found_tags = set()

    for tag, keywords in industry_keywords.items():
        for stock in stock_names:
            for kw in keywords:
                if kw in stock:
                    found_tags.add(tag)
                    break

    return list(found_tags)[:3] if found_tags else ["混合配置"]


@api_bp.route("/fund/<code>", methods=["GET"])
def fund_detail(code: str):
    """获取基金详情。"""
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    result = fund_repo.get_fund_list(page=1, size=100)
    fund = None
    for f in result.get("data", []):
        if f.get("fund_code") == code:
            fund = f
            break

    if not fund:
        return jsonify({"error": "基金不存在"}), 404

    return jsonify(fund)


@api_bp.route("/fund/<code>", methods=["DELETE"])
def fund_delete(code: str):
    """删除基金。"""
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "缺少基金代码"}), 400
    n = fund_repo.delete_fund(code)
    _clear_fund_cache("fund_list", "fund_watchlist")
    return jsonify({"ok": True, "deleted": n})


@api_bp.route("/fund/<code>/watch", methods=["PUT"])
def fund_watch(code: str):
    """关注/取消关注基金。Body: watch (true/false)"""
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"ok": False, "message": "缺少基金代码"}), 400
    body = request.get_json(silent=True) or {}
    watch = body.get("watch", True)
    ok = fund_repo.update_fund_watchlist(code, bool(watch))
    _clear_fund_cache("fund_list", "fund_watchlist")
    if ok:
        return jsonify({"ok": True, "message": "已关注" if watch else "已取消关注"})
    return jsonify({"ok": False, "message": "操作失败"}), 500


# ---------- 基金净值 ----------


@api_bp.route("/fund/nav/latest/<code>", methods=["GET"])
@_cached("nav")
def fund_nav_latest(code: str):
    """最新净值。"""
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400
    result = fund_repo.get_latest_nav(code)
    if result is None:
        return jsonify({"error": "暂无数据"}), 404
    return jsonify(result)


@api_bp.route("/fund/nav/<code>", methods=["GET"])
def fund_nav(code: str):
    """基金净值历史。Query: start?, end?"""
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400
    start_date = request.args.get("start", "").strip() or None
    end_date = request.args.get("end", "").strip() or None
    df = fund_repo.get_fund_nav(code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据"}), 404
    return jsonify(
        {
            "fund_code": code,
            "data": df.to_dict(orient="records"),
        }
    )


@api_bp.route("/fund/holdings/<code>", methods=["GET"])
def fund_holdings(code: str):
    """基金持仓信息"""
    from data import fund_holdings as fh

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    try:
        holdings = fh.get_fund_holdings(code)
        manager_info = fh.get_fund_manager(code)

        return jsonify(
            {"code": 0, "data": {"holdings": holdings, "manager": manager_info}}
        )
    except Exception as e:
        return jsonify({"code": -1, "error": str(e)}), 500


# ---------- 基金分析 ----------


@api_bp.route("/fund/indicators/<code>", methods=["GET"])
@_cached("indicators")
def fund_indicators(code: str):
    """业绩指标。Query: days (默认365)"""
    from analysis.fund_metrics import get_fund_indicators

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400
    try:
        days = int(request.args.get("days", 365))
    except (TypeError, ValueError):
        days = 365
    result = get_fund_indicators(code, days=days)
    if result is None:
        return jsonify({"error": "暂无数据或计算失败"}), 404
    return jsonify({"fund_code": code, **result})


@api_bp.route("/fund/benchmark/<code>", methods=["GET"])
def fund_benchmark(code: str):
    """基准对比。Query: benchmark (默认000300)"""
    from analysis.fund_benchmark import compare_with_benchmark, get_benchmark_data
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400
    benchmark_code = request.args.get("benchmark", "000300").strip()

    fund_df = fund_repo.get_fund_nav(code)
    if fund_df is None or fund_df.empty:
        return jsonify({"error": "基金数据不存在"}), 404

    benchmark_df = get_benchmark_data(benchmark_code)
    if benchmark_df is None or benchmark_df.empty:
        return jsonify({"error": f"基准指数 {benchmark_code} 数据不存在"}), 404

    fund_df = fund_df.sort_values("nav_date")
    fund_nav = fund_df.set_index("nav_date")["unit_nav"]
    benchmark_nav = benchmark_df

    common_idx = fund_nav.index.intersection(benchmark_nav.index)
    if len(common_idx) < 10:
        return jsonify({"error": "基金与基准数据交集不足10天"}), 400

    fund_aligned = fund_nav.loc[common_idx].sort_index()
    bench_aligned = benchmark_nav.loc[common_idx].sort_index()

    fund_base = fund_aligned.iloc[0]
    bench_base = bench_aligned.iloc[0]

    fund_cum_return = ((fund_aligned / fund_base - 1) * 100).tolist()
    bench_cum_return = ((bench_aligned / bench_base - 1) * 100).tolist()
    dates = [str(d)[:10] for d in fund_aligned.index]

    result = compare_with_benchmark(fund_nav, benchmark_nav)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400

    return jsonify(
        {
            "fund_code": code,
            "benchmark": benchmark_code,
            "dates": dates,
            "fund_cum_return": fund_cum_return,
            "benchmark_cum_return": bench_cum_return,
            **result,
        }
    )


@api_bp.route("/fund/cycle/<code>", methods=["GET"])
def fund_cycle(code: str):
    """周期分析。Query: days (默认365)"""
    from analysis.frequency_domain import find_dominant_periods, calc_power_spectrum
    from data import fund_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400
    try:
        days = int(request.args.get("days", 365))
    except (TypeError, ValueError):
        days = 365

    fund_df = fund_repo.get_fund_nav(code)
    if fund_df is None or fund_df.empty:
        return jsonify({"error": "基金数据不存在"}), 404

    if len(fund_df) < 20:
        return jsonify({"error": "数据不足，无法进行周期分析"}), 400

    fund_df = fund_df.sort_values("nav_date").tail(days)
    returns = fund_df["unit_nav"].pct_change().dropna()

    if len(returns) < 20:
        return jsonify({"error": "有效数据不足20条，无法分析"}), 400

    try:
        frequencies, psd = calc_power_spectrum(returns, sampling_rate=1.0)
        dominant_periods = find_dominant_periods(
            frequencies, psd, top_n=5, min_period=5
        )

        def get_period_explanation(period_days):
            if period_days <= 7:
                return "周内短期波动，可能反映交易情绪或资金流动"
            elif period_days <= 14:
                return "双周周期，短期资金调仓效应"
            elif period_days <= 25:
                return "月度周期，与月末资金流动或期权效应相关"
            elif period_days <= 45:
                return "半月~季度周期，可能与财报发布或政策周期相关"
            elif period_days <= 70:
                return "季度周期，与行业事件或资金轮动相关"
            elif period_days <= 120:
                return "半年周期，宏观经济或政策周期影响"
            elif period_days <= 200:
                return "年度周期，季节性因素或市场周期"
            else:
                return "长周期趋势，可能反映经济长波或行业生命周期"

        results = []
        for i, p in enumerate(dominant_periods):
            results.append(
                {
                    "rank": i + 1,
                    "period_days": round(p["period_days"], 1),
                    "power": round(p["power"], 4),
                    "explanation": get_period_explanation(p["period_days"]),
                }
            )

        return jsonify(
            {"fund_code": code, "data_days": len(returns), "dominant_periods": results}
        )
    except Exception as e:
        return jsonify({"error": f"周期分析失败: {e}"}), 500


# ---------- 基金预测 ----------


@api_bp.route("/fund/predict", methods=["POST"])
def fund_predict():
    """预测净值。Body: fund_code"""
    from analysis.fund_lstm import predict as fund_predict_impl

    body = request.get_json(silent=True) or {}
    fund_code = (body.get("fund_code") or "").strip()
    if not fund_code:
        return jsonify({"error": "缺少 fund_code"}), 400
    result = fund_predict_impl(fund_code)
    _clear_fund_cache(f"fund_prediction:{fund_code}")
    return jsonify(result)


@api_bp.route("/fund/prediction/<code>", methods=["GET"])
@_cached("prediction")
def fund_prediction(code: str):
    """获取预测结果。"""
    from analysis.fund_lstm import predict as fund_predict_impl

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    result = fund_predict_impl(code)
    return jsonify(result)


@api_bp.route("/fund/lstm/train", methods=["POST"])
def fund_lstm_train():
    """训练LSTM模型。Body: fund_code, days(默认365), epochs(默认20)"""
    from analysis.fund_lstm import train_model

    body = request.get_json(silent=True) or {}
    fund_code = (body.get("fund_code") or "").strip()
    if not fund_code:
        return jsonify({"error": "缺少基金代码"}), 400

    days = body.get("days", 365)
    epochs = body.get("epochs", 20)

    try:
        result = train_model(fund_code, days=days, epochs=epochs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"训练失败: {e}"}), 500


@api_bp.route("/fund/fit-plot/<code>", methods=["GET"])
def fund_fit_plot(code: str):
    """获取拟合曲线数据，用于前端绘图"""
    from analysis.fund_lstm import get_fit_plot_data

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    data = get_fit_plot_data(code)
    if data is None:
        return jsonify({"error": "暂无拟合数据，请先训练模型"}), 404

    return jsonify(data)


# ---------- 大模型分析 ----------


@api_bp.route("/fund/llm-status", methods=["GET"])
def fund_llm_status():
    """LLM 状态检查。"""
    from analysis.llm.client import is_available as llm_is_available

    available = llm_is_available()
    return jsonify(
        {"available": available, "provider": "MiniMax-M2.5" if available else None}
    )


def _generate_llm_prompt(fund_code: str, analysis_type: str, data: dict) -> str:
    """生成 LLM 分析提示词。"""
    prompts = {
        "profile": f"""请分析基金 {fund_code} 的概况：
基金名称: {data.get("fund_name", "N/A")}
基金类型: {data.get("fund_type", "N/A")}
基金经理: {data.get("manager", "N/A")}
基金规模: {data.get("fund_scale", "N/A")}
成立日期: {data.get("establishment_date", "N/A")}

请给出该基金的概况分析和投资亮点。""",
        "performance": f"""请分析基金 {fund_code} 的业绩表现：
{data.get("metrics", {})}

请分析该基金的收益能力、风险控制能力，并给出评价。""",
        "risk": f"""请分析基金 {fund_code} 的风险评估：
最大回撤: {data.get("max_drawdown", "N/A")}
波动率: {data.get("volatility", "N/A")}
夏普比率: {data.get("sharpe_ratio", "N/A")}

请评估该基金的风险等级和风险特征。""",
    }
    return prompts.get(analysis_type, "")


@api_bp.route("/fund/analysis/profile/<code>", methods=["GET"])
@_cached("llm")
def fund_analysis_profile(code: str):
    """概况分析。"""
    from analysis.llm.client import get_client
    from data import fund_repo
    from data.cache import get_cache

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    cache_key = f"llm_profile:{code}"
    cached = get_cache().get(cache_key)
    if cached:
        return jsonify(cached)

    fund_info = fund_repo.get_fund_info(code)
    if fund_info is None:
        return jsonify({"error": "基金不存在"}), 404

    client = get_client()
    if client is None or not client.is_available():
        return jsonify({"error": "LLM 服务不可用"}), 503

    try:
        prompt = _generate_llm_prompt(code, "profile", fund_info)
        result = client.chat([{"role": "user", "content": prompt}])
        response = {"fund_code": code, "analysis": result}
        get_cache().set(cache_key, response, _FUND_CACHE_TTL["llm"])
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500


@api_bp.route("/fund/analysis/performance/<code>", methods=["GET"])
@_cached("llm")
def fund_analysis_performance(code: str):
    """业绩归因。"""
    from analysis.llm.client import get_client
    from analysis.fund_metrics import get_fund_indicators
    from data.cache import get_cache

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    cache_key = f"llm_performance:{code}"
    cached = get_cache().get(cache_key)
    if cached:
        return jsonify(cached)

    metrics = get_fund_indicators(code, days=365)
    if metrics is None:
        return jsonify({"error": "无法获取业绩数据"}), 404

    client = get_client()
    if client is None or not client.is_available():
        return jsonify({"error": "LLM 服务不可用"}), 503

    try:
        prompt = _generate_llm_prompt(code, "performance", {"metrics": metrics})
        result = client.chat([{"role": "user", "content": prompt}])
        response = {"fund_code": code, "metrics": metrics, "analysis": result}
        get_cache().set(cache_key, response, _FUND_CACHE_TTL["llm"])
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500


@api_bp.route("/fund/analysis/risk/<code>", methods=["GET"])
@_cached("llm")
def fund_analysis_risk(code: str):
    """风险评估。"""
    from analysis.llm.client import get_client
    from analysis.fund_metrics import get_fund_indicators
    from data.cache import get_cache

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    cache_key = f"llm_risk:{code}"
    cached = get_cache().get(cache_key)
    if cached:
        return jsonify(cached)

    metrics = get_fund_indicators(code, days=365)
    if metrics is None:
        return jsonify({"error": "无法获取风险数据"}), 404

    client = get_client()
    if client is None or not client.is_available():
        return jsonify({"error": "LLM 服务不可用"}), 503

    try:
        prompt = _generate_llm_prompt(code, "risk", metrics)
        result = client.chat([{"role": "user", "content": prompt}])
        response = {"fund_code": code, "metrics": metrics, "analysis": result}
        get_cache().set(cache_key, response, _FUND_CACHE_TTL["llm"])
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500


@api_bp.route("/fund/advice/<code>", methods=["GET"])
@_cached("llm")
def fund_advice(code: str):
    """投资建议。"""
    from analysis.llm.client import get_client
    from analysis.fund_metrics import get_fund_indicators
    from analysis.fund_benchmark import compare_with_benchmark, get_benchmark_data
    from data import fund_repo
    from data.cache import get_cache

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    cache_key = f"llm_advice:{code}"
    cached = get_cache().get(cache_key)
    if cached:
        return jsonify(cached)

    fund_info = fund_repo.get_fund_info(code)
    metrics = get_fund_indicators(code, days=365) if fund_info else None

    benchmark_code = "000300"
    benchmark_df = get_benchmark_data(benchmark_code)
    fund_df = fund_repo.get_fund_nav(code)
    comparison = None
    if fund_df is not None and benchmark_df is not None:
        fund_df = fund_df.sort_values("nav_date")
        fund_nav_series = fund_df.set_index("nav_date")["unit_nav"]
        comparison = compare_with_benchmark(fund_nav_series, benchmark_df)

    client = get_client()
    if client is None or not client.is_available():
        return jsonify({"error": "LLM 服务不可用"}), 503

    try:
        prompt = f"""请为基金 {code} 提供投资建议。

基金概况:
- 名称: {fund_info.get("fund_name", "N/A") if fund_info else "N/A"}
- 类型: {fund_info.get("fund_type", "N/A") if fund_info else "N/A"}

业绩指标:
- 年化收益率: {metrics.get("annual_return", "N/A") if metrics else "N/A"}
- 夏普比率: {metrics.get("sharpe_ratio", "N/A") if metrics else "N/A"}
- 最大回撤: {metrics.get("max_drawdown", "N/A") if metrics else "N/A"}
- 卡玛比率: {metrics.get("calmar_ratio", "N/A") if metrics else "N/A"}

基准对比 (沪深300):
- Alpha: {comparison.get("alpha", "N/A") if comparison else "N/A"}
- Beta: {comparison.get("beta", "N/A") if comparison else "N/A"}

请给出综合投资建议，包括是否推荐持有、适合的投资者类型、风险提示等。"""
        result = client.chat([{"role": "user", "content": prompt}])
        response = {"fund_code": code, "advice": result}
        get_cache().set(cache_key, response, _FUND_CACHE_TTL["llm"])
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"分析失败: {e}"}), 500


@api_bp.route("/fund/report/<code>", methods=["GET"])
@_cached("llm")
def fund_report(code: str):
    """完整报告。"""
    from analysis.llm.client import get_client
    from analysis.fund_metrics import get_fund_indicators
    from analysis.fund_benchmark import compare_with_benchmark, get_benchmark_data
    from data import fund_repo
    from data.cache import get_cache

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少基金代码"}), 400

    cache_key = f"llm_report:{code}"
    cached = get_cache().get(cache_key)
    if cached:
        return jsonify(cached)

    fund_info = fund_repo.get_fund_info(code)
    metrics = get_fund_indicators(code, days=365) if fund_info else None

    benchmark_code = "000300"
    benchmark_df = get_benchmark_data(benchmark_code)
    fund_df = fund_repo.get_fund_nav(code)
    comparison = None
    if fund_df is not None and benchmark_df is not None:
        fund_df = fund_df.sort_values("nav_date")
        fund_nav_series = fund_df.set_index("nav_date")["unit_nav"]
        comparison = compare_with_benchmark(fund_nav_series, benchmark_df)

    client = get_client()
    if client is None or not client.is_available():
        return jsonify({"error": "LLM 服务不可用"}), 503

    try:
        prompt = f"""请为基金 {code} 生成一份完整的分析报告，包含以下内容：

1. 基金概况
2. 业绩分析
3. 风险评估
4. 基准对比
5. 投资建议

基金信息:
{str(fund_info) if fund_info else "N/A"}

业绩指标:
{str(metrics) if metrics else "N/A"}

基准对比 (沪深300):
{str(comparison) if comparison else "N/A"}

请用 Markdown 格式输出完整报告。"""
        result = client.chat([{"role": "user", "content": prompt}])
        response = {"fund_code": code, "report": result}
        get_cache().set(cache_key, response, _FUND_CACHE_TTL["llm"])
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"报告生成失败: {e}"}), 500


# ---------- 指数数据 ----------


@api_bp.route("/index/list", methods=["GET"])
@_cached("list")
def index_list():
    """指数列表。"""
    from data import index_repo

    result = index_repo.get_index_list()
    return jsonify({"data": result})


@api_bp.route("/index/data/<code>", methods=["GET"])
@_cached("nav")
def index_data(code: str):
    """指数数据。Query: start?, end?"""
    from data import index_repo

    code = (code or "").strip()
    if not code:
        return jsonify({"error": "缺少指数代码"}), 400
    start_date = request.args.get("start", "").strip() or None
    end_date = request.args.get("end", "").strip() or None
    df = index_repo.get_index_data(code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return jsonify({"error": "暂无数据"}), 404
    return jsonify(
        {
            "index_code": code,
            "data": df.to_dict(orient="records"),
        }
    )


# ---------- 数据同步 ----------


@api_bp.route("/sync/funds", methods=["POST"])
def sync_funds():
    """同步基金数据。Body: fund_codes[] (可选，默认全部)"""
    from data import fund_repo
    import pandas as pd
    from datetime import date, timedelta

    body = request.get_json(silent=True) or {}
    fund_codes = body.get("fund_codes") or []

    if not fund_codes:
        result = fund_repo.get_fund_list(page=1, size=1000)
        fund_codes = [f["fund_code"] for f in result.get("data", [])]

    from data import fund_fetcher

    results = []

    for code in fund_codes:
        try:
            df = fund_fetcher.fetch_fund_nav(code, days=365)
            if df is not None and not df.empty:
                n = fund_repo.upsert_fund_nav(code, df)
                results.append({"code": code, "ok": True, "updated": n})
            else:
                results.append({"code": code, "ok": False, "error": "无数据"})
        except Exception as e:
            results.append({"code": code, "ok": False, "error": str(e)})

    _clear_fund_cache("fund_list", "fund_watchlist")
    success = sum(1 for r in results if r.get("ok"))
    return jsonify(
        {
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results,
        }
    )


@api_bp.route("/sync/status", methods=["GET"])
def sync_status():
    """获取定时同步状态"""
    from server.app import scheduler

    job = scheduler.get_job("sync_funds")
    if job and scheduler.running:
        next_run = (
            job.next_run_time.strftime("%Y-%m-%d %H:%M:%S")
            if job.next_run_time
            else None
        )
        return jsonify(
            {"enabled": True, "next_run": next_run, "schedule": "每天 03:00"}
        )
    return jsonify({"enabled": False, "next_run": None, "schedule": "每天 03:00"})


@api_bp.route("/sync/index", methods=["POST"])
def sync_index():
    """同步指数数据。Body: index_codes[] (可选，默认常用指数)"""
    from data import index_repo
    import pandas as pd

    body = request.get_json(silent=True) or {}
    index_codes = body.get("index_codes") or [
        "000300",
        "000905",
        "000852",
        "000001",
        "399006",
    ]

    try:
        import akshare as ak
    except ImportError:
        return jsonify({"error": "akshare 未安装"}), 503

    results = []

    for code in index_codes:
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            if df is not None and not df.empty:
                df = df.rename(columns={"date": "trade_date", "close": "close_price"})
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                df["daily_return"] = df["close_price"].pct_change()
                df = df[["trade_date", "close_price", "daily_return"]]
                n = index_repo.upsert_index_data(code, df)
                results.append({"code": code, "ok": True, "updated": n})
            else:
                results.append({"code": code, "ok": False, "error": "无数据"})
        except Exception as e:
            results.append({"code": code, "ok": False, "error": str(e)})

    _clear_fund_cache("index_list")
    success = sum(1 for r in results if r.get("ok"))
    return jsonify(
        {
            "total": len(results),
            "success": success,
            "failed": len(results) - success,
            "results": results,
        }
    )
