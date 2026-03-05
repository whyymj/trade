# -*- coding: utf-8 -*-
"""
Flask 应用工厂：创建 app、注册蓝图、配置静态与 SPA 回退。
前后分离：静态资源来自 frontend/dist（Vite 构建产物）。
启动/退出时自动清理分析临时目录，无需手动执行。
支持服务端日志：文件轮转 + 请求访问日志。
支持每日定时同步基金数据。
"""

import atexit
import logging
import math
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, abort, g, redirect, request, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler


def _custom_json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _sync_all_funds():
    """每日定时同步所有基金数据"""
    try:
        from data import fund_repo, fund_fetcher

        result = fund_repo.get_fund_list(page=1, size=1000)
        fund_codes = [f["fund_code"] for f in result.get("data", [])]
        success = 0
        for code in fund_codes:
            try:
                df = fund_fetcher.fetch_fund_nav(code, days=30)
                if df is not None and not df.empty:
                    fund_repo.upsert_fund_nav(code, df)
                    success += 1
            except Exception:
                pass
        print(f"[定时任务] 基金数据同步完成: {success}/{len(fund_codes)}")
    except Exception as e:
        logging.error("定时任务-基金数据同步失败: %s", e)


def _auto_train_watchlist():
    """每日定时训练关注列表中的基金"""
    try:
        from analysis.fund_lstm import auto_train_watchlist_funds

        result = auto_train_watchlist_funds()
        success = sum(
            1 for r in result.get("results", []) if r.get("status") == "success"
        )
        total = len(result.get("results", []))
        print(f"[定时任务] LSTM自动训练完成: {success}/{total}")
    except Exception as e:
        logging.error("定时任务-LSTM自动训练失败: %s", e)


scheduler = BackgroundScheduler()
scheduler.add_job(
    func=_sync_all_funds, trigger="cron", hour=3, minute=0, id="sync_funds"
)
scheduler.add_job(
    func=_auto_train_watchlist, trigger="cron", hour=4, minute=0, id="auto_train"
)


from server.logging_config import (
    init_logging,
    log_request_end,
    log_request_start,
)
from server.routes import api_bp

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

# 前端 SPA 路径前缀，与 /api 等接口区分
APP_PATH_PREFIX = "app"


def _run_cleanup():
    """执行一次分析临时目录清理（静默失败）。"""
    try:
        from analysis.cleanup_temp import cleanup_analysis_temp_dirs

        n = cleanup_analysis_temp_dirs()
        if n > 0:
            print(f"[清理] 已清理 {n} 个分析临时目录")
    except Exception:
        pass


def create_app(static_folder=None):
    """创建并配置 Flask 应用。启动时先清理遗留分析临时目录，退出时再清理一次；再确保数据库表已创建；初始化日志并注册请求日志。"""
    _run_cleanup()
    atexit.register(_run_cleanup)
    atexit.register(lambda: scheduler.shutdown())
    init_logging()
    try:
        from data.schema import create_all_tables

        create_all_tables()
    except Exception as e:
        logging.critical("数据库表初始化失败: %s", e)

    if not scheduler.running:
        scheduler.start()
        print("[定时任务] 基金每日同步已启动 (每天 03:00 执行)")

    # 不设 static_folder，避免 Flask 默认静态路由对 /chart 等返回 404；由下方通配路由统一处理
    dist = str(static_folder or FRONTEND_DIST)
    app = Flask(__name__, static_folder=None)
    app.json.default = _custom_json_default
    app.register_blueprint(api_bp)

    # 注册新闻和市场API蓝图
    from server.routes.news import news_bp
    from server.routes.market import market_bp

    app.register_blueprint(news_bp)
    app.register_blueprint(market_bp)

    # 注册基金行业API蓝图
    from server.routes.fund_industry import fund_industry_bp

    app.register_blueprint(fund_industry_bp)

    # 注册新闻行业分类API蓝图
    from server.routes.news_classification import news_classification_bp

    app.register_blueprint(news_classification_bp)

    # 注册基金-新闻关联API蓝图
    from server.routes.fund_news_association import fund_news_association_bp

    app.register_blueprint(fund_news_association_bp)

    # 注册投资建议API蓝图
    from server.routes.investment_advice import investment_advice_bp

    app.register_blueprint(investment_advice_bp)

    @app.before_request
    def _request_start():
        g._request_start = log_request_start()
        g._request_method = request.method
        g._request_path = request.path

    @app.after_request
    def _request_log(response):
        start = getattr(g, "_request_start", None)
        if start is not None:
            log_request_end(
                getattr(g, "_request_method", "?"),
                getattr(g, "_request_path", "?"),
                response.status_code,
                start,
            )
        return response

    @app.route("/")
    def index():
        """根路径重定向到前端 SPA 前缀，避免与接口冲突。"""
        return redirect(f"/{APP_PATH_PREFIX}", code=302)

    @app.route("/<path:path>")
    def serve_spa(path):
        """静态文件按路径返回；仅对 /app 及 /app/* 做 SPA 回退；favicon.ico 映射为 favicon.svg。"""
        path = path.strip()
        if path == "favicon.ico":
            return send_from_directory(dist, "favicon.svg", mimetype="image/svg+xml")
        # 仅识别前端页面前缀：/app、/app/* 下先按静态文件查找，再回退 index.html
        if path == APP_PATH_PREFIX or path.startswith(APP_PATH_PREFIX + "/"):
            subpath = path[len(APP_PATH_PREFIX) + 1 :].strip()  # 如 "assets/xxx.js"
            if subpath:
                file_path = Path(dist) / subpath
                if file_path.is_file():
                    return send_from_directory(dist, subpath)
            return send_from_directory(dist, "index.html")
        file_path = Path(dist) / path
        if file_path.is_file():
            return send_from_directory(dist, path)
        abort(404)

    return app
