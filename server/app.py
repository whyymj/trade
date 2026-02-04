# -*- coding: utf-8 -*-
"""
Flask 应用工厂：创建 app、注册蓝图、配置静态与 SPA 回退。
前后分离：静态资源来自 frontend/dist（Vite 构建产物）。
启动/退出时自动清理分析临时目录，无需手动执行。
支持服务端日志：文件轮转 + 请求访问日志。
"""
import atexit
from pathlib import Path

from flask import Flask, abort, g, redirect, request, send_from_directory

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
    init_logging()
    try:
        from data.schema import create_all_tables
        create_all_tables()
    except Exception:
        pass
    # 不设 static_folder，避免 Flask 默认静态路由对 /chart 等返回 404；由下方通配路由统一处理
    dist = str(static_folder or FRONTEND_DIST)
    app = Flask(__name__, static_folder=None)
    app.register_blueprint(api_bp)

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
