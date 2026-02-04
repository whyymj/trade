# -*- coding: utf-8 -*-
"""
Flask 应用工厂：创建 app、注册蓝图、配置静态与 SPA 回退。
前后分离：静态资源来自 frontend/dist（Vite 构建产物）。
启动/退出时自动清理分析临时目录，无需手动执行。
支持服务端日志：文件轮转 + 请求访问日志。
"""
import atexit
from pathlib import Path

from flask import Flask, g, request, send_from_directory

from server.logging_config import (
    init_logging,
    log_request_end,
    log_request_start,
)
from server.routes import api_bp

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"


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
    folder = str(static_folder or FRONTEND_DIST)
    app = Flask(__name__, static_folder=folder, static_url_path="")
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
        """SPA 入口。"""
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/<path:path>")
    def serve_spa(path):
        """静态文件或 SPA 回退（Vue Router history）。"""
        file_path = Path(app.static_folder) / path
        if file_path.is_file():
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")

    return app
