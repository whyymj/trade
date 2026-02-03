# -*- coding: utf-8 -*-
"""
Flask 应用工厂：创建 app、注册蓝图、配置静态与 SPA 回退。
前后分离：静态资源来自 frontend/dist（Vite 构建产物）。
"""
from pathlib import Path

from flask import Flask, send_from_directory

from server.routes import api_bp

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"


def create_app(static_folder=None):
    """创建并配置 Flask 应用。启动时确保数据库表已创建（若连接失败则跳过）。"""
    try:
        from data.schema import create_all_tables
        create_all_tables()
    except Exception:
        pass
    folder = str(static_folder or FRONTEND_DIST)
    app = Flask(__name__, static_folder=folder, static_url_path="")
    app.register_blueprint(api_bp)

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
