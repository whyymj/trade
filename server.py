#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端统一启动入口。启动后提供 API（/api/*）与前端静态（frontend/dist）。
环境变量 PORT、FLASK_DEBUG 可用于 Docker 等部署。
"""
import os

from server.app import create_app
from server.utils import get_data_dir

app = create_app()

if __name__ == "__main__":
    print(f"数据目录: {get_data_dir().resolve()}")
    port = int(os.environ.get("PORT", "5050"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
