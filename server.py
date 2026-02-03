#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后端统一启动入口。启动后提供 API（/api/*）与前端静态（frontend/dist）。
"""
from server.app import create_app
from server.utils import get_data_dir

app = create_app()

if __name__ == "__main__":
    print(f"数据目录: {get_data_dir().resolve()}")
    app.run(host="0.0.0.0", port=5050, debug=True)
