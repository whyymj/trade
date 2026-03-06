#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻服务启动脚本
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from services.news.app import app

    port = int(os.getenv("PORT", 8003))
    print(f"Starting News Service on port {port}...")
    app.run(host="0.0.0.0", port=port)
