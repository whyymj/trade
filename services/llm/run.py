#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 服务启动脚本
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.llm.app import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8006))
    app.run(host="0.0.0.0", port=port, debug=True)
