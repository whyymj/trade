# -*- coding: utf-8 -*-
"""支持 python -m server 启动。"""
import os
import sys
from pathlib import Path

# 确保项目根目录在 path 中，从任意目录执行 python -m server 或 python server 均可
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from server.app import create_app
from server.utils import get_data_dir

app = create_app()
if __name__ == "__main__":
    print(f"数据目录: {get_data_dir().resolve()}")
    port = int(os.environ.get("PORT", "5050"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
