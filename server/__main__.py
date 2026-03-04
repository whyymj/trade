# -*- coding: utf-8 -*-
"""支持 python -m server 启动。"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env", override=True)

# 确保项目根目录在 path 中，从任意目录执行 python -m server 或 python server 均可
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def run_tests():
    """运行测试"""
    print("=" * 50)
    print("运行测试...")
    print("=" * 50)
    from tests.test_fund import run_all_tests

    success = run_all_tests()
    if not success:
        print("❌ 测试失败，请修复后再启动")
        sys.exit(1)
    print("✅ 所有测试通过！")
    print()


from server.app import create_app
from server.utils import get_data_dir

if __name__ == "__main__":
    # 检查是否需要运行测试
    run_tests_flag = os.environ.get("RUN_TESTS", "0") == "1"

    if run_tests_flag:
        run_tests()

    print(f"数据目录: {get_data_dir().resolve()}")
    port = int(os.environ.get("PORT", "5050"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=debug)
