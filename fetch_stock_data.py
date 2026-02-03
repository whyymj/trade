#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 config.yaml 全量同步股票日线到数据库（与服务端 /api/sync_all 逻辑一致）。
需在项目根目录执行；依赖 config.yaml 中 mysql 与 stocks 配置。
"""

import sys
from pathlib import Path

# 保证项目根在 path 中
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from server.utils import sync_all_from_config

    print("按 config 全量同步到数据库（先清空再拉取）...")
    results = sync_all_from_config(clear_first=True)
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        print(f"  {r.get('symbol', '')}: {status} - {r.get('message', '')}")
    print("完成。")


if __name__ == "__main__":
    main()
