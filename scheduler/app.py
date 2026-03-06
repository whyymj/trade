#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时任务调度器
独立进程，不随 API 扩容
"""

import os
import sys

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

scheduler = BackgroundScheduler()


def _sync_all_funds():
    """每日定时同步所有基金数据"""
    try:
        from data import fund_repo, fund_fetcher

        result = fund_repo.get_fund_list(page=1, size=1000)
        fund_codes = [f["fund_code"] for f in result.get("data", [])]
        success = 0
        for code in fund_codes:
            try:
                df = fund_fetcher.fetch_fund_nav(code, days=30)
                if df is not None and not df.empty:
                    fund_repo.upsert_fund_nav(code, df)
                    success += 1
            except Exception:
                pass
        print(f"[定时任务] 基金数据同步完成: {success}/{len(fund_codes)}")
    except Exception as e:
        print(f"[定时任务-基金数据同步失败]: {e}")


def _auto_train_watchlist():
    """每日定时训练关注列表中的基金"""
    try:
        from analysis.fund_lstm import auto_train_watchlist_funds

        result = auto_train_watchlist_funds()
        success = sum(
            1 for r in result.get("results", []) if r.get("status") == "success"
        )
        total = len(result.get("results", []))
        print(f"[定时任务] LSTM自动训练完成: {success}/{total}")
    except Exception as e:
        print(f"[定时任务-LSTM自动训练失败]: {e}")


# 添加定时任务
scheduler.add_job(
    func=_sync_all_funds,
    trigger=CronTrigger(hour=3, minute=0),
    id="sync_funds",
    name="基金数据同步",
)

scheduler.add_job(
    func=_auto_train_watchlist,
    trigger=CronTrigger(hour=4, minute=0),
    id="auto_train",
    name="LSTM自动训练",
)

if __name__ == "__main__":
    print("[定时任务] 启动调度器")
    print("[定时任务] 基金每日同步已配置 (每天 03:00 执行)")
    print("[定时任务] LSTM自动训练已配置 (每天 04:00 执行)")
    scheduler.start()
    try:
        import time

        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("[定时任务] 关闭调度器")
