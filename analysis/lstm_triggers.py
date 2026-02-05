# -*- coding: utf-8 -*-
"""
LSTM 训练触发条件：定期与性能衰减。

训练模式：
- 完整训练：用于月度/季度，do_cv_tune=True + 训练后样本外验证（仅更优则部署）
- 增量训练：用于周度，加载当前模型 + 近期数据微调，不跑交叉验证

触发：
- 每周五：周度增量训练
- 每月最后交易日：完整重新训练
- 每季最后交易日：完整重新训练
- 性能衰减：最近 20 日平均误差 > 历史平均 × 1.5 → 完整重新训练
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any, Callable, Optional

from analysis.lstm_versioning import get_recent_accuracy_for_trigger


def is_friday(t: Optional[date | datetime] = None) -> bool:
    """是否为周五（可视为收盘后触发日）。"""
    d = t.date() if isinstance(t, datetime) else (t or date.today())
    return d.weekday() == 4  # Monday=0, Friday=4


def is_last_trading_day_of_month(t: Optional[date | datetime] = None) -> bool:
    """
    是否为当月最后一个交易日（近似：当月最后一个工作日）。
    若需精确交易所日历，可后续接入 exchange_calendars。
    """
    d = t.date() if isinstance(t, datetime) else (t or date.today())
    _, last_day = calendar.monthrange(d.year, d.month)
    last_date = date(d.year, d.month, last_day)
    # 若 last_date 是周末，往前推到周五
    while last_date.weekday() >= 5:
        last_date -= timedelta(days=1)
    return d == last_date


def should_trigger_weekly(t: Optional[date | datetime] = None) -> bool:
    """是否满足周度触发：每周五。"""
    return is_friday(t)


def should_trigger_monthly(t: Optional[date | datetime] = None) -> bool:
    """是否满足月度触发：每月最后一个交易日。"""
    return is_last_trading_day_of_month(t)


def is_last_trading_day_of_quarter(t: Optional[date | datetime] = None) -> bool:
    """是否为当季最后一个交易日（3/6/9/12 月最后一个工作日）。"""
    d = t.date() if isinstance(t, datetime) else (t or date.today())
    if d.month not in (3, 6, 9, 12):
        return False
    return is_last_trading_day_of_month(d)


def should_trigger_quarterly(t: Optional[date | datetime] = None) -> bool:
    """是否满足季度触发：当季最后交易日。"""
    return is_last_trading_day_of_quarter(t)


def should_trigger_performance_decay(
    threshold_multiplier: float = 1.5,
    n_recent_trading_days: int = 20,
    save_dir: Optional[str] = None,
) -> tuple[bool, float, float]:
    """
    是否满足性能衰减触发：最近 n 个交易日平均预测误差 > 历史平均误差 × threshold_multiplier。
    返回 (是否触发, 最近平均误差, 历史平均误差)。
    """
    recent_avg, historical_avg = get_recent_accuracy_for_trigger(
        n_trading_days=n_recent_trading_days,
        save_dir=save_dir,
    )
    if historical_avg <= 0:
        return False, recent_avg, historical_avg
    trigger = recent_avg > historical_avg * threshold_multiplier
    return trigger, recent_avg, historical_avg


def check_triggers(
    t: Optional[date | datetime] = None,
    performance_threshold: float = 1.5,
    save_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    检查所有触发条件，不执行训练。
    返回: {
        "weekly": bool,
        "monthly": bool,
        "performance_decay": bool,
        "performance_recent_avg": float,
        "performance_historical_avg": float,
    }
    """
    t = t or date.today()
    weekly = should_trigger_weekly(t)
    monthly = should_trigger_monthly(t)
    quarterly = should_trigger_quarterly(t)
    perf_trigger, recent_avg, hist_avg = should_trigger_performance_decay(
        threshold_multiplier=performance_threshold,
        save_dir=save_dir,
    )
    return {
        "weekly": weekly,
        "monthly": monthly,
        "quarterly": quarterly,
        "performance_decay": perf_trigger,
        "performance_recent_avg": recent_avg,
        "performance_historical_avg": hist_avg,
    }


def run_triggered_training(
    symbol: str,
    trigger_type: str,
    fetch_hist_fn: Callable[[str, str, str], Any],
    get_date_range_fn: Callable[[], tuple[str, str, Any]],
    run_lstm_pipeline_fn: Callable[..., dict[str, Any]],
    incremental_train_fn: Optional[Callable[..., dict[str, Any]]] = None,
    *,
    full_range_months_weekly: int = 6,
    save_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    根据触发类型执行训练：
    - weekly: 增量训练（加载当前模型，近期数据微调）；若无 incremental_train_fn 则退化为 run_lstm_pipeline do_cv_tune=False
    - monthly: 完整重新训练，使用配置的完整日期范围，do_cv_tune=True
    - performance_decay: 同 monthly，完整重新训练
    """
    start_date, end_date, _ = get_date_range_fn()
    if trigger_type == "weekly":
        from datetime import datetime as dt, timedelta
        end_str = (end_date or "").replace("-", "")[:8]
        if not end_str or len(end_str) != 8:
            return {"error": "无法解析 end_date，周度训练跳过", "trigger": "weekly"}
        end_dt = dt.strptime(end_str, "%Y%m%d").date()
        start_dt = end_dt - timedelta(days=full_range_months_weekly * 31)
        start_date_str = start_dt.strftime("%Y%m%d")
        df = fetch_hist_fn(symbol, start_date_str, end_str)
        if df is None or df.empty:
            return {"error": "周度训练拉取数据失败", "trigger": "weekly"}
        if incremental_train_fn is not None:
            result = incremental_train_fn(df, symbol=symbol, save_dir=save_dir)
            if "error" in result:
                return result
            result["trigger"] = "weekly"
            return result
        result = run_lstm_pipeline_fn(
            df,
            symbol=symbol,
            save_dir=save_dir,
            do_cv_tune=False,
            do_shap=False,
            do_plot=True,
        )
        result["trigger"] = "weekly"
        return result
    if trigger_type in ("monthly", "quarterly", "performance_decay"):
        s = (start_date or "").replace("-", "")[:8]
        e = (end_date or "").replace("-", "")[:8]
        df = fetch_hist_fn(symbol, s, e)
        if df is None or df.empty:
            return {"error": "完整训练拉取数据失败", "trigger": trigger_type}
        result = run_lstm_pipeline_fn(
            df,
            symbol=symbol,
            save_dir=save_dir,
            do_cv_tune=True,
            do_shap=True,
            do_plot=True,
            do_post_training_validation=True,
        )
        result["trigger"] = trigger_type
        return result
    return {"error": f"未知触发类型: {trigger_type}"}
