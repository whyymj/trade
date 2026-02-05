# -*- coding: utf-8 -*-
"""
每日预测流程：获取最新 60 日数据 -> 模型健康检查 -> 预测 -> 记录 -> 可选检查训练触发并异步训练。

- 模型健康度检查：模型存在、未过旧（可配置天数）、可选数据质量
- 预测后可选：检查触发条件，满足则在后台启动训练任务
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from analysis.lstm_versioning import get_current_version_id, list_versions

logger = logging.getLogger(__name__)

# 默认：超过 30 天未训练视为不健康
DEFAULT_MAX_DAYS_WITHOUT_TRAINING = 30


def model_health_check(
    save_dir: Optional[Any] = None,
    *,
    max_days_without_training: int = DEFAULT_MAX_DAYS_WITHOUT_TRAINING,
) -> dict[str, Any]:
    """
    检查模型健康度。返回 { "healthy": bool, "message": str, "details": {...} }。
    - 无当前版本或模型文件缺失 -> 不健康
    - 最后训练时间超过 max_days_without_training 天 -> 不健康并给出警告信息
    """
    details: dict[str, Any] = {}
    try:
        versions = list_versions(save_dir)
        current_id = get_current_version_id(save_dir)
        if not current_id or not versions:
            return {
                "healthy": False,
                "message": "无可用模型或未找到版本",
                "details": {"current_version_id": current_id},
            }
        current_meta = next((v for v in versions if v.get("version_id") == current_id), None)
        if not current_meta:
            return {
                "healthy": False,
                "message": "当前版本元数据缺失",
                "details": {"current_version_id": current_id},
            }
        details["current_version_id"] = current_id
        details["last_training_time"] = current_meta.get("training_time")
        training_time = current_meta.get("training_time")
        if not training_time:
            return {"healthy": True, "message": "模型存在，未记录训练时间", "details": details}
        try:
            if "T" in str(training_time):
                train_dt = datetime.fromisoformat(str(training_time).replace("Z", "+00:00"))
            else:
                train_dt = datetime.strptime(str(training_time)[:19], "%Y-%m-%dT%H:%M:%S")
            days_ago = (datetime.now() - train_dt.replace(tzinfo=None)).days
            details["days_since_training"] = days_ago
            if days_ago >= max_days_without_training:
                return {
                    "healthy": False,
                    "message": f"模型已 {days_ago} 天未更新，超过阈值 {max_days_without_training} 天，建议重新训练",
                    "details": details,
                }
        except Exception as e:
            details["parse_error"] = str(e)
        return {"healthy": True, "message": "模型健康", "details": details}
    except Exception as e:
        logger.exception("model_health_check failed")
        return {"healthy": False, "message": str(e), "details": {}}


def run_training_trigger_async(
    symbol: str,
    check_triggers_fn: Callable[..., dict],
    run_triggered_training_fn: Callable[..., dict],
    fetch_hist_fn: Callable[[str, str, str], Any],
    get_date_range_fn: Callable[[], tuple],
    run_lstm_pipeline_fn: Optional[Callable] = None,
    incremental_train_fn: Optional[Callable] = None,
    save_dir: Optional[Any] = None,
) -> None:
    """
    在后台线程中检查触发条件并执行训练（不阻塞调用方）。
    """
    def _run():
        try:
            triggers = check_triggers_fn(save_dir=save_dir)
            run_type = None
            if triggers.get("monthly"):
                run_type = "monthly"
            elif triggers.get("quarterly"):
                run_type = "quarterly"
            elif triggers.get("performance_decay"):
                run_type = "performance_decay"
            elif triggers.get("weekly"):
                run_type = "weekly"
            if run_type:
                run_triggered_training_fn(
                    symbol=symbol,
                    trigger_type=run_type,
                    fetch_hist_fn=fetch_hist_fn,
                    get_date_range_fn=get_date_range_fn,
                    run_lstm_pipeline_fn=run_lstm_pipeline_fn or (lambda *a, **k: {}),
                    incremental_train_fn=incremental_train_fn,
                    save_dir=save_dir,
                )
        except Exception as e:
            logger.warning("异步训练任务异常: %s", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    logger.info("已启动后台训练检查任务 (symbol=%s)", symbol)
