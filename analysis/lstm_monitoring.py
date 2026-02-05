# -*- coding: utf-8 -*-
"""
LSTM 监控与告警。

- 性能衰减检测：最近 N 日平均预测误差 vs 历史平均，可记录检测历史
- 监控状态：当前版本、最后训练时间、预测次数、准确性指标
- 告警：性能衰减、长时间未训练、低准确率等，支持 webhook 通知
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from analysis.lstm_versioning import (
    PREDICTION_LOG_FILE,
    get_accuracy_records,
    get_current_version_id,
    get_recent_accuracy_for_trigger,
    list_versions,
    _base_dir,
    _load_json_list,
    _save_json_list,
)

logger = logging.getLogger(__name__)

PERFORMANCE_DETECTION_LOG_FILE = "performance_detection_log.json"
ALERT_LOG_FILE = "alert_log.json"
TRAINING_FAILURE_LOG_FILE = "training_failure_log.json"
MAX_DETECTION_LOG_ENTRIES = 100
MAX_ALERT_LOG_ENTRIES = 200
MAX_TRAINING_FAILURE_LOG_ENTRIES = 50

# 告警类型
ALERT_PERFORMANCE_DECAY = "performance_decay"
ALERT_NO_RECENT_TRAINING = "no_recent_training"
ALERT_LOW_ACCURACY = "low_accuracy"
ALERT_CONSECUTIVE_HIGH_ERROR = "consecutive_high_error"
ALERT_TRAINING_FAILURE_COUNT = "training_failure_count"
ALERT_DATA_MISSING = "data_missing"


def run_performance_decay_detection(
    save_dir: Optional[Path | str] = None,
    threshold_multiplier: float = 1.5,
    n_recent_trading_days: int = 20,
    log_result: bool = True,
) -> dict[str, Any]:
    """
    执行性能衰减检测，返回完整报告。
    若 log_result 为 True，将本次检测结果追加到 performance_detection_log。
    """
    base = _base_dir(save_dir)
    recent_avg, historical_avg = get_recent_accuracy_for_trigger(
        n_trading_days=n_recent_trading_days,
        save_dir=base,
    )
    threshold = historical_avg * threshold_multiplier if historical_avg > 0 else 0.0
    triggered = recent_avg > threshold if historical_avg > 0 else False

    records = get_accuracy_records(base)
    with_date = [(r.get("actual_date") or r.get("predict_date") or "", r) for r in records if "error_magnitude" in r]
    with_date.sort(key=lambda x: x[0], reverse=True)
    n_recent = len(with_date[:n_recent_trading_days])
    n_historical = len(records)

    result = {
        "triggered": triggered,
        "recent_avg_error": round(recent_avg, 6),
        "historical_avg_error": round(historical_avg, 6),
        "threshold": round(threshold, 6),
        "threshold_multiplier": threshold_multiplier,
        "n_recent_samples": n_recent,
        "n_historical_samples": n_historical,
        "detected_at": datetime.now().isoformat(),
    }
    if log_result:
        log_path = base / PERFORMANCE_DETECTION_LOG_FILE
        base.mkdir(parents=True, exist_ok=True)
        history = _load_json_list(base, PERFORMANCE_DETECTION_LOG_FILE)
        history = history if isinstance(history, list) else []
        history.insert(0, result)
        history = history[:MAX_DETECTION_LOG_ENTRIES]
        _save_json_list(base, PERFORMANCE_DETECTION_LOG_FILE, history)
    return result


def get_monitoring_status(
    save_dir: Optional[Path | str] = None,
    n_recent_days: int = 20,
) -> dict[str, Any]:
    """
    汇总监控状态：当前版本、最后训练时间、数据范围、验证分数、
    近期预测次数、最近/历史平均预测误差、最近一次性能衰减检测结果。
    """
    base = _base_dir(save_dir)
    current_id = get_current_version_id(base)
    versions = list_versions(base)
    current_meta: Optional[dict] = None
    for v in versions:
        if v.get("version_id") == current_id:
            current_meta = v
            break

    # 预测次数：最近 7 天（优先从 MySQL 统计）
    prediction_count_7d = 0
    try:
        from data.lstm_repo import count_predictions_since_db
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        prediction_count_7d = count_predictions_since_db(cutoff)
    except Exception:
        pass
    if prediction_count_7d == 0:
        pred_log = _load_json_list(base, PREDICTION_LOG_FILE)
        if isinstance(pred_log, list):
            cutoff = (date.today() - timedelta(days=7)).isoformat()
            prediction_count_7d = sum(1 for p in pred_log if (p.get("predict_date") or "") >= cutoff)

    recent_avg, historical_avg = get_recent_accuracy_for_trigger(
        n_trading_days=n_recent_days,
        save_dir=base,
    )
    detection = run_performance_decay_detection(
        save_dir=base,
        threshold_multiplier=1.5,
        n_recent_trading_days=n_recent_days,
        log_result=False,
    )

    # MAE / RMSE / 方向准确率（基于准确性记录）
    records = get_accuracy_records(base)
    errors = [float(r["error_magnitude"]) for r in records if "error_magnitude" in r]
    mae = float(np.mean(errors)) if errors else None
    rmse = float(np.sqrt(np.mean([e * e for e in errors]))) if errors else None
    direction_correct = [r.get("direction_correct") for r in records if "direction_correct" in r]
    direction_accuracy = float(np.mean(direction_correct)) if direction_correct else None

    # 训练失败次数（优先从 MySQL）
    training_failure_count = 0
    try:
        from data.lstm_repo import get_training_failure_count_db
        training_failure_count = get_training_failure_count_db()
    except Exception:
        pass
    if training_failure_count == 0:
        failure_log = _load_json_list(base, TRAINING_FAILURE_LOG_FILE)
        training_failure_count = len(failure_log) if isinstance(failure_log, list) else 0

    return {
        "current_version_id": current_id,
        "last_training_time": current_meta.get("training_time") if current_meta else None,
        "data_start": current_meta.get("data_start") if current_meta else None,
        "data_end": current_meta.get("data_end") if current_meta else None,
        "validation_score": current_meta.get("validation_score") if current_meta else None,
        "metrics": current_meta.get("metrics") if current_meta else None,
        "prediction_count_7d": prediction_count_7d,
        "accuracy_recent_avg_error": round(recent_avg, 6),
        "accuracy_historical_avg_error": round(historical_avg, 6),
        "mae": round(mae, 6) if mae is not None else None,
        "rmse": round(rmse, 6) if rmse is not None else None,
        "direction_accuracy": round(direction_accuracy, 4) if direction_accuracy is not None else None,
        "training_failure_count": training_failure_count,
        "performance_decay": detection,
    }


def check_alerts(
    save_dir: Optional[Path | str] = None,
    *,
    performance_decay_multiplier: float = 1.5,
    max_days_without_training: int = 30,
    min_direction_accuracy: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    检查告警条件，返回告警列表。每项: { "type", "message", "severity", "at", "detail" }。
    - performance_decay: 最近 20 日平均误差 > 历史平均 × multiplier
    - no_recent_training: 当前版本训练时间超过 max_days_without_training 天
    - low_accuracy: 若配置 min_direction_accuracy，且近期方向正确率低于该值则告警
    """
    base = _base_dir(save_dir)
    alerts: list[dict[str, Any]] = []
    now = datetime.now().isoformat()

    # 性能衰减
    triggered, recent_avg, hist_avg = get_recent_accuracy_for_trigger(20, save_dir=base)
    if hist_avg > 0 and recent_avg > hist_avg * performance_decay_multiplier:
        alerts.append({
            "type": ALERT_PERFORMANCE_DECAY,
            "message": f"预测性能衰减：最近20日平均误差 {recent_avg:.4f} 超过历史平均 {hist_avg:.4f} 的 {performance_decay_multiplier} 倍",
            "severity": "high",
            "at": now,
            "detail": {"recent_avg_error": recent_avg, "historical_avg_error": hist_avg},
        })

    # 长时间未训练
    current_id = get_current_version_id(base)
    versions = list_versions(base)
    current_meta = None
    for v in versions:
        if v.get("version_id") == current_id:
            current_meta = v
            break
    if current_meta:
        training_time = current_meta.get("training_time")
        if training_time:
            try:
                if "T" in training_time:
                    train_dt = datetime.fromisoformat(training_time.replace("Z", "+00:00"))
                else:
                    train_dt = datetime.strptime(training_time[:19], "%Y-%m-%dT%H:%M:%S")
                days_ago = (datetime.now() - train_dt.replace(tzinfo=None)).days
                if days_ago >= max_days_without_training:
                    alerts.append({
                        "type": ALERT_NO_RECENT_TRAINING,
                        "message": f"模型已 {days_ago} 天未重新训练，超过阈值 {max_days_without_training} 天",
                        "severity": "medium",
                        "at": now,
                        "detail": {"days_since_training": days_ago},
                    })
            except Exception:
                pass

    # 方向准确率
    if min_direction_accuracy is not None and min_direction_accuracy > 0:
        records = get_accuracy_records(base)
        if records:
            recent = sorted(records, key=lambda r: r.get("actual_date") or r.get("predict_date") or "", reverse=True)[:20]
            correct = sum(1 for r in recent if r.get("direction_correct") == 1)
            total = len(recent)
            if total > 0:
                acc = correct / total
                if acc < min_direction_accuracy:
                    alerts.append({
                        "type": ALERT_LOW_ACCURACY,
                        "message": f"近期方向预测正确率 {acc:.2%} 低于阈值 {min_direction_accuracy:.2%}",
                        "severity": "medium",
                        "at": now,
                        "detail": {"direction_accuracy": acc, "n_samples": total},
                    })

    # 连续 N 天预测误差超过阈值
    records = get_accuracy_records(base)
    if records:
        _, hist_avg_for_threshold = get_recent_accuracy_for_trigger(20, save_dir=base)
        with_date = [(r.get("actual_date") or r.get("predict_date") or "", r) for r in records if "error_magnitude" in r]
        with_date.sort(key=lambda x: x[0], reverse=True)
        recent_5 = with_date[:5]
        if len(recent_5) >= 5:
            threshold = hist_avg_for_threshold * 1.5 if hist_avg_for_threshold > 0 else 0.05
            over = [r for _, r in recent_5 if float(r.get("error_magnitude", 0)) > threshold]
            if len(over) >= 5:
                alerts.append({
                    "type": ALERT_CONSECUTIVE_HIGH_ERROR,
                    "message": f"连续 5 天预测误差超过阈值 {threshold:.4f}",
                    "severity": "high",
                    "at": now,
                    "detail": {"threshold": threshold, "count": len(over)},
                })

    # 训练失败次数 >= 3（优先从 MySQL）
    failure_count = 0
    try:
        from data.lstm_repo import get_training_failure_count_db
        failure_count = get_training_failure_count_db()
    except Exception:
        failure_log = _load_json_list(base, TRAINING_FAILURE_LOG_FILE)
        failure_count = len(failure_log) if isinstance(failure_log, list) else 0
    if failure_count >= 3:
        alerts.append({
            "type": ALERT_TRAINING_FAILURE_COUNT,
            "message": f"训练失败已累计 {failure_count} 次",
            "severity": "high",
            "at": now,
            "detail": {"failure_count": failure_count},
        })

    return alerts


def record_training_failure(
    symbol: str,
    error_message: str,
    save_dir: Optional[Path | str] = None,
) -> None:
    """记录一次训练失败，用于告警（训练失败超过 3 次）；同时写入 MySQL。"""
    try:
        from data.lstm_repo import insert_training_failure_db
        insert_training_failure_db(symbol=symbol, error_message=error_message)
    except Exception:
        pass
    base = _base_dir(save_dir)
    base.mkdir(parents=True, exist_ok=True)
    log = _load_json_list(base, TRAINING_FAILURE_LOG_FILE)
    log = log if isinstance(log, list) else []
    log.insert(0, {
        "symbol": symbol,
        "error": error_message,
        "at": datetime.now().isoformat(),
    })
    _save_json_list(base, TRAINING_FAILURE_LOG_FILE, log[:MAX_TRAINING_FAILURE_LOG_ENTRIES])


def fire_alerts(
    alerts: list[dict[str, Any]],
    save_dir: Optional[Path | str] = None,
    *,
    webhook_url: Optional[str] = None,
    log_alerts: bool = True,
) -> dict[str, Any]:
    """
    执行告警：写入 alert_log，若配置 webhook_url 则对每条告警 POST 请求。
    返回 { "logged": n, "webhook_sent": n, "webhook_errors": [...] }。
    """
    base = _base_dir(save_dir)
    logged = 0
    webhook_sent = 0
    webhook_errors: list[str] = []

    if log_alerts and alerts:
        base.mkdir(parents=True, exist_ok=True)
        history = _load_json_list(base, ALERT_LOG_FILE)
        history = history if isinstance(history, list) else []
        for a in alerts:
            history.insert(0, a)
            logged += 1
        history = history[:MAX_ALERT_LOG_ENTRIES]
        _save_json_list(base, ALERT_LOG_FILE, history)

    if webhook_url and webhook_url.strip():
        try:
            import urllib.request
            for a in alerts:
                try:
                    body = json.dumps(a, ensure_ascii=False).encode("utf-8")
                    req = urllib.request.Request(
                        webhook_url.strip(),
                        data=body,
                        headers={"Content-Type": "application/json; charset=utf-8"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    webhook_sent += 1
                except Exception as e:
                    webhook_errors.append(f"{a.get('type', '')}: {e}")
        except Exception as e:
            webhook_errors.append(str(e))
            logger.warning("LSTM alert webhook failed: %s", e)

    return {"logged": logged, "webhook_sent": webhook_sent, "webhook_errors": webhook_errors}


def get_alert_config_from_env_or_config(load_config_fn: Optional[Callable[[], dict]] = None) -> dict[str, Any]:
    """从环境变量或 config 读取告警配置。"""
    import os
    cfg: dict = (load_config_fn or (lambda: {}))()
    lstm = cfg.get("lstm") or cfg.get("lstm_alerts") or {}
    return {
        "webhook_url": os.environ.get("LSTM_ALERT_WEBHOOK") or lstm.get("webhook_url") or lstm.get("alert_webhook"),
        "performance_decay_multiplier": float(lstm.get("performance_decay_multiplier", 1.5)),
        "max_days_without_training": int(lstm.get("max_days_without_training", 30)),
        "min_direction_accuracy": lstm.get("min_direction_accuracy"),  # None 表示不检查
    }
