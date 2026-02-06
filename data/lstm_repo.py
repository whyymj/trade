# -*- coding: utf-8 -*-
"""
LSTM 训练与预测数据仓储：训练流水、当前版本、预测记录、准确性回填、训练失败。
依赖 data.mysql、data.schema（需先 create_lstm_tables）。
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Optional

from data.mysql import execute, fetch_all, fetch_one


def _json_dumps(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return None


def _json_loads(s: Optional[str]) -> Any:
    if s is None or (isinstance(s, str) and not s.strip()):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _norm_date(s: Optional[str]) -> str:
    v = (s or "").strip().replace("-", "")[:8]
    if len(v) == 8:
        return f"{v[:4]}-{v[4:6]}-{v[6:8]}"
    return s or ""


def _date_to_str(d: Any) -> Optional[str]:
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()
    if isinstance(d, datetime):
        return d.date().isoformat()
    return str(d)[:10]


# ---------- 训练流水 ----------


def insert_training_run(
    version_id: Optional[str],
    symbol: str,
    training_type: str,
    trigger_type: Optional[str],
    data_start: Optional[str],
    data_end: Optional[str],
    params: Optional[dict],
    metrics: Optional[dict],
    validation_deployed: bool = False,
    validation_reason: Optional[str] = None,
    holdout_metrics: Optional[dict] = None,
    duration_seconds: Optional[int] = None,
) -> int:
    """插入一条训练流水，返回自增 id。"""
    sql = """
    INSERT INTO lstm_training_run (
        version_id, symbol, training_type, trigger_type,
        data_start, data_end, params_json, metrics_json,
        validation_deployed, validation_reason, holdout_metrics_json, duration_seconds
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    data_start_val = data_start[:10] if data_start else None
    data_end_val = data_end[:10] if data_end else None
    execute(sql, (
        version_id or None,
        (symbol or "").strip(),
        (training_type or "full").strip(),
        (trigger_type or "").strip() or None,
        data_start_val,
        data_end_val,
        _json_dumps(params),
        _json_dumps(metrics),
        1 if validation_deployed else 0,
        (validation_reason or "").strip() or None,
        _json_dumps(holdout_metrics),
        duration_seconds,
    ))
    row = fetch_one("SELECT LAST_INSERT_ID() AS id")
    return int(row["id"]) if row and row.get("id") else 0


def list_training_runs(
    symbol: Optional[str] = None,
    limit: int = 50,
    dedupe_by_symbol: bool = False,
) -> list[dict[str, Any]]:
    """查询训练流水，按创建时间倒序。dedupe_by_symbol=True 时每只股票只返回最新一条。"""
    if symbol:
        sql = """
        SELECT id, version_id, symbol, training_type, trigger_type,
               data_start, data_end, params_json, metrics_json,
               validation_deployed, validation_reason, holdout_metrics_json,
               duration_seconds, created_at
        FROM lstm_training_run
        WHERE symbol = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        rows = fetch_all(sql, (symbol.strip(), limit))
    elif dedupe_by_symbol:
        sql = """
        SELECT t.id, t.version_id, t.symbol, t.training_type, t.trigger_type,
               t.data_start, t.data_end, t.params_json, t.metrics_json,
               t.validation_deployed, t.validation_reason, t.holdout_metrics_json,
               t.duration_seconds, t.created_at
        FROM lstm_training_run t
        INNER JOIN (
            SELECT symbol, MAX(id) AS max_id FROM lstm_training_run GROUP BY symbol
        ) latest ON t.symbol = latest.symbol AND t.id = latest.max_id
        ORDER BY t.created_at DESC
        LIMIT %s
        """
        rows = fetch_all(sql, (limit,))
    else:
        sql = """
        SELECT id, version_id, symbol, training_type, trigger_type,
               data_start, data_end, params_json, metrics_json,
               validation_deployed, validation_reason, holdout_metrics_json,
               duration_seconds, created_at
        FROM lstm_training_run
        ORDER BY created_at DESC
        LIMIT %s
        """
        rows = fetch_all(sql, (limit,))
    out = []
    for r in rows:
        d = dict(r)
        d["params"] = _json_loads(d.pop("params_json", None))
        d["metrics"] = _json_loads(d.pop("metrics_json", None))
        d["holdout_metrics"] = _json_loads(d.pop("holdout_metrics_json", None))
        d["data_start"] = _date_to_str(d.get("data_start"))
        d["data_end"] = _date_to_str(d.get("data_end"))
        d["created_at"] = d.get("created_at").isoformat() if d.get("created_at") else None
        out.append(d)
    return out


def dedupe_training_runs_keep_latest() -> int:
    """
    数据库去重：每只股票只保留最新一条训练流水，删除同 symbol 的旧记录。
    返回删除的行数。
    """
    sql_delete = """
    DELETE t FROM lstm_training_run t
    LEFT JOIN (
        SELECT symbol, MAX(id) AS max_id FROM lstm_training_run GROUP BY symbol
    ) latest ON t.symbol = latest.symbol AND t.id = latest.max_id
    WHERE latest.max_id IS NULL
    """
    return execute(sql_delete) or 0


def delete_training_runs_by_symbols(symbols: list[str]) -> int:
    """删除指定股票的全部训练流水，返回删除的行数。"""
    if not symbols:
        return 0
    placeholders = ", ".join(["%s"] * len(symbols))
    sql = f"DELETE FROM lstm_training_run WHERE symbol IN ({placeholders})"
    return execute(sql, tuple((s or "").strip() for s in symbols if (s or "").strip())) or 0


# ---------- 模型版本（替代本地 versions 目录） ----------


def insert_model_version(
    version_id: str,
    training_time: Optional[str],
    data_start: Optional[str],
    data_end: Optional[str],
    metadata: Optional[dict],
    model_bytes: bytes,
    symbol: str = "",
    years: int = 1,
) -> None:
    """将模型版本写入数据库（按股票+年份：version_id、symbol、years、元数据 JSON、权重 BLOB）。"""
    data_start_val = (data_start or "")[:10] if data_start else None
    data_end_val = (data_end or "")[:10] if data_end else None
    symbol_val = (symbol or "").strip() or ""
    years_val = 1 if years not in (1, 2, 3) else int(years)
    sql = """
    INSERT INTO lstm_model_version (version_id, symbol, years, training_time, data_start, data_end, metadata_json, model_blob)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    execute(sql, (
        (version_id or "").strip(),
        symbol_val,
        years_val,
        (training_time or "").strip() or None,
        data_start_val,
        data_end_val,
        _json_dumps(metadata),
        model_bytes,
    ))


def get_model_version_symbol_years(version_id: str) -> Optional[tuple[str, int]]:
    """从数据库读取指定版本的 symbol、years，不存在返回 None。返回 (symbol, years)。"""
    row = fetch_one(
        "SELECT symbol, years FROM lstm_model_version WHERE version_id = %s",
        ((version_id or "").strip(),),
    )
    if not row:
        return None
    sym = (row.get("symbol") or "").strip()
    y = row.get("years")
    if y is None:
        y = 1
    return (sym, int(y))


def get_model_version(version_id: str) -> Optional[dict[str, Any]]:
    """从数据库读取指定版本：返回 { metadata, model_blob (bytes) }，不存在返回 None。"""
    row = fetch_one(
        "SELECT metadata_json, model_blob FROM lstm_model_version WHERE version_id = %s",
        ((version_id or "").strip(),),
    )
    if not row or row.get("model_blob") is None:
        return None
    return {
        "metadata": _json_loads(row.get("metadata_json")),
        "model_blob": row["model_blob"],
    }


def get_version_date_range(version_id: str) -> Optional[tuple[str, str]]:
    """返回指定版本的训练数据日期范围 (data_start, data_end)，格式 YYYY-MM-DD。不存在返回 None。"""
    row = fetch_one(
        "SELECT data_start, data_end FROM lstm_model_version WHERE version_id = %s",
        ((version_id or "").strip(),),
    )
    if not row or (row.get("data_start") is None and row.get("data_end") is None):
        return None
    start = _date_to_str(row.get("data_start")) if row.get("data_start") else None
    end = _date_to_str(row.get("data_end")) if row.get("data_end") else None
    if not start or not end:
        return None
    return (start, end)


def list_model_versions_from_db(
    symbol: Optional[str] = None,
    years: Optional[int] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """列出模型版本（不含 model_blob），按 created_at 倒序。可按 symbol、years 筛选。"""
    if symbol is not None and years is not None:
        sql = """
        SELECT version_id, symbol, years, training_time, data_start, data_end, metadata_json, created_at
        FROM lstm_model_version
        WHERE symbol = %s AND years = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        rows = fetch_all(sql, ((symbol or "").strip(), int(years), limit))
    else:
        sql = """
        SELECT version_id, symbol, years, training_time, data_start, data_end, metadata_json, created_at
        FROM lstm_model_version
        ORDER BY created_at DESC
        LIMIT %s
        """
        rows = fetch_all(sql, (limit,))
    out = []
    for r in rows:
        meta = _json_loads(r.get("metadata_json"))
        out.append({
            "version_id": r.get("version_id"),
            "symbol": r.get("symbol"),
            "years": r.get("years"),
            "training_time": r.get("training_time"),
            "data_start": _date_to_str(r.get("data_start")),
            "data_end": _date_to_str(r.get("data_end")),
            "validation_score": meta.get("validation_score") if isinstance(meta, dict) else None,
            "metrics": meta.get("metrics") if isinstance(meta, dict) else None,
            "path": "db",
        })
    return out


def delete_model_version(version_id: str) -> None:
    """删除指定版本。"""
    execute("DELETE FROM lstm_model_version WHERE version_id = %s", ((version_id or "").strip(),))


# ---------- 当前版本（按股票+年份） ----------


def set_current_version_db(version_id: str, symbol: str = "", years: int = 1) -> None:
    """将指定 (symbol, years) 的当前版本写入 lstm_current_version_per_symbol 表。"""
    symbol_val = (symbol or "").strip() or ""
    years_val = 1 if years not in (1, 2, 3) else int(years)
    sql = """
    INSERT INTO lstm_current_version_per_symbol (symbol, years, version_id)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE version_id = VALUES(version_id)
    """
    execute(sql, (symbol_val, years_val, (version_id or "").strip(),))


def get_current_version_from_db(symbol: Optional[str] = None, years: Optional[int] = None) -> Optional[str]:
    """从数据库读取当前版本号。若提供 symbol 与 years，从 lstm_current_version_per_symbol 读；否则从旧表 lstm_current_version 读（兼容）。"""
    if symbol is not None and symbol != "" and years is not None and years in (1, 2, 3):
        row = fetch_one(
            "SELECT version_id FROM lstm_current_version_per_symbol WHERE symbol = %s AND years = %s",
            ((symbol or "").strip(), int(years),),
        )
        return (row.get("version_id") or "").strip() or None if row else None
    row = fetch_one("SELECT version_id FROM lstm_current_version WHERE id = 1")
    return (row.get("version_id") or "").strip() or None if row else None


def delete_current_version_for_symbols(symbols: list[str]) -> None:
    """删除指定股票在 lstm_current_version_per_symbol 中的全部当前版本记录（清理训练数据时用）。"""
    if not symbols:
        return
    placeholders = ", ".join(["%s"] * len(symbols))
    sql = f"DELETE FROM lstm_current_version_per_symbol WHERE symbol IN ({placeholders})"
    execute(sql, tuple((s or "").strip() for s in symbols if (s or "").strip()))


# ---------- 预测记录 ----------


def insert_prediction_log(
    symbol: str,
    predict_date: str,
    direction: int,
    magnitude: float,
    prob_up: float,
    model_version_id: Optional[str] = None,
    source: str = "lstm",
) -> None:
    """写入预测记录，同 (symbol, predict_date) 则 REPLACE。"""
    predict_date = _norm_date(predict_date) or (predict_date or "")[:10]
    sql = """
    INSERT INTO lstm_prediction_log (symbol, predict_date, direction, magnitude, prob_up, model_version_id, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE direction = VALUES(direction), magnitude = VALUES(magnitude), prob_up = VALUES(prob_up),
    model_version_id = VALUES(model_version_id), source = VALUES(source)
    """
    execute(sql, (
        (symbol or "").strip(),
        predict_date,
        int(direction),
        float(magnitude),
        float(prob_up),
        (model_version_id or "").strip() or None,
        (source or "lstm").strip()[:16],
    ))


def get_predictions_pending_accuracy_from_db(symbol: Optional[str] = None) -> list[dict[str, Any]]:
    """获取尚未回填准确性的预测记录（在 lstm_prediction_log 且不在 lstm_accuracy_record）。"""
    if symbol:
        sql = """
        SELECT p.symbol, p.predict_date, p.direction, p.magnitude, p.prob_up, p.model_version_id, p.source
        FROM lstm_prediction_log p
        LEFT JOIN lstm_accuracy_record a ON p.symbol = a.symbol AND p.predict_date = a.predict_date
        WHERE a.id IS NULL AND p.symbol = %s
        ORDER BY p.predict_date DESC
        """
        rows = fetch_all(sql, (symbol.strip(),))
    else:
        sql = """
        SELECT p.symbol, p.predict_date, p.direction, p.magnitude, p.prob_up, p.model_version_id, p.source
        FROM lstm_prediction_log p
        LEFT JOIN lstm_accuracy_record a ON p.symbol = a.symbol AND p.predict_date = a.predict_date
        WHERE a.id IS NULL
        ORDER BY p.predict_date DESC
        """
        rows = fetch_all(sql)
    return [dict(r) for r in rows]


def count_predictions_since_db(since_date: str) -> int:
    """统计某日及之后的预测条数（用于监控 7 日预测次数）。"""
    since_date = (since_date or "").strip()[:10]
    row = fetch_one("SELECT COUNT(*) AS n FROM lstm_prediction_log WHERE predict_date >= %s", (since_date,))
    return int(row["n"]) if row and row.get("n") is not None else 0


def _row_to_prediction_dict(row: dict) -> dict[str, Any]:
    """将预测记录行转为统一结构。"""
    return {
        "symbol": row.get("symbol"),
        "predict_date": _date_to_str(row.get("predict_date")),
        "direction": int(row.get("direction", 0)),
        "magnitude": float(row.get("magnitude", 0)),
        "prob_up": float(row.get("prob_up", 0.5)),
        "prob_down": round(1.0 - float(row.get("prob_up", 0.5)), 4),
        "direction_label": "涨" if int(row.get("direction", 0)) == 1 else "跌",
        "source": (row.get("source") or "lstm").strip(),
    }


def get_last_prediction_for_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """获取指定股票最近一次预测记录，用于刷新页面后恢复展示。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return None
    row = fetch_one(
        "SELECT symbol, predict_date, direction, magnitude, prob_up, model_version_id, source "
        "FROM lstm_prediction_log WHERE symbol = %s ORDER BY predict_date DESC LIMIT 1",
        (symbol,),
    )
    if not row:
        return None
    return _row_to_prediction_dict(row)


def get_all_last_predictions() -> list[dict[str, Any]]:
    """获取每只股票最近一次预测记录（用于按股票分别展示）。"""
    sql = """
    SELECT p.symbol, p.predict_date, p.direction, p.magnitude, p.prob_up, p.source
    FROM lstm_prediction_log p
    INNER JOIN (
        SELECT symbol, MAX(predict_date) AS md FROM lstm_prediction_log GROUP BY symbol
    ) q ON p.symbol = q.symbol AND p.predict_date = q.md
    ORDER BY p.symbol
    """
    rows = fetch_all(sql, ())
    return [_row_to_prediction_dict(dict(r)) for r in rows]


# ---------- 拟合曲线图（按股票，存 DB） ----------


def save_lstm_plot(symbol: str, plot_bytes: bytes) -> None:
    """保存或更新某股票的拟合曲线图（PNG 二进制）。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return
    sql = """
    INSERT INTO lstm_plot (symbol, plot_blob) VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE plot_blob = VALUES(plot_blob)
    """
    execute(sql, (symbol, plot_bytes))


def get_lstm_plot(symbol: str) -> Optional[bytes]:
    """按股票读取拟合曲线图 PNG 二进制，不存在返回 None。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return None
    row = fetch_one("SELECT plot_blob FROM lstm_plot WHERE symbol = %s", (symbol,))
    if not row or row.get("plot_blob") is None:
        return None
    blob = row["plot_blob"]
    return bytes(blob) if blob is not None else None


def delete_lstm_plot_for_symbol(symbol: str) -> None:
    """删除指定股票的拟合曲线图（lstm_plot 表）。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return
    execute("DELETE FROM lstm_plot WHERE symbol = %s", (symbol,))


# ---------- 拟合曲线图缓存（按股票+年份） ----------


def save_lstm_plot_cache(symbol: str, years: int, plot_bytes: bytes) -> None:
    """按 (symbol, years) 保存拟合曲线图缓存。years 应为 1/2/3。"""
    symbol = (symbol or "").strip()
    if not symbol or years not in (1, 2, 3):
        return
    sql = """
    INSERT INTO lstm_plot_cache (symbol, years, plot_blob) VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE plot_blob = VALUES(plot_blob), created_at = CURRENT_TIMESTAMP
    """
    execute(sql, (symbol, int(years), plot_bytes))


def get_lstm_plot_cache(symbol: str, years: int) -> Optional[bytes]:
    """按 (symbol, years) 读取拟合曲线图缓存，不存在返回 None。"""
    symbol = (symbol or "").strip()
    if not symbol or years not in (1, 2, 3):
        return None
    row = fetch_one(
        "SELECT plot_blob FROM lstm_plot_cache WHERE symbol = %s AND years = %s",
        (symbol, int(years),),
    )
    if not row or row.get("plot_blob") is None:
        return None
    blob = row["plot_blob"]
    return bytes(blob) if blob is not None else None


def delete_lstm_plot_cache_for_symbol(symbol: str) -> None:
    """删除指定股票在 lstm_plot_cache 中所有年份的缓存（清理训练数据时用）。"""
    symbol = (symbol or "").strip()
    if not symbol:
        return
    execute("DELETE FROM lstm_plot_cache WHERE symbol = %s", (symbol,))


# ---------- 准确性记录 ----------


def insert_accuracy_record(
    symbol: str,
    predict_date: str,
    actual_date: str,
    pred_direction: int,
    pred_magnitude: float,
    actual_direction: int,
    actual_magnitude: float,
    error_magnitude: float,
    direction_correct: int,
) -> None:
    """写入一条准确性回填记录，同 (symbol, predict_date) 则 REPLACE。"""
    pd_str = _norm_date(predict_date)
    ad_str = _norm_date(actual_date)
    sql = """
    INSERT INTO lstm_accuracy_record (
        symbol, predict_date, actual_date, pred_direction, pred_magnitude,
        actual_direction, actual_magnitude, error_magnitude, direction_correct
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE actual_date = VALUES(actual_date), pred_direction = VALUES(pred_direction),
    pred_magnitude = VALUES(pred_magnitude), actual_direction = VALUES(actual_direction),
    actual_magnitude = VALUES(actual_magnitude), error_magnitude = VALUES(error_magnitude),
    direction_correct = VALUES(direction_correct)
    """
    execute(sql, (
        (symbol or "").strip(),
        pd_str or predict_date[:10],
        ad_str or actual_date[:10],
        int(pred_direction), float(pred_magnitude), int(actual_direction), float(actual_magnitude),
        float(error_magnitude), int(direction_correct),
    ))


def get_accuracy_records_from_db(symbol: Optional[str] = None, limit: int = 500) -> list[dict[str, Any]]:
    """查询准确性记录，按 actual_date 倒序。"""
    if symbol:
        sql = """
        SELECT symbol, predict_date, actual_date, pred_direction, pred_magnitude,
               actual_direction, actual_magnitude, error_magnitude, direction_correct, created_at
        FROM lstm_accuracy_record WHERE symbol = %s ORDER BY actual_date DESC, predict_date DESC LIMIT %s
        """
        rows = fetch_all(sql, (symbol.strip(), limit))
    else:
        sql = """
        SELECT symbol, predict_date, actual_date, pred_direction, pred_magnitude,
               actual_direction, actual_magnitude, error_magnitude, direction_correct, created_at
        FROM lstm_accuracy_record ORDER BY actual_date DESC, predict_date DESC LIMIT %s
        """
        rows = fetch_all(sql, (limit,))
    out = []
    for r in rows:
        d = dict(r)
        d["predict_date"] = _date_to_str(d.get("predict_date"))
        d["actual_date"] = _date_to_str(d.get("actual_date"))
        out.append(d)
    return out


# ---------- 训练失败 ----------


def insert_training_failure_db(symbol: str, error_message: str) -> None:
    """记录一次训练失败。"""
    execute(
        "INSERT INTO lstm_training_failure (symbol, error_message) VALUES (%s, %s)",
        ((symbol or "").strip(), (error_message or "").strip()[:65535]),
    )


def get_training_failure_count_db() -> int:
    """返回训练失败记录条数（用于告警）。"""
    row = fetch_one("SELECT COUNT(*) AS n FROM lstm_training_failure")
    return int(row["n"]) if row and row.get("n") is not None else 0
