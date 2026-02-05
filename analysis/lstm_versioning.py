# -*- coding: utf-8 -*-
"""
LSTM 模型版本管理与预测准确性记录。

- 版本化保存：训练时间、数据范围、验证分数，只保留最新 1 个版本
- 预测日志：每次预测写入一条记录
- 准确性记录：有实际数据后回填误差，用于性能衰减触发
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from analysis.lstm_constants import DEFAULT_MODEL_DIR, FORECAST_DAYS

MAX_VERSIONS = 1  # 只保留最新一个版本
VERSIONS_DIR_NAME = "versions"
CURRENT_VERSION_FILE = "current_version.json"
PREDICTION_LOG_FILE = "prediction_log.json"
ACCURACY_RECORDS_FILE = "accuracy_records.json"


def _base_dir(save_dir: Optional[Path | str] = None) -> Path:
    return Path(save_dir or DEFAULT_MODEL_DIR or "")


def _versions_dir(base: Path) -> Path:
    return base / VERSIONS_DIR_NAME


def _get_version_metadata_path(version_dir: Path) -> Path:
    return version_dir / "lstm_metadata.json"


def _get_model_path(version_dir: Path) -> Path:
    return version_dir / "lstm_model.pt"


def _version_id_from_dir_name(name: str) -> str:
    return name


def list_versions(save_dir: Optional[Path | str] = None) -> list[dict[str, Any]]:
    """
    列出已保存的版本（从数据库读取，按 created_at 倒序）。
    每项含 version_id, training_time, data_start, data_end, validation_score, path。
    """
    try:
        from data.lstm_repo import list_model_versions_from_db
        return list_model_versions_from_db(limit=MAX_VERSIONS * 2)
    except Exception:
        return []


def get_current_version_id(save_dir: Optional[Path | str] = None) -> Optional[str]:
    """返回当前使用的版本 ID（从数据库读取）。"""
    try:
        from data.lstm_repo import get_current_version_from_db
        return get_current_version_from_db()
    except Exception:
        return None


def get_current_version_path(save_dir: Optional[Path | str] = None) -> Optional[Path]:
    """当前使用数据库存储时恒为 None；保留接口兼容。"""
    return None


def _get_version_from_db(version_id: str) -> Optional[tuple[Any, dict]]:
    """从数据库加载指定版本的 state_dict 与 metadata，返回 (state_dict, metadata) 或 None。"""
    try:
        from data.lstm_repo import get_model_version
        import torch
        import io
        row = get_model_version(version_id)
        if not row or not row.get("model_blob"):
            return None
        buf = io.BytesIO(row["model_blob"])
        ckpt = torch.load(buf, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict")
        meta = row.get("metadata") or ckpt.get("metadata")
        if state_dict is None or meta is None:
            return None
        return state_dict, meta
    except Exception:
        return None


def get_current_model_from_db(save_dir: Optional[Path | str] = None) -> Optional[tuple[Any, dict]]:
    """从数据库加载当前版本的 state_dict 与 metadata，返回 (state_dict, metadata) 或 None。"""
    vid = get_current_version_id(save_dir)
    if not vid:
        return None
    return _get_version_from_db(vid)


def set_current_version(version_id: str, save_dir: Optional[Path | str] = None) -> None:
    """将指定版本设为当前（仅写数据库，需确保该版本已存在于 lstm_model_version）。"""
    try:
        from data.lstm_repo import get_model_version, set_current_version_db
        if get_model_version(version_id) is None:
            raise FileNotFoundError(f"版本不存在: {version_id}")
        set_current_version_db(version_id)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(f"版本不存在或数据库异常: {version_id}") from e


def _prune_versions(base: Path, keep: int = MAX_VERSIONS) -> None:
    """保留最近 keep 个版本（默认 1），从数据库删除更旧版本。"""
    versions = list_versions(base)
    if len(versions) <= keep:
        return
    try:
        from data.lstm_repo import delete_model_version, set_current_version_db
        for v in versions[keep:]:
            delete_model_version(v["version_id"])
        current = get_current_version_id(base)
        remaining = list_versions(base)
        if current and not any(r["version_id"] == current for r in remaining) and remaining:
            set_current_version_db(remaining[0]["version_id"])
    except Exception:
        pass


def save_versioned_model(
    save_dir: Path | str,
    state_dict: Any,
    metadata: dict[str, Any],
    *,
    training_time: Optional[str] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
    validation_score: Optional[dict[str, float] | float] = None,
    promote_to_current: bool = True,
) -> str:
    """
    将模型与元数据保存到数据库（lstm_model_version 表）。
    若 promote_to_current=True，更新当前版本并裁剪至只保留最新 1 个版本。
    返回 version_id。
    """
    import torch
    import io
    from data.lstm_repo import insert_model_version, set_current_version_db

    base = _base_dir(save_dir)
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = dict(metadata)
    meta["training_time"] = training_time or datetime.now().isoformat()
    meta["data_start"] = data_start
    meta["data_end"] = data_end
    meta["validation_score"] = validation_score

    buf = io.BytesIO()
    torch.save({"state_dict": state_dict, "metadata": meta}, buf)
    model_bytes = buf.getvalue()

    insert_model_version(
        version_id=version_id,
        training_time=meta.get("training_time"),
        data_start=data_start,
        data_end=data_end,
        metadata=meta,
        model_bytes=model_bytes,
    )

    if promote_to_current:
        set_current_version_db(version_id)
        _prune_versions(base, MAX_VERSIONS)
    return version_id


def remove_version(version_id: str, save_dir: Optional[Path | str] = None) -> None:
    """从数据库删除指定版本。"""
    try:
        from data.lstm_repo import delete_model_version
        delete_model_version(version_id)
    except Exception:
        pass


# ---------- 预测日志与准确性记录 ----------


def _load_json_list(base: Path, filename: str) -> list[dict[str, Any]]:
    path = base / filename
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_json_list(base: Path, filename: str, data: list[dict[str, Any]]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    path = base / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record_prediction(
    symbol: str,
    predict_date: str,
    direction: int,
    magnitude: float,
    prob_up: float,
    model_version_id: Optional[str] = None,
    save_dir: Optional[Path | str] = None,
    source: str = "lstm",
) -> None:
    """记录一次预测，用于后续回填准确性。predict_date 格式 YYYY-MM-DD 或 YYYYMMDD。"""
    base = _base_dir(save_dir)
    records = _load_json_list(base, PREDICTION_LOG_FILE)
    rec = {
        "symbol": symbol,
        "predict_date": _normalize_date(predict_date),
        "direction": direction,
        "magnitude": magnitude,
        "prob_up": prob_up,
        "model_version_id": model_version_id or get_current_version_id(base),
        "recorded_at": datetime.now().isoformat(),
    }
    records.append(rec)
    _save_json_list(base, PREDICTION_LOG_FILE, records)
    try:
        from data.lstm_repo import insert_prediction_log
        insert_prediction_log(
            symbol=symbol,
            predict_date=rec["predict_date"],
            direction=direction,
            magnitude=magnitude,
            prob_up=prob_up,
            model_version_id=rec.get("model_version_id"),
            source=(source or "lstm").strip()[:16],
        )
    except Exception:
        pass


def _normalize_date(s: str) -> str:
    s = (s or "").strip().replace("-", "")[:8]
    if len(s) == 8:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def get_accuracy_records(save_dir: Optional[Path | str] = None) -> list[dict[str, Any]]:
    """返回已回填的准确性记录列表；优先从 MySQL 读取，无则从本地 JSON 读取。"""
    try:
        from data.lstm_repo import get_accuracy_records_from_db
        rows = get_accuracy_records_from_db(symbol=None, limit=2000)
        if rows:
            return rows
    except Exception:
        pass
    return _load_json_list(_base_dir(save_dir), ACCURACY_RECORDS_FILE)


def get_predictions_pending_accuracy(save_dir: Optional[Path | str] = None) -> list[dict[str, Any]]:
    """返回尚未回填准确性的预测记录；优先从 MySQL 读取。"""
    try:
        from data.lstm_repo import get_predictions_pending_accuracy_from_db
        rows = get_predictions_pending_accuracy_from_db(symbol=None)
        if rows is not None:
            return [{"symbol": r["symbol"], "predict_date": r.get("predict_date") or "", "direction": r.get("direction"), "magnitude": r.get("magnitude"), "prob_up": r.get("prob_up"), "model_version_id": r.get("model_version_id")} for r in rows]
    except Exception:
        pass
    base = _base_dir(save_dir)
    predictions = _load_json_list(base, PREDICTION_LOG_FILE)
    accuracy = _load_json_list(base, ACCURACY_RECORDS_FILE)
    filled = {(r["symbol"], r["predict_date"]) for r in accuracy}
    return [p for p in predictions if (p["symbol"], p["predict_date"]) not in filled]


def update_accuracy_for_symbol(
    symbol: str,
    as_of_date: str,
    fetch_hist_fn: Callable[[str, str, str], Any],
    save_dir: Optional[Path | str] = None,
) -> int:
    """
    根据 as_of_date 之前的实际行情，对「预测日 + FORECAST_DAYS <= as_of_date」的预测回填准确性。
    fetch_hist_fn(symbol, start_date, end_date) 返回 DataFrame，需含日期与收盘。
    返回本次新增的准确性记录条数。
    """
    base = _base_dir(save_dir)
    pending = [
        p for p in get_predictions_pending_accuracy(base)
        if p["symbol"] == symbol
    ]
    if not pending:
        return 0
    as_of = _normalize_date(as_of_date).replace("-", "")
    date_col = "日期"
    # 按预测日分组，只处理「预测日 + 5 个交易日 <= as_of」的
    from datetime import datetime as dt, timedelta
    def parse_d(s: str) -> dt:
        s = s.replace("-", "")[:8]
        return dt.strptime(s, "%Y%m%d") if len(s) == 8 else dt.min
    new_records = []
    min_start = None
    max_end = None
    for p in pending:
        pd_str = p["predict_date"].replace("-", "")
        pd_dt = parse_d(p["predict_date"])
        # 实际到期日约等于预测日 + 5 个交易日（近似为 +7 日历日）
        end_dt = pd_dt + timedelta(days=7)
        end_str = end_dt.strftime("%Y%m%d")
        if end_str > as_of:
            continue
        if min_start is None or pd_str < min_start:
            min_start = pd_str
        if max_end is None or end_str > max_end:
            max_end = end_str
        new_records.append((p, pd_str, end_str))
    if not new_records:
        return 0
    start_range = (min_start or "")[:8]
    end_range = (as_of or "").replace("-", "")[:8]  # 拉到 as_of 确保包含所有实际到期日
    df = fetch_hist_fn(symbol, start_range, end_range)
    if df is None or df.empty:
        return 0
    if date_col not in df.columns:
        date_col = df.columns[0]
    close_col = "收盘" if "收盘" in df.columns else ("close" if "close" in df.columns else df.columns[1])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = df[date_col].astype(str).str.replace("-", "").str[:8]
    close_series = df[close_col] if close_col in df.columns else df.iloc[:, 1]
    close_series = close_series.astype(float)
    dates = df[date_col].tolist()
    n_new = 0
    accuracy_list = get_accuracy_records(base)
    for p, pd_str, end_str in new_records:
        # 找预测日对应的行和预测日+5 的收盘
        try:
            idx_pd = dates.index(pd_str) if pd_str in dates else None
            if idx_pd is None:
                continue
            idx_end = min(idx_pd + FORECAST_DAYS, len(dates) - 1)
            actual_end_date = dates[idx_end]
            close_start = float(close_series.iloc[idx_pd])
            close_end = float(close_series.iloc[idx_end])
            actual_magnitude = (close_end / close_start) - 1.0
            actual_direction = 1 if actual_magnitude > 0 else 0
            pred_mag = float(p.get("magnitude", 0))
            pred_dir = int(p.get("direction", 0))
            error_magnitude = abs(actual_magnitude - pred_mag)
            direction_correct = 1 if pred_dir == actual_direction else 0
        except Exception:
            continue
        actual_date_str = actual_end_date if len(actual_end_date) == 8 else f"{actual_end_date[:4]}-{actual_end_date[4:6]}-{actual_end_date[6:8]}"
        accuracy_list.append({
            "symbol": symbol,
            "predict_date": p["predict_date"],
            "actual_date": actual_date_str,
            "pred_direction": pred_dir,
            "pred_magnitude": pred_mag,
            "actual_direction": actual_direction,
            "actual_magnitude": actual_magnitude,
            "error_magnitude": error_magnitude,
            "direction_correct": direction_correct,
        })
        try:
            from data.lstm_repo import insert_accuracy_record
            insert_accuracy_record(
                symbol=symbol,
                predict_date=p["predict_date"],
                actual_date=actual_date_str,
                pred_direction=pred_dir,
                pred_magnitude=pred_mag,
                actual_direction=actual_direction,
                actual_magnitude=actual_magnitude,
                error_magnitude=error_magnitude,
                direction_correct=direction_correct,
            )
        except Exception:
            pass
        n_new += 1
    if n_new > 0:
        _save_json_list(base, ACCURACY_RECORDS_FILE, accuracy_list)
    return n_new


def get_recent_accuracy_for_trigger(
    n_trading_days: int = 20,
    save_dir: Optional[Path | str] = None,
) -> tuple[float, float]:
    """
    返回 (最近 n 个交易日的平均 error_magnitude, 历史平均 error_magnitude)。
    若记录不足则返回 (0.0, 0.0)。
    """
    records = get_accuracy_records(save_dir)
    if not records:
        return 0.0, 0.0
    all_errors = [float(r["error_magnitude"]) for r in records if "error_magnitude" in r]
    if not all_errors:
        return 0.0, 0.0
    historical_avg = sum(all_errors) / len(all_errors)
    # 按 actual_date 或 predict_date 取最近 n 条
    with_date = [(r.get("actual_date") or r.get("predict_date") or "", r) for r in records]
    with_date.sort(key=lambda x: x[0], reverse=True)
    recent = with_date[:n_trading_days]
    recent_errors = [float(r["error_magnitude"]) for _, r in recent if "error_magnitude" in r]
    if not recent_errors:
        return 0.0, historical_avg
    recent_avg = sum(recent_errors) / len(recent_errors)
    return recent_avg, historical_avg
