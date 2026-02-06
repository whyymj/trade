# -*- coding: utf-8 -*-
"""
LSTM 深度学习模型：多步预测与可解释性

- 输入：过去 60 个交易日的 [收盘价、成交量、技术指标]
- 输出：未来 5 日价格方向（分类）与涨跌幅（回归）
- 架构：Input -> LSTM -> Dropout -> 全连接（双头：分类 + 回归）
- 支持交叉验证、超参数优化、SHAP/特征重要性、模型保存与加载、评估指标与可视化
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# 技术指标在 technical 模块
from analysis.technical import (
    calc_bollinger_bands,
    calc_macd,
    calc_mfi,
    calc_obv,
    calc_rsi,
    calc_rolling_volatility,
)

try:
    from analysis.lstm_versioning import (
        get_current_version_id,
        get_current_version_path,
        get_current_model_from_db,
        _get_version_from_db,
        remove_version,
        save_versioned_model,
        set_current_version,
        _prune_versions,
        _versions_dir,
        MAX_VERSIONS,
    )
except ImportError:
    get_current_version_path = None  # type: ignore[assignment]
    get_current_model_from_db = None  # type: ignore[assignment]
    _get_version_from_db = None  # type: ignore[assignment]
    save_versioned_model = None  # type: ignore[assignment]
    get_current_version_id = None
    set_current_version = None
    remove_version = None
    _prune_versions = None
    _versions_dir = None
    MAX_VERSIONS = 1

try:
    from analysis.lstm_validation import (
        evaluate_model_on_holdout,
        should_deploy_new_model,
    )
except ImportError:
    evaluate_model_on_holdout = None  # type: ignore[assignment]
    should_deploy_new_model = None  # type: ignore[assignment]

# 延迟导入 PyTorch，便于在无 GPU 环境下跳过
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    TimeSeriesSplit = None  # type: ignore[assignment]
    accuracy_score = None  # type: ignore[assignment]
    recall_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    shap = None  # type: ignore[assignment]

from analysis.lstm_constants import DEFAULT_MODEL_DIR, FORECAST_DAYS

SEQ_LEN = 60

# 本机/CPU 友好预设：减少超参组合与 epoch，缩短单次训练时间（约 2～8 分钟/次，视样本量而定）
# 默认网格：lr/hidden/epochs 组合多、5 折 CV，CPU 上完整训练可能需 20～60 分钟
CPU_FRIENDLY_PARAM_GRID = {
    "lr": [5e-4],
    "hidden_size": [32],
    "epochs": [25],
    "batch_size": 32,
}
CPU_FRIENDLY_CV_SPLITS = 3
DEFAULT_FEATURE_NAMES = [
    "close_norm",
    "volume_norm",
    "rsi",
    "macd_hist",
    "bb_position",
    "volatility",
    "obv_norm",
    "mfi",
    "aroon_up",
]


def _ensure_torch():
    if not _TORCH_AVAILABLE:
        raise RuntimeError("LSTM 模块需要 PyTorch，请安装: pip install torch")
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("LSTM 模块需要 scikit-learn，请安装: pip install scikit-learn")


def build_features_from_df(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Series, np.ndarray, np.ndarray]:
    """
    从日线 DataFrame 构建 LSTM 特征矩阵与目标。
    - 列名支持中文：收盘、成交量、最高、最低、开盘 等。
    - 返回: (X, feature_names, y_info, y_direction, y_magnitude)
      X: (n_samples, SEQ_LEN, n_features)
      y_direction: 0/1，y_magnitude: 未来5日涨跌幅
    """
    close_col = "收盘" if "收盘" in df.columns else "close"
    vol_col = "成交量" if "成交量" in df.columns else "volume"
    high_col = "最高" if "最高" in df.columns else "high"
    low_col = "最低" if "最低" in df.columns else "low"
    open_col = "开盘" if "开盘" in df.columns else "open"

    def _to_series(s: pd.Series | pd.DataFrame) -> pd.Series:
        if isinstance(s, pd.DataFrame):
            out = s.squeeze(axis=1)
            assert isinstance(out, pd.Series)
            return out
        return s

    close = _to_series(df[close_col].astype(float))
    volume = _to_series(df[vol_col].astype(float))
    high = _to_series(df[high_col].astype(float))
    low = _to_series(df[low_col].astype(float))

    # 技术指标
    rsi = calc_rsi(close, period=14)
    macd = calc_macd(close, 12, 26, 9)
    bb = calc_bollinger_bands(close, period=20, num_std=2.0)
    returns = close.pct_change()
    vol = calc_rolling_volatility(returns, window=20, annualize=True)
    obv = calc_obv(close, volume)
    mfi = calc_mfi(high, low, close, volume, period=14)
    try:
        from analysis.technical import calc_aroon
        aroon = calc_aroon(high, low, period=20)
        aroon_up = aroon["aroon_up"]
    except Exception:
        aroon_up = pd.Series(np.nan, index=close.index)

    # 标准化与组合
    close_norm = (close - close.rolling(SEQ_LEN, min_periods=1).min()) / (
        close.rolling(SEQ_LEN, min_periods=1).max() - close.rolling(SEQ_LEN, min_periods=1).min() + 1e-8
    )
    vol_min = volume.rolling(SEQ_LEN, min_periods=1).min()
    vol_max = volume.rolling(SEQ_LEN, min_periods=1).max()
    volume_norm = (volume - vol_min) / (vol_max - vol_min + 1e-8)
    bb_mid, bb_upper, bb_lower = bb["mid"], bb["upper"], bb["lower"]
    bb_position = (close - bb_mid) / (bb_upper - bb_lower + 1e-8)
    obv_min = obv.rolling(SEQ_LEN, min_periods=1).min()
    obv_max = obv.rolling(SEQ_LEN, min_periods=1).max()
    obv_norm = (obv - obv_min) / (obv_max - obv_min + 1e-8)
    # 填充 NaN
    rsi = rsi.fillna(50)
    macd_hist = macd["hist"].fillna(0)
    bb_position = bb_position.fillna(0)
    vol = vol.fillna(0)
    obv_norm = obv_norm.fillna(0.5)
    mfi = mfi.fillna(50)
    aroon_up = aroon_up.fillna(50)

    n = len(df)
    need = SEQ_LEN + FORECAST_DAYS
    if n < need:
        return (
            np.zeros((0, SEQ_LEN, len(DEFAULT_FEATURE_NAMES))),
            DEFAULT_FEATURE_NAMES.copy(),
            pd.Series(dtype=object),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )

    _macd = np.asarray(macd_hist.values, dtype=np.float64)
    _bb = np.asarray(bb_position.values, dtype=np.float64)
    _vol = np.asarray(vol.values, dtype=np.float64)
    feature_list = [
        close_norm.values,
        volume_norm.values,
        np.asarray(rsi.values, dtype=np.float64) / 100.0,
        np.clip(_macd, -1e2, 1e2) / (float(np.nanmax(np.abs(_macd))) + 1e-8),
        np.clip(_bb, -2, 2) / 2.0,
        np.clip(_vol, 0, 2) / 2.0,
        obv_norm.values,
        np.asarray(mfi.values, dtype=np.float64) / 100.0,
        np.asarray(aroon_up.values, dtype=np.float64) / 100.0,
    ]
    # (n_features, T)
    F = np.stack(feature_list, axis=1)

    X_list = []
    y_direction_list = []
    y_magnitude_list = []
    end_dates = []

    for i in range(SEQ_LEN, n - FORECAST_DAYS):
        X_list.append(F[i - SEQ_LEN : i])  # (SEQ_LEN, n_features)
        future_ret = (close.iloc[i + FORECAST_DAYS - 1] / close.iloc[i - 1]) - 1.0
        y_direction_list.append(1 if future_ret > 0 else 0)
        y_magnitude_list.append(future_ret)
        date_val = df["日期"].iloc[i] if "日期" in df.columns else (df.index[i] if hasattr(df.index[i], "strftime") else i)
        end_dates.append(date_val)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_dir = np.array(y_direction_list, dtype=np.int64)
    y_mag = np.array(y_magnitude_list, dtype=np.float32)

    y_info = pd.Series(
        [
            {"direction": int(d), "magnitude": float(m), "end_date": str(e)}
            for d, m, e in zip(y_direction_list, y_magnitude_list, end_dates)
        ],
        index=end_dates,
    )
    return X, DEFAULT_FEATURE_NAMES.copy(), y_info, y_dir, y_mag


if _TORCH_AVAILABLE and nn is not None and torch is not None:

    class LSTMDualHead(nn.Module):
        """LSTM + Dropout + 全连接双头（分类 + 回归）。"""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.3,
            num_classes: int = 2,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.dropout = nn.Dropout(p=dropout)
            self.fc_shared = nn.Linear(hidden_size, hidden_size)
            self.fc_direction = nn.Linear(hidden_size, num_classes)
            self.fc_magnitude = nn.Linear(hidden_size, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            last_h = lstm_out[:, -1]
            h = self.dropout(torch.relu(self.fc_shared(last_h)))
            direction_logits = self.fc_direction(h)
            magnitude = self.fc_magnitude(h).squeeze(-1)
            return direction_logits, magnitude
else:
    LSTMDualHead = None  # type: ignore[misc, assignment]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight_cls: float = 1.0,
    weight_reg: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    criterion_cls = nn.CrossEntropyLoss()
    for X_b, y_dir_b, y_mag_b in loader:
        X_b = X_b.to(device)
        y_dir_b = y_dir_b.to(device)
        y_mag_b = y_mag_b.to(device)
        optimizer.zero_grad()
        logits, mag_pred = model(X_b)
        loss_cls = criterion_cls(logits, y_dir_b)
        loss_reg = nn.functional.mse_loss(mag_pred, y_mag_b)
        loss = weight_cls * loss_cls + weight_reg * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_dir_true, all_dir_pred = [], []
    all_mag_true, all_mag_pred = [], []
    with torch.no_grad():
        for X_b, y_dir_b, y_mag_b in loader:
            X_b = X_b.to(device)
            logits, mag_pred = model(X_b)
            all_dir_true.extend(y_dir_b.numpy().tolist())
            all_dir_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_mag_true.extend(y_mag_b.numpy().tolist())
            all_mag_pred.extend(mag_pred.cpu().numpy().tolist())
    all_dir_true = np.array(all_dir_true)
    all_dir_pred = np.array(all_dir_pred)
    all_mag_true = np.array(all_mag_true)
    all_mag_pred = np.array(all_mag_pred)

    acc = accuracy_score(all_dir_true, all_dir_pred)
    rec = recall_score(all_dir_true, all_dir_pred, average="binary", zero_division=0)
    f1 = f1_score(all_dir_true, all_dir_pred, average="binary", zero_division=0)
    mse = mean_squared_error(all_mag_true, all_mag_pred)
    return {
        "accuracy": float(acc),
        "recall": float(rec),
        "f1": float(f1),
        "mse": float(mse),
    }


def cross_validate_and_tune(
    X: np.ndarray,
    y_direction: np.ndarray,
    y_magnitude: np.ndarray,
    feature_names: list[str],
    n_splits: int = 5,
    param_grid: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    时间序列交叉验证 + 超参数搜索（学习率、隐藏层大小、epochs）。
    返回每折指标与最佳超参及对应模型在全体数据上的复现结果（仅汇报用，不返回模型）。
    """
    _ensure_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)

    param_grid = param_grid or {}
    lr_list = param_grid.get("lr", [1e-3, 5e-4])
    hidden_list = param_grid.get("hidden_size", [32, 64])
    epochs_list = param_grid.get("epochs", [30, 50])
    batch_size = param_grid.get("batch_size", 32)

    n_features = X.shape[2]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    cv_results: list[dict[str, Any]] = []

    for lr in lr_list:
        for hidden in hidden_list:
            for epochs in epochs_list:
                fold_metrics: list[dict[str, float]] = []
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_dir_tr, y_dir_val = y_direction[train_idx], y_direction[val_idx]
                    y_mag_tr, y_mag_val = y_magnitude[train_idx], y_magnitude[val_idx]

                    train_ds = TensorDataset(
                        torch.from_numpy(X_tr),
                        torch.from_numpy(y_dir_tr),
                        torch.from_numpy(y_mag_tr),
                    )
                    val_ds = TensorDataset(
                        torch.from_numpy(X_val),
                        torch.from_numpy(y_dir_val),
                        torch.from_numpy(y_mag_val),
                    )
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_ds, batch_size=batch_size)

                    model = LSTMDualHead(
                        input_size=n_features,
                        hidden_size=hidden,
                        num_layers=1,
                        dropout=0.3,
                        num_classes=2,
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    for _ in range(epochs):
                        train_epoch(model, train_loader, optimizer, device)
                    metrics = evaluate(model, val_loader, device)
                    fold_metrics.append(metrics)

                avg_f1 = np.mean([m["f1"] for m in fold_metrics])
                avg_mse = np.mean([m["mse"] for m in fold_metrics])
                score = avg_f1 - 0.1 * np.log1p(avg_mse)
                cv_results.append({
                    "params": {"lr": lr, "hidden_size": hidden, "epochs": epochs},
                    "fold_metrics": fold_metrics,
                    "avg_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
                    "avg_recall": float(np.mean([m["recall"] for m in fold_metrics])),
                    "avg_f1": float(avg_f1),
                    "avg_mse": float(avg_mse),
                    "score": float(score),
                })
                if score > best_score:
                    best_score = score
                    best_params = {"lr": lr, "hidden_size": hidden, "epochs": epochs}

    return {
        "cv_results": cv_results,
        "best_params": best_params,
        "best_score": float(best_score),
        "n_splits": n_splits,
    }


def train_and_save(
    X: np.ndarray,
    y_direction: np.ndarray,
    y_magnitude: np.ndarray,
    feature_names: list[str],
    save_dir: Optional[os.PathLike | str] = None,
    *,
    lr: float = 5e-4,
    hidden_size: int = 64,
    epochs: int = 50,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    seed: int = 42,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
    validation_score: Optional[dict[str, float] | float] = None,
    use_versioning: bool = True,
    promote_to_current: bool = True,
    symbol: str = "",
    years: int = 1,
) -> tuple[nn.Module, dict[str, Any], dict[str, float]]:
    """
    在全部数据上训练并保存模型与元数据。
    若 use_versioning 为 True（默认），则写入数据库并只保留最新 1 个版本。
    返回: (model, metadata, final_metrics)；metadata 中含 version_id（当使用版本化时）。
    """
    _ensure_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir or DEFAULT_MODEL_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_features = X.shape[2]
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_direction),
        torch.from_numpy(y_magnitude),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMDualHead(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        num_classes=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device)

    final_metrics = evaluate(model, DataLoader(dataset, batch_size=batch_size), device)
    model.eval()
    with torch.no_grad():
        logits, mag_pred = model(torch.from_numpy(X).to(device))
        last_dir_pred = logits.argmax(dim=1).cpu().numpy()
        last_mag_pred = mag_pred.cpu().numpy()
    metadata = {
        "feature_names": feature_names,
        "seq_len": SEQ_LEN,
        "forecast_days": FORECAST_DAYS,
        "hidden_size": hidden_size,
        "n_features": n_features,
        "lr": lr,
        "epochs": epochs,
        "metrics": final_metrics,
        "last_dir_pred": last_dir_pred.tolist(),
        "last_mag_pred": last_mag_pred.tolist(),
    }
    if save_versioned_model is not None:
        version_id = save_versioned_model(
            save_dir,
            model.cpu().state_dict(),
            metadata,
            training_time=None,
            data_start=data_start,
            data_end=data_end,
            validation_score=validation_score or final_metrics,
            promote_to_current=promote_to_current if use_versioning else False,
            symbol=(symbol or "").strip(),
            years=1 if years not in (1, 2, 3) else int(years),
        )
        metadata["version_id"] = version_id
        model.to(device)
        return model, metadata, final_metrics
    raise RuntimeError("LSTM 模型需写入数据库，请确保 data.lstm_repo 可用并已执行 create_lstm_tables。")


def load_model(
    save_dir: Optional[os.PathLike | str] = None,
    device: Optional[torch.device] = None,
    version_id: Optional[str] = None,
    symbol: Optional[str] = None,
    years: Optional[int] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """加载已保存的 LSTM 模型与元数据。若提供 symbol、years 则加载该股票该年份的当前模型；否则按 version_id 或全局当前版本。"""
    _ensure_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, metadata = None, None
    if get_current_model_from_db is not None and _get_version_from_db is not None:
        if version_id:
            pair = _get_version_from_db(version_id)
        else:
            pair = get_current_model_from_db(save_dir, symbol=symbol, years=years)
        if pair is not None:
            state_dict, metadata = pair

    if state_dict is None or metadata is None:
        raise FileNotFoundError(
            "未找到已保存的模型，请先完成一次 LSTM 训练。模型存于数据库 lstm_model_version 表。"
        )

    n_features = metadata["n_features"]
    hidden_size = metadata["hidden_size"]
    model = LSTMDualHead(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.3,
        num_classes=2,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, metadata


def incremental_train_and_save(
    df: pd.DataFrame,
    symbol: str = "",
    save_dir: Optional[os.PathLike | str] = None,
    *,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """
    增量训练：加载当前模型，仅用新数据微调若干 epoch，保存为新版本。
    用于周度增量更新，无需交叉验证，速度快。
    返回: { "version_id", "metrics", "n_samples", "data_start", "data_end" } 或 {"error": "..."}。
    """
    _ensure_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir or DEFAULT_MODEL_DIR)
    try:
        model, metadata = load_model(save_dir=save_dir, device=device)
    except FileNotFoundError as e:
        return {"error": f"无现有模型可做增量训练: {e}"}

    X, feature_names, y_info, y_dir, y_mag = build_features_from_df(df)
    if len(X) == 0:
        return {"error": "样本不足，需要至少 65 个交易日数据"}

    date_col = "日期" if "日期" in df.columns else df.columns[0]
    dates = df[date_col].astype(str).tolist()
    data_start = str(dates[0])[:10] if dates else None
    data_end = str(dates[-1])[:10] if dates else None

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_dir),
        torch.from_numpy(y_mag),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device)

    final_metrics = evaluate(model, DataLoader(dataset, batch_size=batch_size), device)
    model.eval()
    with torch.no_grad():
        logits, mag_pred = model(torch.from_numpy(X).to(device))
        last_dir_pred = logits.argmax(dim=1).cpu().numpy()
        last_mag_pred = mag_pred.cpu().numpy()

    meta = {
        "feature_names": metadata.get("feature_names", feature_names),
        "seq_len": metadata.get("seq_len", SEQ_LEN),
        "forecast_days": metadata.get("forecast_days", FORECAST_DAYS),
        "hidden_size": metadata.get("hidden_size", 64),
        "n_features": int(X.shape[2]),
        "lr": lr,
        "epochs": epochs,
        "metrics": final_metrics,
        "last_dir_pred": last_dir_pred.tolist(),
        "last_mag_pred": last_mag_pred.tolist(),
        "training_type": "incremental",
    }
    version_id = None
    if save_versioned_model is not None:
        version_id = save_versioned_model(
            save_dir,
            model.cpu().state_dict(),
            meta,
            training_time=None,
            data_start=data_start,
            data_end=data_end,
            validation_score=final_metrics,
        )
        model.to(device)

    # 拟合曲线图改存数据库，见 run_lstm_pipeline 的 do_plot 与 generate_fit_plot_for_symbol
    return {
        "version_id": version_id,
        "metrics": final_metrics,
        "n_samples": int(X.shape[0]),
        "data_start": data_start,
        "data_end": data_end,
        "training_type": "incremental",
    }


def compute_feature_importance_and_shap(
    model: nn.Module,
    X: np.ndarray,
    feature_names: list[str],
    device: torch.device,
    n_background: int = 100,
    n_explain: int = 200,
) -> dict[str, Any]:
    """
    特征重要性（基于梯度 L2 范数）与 SHAP（若可用）。
    返回 feature_importance 与可选的 shap_values / shap_summary_plot_path。
    """
    _ensure_torch()
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    X_t.requires_grad = True
    logits, mag = model(X_t)
    # 用分类头梯度作为特征重要性
    (logits.sum() + mag.sum()).backward()
    grad = X_t.grad.cpu().numpy()
    # (n_samples, seq_len, n_features) -> 按特征聚合
    importance = np.sqrt(np.mean(grad ** 2, axis=(0, 1)))
    importance = importance / (importance.sum() + 1e-8)
    feature_importance = dict(zip(feature_names, importance.tolist()))

    out: dict[str, Any] = {"feature_importance": feature_importance}
    if not _SHAP_AVAILABLE or X.shape[0] < n_background:
        return out

    try:
        background = X[: min(n_background, len(X))]
        to_explain = X[: min(n_explain, len(X))]
        background_t = torch.from_numpy(background).float().to(device)

        class _ModelWrap(nn.Module):
            def __init__(self, m: nn.Module):
                super().__init__()
                self._m = m
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                logits, _ = self._m(x)
                return logits

        wrapped = _ModelWrap(model)
        explainer = shap.DeepExplainer(wrapped, background_t)
        to_explain_t = torch.from_numpy(to_explain).float().to(device)
        shap_values = explainer.shap_values(to_explain_t)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        out["shap_values"] = shap_values.tolist() if hasattr(shap_values, "tolist") else shap_values
        out["shap_background_shape"] = list(background.shape)
        out["shap_explain_shape"] = list(to_explain.shape)
    except Exception as e:
        out["shap_error"] = str(e)
    return out


def plot_predictions_vs_actual(
    dates: list[Any],
    y_true_dir: np.ndarray,
    y_pred_dir: np.ndarray,
    y_true_mag: np.ndarray,
    y_pred_mag: np.ndarray,
    save_path: Optional[os.PathLike | str] = None,
    return_bytes: bool = False,
) -> Optional[str] | Optional[bytes]:
    """
    绘制预测 vs 实际：方向（0/1）与幅度（涨跌幅）。
    save_path 给定则保存到文件并返回路径；return_bytes=True 时写入内存并返回 PNG bytes（不写文件）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except ImportError:
        return None
    n = len(dates)
    if n == 0:
        return None
    x_axis = range(n)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(x_axis, y_true_dir, label="实际方向", color="blue", alpha=0.7)
    ax1.plot(x_axis, y_pred_dir, label="预测方向", color="orange", alpha=0.7)
    ax1.set_ylabel("方向 (0=跌, 1=涨)")
    ax1.legend(loc="upper right")
    ax1.set_title("未来5日价格方向：预测 vs 实际")
    ax2.plot(x_axis, y_true_mag, label="实际涨跌幅", color="blue", alpha=0.7)
    ax2.plot(x_axis, y_pred_mag, label="预测涨跌幅", color="orange", alpha=0.7)
    ax2.set_ylabel("涨跌幅")
    ax2.legend(loc="upper right")
    ax2.set_title("未来5日涨跌幅：预测 vs 实际")
    plt.xlabel("样本索引")
    plt.tight_layout()
    if return_bytes:
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        return buf.getvalue()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        return str(save_path)
    plt.close()
    return None


def generate_fit_plot_for_symbol(
    symbol: str,
    save_dir: Optional[os.PathLike | str] = None,
    fetch_hist_fn: Optional[Any] = None,
    get_date_range_fn: Optional[Any] = None,
    years: Optional[int] = None,
    return_bytes_only: bool = False,
) -> bool | Optional[bytes]:
    """
    用指定模型与该股票历史数据生成「预测 vs 实际」曲线图。
    - years=None：使用全局当前模型，get_date_range_fn() 取日期范围，生成后写入数据库，返回 True/False。
    - years=1|2|3 且 return_bytes_only=True：使用该股票该年份的当前模型，用该版本训练日期范围取数，生成图并直接返回 bytes（不写入库）。
    """
    if not symbol or not fetch_hist_fn:
        return False if not return_bytes_only else None
    if not return_bytes_only and not get_date_range_fn:
        return False
    _ensure_torch()
    import torch
    device = torch.device("cpu")
    save_dir = Path(save_dir or DEFAULT_MODEL_DIR)

    if years is not None and years in (1, 2, 3) and return_bytes_only:
        try:
            model, _ = load_model(save_dir=save_dir, device=device, symbol=symbol, years=years)
        except FileNotFoundError:
            return None
        try:
            from data.lstm_repo import get_current_version_from_db, get_version_date_range
        except ImportError:
            return None
        version_id = get_current_version_from_db(symbol=symbol, years=years)
        if not version_id:
            return None
        date_range = get_version_date_range(version_id)
        if not date_range:
            return None
        start_date, end_date = date_range
        df = fetch_hist_fn(symbol, start_date, end_date)
        if df is None or df.empty or len(df) < 65:
            return None
        X, _, _, y_dir, y_mag = build_features_from_df(df)
        if len(X) == 0:
            return None
        model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(device)
            logits, mag_pred = model(x_t)
            pred_dir = logits.argmax(dim=1).cpu().numpy()
            pred_mag = mag_pred.cpu().numpy().flatten()
        plot_bytes = plot_predictions_vs_actual(
            list(range(len(y_dir))), y_dir, pred_dir, y_mag, pred_mag, return_bytes=True
        )
        return plot_bytes if isinstance(plot_bytes, bytes) else None

    try:
        model, _ = load_model(save_dir=save_dir, device=device)
    except FileNotFoundError:
        return False
    start_date, end_date, _ = get_date_range_fn()
    df = fetch_hist_fn(symbol, start_date, end_date)
    if df is None or df.empty or len(df) < 65:
        return False
    X, _, _, y_dir, y_mag = build_features_from_df(df)
    if len(X) == 0:
        return False
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        logits, mag_pred = model(x_t)
        pred_dir = logits.argmax(dim=1).cpu().numpy()
        pred_mag = mag_pred.cpu().numpy().flatten()
    plot_bytes = plot_predictions_vs_actual(
        list(range(len(y_dir))), y_dir, pred_dir, y_mag, pred_mag, return_bytes=True
    )
    if not isinstance(plot_bytes, bytes):
        return False
    return True


def run_lstm_pipeline(
    df: pd.DataFrame,
    symbol: str = "",
    save_dir: Optional[os.PathLike | str] = None,
    do_cv_tune: bool = True,
    do_shap: bool = True,
    do_plot: bool = True,
    param_grid: Optional[dict[str, Any]] = None,
    do_post_training_validation: bool = True,
    fast_training: bool = False,
    years: int = 1,
) -> dict[str, Any]:
    """
    端到端：特征构建 -> 交叉验证/超参优化 -> 训练保存 ->（可选）样本外验证，仅更优则部署 -> 可解释性 -> 可视化。
    当 do_post_training_validation=True 时，保留最近约 3 个月作测试集，新模型仅当显著优于旧模型才设为当前版本。
    fast_training=True 时使用本机/CPU 友好预设（更少超参与折数），适合无 GPU 或希望快速跑通。
    """
    _ensure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fast_training and param_grid is None:
        param_grid = CPU_FRIENDLY_PARAM_GRID
    save_dir = Path(save_dir or DEFAULT_MODEL_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, feature_names, y_info, y_dir, y_mag = build_features_from_df(df)
    if len(X) == 0:
        return {"error": "样本不足，需要至少 65 个交易日数据"}

    date_col = "日期" if "日期" in df.columns else df.columns[0]
    dates = df[date_col].astype(str).tolist()
    data_start = str(dates[0])[:10] if dates else None
    data_end = str(dates[-1])[:10] if dates else None

    result: dict[str, Any] = {
        "symbol": symbol,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[2]),
        "feature_names": feature_names,
        "data_start": data_start,
        "data_end": data_end,
    }

    # 训练前记录该 (symbol, years) 的当前版本，用于验证阶段对比
    sym_val = (symbol or "").strip()
    years_val = 1 if years not in (1, 2, 3) else int(years)
    old_version_id = get_current_version_id(save_dir, symbol=sym_val, years=years_val) if get_current_version_id else None
    promote = not do_post_training_validation

    validation_score: Optional[dict[str, float] | float] = None
    n_splits = CPU_FRIENDLY_CV_SPLITS if fast_training else 5
    if do_cv_tune:
        cv_result = cross_validate_and_tune(
            X, y_dir, y_mag, feature_names,
            n_splits=n_splits,
            param_grid=param_grid,
            device=device,
        )
        result["cross_validation"] = cv_result
        best = cv_result["best_params"]
        lr, hidden_size, epochs = best["lr"], best["hidden_size"], best["epochs"]
        best_entry = max(cv_result["cv_results"], key=lambda x: x["score"])
        validation_score = {
            "best_score": cv_result["best_score"],
            "avg_f1": best_entry["avg_f1"],
            "avg_mse": best_entry["avg_mse"],
        }
    else:
        lr, hidden_size, epochs = 5e-4, 64, 50

    model, metadata, metrics = train_and_save(
        X, y_dir, y_mag, feature_names,
        save_dir=save_dir,
        lr=lr,
        hidden_size=hidden_size,
        epochs=epochs,
        device=device,
        data_start=data_start,
        data_end=data_end,
        validation_score=validation_score,
        promote_to_current=promote,
        symbol=sym_val,
        years=years_val,
    )
    result["metrics"] = metrics
    result["metadata"] = metadata

    # 训练后验证：样本外测试（最近约 3 个月），仅新模型显著优于旧模型才部署
    if do_post_training_validation and evaluate_model_on_holdout and should_deploy_new_model and set_current_version and remove_version and _prune_versions:
        new_version_id = metadata.get("version_id")
        n_holdout = min(66, max(10, len(X) // 3))  # 约 3 个月或至少 10 条
        X_holdout = X[-n_holdout:]
        y_dir_holdout = y_dir[-n_holdout:]
        y_mag_holdout = y_mag[-n_holdout:]
        new_holdout_metrics = evaluate_model_on_holdout(model, X_holdout, y_dir_holdout, y_mag_holdout, device=device)
        old_holdout_metrics: dict[str, float] = {}
        old_model = None
        if old_version_id and old_version_id != new_version_id:
            try:
                old_model, _ = load_model(save_dir=save_dir, device=device, version_id=old_version_id)
                old_holdout_metrics = evaluate_model_on_holdout(old_model, X_holdout, y_dir_holdout, y_mag_holdout, device=device)
            except Exception:
                pass
        deploy, reason = should_deploy_new_model(new_holdout_metrics, old_holdout_metrics) if old_holdout_metrics else (True, "无旧模型，直接部署")
        if deploy:
            set_current_version(new_version_id, save_dir, symbol=sym_val, years=years_val)
            _prune_versions(save_dir, MAX_VERSIONS, symbol=sym_val, years=years_val)
        else:
            remove_version(new_version_id, save_dir)
        result["validation"] = {
            "deployed": deploy,
            "reason": reason,
            "new_holdout_metrics": new_holdout_metrics,
            "old_holdout_metrics": old_holdout_metrics,
            "n_holdout": n_holdout,
        }

    if do_shap:
        interpret = compute_feature_importance_and_shap(
            model, X, feature_names, device,
            n_background=min(100, len(X) // 2),
            n_explain=min(200, len(X)),
        )
        result["interpretability"] = interpret

    if do_plot:
        dates = list(range(len(y_dir)))
        pred_dir = np.array(metadata.get("last_dir_pred", []))
        pred_mag = np.array(metadata.get("last_mag_pred", []))
        if len(pred_dir) == len(y_dir) and len(pred_mag) == len(y_mag):
            plot_bytes = plot_predictions_vs_actual(
                dates, y_dir, pred_dir, y_mag, pred_mag, return_bytes=True
            )
            if isinstance(plot_bytes, bytes) and symbol and years_val in (1, 2, 3):
                try:
                    from data.lstm_repo import save_lstm_plot_cache
                    save_lstm_plot_cache(symbol, years_val, plot_bytes)
                except Exception:
                    pass
            result["plot_path"] = "db" if isinstance(plot_bytes, bytes) else ""

    return result


