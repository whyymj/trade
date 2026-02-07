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

from analysis.lstm_constants import DEFAULT_MODEL_DIR, FORECAST_DAYS, PREDICTION_OFFSET_DAYS
from analysis.lstm_losses import get_regression_criterion
from analysis.lstm_training import DEFAULT_IMPROVED_CONFIG, 改进的训练策略
from analysis.lstm_volatility_features import (
    VOLATILITY_FEATURE_NAMES,
    构建波动增强特征,
    波动增强特征到数组,
)

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
BASE_FEATURE_NAMES = [
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
DEFAULT_FEATURE_NAMES = BASE_FEATURE_NAMES + VOLATILITY_FEATURE_NAMES


def _fit_y_mag_scale(y_mag: np.ndarray) -> tuple[float, float]:
    """用训练集拟合涨跌幅的均值与标准差，用于标准化。返回 (mean, std)，std 过小则用 1.0。"""
    mean = float(np.mean(y_mag))
    std = float(np.std(y_mag))
    return mean, (std if std >= 1e-8 else 1.0)


def _standardize_y_mag(y_mag: np.ndarray, mean: float, std: float) -> np.ndarray:
    """标准化涨跌幅。"""
    return ((y_mag - mean) / std).astype(np.float32)


def _inverse_standardize_mag(pred: np.ndarray, mean: float, std: float) -> np.ndarray:
    """将模型输出的标准化预测反变换为原始涨跌幅量纲。"""
    return (pred * std + mean).astype(np.float64)


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
    open_ = _to_series(df[open_col].astype(float))

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

    # 波动增强特征（缓解预测过度平滑）
    vol_df = 构建波动增强特征(close, high, low, open_, volume)
    F_vol, _ = 波动增强特征到数组(vol_df, SEQ_LEN)
    F = np.hstack([F, F_vol.astype(np.float32)])

    X_list = []
    y_direction_list = []
    y_magnitude_list = []  # 每个元素为长度 FORECAST_DAYS 的逐日涨跌幅
    end_dates = []

    for i in range(SEQ_LEN, n - FORECAST_DAYS):
        X_list.append(F[i - SEQ_LEN : i])  # (SEQ_LEN, n_features)
        # 未来 5 日逐日收益率：ret_d[t] = close[i+t]/close[i+t-1] - 1
        daily_rets = []
        for d in range(FORECAST_DAYS):
            ret_d = (float(close.iloc[i + d]) / float(close.iloc[i + d - 1])) - 1.0
            daily_rets.append(ret_d)
        cum_ret = (float(close.iloc[i + FORECAST_DAYS - 1]) / float(close.iloc[i - 1])) - 1.0
        y_direction_list.append(1 if cum_ret > 0 else 0)
        y_magnitude_list.append(daily_rets)
        date_val = df["日期"].iloc[i] if "日期" in df.columns else (df.index[i] if hasattr(df.index[i], "strftime") else i)
        end_dates.append(date_val)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_dir = np.array(y_direction_list, dtype=np.int64)
    y_mag = np.array(y_magnitude_list, dtype=np.float32)  # (N, FORECAST_DAYS)

    y_info = pd.Series(
        [
            {"direction": int(d), "magnitude": float((1 + m[0]) * (1 + m[1]) * (1 + m[2]) * (1 + m[3]) * (1 + m[4]) - 1), "end_date": str(e)}
            for d, m, e in zip(y_direction_list, y_magnitude_list, end_dates)
        ],
        index=end_dates,
    )
    feature_names = BASE_FEATURE_NAMES + VOLATILITY_FEATURE_NAMES
    return X, feature_names, y_info, y_dir, y_mag


if _TORCH_AVAILABLE and nn is not None and torch is not None:

    class SeqAttention(nn.Module):
        """序列注意力：对 seq_len 维做 softmax，得到每个时间步权重。"""

        def __init__(self, hidden_size: int):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            # seq: (batch, seq_len, hidden_size) -> weights: (batch, seq_len, 1)
            scores = self.attn(seq)
            return torch.softmax(scores, dim=1)

    class LSTMDualHead(nn.Module):
        """LSTM + Dropout + 全连接双头（分类 + 回归）。回归头可输出多步（如 5 日逐日涨跌幅）。"""

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            dropout: float = 0.3,
            num_classes: int = 2,
            n_magnitude_outputs: int = 5,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.n_magnitude_outputs = n_magnitude_outputs
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
            self.fc_magnitude = nn.Linear(hidden_size, n_magnitude_outputs)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            last_h = lstm_out[:, -1]
            h = self.dropout(torch.relu(self.fc_shared(last_h)))
            direction_logits = self.fc_direction(h)
            magnitude = self.fc_magnitude(h)  # (batch, n_magnitude_outputs)
            if magnitude.shape[-1] == 1:
                magnitude = magnitude.squeeze(-1)
            return direction_logits, magnitude

    class LSTMDualHeadEnhanced(nn.Module):
        """
        增强 LSTM：双向 LSTM + 注意力 + 跳跃连接 + 更深全连接，输出 (方向 logits, 多步涨跌幅)。
        """

        def __init__(
            self,
            input_size: int,
            hidden_size: int = 128,
            num_layers: int = 3,
            dropout: float = 0.2,
            num_classes: int = 2,
            n_magnitude_outputs: int = 5,
        ):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.n_magnitude_outputs = n_magnitude_outputs
            self.bidirectional = True
            hidden_out = hidden_size * 2  # 双向

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True,
            )
            self.attention = SeqAttention(hidden_out)
            self.skip_connection = nn.Sequential(
                nn.Linear(input_size, hidden_out),
                nn.ReLU(),
            )
            self.fc_shared = nn.Sequential(
                nn.Linear(hidden_out * 2, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.fc_direction = nn.Linear(64, num_classes)
            self.fc_magnitude = nn.Linear(64, n_magnitude_outputs)
            self._init_weights()

        def _init_weights(self) -> None:
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
            for m in self.fc_shared:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            nn.init.normal_(self.fc_direction.weight, mean=0, std=0.01)
            nn.init.normal_(self.fc_magnitude.weight, mean=0, std=0.01)
            if self.fc_direction.bias is not None:
                nn.init.constant_(self.fc_direction.bias, 0)
            if self.fc_magnitude.bias is not None:
                nn.init.constant_(self.fc_magnitude.bias, 0)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
            weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            context = (lstm_out * weights).sum(dim=1)  # (batch, hidden*2)
            skip = self.skip_connection(x[:, -1, :])  # (batch, hidden*2)
            combined = torch.cat([context, skip], dim=1)  # (batch, hidden*4)
            h = self.fc_shared(combined)  # (batch, 64)
            direction_logits = self.fc_direction(h)
            magnitude = self.fc_magnitude(h)  # (batch, n_magnitude_outputs)
            if magnitude.shape[-1] == 1:
                magnitude = magnitude.squeeze(-1)
            return direction_logits, magnitude
else:
    LSTMDualHead = None  # type: ignore[misc, assignment]
    LSTMDualHeadEnhanced = None  # type: ignore[misc, assignment]
    SeqAttention = None  # type: ignore[misc, assignment]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weight_cls: float = 1.0,
    weight_reg: float = 1.0,
    reg_criterion: Optional[Any] = None,
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
        if reg_criterion is None:
            loss_reg = nn.functional.mse_loss(mag_pred, y_mag_b)
        else:
            out = reg_criterion(mag_pred, y_mag_b)
            loss_reg = out[0] if isinstance(out, tuple) else out
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
    reg_loss_type: str = "mse",
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
    reg_criterion = get_regression_criterion(reg_loss_type)
    if reg_criterion is not None:
        reg_criterion = reg_criterion.to(device)

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
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)  # 时间序列保持顺序
                    val_loader = DataLoader(val_ds, batch_size=batch_size)

                    model = LSTMDualHead(
                        input_size=n_features,
                        hidden_size=hidden,
                        num_layers=1,
                        dropout=0.3,
                        num_classes=2,
                        n_magnitude_outputs=FORECAST_DAYS,
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    for _ in range(epochs):
                        train_epoch(model, train_loader, optimizer, device, reg_criterion=reg_criterion)
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
    use_enhanced_model: bool = True,
    reg_loss_type: str = "full",
    use_improved_training: bool = True,
    stop_event: Optional[Any] = None,
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 时间序列保持顺序

    if use_enhanced_model and LSTMDualHeadEnhanced is not None:
        enh_hidden = 128
        enh_layers = 3
        model = LSTMDualHeadEnhanced(
            input_size=n_features,
            hidden_size=enh_hidden,
            num_layers=enh_layers,
            dropout=0.2,
            num_classes=2,
            n_magnitude_outputs=FORECAST_DAYS,
        ).to(device)
        _hidden_size, _num_layers = enh_hidden, enh_layers
        _model_type = "enhanced"
    else:
        model = LSTMDualHead(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.3,
            num_classes=2,
            n_magnitude_outputs=FORECAST_DAYS,
        ).to(device)
        _hidden_size, _num_layers = hidden_size, 1
        _model_type = "basic"
    training_loss_history: list[float] = []
    y_mag_mean: Optional[float] = None
    y_mag_std: Optional[float] = None
    if use_improved_training:
        n_total = len(X)
        n_val = max(20, int(0.2 * n_total))
        n_train = n_total - n_val
        y_mag_train = y_magnitude[:n_train]
        y_mag_mean, y_mag_std = _fit_y_mag_scale(y_mag_train)
        y_mag_tr_std = _standardize_y_mag(y_magnitude[:n_train], y_mag_mean, y_mag_std)
        y_mag_val_std = _standardize_y_mag(y_magnitude[n_train:], y_mag_mean, y_mag_std)
        train_dataset = TensorDataset(
            torch.from_numpy(X[:n_train]),
            torch.from_numpy(y_direction[:n_train]),
            torch.from_numpy(y_mag_tr_std),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X[n_train:]),
            torch.from_numpy(y_direction[n_train:]),
            torch.from_numpy(y_mag_val_std),
        )
        train_loader_split = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列保持顺序
        val_loader_split = DataLoader(val_dataset, batch_size=batch_size)
        improved_config = dict(DEFAULT_IMPROVED_CONFIG)
        improved_config["max_epochs"] = epochs
        improved_config["reg_loss_type"] = reg_loss_type
        _, training_loss_history, _improved_metrics, _best_epoch = 改进的训练策略(
            model, train_loader_split, val_loader_split, device, improved_config, stop_event=stop_event
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        reg_criterion = get_regression_criterion(reg_loss_type)
        if reg_criterion is not None:
            reg_criterion = reg_criterion.to(device)
        for _ in range(epochs):
            if stop_event is not None and stop_event.is_set():
                break
            loss = train_epoch(model, loader, optimizer, device, reg_criterion=reg_criterion)
            training_loss_history.append(loss)

    model.eval()
    with torch.no_grad():
        logits, mag_pred = model(torch.from_numpy(X).to(device))
        last_dir_pred = logits.argmax(dim=1).cpu().numpy()
        mag_pred_np = mag_pred.cpu().numpy()
    if y_mag_mean is not None and y_mag_std is not None:
        mag_pred_raw = _inverse_standardize_mag(mag_pred_np, y_mag_mean, y_mag_std)
        all_mag_true = y_magnitude.flatten() if y_magnitude.ndim > 1 else y_magnitude
        all_mag_pred_flat = mag_pred_raw.flatten() if mag_pred_raw.ndim > 1 else mag_pred_raw
        final_metrics = {
            "accuracy": float(accuracy_score(y_direction, last_dir_pred)),
            "recall": float(recall_score(y_direction, last_dir_pred, average="binary", zero_division=0)),
            "f1": float(f1_score(y_direction, last_dir_pred, average="binary", zero_division=0)),
            "mse": float(mean_squared_error(all_mag_true, all_mag_pred_flat)),
        }
        last_mag_flat = mag_pred_raw.flatten() if mag_pred_raw.ndim > 1 else mag_pred_raw
    else:
        mag_pred_raw = mag_pred_np
        final_metrics = evaluate(model, DataLoader(dataset, batch_size=batch_size), device)
        last_mag_flat = mag_pred_np.flatten() if mag_pred_np.ndim > 1 else mag_pred_np
    stopped = stop_event is not None and stop_event.is_set()
    metadata = {
        "feature_names": feature_names,
        "seq_len": SEQ_LEN,
        "forecast_days": FORECAST_DAYS,
        "n_magnitude_outputs": FORECAST_DAYS,
        "hidden_size": _hidden_size,
        "n_features": n_features,
        "num_layers": _num_layers,
        "model_type": _model_type,
        "reg_loss_type": reg_loss_type,
        "use_improved_training": use_improved_training,
        "lr": lr,
        "epochs": epochs,
        "metrics": final_metrics,
        "last_dir_pred": last_dir_pred.tolist(),
        "last_mag_pred": last_mag_flat.tolist() if hasattr(last_mag_flat, "tolist") else list(last_mag_flat),
        "training_loss_history": training_loss_history,
        "training_stopped": stopped,
    }
    if y_mag_mean is not None and y_mag_std is not None:
        metadata["y_mag_mean"] = y_mag_mean
        metadata["y_mag_std"] = y_mag_std
    try:
        from analysis.lstm_diagnostics import 诊断LSTM预测平淡问题
        训练数据 = {"X": X, "训练损失": training_loss_history}
        pred_for_diag = mag_pred_raw
        last_mag_1d = pred_for_diag.mean(axis=1) if pred_for_diag.ndim > 1 else pred_for_diag
        y_mag_1d = y_magnitude.mean(axis=1) if y_magnitude.ndim > 1 else y_magnitude
        诊断结果 = 诊断LSTM预测平淡问题(last_mag_1d, y_mag_1d, model, 训练数据)
        metadata["diagnostics"] = 诊断结果
    except Exception:
        pass
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
    model_type = metadata.get("model_type", "basic")
    n_mag_out = metadata.get("n_magnitude_outputs", FORECAST_DAYS)
    if state_dict and "fc_magnitude.weight" in state_dict:
        n_mag_out = int(state_dict["fc_magnitude.weight"].shape[0])
    if model_type == "enhanced" and LSTMDualHeadEnhanced is not None:
        num_layers = metadata.get("num_layers", 3)
        model = LSTMDualHeadEnhanced(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            num_classes=2,
            n_magnitude_outputs=n_mag_out,
        ).to(device)
    else:
        model = LSTMDualHead(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.3,
            num_classes=2,
            n_magnitude_outputs=n_mag_out,
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
    reg_loss_type: str = "full",
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

    inc_y_mean = metadata.get("y_mag_mean")
    inc_y_std = metadata.get("y_mag_std")
    if inc_y_mean is not None and inc_y_std is not None:
        y_mag_train = _standardize_y_mag(y_mag, inc_y_mean, inc_y_std)
    else:
        y_mag_train = y_mag

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_dir),
        torch.from_numpy(y_mag_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 时间序列保持顺序
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_criterion = get_regression_criterion(reg_loss_type)
    if reg_criterion is not None:
        reg_criterion = reg_criterion.to(device)
    for _ in range(epochs):
        train_epoch(model, loader, optimizer, device, reg_criterion=reg_criterion)

    model.eval()
    with torch.no_grad():
        logits, mag_pred = model(torch.from_numpy(X).to(device))
        last_dir_pred = logits.argmax(dim=1).cpu().numpy()
        mag_pred_np = mag_pred.cpu().numpy()
    if inc_y_mean is not None and inc_y_std is not None:
        last_mag_pred = _inverse_standardize_mag(mag_pred_np, inc_y_mean, inc_y_std)
        all_mag_true = y_mag.flatten() if y_mag.ndim > 1 else y_mag
        all_mag_pred_flat = last_mag_pred.flatten() if last_mag_pred.ndim > 1 else last_mag_pred
        final_metrics = {
            "accuracy": float(accuracy_score(y_dir, last_dir_pred)),
            "recall": float(recall_score(y_dir, last_dir_pred, average="binary", zero_division=0)),
            "f1": float(f1_score(y_dir, last_dir_pred, average="binary", zero_division=0)),
            "mse": float(mean_squared_error(all_mag_true, all_mag_pred_flat)),
        }
    else:
        final_metrics = evaluate(model, DataLoader(dataset, batch_size=batch_size), device)
        last_mag_pred = mag_pred_np
    last_mag_flat = last_mag_pred.flatten() if last_mag_pred.ndim > 1 else last_mag_pred

    meta = {
        "feature_names": metadata.get("feature_names", feature_names),
        "seq_len": metadata.get("seq_len", SEQ_LEN),
        "forecast_days": metadata.get("forecast_days", FORECAST_DAYS),
        "hidden_size": metadata.get("hidden_size", 64),
        "num_layers": metadata.get("num_layers", 1),
        "model_type": metadata.get("model_type", "basic"),
        "n_features": int(X.shape[2]),
        "lr": lr,
        "epochs": epochs,
        "metrics": final_metrics,
        "last_dir_pred": last_dir_pred.tolist(),
        "last_mag_pred": last_mag_flat.tolist() if hasattr(last_mag_flat, "tolist") else list(last_mag_flat),
        "training_type": "incremental",
    }
    if inc_y_mean is not None and inc_y_std is not None:
        meta["y_mag_mean"] = inc_y_mean
        meta["y_mag_std"] = inc_y_std
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


def get_fit_plot_data(
    symbol: str,
    save_dir: Optional[os.PathLike | str] = None,
    fetch_hist_fn: Optional[Any] = None,
    years: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """
    返回「预测 vs 实际」曲线图所需数据，供前端 ECharts 绘制。
    返回: { dates, actual_dir, pred_dir, actual_mag, pred_mag, dates_price, actual_price, predicted_price } 或 None。
    其中 dates_price/actual_price/predicted_price 为按 5 日窗口结束日对齐的「实际价格」与「预测价格」曲线。
    若 lstm_constants.PREDICTION_OFFSET_DAYS > 0，会将预测曲线提前 N 天展示（在时间轴上左移）。
    """
    if not symbol or not fetch_hist_fn or years not in (1, 2, 3):
        return None
    _ensure_torch()
    import torch
    device = torch.device("cpu")
    save_dir = Path(save_dir or DEFAULT_MODEL_DIR)
    try:
        model, metadata_plot = load_model(save_dir=save_dir, device=device, symbol=symbol, years=years)
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
    X, _, y_info, y_dir, y_mag = build_features_from_df(df)
    if len(X) == 0:
        return None
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        logits, mag_pred = model(x_t)
        pred_dir = logits.argmax(dim=1).cpu().numpy()
        mag_np = mag_pred.cpu().numpy()
        y_mean = metadata_plot.get("y_mag_mean")
        y_std = metadata_plot.get("y_mag_std")
        if y_mean is not None and y_std is not None:
            mag_np = _inverse_standardize_mag(mag_np, y_mean, y_std)
        pred_mag = (mag_np.mean(axis=1) if mag_np.ndim > 1 else mag_np.flatten()).tolist()
    dates = [str(d)[:10] for d in y_info.index.tolist()]
    # y_mag 可能为 (N,5) 逐日或 (N,) 单值；图表用「5 日累计」一维
    if y_mag.ndim == 2 and y_mag.shape[1] == FORECAST_DAYS:
        actual_mag_1d = np.array([float((1 + m[0]) * (1 + m[1]) * (1 + m[2]) * (1 + m[3]) * (1 + m[4]) - 1) for m in y_mag])
    else:
        actual_mag_1d = np.asarray(y_mag).flatten()
    # 预测价格曲线：每个样本对应 5 日窗口的结束日，蓝线=当日实际收盘价，红线=起点价×(1+预测涨跌幅)，同一 k 同一结束日，无故意错位
    # 注意：上方「方向/涨跌幅」横轴为窗口起始日，此处为结束日，故与上两块图相比会有约 5 日的视觉错位
    close_col = "收盘" if "收盘" in df.columns else "close"
    date_col = "日期" if "日期" in df.columns else None
    close_series = df[close_col].astype(float)
    n_samples = len(X)
    actual_price_list: list[float] = []
    predicted_price_list: list[float] = []
    dates_price: list[str] = []
    for k in range(n_samples):
        idx_start = SEQ_LEN + k - 1  # 5 日窗口前一日收盘（预测起点）
        idx_end = SEQ_LEN + k + FORECAST_DAYS - 1  # 5 日窗口最后一日收盘
        start_price = float(close_series.iloc[idx_start])
        end_price = float(close_series.iloc[idx_end])
        actual_price_list.append(end_price)
        predicted_price_list.append(start_price * (1.0 + float(pred_mag[k])))
        if date_col:
            dates_price.append(str(df[date_col].iloc[idx_end])[:10])
        else:
            dates_price.append(str(df.index[idx_end])[:10] if hasattr(df.index[idx_end], "__str__") else str(idx_end))
    result: dict[str, Any] = {
        "dates": dates,
        "actual_dir": y_dir.tolist(),
        "pred_dir": pred_dir.tolist(),
        "actual_mag": actual_mag_1d.tolist(),
        "pred_mag": pred_mag,
        "dates_price": dates_price,
        "actual_price": actual_price_list,
        "predicted_price": predicted_price_list,
    }
    # 预测提前：若 PREDICTION_OFFSET_DAYS=N，在日期 j 处显示 pred[j+N]，使预测曲线在时间轴上左移 N 天
    offset = PREDICTION_OFFSET_DAYS if isinstance(PREDICTION_OFFSET_DAYS, int) else 0
    if offset > 0 and n_samples > offset:
        result["dates"] = dates[: n_samples - offset]
        result["actual_dir"] = y_dir.tolist()[: n_samples - offset]
        result["pred_dir"] = pred_dir.tolist()[offset:]
        result["actual_mag"] = actual_mag_1d.tolist()[: n_samples - offset]
        result["pred_mag"] = pred_mag[offset:]
        result["dates_price"] = dates_price[: n_samples - offset]
        result["actual_price"] = actual_price_list[: n_samples - offset]
        result["predicted_price"] = predicted_price_list[offset:]
    try:
        from analysis.lstm_diagnostics import 诊断LSTM预测平淡问题
        训练数据 = {"X": X}
        诊断结果 = 诊断LSTM预测平淡问题(np.array(pred_mag) if isinstance(pred_mag, list) else pred_mag, actual_mag_1d, model, 训练数据)
        result["diagnostics"] = 诊断结果
    except Exception:
        pass
    return result


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
    use_enhanced_model: bool = True,
    reg_loss_type: str = "full",
    use_improved_training: bool = True,
    stop_event: Optional[Any] = None,
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
    # 使用改进训练时跳过交叉验证，用固定超参直接训练，显著提速（CV 约 40 次子训练）
    skip_cv = use_improved_training
    n_splits = CPU_FRIENDLY_CV_SPLITS if fast_training else 5
    if do_cv_tune and not skip_cv:
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
        lr, hidden_size, epochs = 5e-4, 128 if use_enhanced_model else 64, 50

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
        use_enhanced_model=use_enhanced_model,
        reg_loss_type=reg_loss_type,
        use_improved_training=use_improved_training,
        stop_event=stop_event,
    )
    result["metrics"] = metrics
    result["metadata"] = metadata
    result["diagnostics"] = metadata.get("diagnostics") or {}

    # 训练后验证：样本外测试（最近约 3 个月），仅新模型显著优于旧模型才部署
    if do_post_training_validation and evaluate_model_on_holdout and should_deploy_new_model and set_current_version and remove_version and _prune_versions:
        new_version_id = metadata.get("version_id")
        n_holdout = min(66, max(10, len(X) // 3))  # 约 3 个月或至少 10 条
        X_holdout = X[-n_holdout:]
        y_dir_holdout = y_dir[-n_holdout:]
        y_mag_holdout = y_mag[-n_holdout:]
        y_mag_mean = metadata.get("y_mag_mean")
        y_mag_std = metadata.get("y_mag_std")
        new_holdout_metrics = evaluate_model_on_holdout(
            model, X_holdout, y_dir_holdout, y_mag_holdout, device=device,
            y_mag_mean=y_mag_mean, y_mag_std=y_mag_std,
        )
        old_holdout_metrics: dict[str, float] = {}
        old_model = None
        if old_version_id and old_version_id != new_version_id:
            try:
                old_model, old_meta = load_model(save_dir=save_dir, device=device, version_id=old_version_id)
                old_holdout_metrics = evaluate_model_on_holdout(
                    old_model, X_holdout, y_dir_holdout, y_mag_holdout, device=device,
                    y_mag_mean=old_meta.get("y_mag_mean"), y_mag_std=old_meta.get("y_mag_std"),
                )
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

    return result


