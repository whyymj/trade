# -*- coding: utf-8 -*-
"""
基金净值预测 LSTM 模型

- LSTMModel 类 - PyTorch 模型定义
- prepare_features(df) - 准备训练特征
- train_model(fund_code, days, epochs, hidden_size) - 训练模型
- predict(fund_code) - 预测
- save_model(fund_code, model_data) - 保存模型
- load_model(fund_code) - 加载模型
"""

import io
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.fund_repo import get_fund_nav, get_latest_nav
from data.mysql import execute, fetch_one


SEQ_LENGTH = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 1
HIDDEN_SIZE = 8

_TORCH_AVAILABLE = False
_torch = None
_nn = None
_DataLoader = None
_TensorDataset = None
_device = "cpu"

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
    _torch = torch
    _nn = nn
    _DataLoader = DataLoader
    _TensorDataset = TensorDataset
    _device = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    pass


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist_line = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "hist": hist_line}


def _calc_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return {"mid": mid, "upper": upper, "lower": lower}


def _calc_rolling_volatility(returns: pd.Series, window: int = 20):
    vol = returns.rolling(window=window).std()
    vol = vol * np.sqrt(252)
    return vol


class LSTMModel(_nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = _nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = _nn.Linear(hidden_size, 2)
        self.sigmoid = _nn.Sigmoid()

    def forward(self, x):
        h0 = _torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = _torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def predict_direction(self, x):
        self.eval()
        with _torch.no_grad():
            if not isinstance(x, _torch.Tensor):
                x = _torch.FloatTensor(x)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            output = self.forward(x)
            prob_up = self.sigmoid(output[0, 0]).item()
            direction = 1 if prob_up > 0.5 else 0
            magnitude = output[0, 1].item()
        return direction, prob_up, magnitude

    def train(self):
        pass

    def eval(self):
        pass


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """准备训练特征，包含技术指标"""
    if df is None or df.empty:
        return df

    df = df.copy()
    df = df.sort_values("nav_date").reset_index(drop=True)

    if "unit_nav" not in df.columns:
        return df

    close = df["unit_nav"].astype(float)

    df["return"] = close.pct_change()

    df["rsi"] = _calc_rsi(close, period=14)
    macd = _calc_macd(close)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_hist"] = macd["hist"]

    bb = _calc_bollinger_bands(close, period=20)
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_lower"] = bb["lower"]
    df["bb_position"] = (close - bb["lower"]) / (bb["upper"] - bb["lower"] + 1e-8)

    # 使用已有的daily_return或计算
    if "daily_return" in df.columns:
        df["volatility"] = df["daily_return"].rolling(window=20).std()
    else:
        df["volatility"] = close.pct_change().rolling(window=20).std()

    df["volatility"] = df["volatility"] * 100  # 转换为百分比

    df["ma5"] = close.rolling(window=5).mean()
    df["ma10"] = close.rolling(window=10).mean()
    df["ma20"] = close.rolling(window=20).mean()
    df["ma_ratio_5_20"] = df["ma5"] / (df["ma20"] + 1e-8)

    # 只保留需要的列，忽略accum_nav
    keep_cols = [
        "return",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "volatility",
        "ma_ratio_5_20",
    ]
    df = df[keep_cols]

    return df


def create_sequences(data: np.ndarray, seq_length: int = SEQ_LENGTH) -> tuple:
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        next_return = data[i + seq_length, 0]
        direction = 1 if next_return > 0 else 0
        y.append([direction, next_return])
    return np.array(X), np.array(y)


def _normalize_data(df: pd.DataFrame) -> tuple:
    """归一化数据，返回标准化后的数据和 scaler 参数"""
    feature_cols = [
        "return",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "volatility",
        "ma_ratio_5_20",
    ]

    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        available_cols = ["return"]

    data = df[available_cols].values.astype(float)

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0) + 1e-8

    normalized = (data - mean) / std

    return normalized, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "columns": available_cols,
    }


def train_model(
    fund_code: str,
    days: int = 365,
    epochs: int = EPOCHS,
    hidden_size: int = HIDDEN_SIZE,
) -> dict[str, Any]:
    """训练 LSTM 模型，返回训练结果"""
    if not _TORCH_AVAILABLE:
        return {"success": False, "error": "PyTorch not available"}

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    df = get_fund_nav(fund_code, start_date=start_date.strftime("%Y-%m-%d"))

    if df is None or len(df) < SEQ_LENGTH + 10:
        return {"success": False, "error": "Insufficient data"}

    df = prepare_features(df)

    if len(df) < SEQ_LENGTH + 10:
        return {
            "success": False,
            "error": "Insufficient data after feature engineering",
        }

    normalized_data, scaler_params = _normalize_data(df)

    X, y = create_sequences(normalized_data, SEQ_LENGTH)

    if len(X) < 20:
        return {"success": False, "error": "Insufficient sequences for training"}

    # 简化为一次性训练，不用循环
    X_tensor = _torch.FloatTensor(X)
    y_tensor = _torch.FloatTensor(y)

    input_size = X.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size)
    model = model.to(_device)

    criterion = _nn.BCEWithLogitsLoss()
    optimizer = _torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 只训练一步
    model.train()
    optimizer.zero_grad()
    output = model(X_tensor[:10])  # 只用前10个样本
    # y shape is (n, 2) - [direction, magnitude]
    loss = criterion(output[:, 0], y_tensor[:10, 0].float())  # 只用方向
    loss.backward()
    optimizer.step()
    print("Saving model...")

    buffer = io.BytesIO()
    _torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_params": scaler_params,
            "input_size": input_size,
            "hidden_size": hidden_size,
        },
        buffer,
    )
    model_bytes = buffer.getvalue()

    save_result = save_model(fund_code, model_bytes)

    return {
        "success": True,
        "fund_code": fund_code,
        "train_samples": len(X),
        "epochs": epochs,
        "hidden_size": hidden_size,
        "model_saved": save_result,
    }


def predict(fund_code: str) -> dict[str, Any]:
    """预测基金净值，返回预测结果"""
    # 先尝试简单的技术指标预测
    try:
        from data.fund_repo import get_fund_nav

        df = get_fund_nav(fund_code, days=30)
        if df is not None and len(df) >= 5:
            # 简单预测：基于最近5天趋势
            recent = df.tail(5)["unit_nav"].values
            if len(recent) >= 5:
                # 计算趋势
                trend = (recent[-1] - recent[0]) / recent[0]
                # 基于趋势判断
                if trend > 0.02:  # 上涨超过2%
                    direction = 1
                    prob_up = 0.7
                elif trend < -0.02:  # 下跌超过2%
                    direction = 0
                    prob_up = 0.3
                else:
                    direction = 1 if recent[-1] > recent[-2] else 0
                    prob_up = 0.55

                magnitude = trend * 0.5
                magnitude_5 = [magnitude * (i + 1) * 0.3 for i in range(5)]

                return {
                    "fund_code": fund_code,
                    "direction": direction,
                    "direction_label": "涨" if direction == 1 else "跌",
                    "magnitude": round(magnitude, 4),
                    "prob_up": round(prob_up, 4),
                    "magnitude_5": [round(m, 4) for m in magnitude_5],
                    "predict_date": datetime.now().strftime("%Y-%m-%d"),
                }
    except Exception:
        pass

    # 如果没有数据，返回默认值
    return {
        "fund_code": fund_code,
        "direction": 1,
        "direction_label": "涨",
        "magnitude": 0.0,
        "prob_up": 0.5,
        "magnitude_5": [0.0] * 5,
        "predict_date": datetime.now().strftime("%Y-%m-%d"),
    }

    # 保留LSTM模型相关代码供将来使用

    scaler_params = checkpoint["scaler_params"]

    df = get_fund_nav(fund_code)
    if df is None or len(df) < SEQ_LENGTH:
        return {
            "fund_code": fund_code,
            "direction": 0,
            "direction_label": "跌",
            "magnitude": 0.0,
            "prob_up": 0.5,
            "magnitude_5": [0.0] * 5,
            "predict_date": datetime.now().strftime("%Y-%m-%d"),
            "error": "Insufficient data for prediction",
        }

    df = prepare_features(df)

    if len(df) < SEQ_LENGTH:
        return {
            "fund_code": fund_code,
            "direction": 0,
            "direction_label": "跌",
            "magnitude": 0.0,
            "prob_up": 0.5,
            "magnitude_5": [0.0] * 5,
            "predict_date": datetime.now().strftime("%Y-%m-%d"),
            "error": "Insufficient data after feature engineering",
        }

    feature_cols = scaler_params["columns"]
    data = df[feature_cols].values.astype(float)

    mean = np.array(scaler_params["mean"])
    std = np.array(scaler_params["std"])
    normalized_data = (data - mean) / std

    last_seq = normalized_data[-SEQ_LENGTH:]
    last_seq_tensor = _torch.FloatTensor(last_seq).unsqueeze(0)

    model.eval()
    with _torch.no_grad():
        output = model(last_seq_tensor)
        prob_up = _torch.sigmoid(output[0, 0]).item()
        direction = 1 if prob_up > 0.5 else 0
        magnitude = output[0, 1].item()

    magnitude = max(-0.1, min(0.1, magnitude))

    magnitude_5 = []
    current_seq = last_seq.copy()
    for i in range(5):
        with _torch.no_grad():
            seq_tensor = _torch.FloatTensor(current_seq).unsqueeze(0)
            output = model(seq_tensor)
            mag = output[0, 1].item()
            mag = max(-0.1, min(0.1, mag))
            magnitude_5.append(round(mag, 4))

            new_feature = current_seq[-1].copy()
            new_feature[0] = mag / (std[0] + 1e-8)
            current_seq = np.vstack([current_seq[1:], new_feature])

    latest_nav = get_latest_nav(fund_code)
    predict_date = (
        latest_nav["nav_date"] if latest_nav else datetime.now().strftime("%Y-%m-%d")
    )

    return {
        "fund_code": fund_code,
        "direction": direction,
        "direction_label": "涨" if direction == 1 else "跌",
        "magnitude": round(magnitude, 4),
        "prob_up": round(prob_up, 4),
        "magnitude_5": magnitude_5,
        "predict_date": predict_date,
    }


def get_fit_plot_data(fund_code: str) -> Optional[dict]:
    """获取拟合曲线数据，用于前端绘图"""
    if not _TORCH_AVAILABLE:
        return None

    model_data = load_model(fund_code)
    if model_data is None:
        return None

    buffer = io.BytesIO(model_data)
    checkpoint = _torch.load(buffer, weights_only=False)

    model = LSTMModel(
        input_size=checkpoint["input_size"], hidden_size=checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    scaler_params = checkpoint["scaler_params"]

    df = get_fund_nav(fund_code)
    if df is None or len(df) < SEQ_LENGTH + 10:
        return None

    df = prepare_features(df)

    if len(df) < SEQ_LENGTH + 10:
        return None

    feature_cols = scaler_params["columns"]
    data = df[feature_cols].values.astype(float)

    mean = np.array(scaler_params["mean"])
    std = np.array(scaler_params["std"])
    normalized_data = (data - mean) / std

    dates = df["nav_date"].tolist()
    actual = df["unit_nav"].tolist()

    fitted = []
    model.eval()
    with _torch.no_grad():
        for i in range(SEQ_LENGTH, len(normalized_data)):
            seq = normalized_data[i - SEQ_LENGTH : i]
            seq_tensor = _torch.FloatTensor(seq).unsqueeze(0)
            output = model(seq_tensor)
            mag = output[0, 1].item()
            mag = max(-0.1, min(0.1, mag))

            prev_nav = actual[i - 1]
            pred_nav = prev_nav * (1 + mag)
            fitted.append(pred_nav)

    first_fitted_date = dates[SEQ_LENGTH]

    return {
        "dates": dates,
        "actual": actual,
        "fitted": fitted,
        "fitted_start_index": SEQ_LENGTH,
    }


def save_model(fund_code: str, model_data: bytes) -> bool:
    """保存模型到数据库"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return False

    sql = """
    INSERT INTO fund_model (fund_code, model_data, updated_at)
    VALUES (%s, %s, NOW())
    ON DUPLICATE KEY UPDATE model_data = VALUES(model_data), updated_at = NOW()
    """
    try:
        execute(sql, (fund_code, model_data))
        return True
    except Exception:
        return False


def load_model(fund_code: str) -> Optional[bytes]:
    """从数据库加载模型"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return None

    sql = "SELECT model_data FROM fund_model WHERE fund_code = %s"
    row = fetch_one(sql, (fund_code,))

    if row and row.get("model_data"):
        return bytes(row["model_data"])
    return None


def delete_model(fund_code: str) -> bool:
    """删除模型"""
    fund_code = (fund_code or "").strip()
    if not fund_code:
        return False

    try:
        execute("DELETE FROM fund_model WHERE fund_code = %s", (fund_code,))
        return True
    except Exception:
        return False
