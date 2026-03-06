#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 预测模块
提供模型预测功能
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from services.stock.analysis.lstm_model import LSTMModel
from services.stock.analysis.lstm_training import (
    SEQ_LEN,
    build_features_from_df,
    get_stock_data,
)

try:
    from data.lstm_repo import get_model_version

    _LSTM_REPO_AVAILABLE = True
except Exception:
    _LSTM_REPO_AVAILABLE = False


def load_model(
    symbol: str,
    device: Optional[torch.device] = None,
) -> tuple[LSTMModel, dict[str, Any]]:
    """加载已训练的模型

    Args:
        symbol: 股票代码
        device: 设备

    Returns:
        (model, metadata)
    """
    if not _LSTM_REPO_AVAILABLE:
        raise FileNotFoundError("LSTM repo 不可用")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    version_id = None
    try:
        from data.lstm_repo import get_current_version_from_db

        version_id = get_current_version_from_db(symbol=symbol, years=1)
    except Exception:
        pass

    if not version_id:
        raise FileNotFoundError(f"未找到 {symbol} 的训练模型，请先执行训练")

    model_info = get_model_version(version_id)
    if model_info is None:
        raise FileNotFoundError(f"未找到版本 {version_id} 的模型")

    metadata = model_info.get("metadata", {})
    n_features = metadata.get("n_features", 10)
    hidden_size = metadata.get("hidden_size", 64)

    model = LSTMModel(
        input_size=n_features,
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=2,
        n_magnitude_outputs=5,
    ).to(device)

    model_path = model_info.get("model_path")
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    return model, metadata


def predict(symbol: str) -> dict[str, Any]:
    """预测股票走势

    Args:
        symbol: 股票代码

    Returns:
        预测结果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, metadata = load_model(symbol, device=device)
    except FileNotFoundError:
        return {
            "symbol": symbol,
            "error": "模型未训练",
            "direction": None,
            "magnitude": None,
            "confidence": None,
        }

    df = get_stock_data(symbol, days=SEQ_LEN + 10)
    if df is None or len(df) < SEQ_LEN:
        return {
            "symbol": symbol,
            "error": "数据不足",
            "direction": None,
            "magnitude": None,
            "confidence": None,
        }

    X, _, _, _, _ = build_features_from_df(df)
    if len(X) == 0:
        return {
            "symbol": symbol,
            "error": "无法构建特征",
            "direction": None,
            "magnitude": None,
            "confidence": None,
        }

    last_X = X[-1:].copy()
    X_tensor = torch.from_numpy(last_X).float().to(device)

    with torch.no_grad():
        direction_logits, magnitude_pred = model(X_tensor)

    direction_prob = torch.softmax(direction_logits, dim=1)
    pred_class = direction_logits.argmax(dim=1).item()
    confidence = direction_prob[0, pred_class].item()

    magnitude = magnitude_pred[0].cpu().numpy()
    avg_magnitude = float(np.mean(magnitude))

    direction_label = "上涨" if pred_class == 1 else "下跌"

    return {
        "symbol": symbol,
        "direction": direction_label,
        "direction_code": pred_class,
        "magnitude": avg_magnitude,
        "confidence": confidence,
        "daily_magnitude": magnitude.tolist(),
    }
