#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM 分析模块
提供 LSTM 模型、训练和预测功能
"""

from services.stock.analysis.lstm_model import LSTMModel

try:
    from services.stock.analysis.lstm_training import train_model
except Exception:
    train_model = None
try:
    from services.stock.analysis.lstm_predict import predict
except Exception:
    predict = None

__all__ = ["LSTMModel", "train_model", "predict"]
