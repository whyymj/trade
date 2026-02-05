# -*- coding: utf-8 -*-
"""LSTM 模块共用常量，避免 lstm_model 与 lstm_versioning 循环导入。"""
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "analysis_temp" / "lstm"
FORECAST_DAYS = 5
