# -*- coding: utf-8 -*-
"""LSTM 模块共用常量，避免 lstm_model 与 lstm_versioning 循环导入。"""
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "analysis_temp" / "lstm"
FORECAST_DAYS = 5

# 预测提前天数。设为 N 时，在拟合图中将预测曲线提前 N 天展示（在时间轴上左移：
# 日期 j 处显示的是「窗口 j+N」的预测值，与「日期 j」的实际值对比）。设为 0 表示不提前。
PREDICTION_OFFSET_DAYS = 5
