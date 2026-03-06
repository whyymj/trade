#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock 服务路由模块
"""

from services.stock.routes.lstm import lstm_bp
from services.stock.routes.stock import stock_bp

__all__ = ["lstm_bp", "stock_bp"]
