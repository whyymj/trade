#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 客户端模块
"""

from .deepseek import DeepSeekClient
from .minimax import MiniMaxClient

__all__ = ["DeepSeekClient", "MiniMaxClient"]
