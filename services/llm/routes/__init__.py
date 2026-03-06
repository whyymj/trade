#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路由模块
"""

from flask import Blueprint
from .llm import llm_bp

__all__ = ["llm_bp"]
