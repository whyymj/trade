#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻服务路由模块
"""

from flask import Blueprint

from .news import news_bp

__all__ = ["news_bp"]
