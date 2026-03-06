#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻数据模块
"""

from .news_crawler import NewsCrawler, get_crawler
from .news_repo import NewsRepo, get_repo

__all__ = ["NewsCrawler", "NewsRepo", "get_crawler", "get_repo"]
