#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻 API 路由
"""

from datetime import datetime
from flask import Blueprint, request, jsonify
from shared.cache import get_cache
from shared.messaging import get_mq, NEWS_CRAWLED
from services.news.data import NewsCrawler, NewsRepo

news_bp = Blueprint("news", __name__)
cache = get_cache()
repo = NewsRepo()


@news_bp.route("/list", methods=["GET"])
def get_news_list():
    """获取新闻列表"""
    days = int(request.args.get("days", 1))
    category = request.args.get("category")
    limit = int(request.args.get("limit", 100))

    cache_key = f"news_list_{days}_{category}_{limit}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_news(days=days, category=category, limit=limit)
    cache.set(cache_key, result, ttl=300)

    return jsonify({"success": True, "data": result, "cached": False})


@news_bp.route("/latest", methods=["GET"])
def get_latest_news():
    """获取最新新闻"""
    limit = int(request.args.get("limit", 20))

    cache_key = f"news_latest_{limit}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_latest_news(limit=limit)
    cache.set(cache_key, result, ttl=300)

    return jsonify({"success": True, "data": result, "cached": False})


@news_bp.route("/detail/<int:news_id>", methods=["GET"])
def get_news_detail(news_id):
    """获取新闻详情"""
    cache_key = f"news_detail_{news_id}"
    result = cache.get(cache_key)
    if result:
        return jsonify({"success": True, "data": result, "cached": True})

    result = repo.get_news_by_id(news_id)

    if not result:
        return jsonify({"success": False, "message": "News not found"}), 404

    cache.set(cache_key, result, ttl=3600)

    return jsonify({"success": True, "data": result, "cached": False})


@news_bp.route("/sync", methods=["POST"])
def sync_news():
    """同步新闻"""
    crawler = NewsCrawler()
    repo = NewsRepo()

    news_list = crawler.fetch_today()

    if not news_list:
        return jsonify(
            {
                "success": True,
                "message": "No news to sync",
                "data": {"fetched": 0, "saved": 0},
            }
        )

    saved = repo.save_news(news_list)

    mq = get_mq()
    mq.publish(NEWS_CRAWLED, {"count": saved, "timestamp": datetime.now().isoformat()})

    return jsonify(
        {
            "success": True,
            "message": "News synced successfully",
            "data": {"fetched": len(news_list), "saved": saved},
        }
    )
