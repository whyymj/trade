#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 路由 - 通用对话、新闻分析、行业分类
"""

import hashlib
import json
import logging

from flask import Blueprint, jsonify, request

from shared.cache import get_cache
from services.llm.llm import DeepSeekClient, MiniMaxClient
from services.llm.app import rate_limit

llm_bp = Blueprint("llm", __name__)
cache = get_cache()
deepseek = DeepSeekClient()
minimax = MiniMaxClient()

logger = logging.getLogger(__name__)


def _get_cache_key(messages: list, provider: str) -> str:
    """生成缓存键"""
    content = json.dumps(messages, sort_keys=True)
    return f"llm:{provider}:{hashlib.md5(content.encode()).hexdigest()}"


@llm_bp.route("/chat", methods=["POST"])
@rate_limit(max_calls=100, per_minute=1)
def chat():
    """通用对话 API"""
    data = request.get_json()
    provider = data.get("provider", "deepseek")
    messages = data.get("messages", [])

    if not messages:
        return jsonify({"success": False, "message": "Messages required"}), 400

    cache_key = _get_cache_key(messages, provider)
    cached = cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "data": cached, "cached": True})

    try:
        if provider == "deepseek":
            result = deepseek.chat(messages)
        elif provider == "minimax":
            result = minimax.chat(messages)
        else:
            return jsonify({"success": False, "message": "Invalid provider"}), 400

        cache.set(cache_key, result, ttl=86400)

        return jsonify({"success": True, "data": result, "cached": False})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@llm_bp.route("/analyze-news", methods=["POST"])
@rate_limit(max_calls=50, per_minute=1)
def analyze_news():
    """新闻分析 API"""
    data = request.get_json()
    news_list = data.get("news", [])

    if not news_list:
        return jsonify({"success": False, "message": "News required"}), 400

    try:
        messages = [
            {
                "role": "system",
                "content": "你是一个金融新闻分析助手，请提取新闻中的关键信息。",
            },
            {"role": "user", "content": f"请分析以下新闻：{news_list[:5]}"},
        ]
        key_info = minimax.chat(messages)

        messages = [
            {
                "role": "system",
                "content": "你是一个资深的金融分析师，请对新闻进行深度分析。",
            },
            {"role": "user", "content": f"新闻关键信息：{key_info}\n请进行深度分析。"},
        ]
        deep_analysis = deepseek.chat(messages)

        return jsonify(
            {
                "success": True,
                "data": {"key_info": key_info, "deep_analysis": deep_analysis},
            }
        )
    except Exception as e:
        logger.error(f"Analyze news error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500


@llm_bp.route("/classify-industry", methods=["POST"])
@rate_limit(max_calls=50, per_minute=1)
def classify_industry():
    """行业分类 API"""
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"success": False, "message": "Text required"}), 400

    cache_key = f"industry:{hashlib.md5(text.encode()).hexdigest()}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify({"success": True, "data": {"industry": cached}, "cached": True})

    try:
        messages = [
            {
                "role": "system",
                "content": "你是一个行业分类专家，请将文本分类到以下行业之一：宏观、行业、全球、政策、公司。",
            },
            {"role": "user", "content": text},
        ]
        result = deepseek.chat(messages)
        industry = result.strip()

        cache.set(cache_key, industry, ttl=86400)

        return jsonify(
            {"success": True, "data": {"industry": industry}, "cached": False}
        )
    except Exception as e:
        logger.error(f"Classify industry error: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500
