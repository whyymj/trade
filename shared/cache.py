#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis 缓存实现
提供与原有 data/cache.py 兼容的接口
"""

import json
import os
from typing import Any, Optional

import redis
from dotenv import load_dotenv

load_dotenv()


class RedisCache:
    """Redis 缓存客户端"""

    def __init__(self):
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        data = self.redis.get(key)
        if data:
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data
        return None

    def set(self, key: str, value: Any, ttl: int = 1800) -> None:
        """设置缓存"""
        try:
            serialized = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(value)
        self.redis.setex(key, ttl, serialized)

    def delete(self, key: str) -> bool:
        """删除缓存"""
        return bool(self.redis.delete(key))

    def clear(self, pattern: str = None) -> int:
        """清空缓存"""
        if pattern:
            count = 0
            for key in self.redis.scan_iter(match=pattern):
                self.redis.delete(key)
                count += 1
            return count
        else:
            return self.redis.flushdb()

    def cleanup_expired(self) -> int:
        """清理过期缓存（Redis 自动处理，返回 0）"""
        return 0


_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """获取缓存实例（单例）"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance
