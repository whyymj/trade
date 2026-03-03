# -*- coding: utf-8 -*-
"""
内存缓存实现，支持 TTL 和线程安全。
"""

import threading
import time
from typing import Any, Optional


class SimpleCache:
    """线程安全的内存缓存，支持 TTL"""

    def __init__(self, default_ttl: int = 300):
        self._cache: dict[str, tuple[Any, Optional[float]]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值，过期返回 None"""
        with self._lock:
            if key not in self._cache:
                return None
            value, expires_at = self._cache[key]
            if expires_at is not None and time.time() > expires_at:
                del self._cache[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值，可选指定 TTL（秒）"""
        if ttl is None:
            ttl = self._default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        with self._lock:
            self._cache[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        """删除缓存，返回是否成功"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        count = 0
        now = time.time()
        with self._lock:
            expired_keys = [
                k
                for k, (_, exp) in self._cache.items()
                if exp is not None and now > exp
            ]
            for key in expired_keys:
                del self._cache[key]
                count += 1
        return count


_cache_instance: Optional[SimpleCache] = None
_cache_lock = threading.Lock()


def get_cache() -> SimpleCache:
    """获取全局缓存实例（单例）"""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = SimpleCache()
    return _cache_instance
