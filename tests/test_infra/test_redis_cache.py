#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Redis 缓存
"""

import time

import pytest
from shared.cache import get_cache


def test_cache_set_get():
    """测试缓存设置和获取"""
    cache = get_cache()
    cache.set("test_key", {"value": 123}, ttl=60)
    result = cache.get("test_key")
    assert result == {"value": 123}


def test_cache_delete():
    """测试缓存删除"""
    cache = get_cache()
    cache.set("test_key_delete", "value")
    cache.delete("test_key_delete")
    assert cache.get("test_key_delete") is None


def test_cache_ttl():
    """测试 TTL"""
    cache = get_cache()
    cache.set("test_key_ttl", "value", ttl=1)
    time.sleep(2)
    assert cache.get("test_key_ttl") is None


def test_cache_clear_pattern():
    """测试按模式清空缓存"""
    cache = get_cache()
    cache.set("test:pattern:1", "value1")
    cache.set("test:pattern:2", "value2")
    cache.set("test:other", "value3")
    cache.clear(pattern="test:pattern:*")
    assert cache.get("test:pattern:1") is None
    assert cache.get("test:pattern:2") is None
    assert cache.get("test:other") == "value3"


def test_cache_clear_all():
    """测试清空所有缓存"""
    cache = get_cache()
    cache.set("test_all_key", "value")
    cache.clear()
    assert cache.get("test_all_key") is None


@pytest.mark.parametrize(
    "key,value,ttl",
    [
        ("str_key", "string_value", 60),
        ("int_key", 123, 60),
        ("list_key", [1, 2, 3], 60),
        ("dict_key", {"a": 1, "b": 2}, 60),
    ],
)
def test_cache_various_types(key, value, ttl):
    """测试不同类型的数据"""
    cache = get_cache()
    cache.set(key, value, ttl=ttl)
    result = cache.get(key)
    assert result == value
