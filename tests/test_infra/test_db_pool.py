#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 MySQL 连接池
"""

import pytest
from shared.db import get_connection, fetch_all, fetch_one, execute, test_connection


def test_connection_pool():
    """测试连接池"""
    with get_connection() as conn:
        assert conn is not None


def test_fetch_all():
    """测试查询"""
    result = fetch_all("SELECT 1 as test")
    assert len(result) == 1
    assert result[0]["test"] == 1


def test_fetch_one():
    """测试查询单行"""
    result = fetch_one("SELECT 1 as test")
    assert result is not None
    assert result["test"] == 1


def test_execute():
    """测试执行"""
    execute(
        "CREATE TABLE IF NOT EXISTS test_table (id INT PRIMARY KEY, value VARCHAR(10))"
    )
    execute(
        "INSERT INTO test_table (id, value) VALUES (1, 'test') ON DUPLICATE KEY UPDATE value = 'test'"
    )
    execute("DROP TABLE IF EXISTS test_table")


def test_test_connection():
    """测试数据库连接测试"""
    result = test_connection()
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT 2 as num", [{"num": 2}]),
        ("SELECT NULL as val", [{"val": None}]),
    ],
)
def test_fetch_all_parametrized(sql, expected):
    """参数化测试查询"""
    result = fetch_all(sql)
    assert result == expected
