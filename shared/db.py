#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MySQL 连接池实现，基于 SQLAlchemy
提供与原有 data/mysql.py 兼容的接口
"""

import os
from contextlib import contextmanager
from typing import Any, Callable, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()


_config_cache: Optional[dict] = None


def load_mysql_config() -> dict:
    """加载 MySQL 配置（进程内缓存），环境变量优先"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    defaults = {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "trade_cache",
        "charset": "utf8mb4",
    }

    _config_cache = {
        **defaults,
        "host": os.getenv("MYSQL_HOST", defaults["host"]),
        "port": int(os.getenv("MYSQL_PORT", defaults["port"])),
        "user": os.getenv("MYSQL_USER", defaults["user"]),
        "password": os.getenv("MYSQL_PASSWORD", defaults["password"]),
        "database": os.getenv("MYSQL_DATABASE", defaults["database"]),
        "charset": defaults["charset"],
    }
    return _config_cache


def _get_db_url() -> str:
    """获取数据库连接 URL"""
    cfg = load_mysql_config()
    return (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        f"?charset={cfg['charset']}"
    )


engine = create_engine(
    _get_db_url(),
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)


@contextmanager
def get_connection():
    """获取数据库连接（上下文管理器）"""
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute(sql: str, args: Optional[tuple] = None) -> int:
    """执行一条 SQL（INSERT/UPDATE/DELETE），返回受影响行数"""
    with get_connection() as conn:
        result = conn.execute(text(sql), args or ())
        return result.rowcount


def execute_many(sql: str, args_list: list) -> int:
    """批量执行同一条 SQL，返回总受影响行数"""
    total = 0
    with get_connection() as conn:
        for args in args_list:
            result = conn.execute(text(sql), args or ())
            total += result.rowcount
    return total


def fetch_one(sql: str, args: Optional[tuple] = None) -> Optional[dict[str, Any]]:
    """查询单行，返回字典或 None"""
    with get_connection() as conn:
        result = conn.execute(text(sql), args or ())
        row = result.fetchone()
        if row:
            return dict(row._mapping)
        return None


def fetch_all(sql: str, args: Optional[tuple] = None) -> list[dict[str, Any]]:
    """查询多行，返回字典列表"""
    with get_connection() as conn:
        result = conn.execute(text(sql), args or ())
        return [dict(row._mapping) for row in result]


def run_connection(fn: Callable[[Any], Any]) -> Any:
    """在单次连接内执行自定义逻辑"""
    with get_connection() as conn:
        return fn(conn)


def test_connection() -> bool:
    """测试数据库是否可连接"""
    try:
        with get_connection() as conn:
            result = conn.execute(text("SELECT 1 AS ok"))
            row = result.fetchone()
            return row is not None and row._mapping["ok"] == 1
    except Exception:
        return False
