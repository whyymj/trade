"""
本地 MySQL 连接与基本访问。依赖 PyMySQL、PyYAML，配置见项目根目录 config.yaml 的 mysql 段。
"""
import os
import yaml  # type: ignore
import pymysql  # type: ignore
from contextlib import contextmanager
from typing import Any, Optional


def load_mysql_config() -> dict:
    """从 config.yaml 读取 MySQL 配置，未配置则使用默认本地参数。"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml",
    )
    defaults = {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "trade_cache",
        "charset": "utf8mb4",
    }
    if not os.path.exists(config_path):
        return defaults
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    mysql_cfg = cfg.get("mysql") or {}
    return {**defaults, **mysql_cfg}


@contextmanager
def get_connection():
    """获取数据库连接的上下文管理器，用毕自动关闭。"""
    cfg = load_mysql_config()
    conn = pymysql.connect(
        host=cfg["host"],
        port=int(cfg["port"]),
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        charset=cfg["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute(sql: str, args: Optional[tuple] = None) -> int:
    """执行一条 SQL（INSERT/UPDATE/DELETE），返回受影响行数。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            affected = cur.execute(sql, args or ())
            return affected


def execute_many(sql: str, args_list: list) -> int:
    """批量执行同一条 SQL，返回总受影响行数。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, args_list)
            return cur.rowcount


def fetch_one(sql: str, args: Optional[tuple] = None) -> Optional[dict[str, Any]]:
    """查询单行，返回字典或 None。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args or ())
            return cur.fetchone()


def fetch_all(sql: str, args: Optional[tuple] = None) -> list[dict[str, Any]]:
    """查询多行，返回字典列表。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args or ())
            return cur.fetchall()


def test_connection() -> bool:
    """测试数据库是否可连接。"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                row = cur.fetchone()
                return row is not None and row.get("ok") == 1
    except Exception:
        return False
