"""
本地 MySQL 连接与基本访问。依赖 PyMySQL、PyYAML，配置见项目根目录 config.yaml 的 mysql 段。
优化：配置结果进程内缓存，减少重复读文件；批量操作可用 run_connection 单连接多语句。
"""
import os
import yaml  # type: ignore
import pymysql  # type: ignore
from contextlib import contextmanager
from typing import Any, Callable, Optional

_config_cache: Optional[dict] = None


def load_mysql_config() -> dict:
    """从 config.yaml 读取 MySQL 配置（进程内缓存），未配置则使用默认本地参数。
    环境变量 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE 会覆盖文件与默认值，便于 Docker 等部署。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
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
        _config_cache = {**defaults}
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        mysql_cfg = cfg.get("mysql") or {}
        _config_cache = {**defaults, **mysql_cfg}
    # 环境变量覆盖（Docker / 生产部署）
    if os.environ.get("MYSQL_HOST"):
        _config_cache["host"] = os.environ["MYSQL_HOST"]
    if os.environ.get("MYSQL_PORT"):
        _config_cache["port"] = int(os.environ["MYSQL_PORT"])
    if os.environ.get("MYSQL_USER"):
        _config_cache["user"] = os.environ["MYSQL_USER"]
    if os.environ.get("MYSQL_PASSWORD") is not None:
        _config_cache["password"] = os.environ["MYSQL_PASSWORD"]
    if os.environ.get("MYSQL_DATABASE"):
        _config_cache["database"] = os.environ["MYSQL_DATABASE"]
    return _config_cache


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


def run_connection(fn: Callable[[Any], Any]) -> Any:
    """
    在单次连接内执行自定义逻辑，用于需多句 SQL 的批量操作，减少连接次数。
    fn(conn) 可多次使用 conn.cursor() 执行 execute/executemany/fetch；返回 fn 的返回值。
    """
    with get_connection() as conn:
        return fn(conn)


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
