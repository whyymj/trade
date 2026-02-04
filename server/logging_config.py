# -*- coding: utf-8 -*-
"""
服务端日志配置：文件轮转 + 控制台输出，请求访问日志。
"""
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.environ.get("LOG_DIR", str(ROOT / "logs")))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", "10")) * 1024 * 1024  # 默认 10MB
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "5"))

# 应用主日志
APP_LOG_NAME = "trade_app"
# 请求访问日志单独文件（可选，不设置则与主日志同文件）
ACCESS_LOG_NAME = "trade_access"


def _make_handler(
    log_path: Path,
    formatter: logging.Formatter,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> RotatingFileHandler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    h.setFormatter(formatter)
    return h


def init_logging(log_dir: Path | None = None) -> logging.Logger:
    """
    初始化服务端日志：主日志写文件并输出到控制台，请求日志单独文件。
    返回主 logger，供业务代码使用。
    """
    dir_path = Path(log_dir) if log_dir else LOG_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_fmt = logging.Formatter(
        "[%(levelname)s] %(name)s: %(message)s"
    )

    # 主 logger：应用与错误
    root = logging.getLogger(APP_LOG_NAME)
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    if not root.handlers:
        root.addHandler(_make_handler(dir_path / "app.log", fmt))
        ch = logging.StreamHandler()
        ch.setFormatter(console_fmt)
        root.addHandler(ch)

    # 请求访问 logger（单独文件，便于排查接口）
    access_logger = logging.getLogger(ACCESS_LOG_NAME)
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False
    if not access_logger.handlers:
        access_logger.addHandler(_make_handler(dir_path / "access.log", fmt))
        ch = logging.StreamHandler()
        ch.setFormatter(console_fmt)
        access_logger.addHandler(ch)

    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """获取业务用 logger，name 为 None 时返回主 logger。"""
    if name:
        return logging.getLogger(f"{APP_LOG_NAME}.{name}")
    return logging.getLogger(APP_LOG_NAME)


def get_access_logger() -> logging.Logger:
    """获取请求访问日志 logger。"""
    return logging.getLogger(ACCESS_LOG_NAME)


def log_request_start() -> float:
    """请求开始时调用，返回时间戳供 after_request 计算耗时。"""
    return time.perf_counter()


def log_request_end(
    method: str,
    path: str,
    status_code: int,
    start_time: float,
    extra: str = "",
) -> None:
    """请求结束时调用，写入访问日志。"""
    duration_ms = (time.perf_counter() - start_time) * 1000
    get_access_logger().info(
        "%s %s %s %.2fms%s",
        method,
        path,
        status_code,
        duration_ms,
        f" {extra}" if extra else "",
    )
