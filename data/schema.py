# -*- coding: utf-8 -*-
"""
数据库表结构定义与创建。依赖 data.mysql。
表设计（便于每日增量更新）：
- stock_meta: 股票元信息 + last_trade_date（该股已入库的最新交易日），日更时按此拉取增量
- stock_daily: 日线行情，按 (symbol, trade_date) 唯一，INSERT ON DUPLICATE KEY UPDATE 即可日更
"""
from data.mysql import execute

# akshare 列名（中文）-> 表字段名 映射；顺序与表 stock_daily 一致（open, high, low, close...）
AKSHARE_DAILY_COLUMNS = [
    ("日期", "trade_date"),
    ("开盘", "open"),
    ("最高", "high"),
    ("最低", "low"),
    ("收盘", "close"),
    ("成交量", "volume"),
    ("成交额", "amount"),
    ("振幅", "amplitude"),
    ("涨跌幅", "change_pct"),
    ("涨跌额", "change_amt"),
    ("换手率", "turnover_rate"),
]


def create_stock_meta_table() -> None:
    """股票元信息表：symbol 唯一；last_trade_date 为该股已入库的最新交易日，供日更增量拉取。"""
    sql = """
    CREATE TABLE IF NOT EXISTS stock_meta (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        name VARCHAR(128) DEFAULT NULL,
        market VARCHAR(8) NOT NULL DEFAULT 'a',
        last_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最新交易日，日更时从此日期之后拉取',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol (symbol),
        KEY idx_market (market),
        KEY idx_last_trade_date (last_trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_stock_daily_table() -> None:
    """日线行情表：按 (symbol, trade_date) 唯一，索引支持按股、按日期区间查询。"""
    sql = """
    CREATE TABLE IF NOT EXISTS stock_daily (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        trade_date DATE NOT NULL,
        open DECIMAL(14,4) DEFAULT NULL,
        high DECIMAL(14,4) DEFAULT NULL,
        low DECIMAL(14,4) DEFAULT NULL,
        close DECIMAL(14,4) DEFAULT NULL,
        volume BIGINT UNSIGNED DEFAULT NULL,
        amount DECIMAL(20,2) DEFAULT NULL,
        amplitude DECIMAL(10,4) DEFAULT NULL,
        change_pct DECIMAL(10,4) DEFAULT NULL,
        change_amt DECIMAL(14,4) DEFAULT NULL,
        turnover_rate DECIMAL(10,4) DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol_date (symbol, trade_date),
        KEY idx_symbol (symbol),
        KEY idx_trade_date (trade_date),
        KEY idx_symbol_date (symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def migrate_stock_meta_last_trade_date() -> None:
    """为已存在的 stock_meta 表增加 last_trade_date 列（无则跳过）。"""
    try:
        execute("""
            ALTER TABLE stock_meta
            ADD COLUMN last_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最新交易日'
            AFTER market
        """)
    except Exception:
        pass

def create_all_tables() -> None:
    """创建全部业务表（幂等）；并执行可选迁移。"""
    create_stock_meta_table()
    migrate_stock_meta_last_trade_date()
    create_stock_daily_table()
