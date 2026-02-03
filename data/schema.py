# -*- coding: utf-8 -*-
"""
数据库表结构定义与创建。依赖 data.mysql。
表设计（便于每日增量更新）：
- stock_meta: 股票元信息 + first_trade_date / last_trade_date（已有数据时间范围），更新时只拉取范围内没有的
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
    """股票元信息表：symbol 唯一；first/last_trade_date 为已有数据时间范围；remark 为用户手动输入的说明。"""
    sql = """
    CREATE TABLE IF NOT EXISTS stock_meta (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        name VARCHAR(128) DEFAULT NULL,
        market VARCHAR(8) NOT NULL DEFAULT 'a',
        first_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最早交易日',
        last_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最新交易日',
        remark VARCHAR(512) DEFAULT NULL COMMENT '用户手动输入的说明',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol (symbol),
        KEY idx_market (market),
        KEY idx_last_trade_date (last_trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_stock_daily_table() -> None:
    """日线行情表：按 (symbol, trade_date) 唯一。idx_symbol_date 可兼做按 symbol 查询，不再单独建 idx_symbol 减少写入开销。"""
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
        KEY idx_trade_date (trade_date),
        KEY idx_symbol_date (symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def migrate_stock_meta_last_trade_date() -> None:
    """为已存在的 stock_meta 表增加 last_trade_date 列与索引（列已存在则跳过）。"""
    try:
        execute("""
            ALTER TABLE stock_meta
            ADD COLUMN last_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最新交易日'
        """)
    except Exception:
        pass
    try:
        execute("ALTER TABLE stock_meta ADD KEY idx_last_trade_date (last_trade_date)")
    except Exception:
        pass


def migrate_stock_meta_first_trade_date() -> None:
    """为已存在的 stock_meta 表增加 first_trade_date 列（列已存在则跳过）。"""
    try:
        execute("""
            ALTER TABLE stock_meta
            ADD COLUMN first_trade_date DATE DEFAULT NULL COMMENT '该股已入库的最早交易日'
        """)
    except Exception:
        pass


def migrate_stock_meta_remark() -> None:
    """为已存在的 stock_meta 表增加 remark 列（用户说明）。"""
    try:
        execute("""
            ALTER TABLE stock_meta
            ADD COLUMN remark VARCHAR(512) DEFAULT NULL COMMENT '用户手动输入的说明'
        """)
    except Exception:
        pass


def migrate_backfill_first_trade_date() -> None:
    """为已有 last_trade_date 但 first_trade_date 为空的记录从 stock_daily 回填最早交易日。"""
    try:
        execute("""
            UPDATE stock_meta m
            SET first_trade_date = (
                SELECT MIN(d.trade_date) FROM stock_daily d WHERE d.symbol = m.symbol
            )
            WHERE m.first_trade_date IS NULL AND m.last_trade_date IS NOT NULL
        """)
    except Exception:
        pass


def create_all_tables() -> None:
    """创建全部业务表（幂等）；并执行可选迁移。"""
    create_stock_meta_table()
    migrate_stock_meta_last_trade_date()
    migrate_stock_meta_first_trade_date()
    migrate_stock_meta_remark()
    migrate_backfill_first_trade_date()
    create_stock_daily_table()


if __name__ == "__main__":
    """命令行执行：创建表并执行迁移。用法：python -m data.schema"""
    from data.mysql import test_connection
    if not test_connection():
        print("数据库连接失败，请检查 config.yaml 中 mysql 配置。")
        exit(1)
    create_all_tables()
    print("表已创建，迁移已执行。")
