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


# ---------- LSTM 训练相关表 ----------


def create_lstm_training_run_table() -> None:
    """LSTM 训练流水：每次训练（完整/增量）记录参数、指标、验证结果。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_training_run (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        version_id VARCHAR(32) DEFAULT NULL COMMENT '保存后的版本号，失败或未部署可为空',
        symbol VARCHAR(32) NOT NULL,
        training_type VARCHAR(16) NOT NULL COMMENT 'full|incremental',
        trigger_type VARCHAR(32) DEFAULT NULL COMMENT 'manual|weekly|monthly|quarterly|performance_decay',
        data_start DATE DEFAULT NULL,
        data_end DATE DEFAULT NULL,
        params_json JSON DEFAULT NULL COMMENT 'lr, hidden_size, epochs, batch_size, do_cv_tune 等',
        metrics_json JSON DEFAULT NULL COMMENT 'accuracy, f1, mse, recall',
        validation_deployed TINYINT(1) NOT NULL DEFAULT 0 COMMENT '是否通过样本外验证并部署',
        validation_reason VARCHAR(512) DEFAULT NULL,
        holdout_metrics_json JSON DEFAULT NULL,
        duration_seconds INT UNSIGNED DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_symbol (symbol),
        KEY idx_version_id (version_id),
        KEY idx_created_at (created_at),
        KEY idx_training_type (training_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LSTM训练流水'
    """
    execute(sql)


def create_lstm_current_version_table() -> None:
    """当前使用的 LSTM 模型版本（单行，id=1）。保留用于兼容，实际使用 per_symbol 表。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_current_version (
        id TINYINT UNSIGNED NOT NULL PRIMARY KEY DEFAULT 1,
        version_id VARCHAR(32) NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        CONSTRAINT single_row CHECK (id = 1)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='当前LSTM模型版本(兼容)'
    """
    execute(sql)


def create_lstm_current_version_per_symbol_table() -> None:
    """按股票+年份的当前 LSTM 模型版本：(symbol, years) -> version_id。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_current_version_per_symbol (
        symbol VARCHAR(32) NOT NULL,
        years TINYINT UNSIGNED NOT NULL COMMENT '1/2/3 年',
        version_id VARCHAR(32) NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, years),
        KEY idx_version_id (version_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='按股票+年份的当前LSTM版本'
    """
    execute(sql)


def create_lstm_prediction_log_table() -> None:
    """预测记录：按 (symbol, predict_date, years) 写入，支持 1/2/3 年模型分别记录。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_prediction_log (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        predict_date DATE NOT NULL,
        years TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '1/2/3 年模型',
        direction TINYINT NOT NULL COMMENT '0跌 1涨',
        magnitude DECIMAL(12,6) NOT NULL,
        prob_up DECIMAL(8,4) NOT NULL,
        model_version_id VARCHAR(32) DEFAULT NULL,
        source VARCHAR(16) NOT NULL DEFAULT 'lstm' COMMENT 'lstm|arima|technical',
        magnitude_5 JSON DEFAULT NULL COMMENT '5日逐日涨跌幅',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol_predict_years (symbol, predict_date, years),
        KEY idx_symbol (symbol),
        KEY idx_symbol_years (symbol, years),
        KEY idx_predict_date (predict_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LSTM预测记录(按股票+日期+年份)'
    """
    execute(sql)


def migrate_lstm_prediction_log_years() -> None:
    """为已存在的 lstm_prediction_log 表增加 years 列并调整唯一键。"""
    try:
        execute(
            "ALTER TABLE lstm_prediction_log ADD COLUMN years TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '1/2/3 年模型'"
        )
    except Exception:
        pass
    try:
        execute("ALTER TABLE lstm_prediction_log DROP INDEX uk_symbol_predict")
    except Exception:
        pass
    try:
        execute(
            "ALTER TABLE lstm_prediction_log ADD UNIQUE KEY uk_symbol_predict_years (symbol, predict_date, years)"
        )
    except Exception:
        pass
    try:
        execute(
            "ALTER TABLE lstm_prediction_log ADD KEY idx_symbol_years (symbol, years)"
        )
    except Exception:
        pass


def migrate_lstm_prediction_log_magnitude_5() -> None:
    """为 lstm_prediction_log 增加 magnitude_5 列（JSON 数组，5 日逐日涨跌幅）。"""
    try:
        execute(
            "ALTER TABLE lstm_prediction_log ADD COLUMN magnitude_5 JSON DEFAULT NULL COMMENT '5日逐日涨跌幅'"
        )
    except Exception:
        pass


def create_lstm_accuracy_record_table() -> None:
    """准确性回填：有实际行情后写入，用于监控与性能衰减判断。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_accuracy_record (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        predict_date DATE NOT NULL,
        actual_date DATE NOT NULL,
        pred_direction TINYINT NOT NULL,
        pred_magnitude DECIMAL(12,6) NOT NULL,
        actual_direction TINYINT NOT NULL,
        actual_magnitude DECIMAL(12,6) NOT NULL,
        error_magnitude DECIMAL(12,6) NOT NULL,
        direction_correct TINYINT NOT NULL COMMENT '0|1',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol_predict (symbol, predict_date),
        KEY idx_symbol (symbol),
        KEY idx_actual_date (actual_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LSTM预测准确性回填'
    """
    execute(sql)


def create_lstm_training_failure_table() -> None:
    """训练失败记录：用于告警（失败≥3次）。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_training_failure (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_symbol (symbol),
        KEY idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LSTM训练失败记录'
    """
    execute(sql)


def create_lstm_model_version_table() -> None:
    """LSTM 模型版本：按股票+年份存储，version_id、元数据 JSON、模型权重 BLOB。"""
    sql = """
    CREATE TABLE IF NOT EXISTS lstm_model_version (
        version_id VARCHAR(32) NOT NULL PRIMARY KEY,
        symbol VARCHAR(32) NOT NULL DEFAULT '',
        years TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '1/2/3 年',
        training_time VARCHAR(64) DEFAULT NULL,
        data_start DATE DEFAULT NULL,
        data_end DATE DEFAULT NULL,
        metadata_json JSON DEFAULT NULL,
        model_blob LONGBLOB DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_created_at (created_at),
        KEY idx_symbol_years (symbol, years)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='LSTM模型版本(按股票+年份)'
    """
    execute(sql)


def migrate_lstm_model_version_symbol_years() -> None:
    """为已存在的 lstm_model_version 表增加 symbol、years 列（按股票+年份存储迁移）。"""
    try:
        execute(
            "ALTER TABLE lstm_model_version ADD COLUMN symbol VARCHAR(32) NOT NULL DEFAULT ''"
        )
    except Exception:
        pass
    try:
        execute(
            "ALTER TABLE lstm_model_version ADD COLUMN years TINYINT UNSIGNED NOT NULL DEFAULT 1 COMMENT '1/2/3 年'"
        )
    except Exception:
        pass
    try:
        execute(
            "ALTER TABLE lstm_model_version ADD KEY idx_symbol_years (symbol, years)"
        )
    except Exception:
        pass


def create_lstm_tables() -> None:
    """创建 LSTM 相关全部表（幂等）。"""
    create_lstm_training_run_table()
    create_lstm_current_version_table()
    create_lstm_current_version_per_symbol_table()
    create_lstm_prediction_log_table()
    create_lstm_accuracy_record_table()
    create_lstm_training_failure_table()
    create_lstm_model_version_table()
    migrate_lstm_model_version_symbol_years()
    migrate_lstm_prediction_log_years()
    migrate_lstm_prediction_log_magnitude_5()


def create_all_tables() -> None:
    """创建全部业务表（幂等）；并执行可选迁移。"""
    create_stock_meta_table()
    migrate_stock_meta_last_trade_date()
    migrate_stock_meta_first_trade_date()
    migrate_stock_meta_remark()
    migrate_backfill_first_trade_date()
    create_stock_daily_table()
    create_lstm_tables()
    create_fund_tables()
    create_news_tables()
    create_market_tables()


# ---------- 基金相关表 ----------


def create_fund_meta_table() -> None:
    """基金基本信息表"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_meta (
        fund_code VARCHAR(10) PRIMARY KEY,
        fund_name VARCHAR(128) DEFAULT NULL,
        fund_type VARCHAR(32) DEFAULT NULL,
        manager VARCHAR(128) DEFAULT NULL,
        establishment_date DATE DEFAULT NULL,
        fund_scale DECIMAL(20,2) DEFAULT NULL,
        watchlist TINYINT(1) DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY idx_watchlist (watchlist),
        KEY idx_fund_type (fund_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_fund_nav_table() -> None:
    """基金净值表"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_nav (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        fund_code VARCHAR(10) NOT NULL,
        nav_date DATE NOT NULL,
        unit_nav DECIMAL(10,4) DEFAULT NULL,
        accum_nav DECIMAL(10,4) DEFAULT NULL,
        daily_return DECIMAL(10,4) DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_fund_date (fund_code, nav_date),
        KEY idx_code_date (fund_code, nav_date),
        KEY idx_nav_date (nav_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_index_data_table() -> None:
    """指数数据表"""
    sql = """
    CREATE TABLE IF NOT EXISTS index_data (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        index_code VARCHAR(20) NOT NULL,
        trade_date DATE NOT NULL,
        close_price DECIMAL(10,2) DEFAULT NULL,
        daily_return DECIMAL(10,4) DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_index_date (index_code, trade_date),
        KEY idx_code_date (index_code, trade_date),
        KEY idx_trade_date (trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_fund_prediction_table() -> None:
    """预测记录表"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_prediction (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        fund_code VARCHAR(10) NOT NULL,
        predict_date DATE NOT NULL,
        direction TINYINT NOT NULL,
        magnitude DECIMAL(12,6) DEFAULT NULL,
        prob_up DECIMAL(8,4) DEFAULT NULL,
        magnitude_5 JSON DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_fund_predict (fund_code, predict_date),
        KEY idx_code_date (fund_code, predict_date),
        KEY idx_predict_date (predict_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_fund_model_table() -> None:
    """模型存储表"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_model (
        fund_code VARCHAR(10) PRIMARY KEY,
        model_data LONGBLOB DEFAULT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_fund_tables() -> None:
    """创建基金相关全部表（幂等）"""
    create_fund_meta_table()
    create_fund_nav_table()
    create_index_data_table()
    create_fund_prediction_table()
    create_fund_model_table()


# ---------- 新闻相关表 ----------


def create_news_data_table() -> None:
    """新闻数据表"""
    sql = """
    CREATE TABLE IF NOT EXISTS news_data (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        news_date DATE COMMENT '新闻日期',
        title VARCHAR(256) NOT NULL,
        content TEXT,
        source VARCHAR(32),
        url VARCHAR(512),
        published_at DATETIME,
        category VARCHAR(32) DEFAULT 'general',
        sentiment VARCHAR(16) DEFAULT 'neutral',
        importance TINYINT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_url (url),
        KEY idx_news_date (news_date),
        KEY idx_category (category),
        KEY idx_published_at (published_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_news_analysis_table() -> None:
    """新闻分析结果表"""
    sql = """
    CREATE TABLE IF NOT EXISTS news_analysis (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        analysis_date DATE NOT NULL,
        news_count INT DEFAULT 0,
        summary TEXT,
        deep_analysis TEXT,
        market_impact VARCHAR(16) DEFAULT 'neutral',
        key_events JSON,
        investment_advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_date (analysis_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


# ---------- 市场数据相关表 ----------


def create_macro_data_table() -> None:
    """宏观经济数据表"""
    sql = """
    CREATE TABLE IF NOT EXISTS macro_data (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        indicator VARCHAR(32) NOT NULL,
        period VARCHAR(16) NOT NULL,
        value DECIMAL(20,4),
        unit VARCHAR(16),
        source VARCHAR(32),
        publish_date DATE,
        trade_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_indicator_period (indicator, period)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_money_flow_table() -> None:
    """资金流向表"""
    sql = """
    CREATE TABLE IF NOT EXISTS money_flow (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        trade_date DATE NOT NULL,
        north_money DECIMAL(20,2),
        north_buy DECIMAL(20,2),
        north_sell DECIMAL(20,2),
        main_money DECIMAL(20,2),
        margin_balance DECIMAL(20,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_trade_date (trade_date),
        KEY idx_trade_date (trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_market_sentiment_table() -> None:
    """市场情绪表"""
    sql = """
    CREATE TABLE IF NOT EXISTS market_sentiment (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        trade_date DATE NOT NULL,
        volume DECIMAL(20,2),
        up_count INT DEFAULT 0,
        down_count INT DEFAULT 0,
        turnover_rate DECIMAL(10,4),
        advance_count INT DEFAULT 0,
        decline_count INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_trade_date (trade_date),
        KEY idx_trade_date (trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_global_macro_table() -> None:
    """全球宏观数据表"""
    sql = """
    CREATE TABLE IF NOT EXISTS global_macro (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        trade_date DATE NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        close_price DECIMAL(14,4),
        change_pct DECIMAL(10,4),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_symbol_date (symbol, trade_date),
        KEY idx_trade_date (trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_news_tables() -> None:
    """创建新闻相关全部表"""
    create_news_data_table()
    create_news_analysis_table()
    create_news_industry_classification_table()
    create_fund_news_association_table()


def create_fund_news_association_table() -> None:
    """基金-新闻关联表"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_news_association (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        fund_code VARCHAR(16) NOT NULL,
        fund_name VARCHAR(256),
        news_id BIGINT DEFAULT 0,
        news_title VARCHAR(512),
        news_source VARCHAR(64),
        news_url VARCHAR(1024),
        industry VARCHAR(64),
        industry_code VARCHAR(8),
        match_type VARCHAR(16) DEFAULT 'industry',
        match_score DECIMAL(5,4) DEFAULT 0.0,
        created_at DATETIME,
        created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_fund_news (fund_code, news_id),
        KEY idx_fund_code (fund_code),
        KEY idx_industry_code (industry_code),
        KEY idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_news_industry_classification_table() -> None:
    """新闻行业分类表"""
    sql = """
    CREATE TABLE IF NOT EXISTS news_industry_classification (
        id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        news_id BIGINT DEFAULT 0,
        title VARCHAR(512),
        source VARCHAR(64),
        url VARCHAR(1024),
        published_at DATETIME,
        news_date DATE,
        original_category VARCHAR(32) DEFAULT 'general',
        industry VARCHAR(64),
        industry_code VARCHAR(8),
        confidence DECIMAL(5,4) DEFAULT 0.0,
        classified_at DATETIME,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_news_id (news_id),
        KEY idx_industry (industry_code),
        KEY idx_classified_at (classified_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute(sql)


def create_market_tables() -> None:
    """创建市场数据相关全部表"""
    create_macro_data_table()
    create_money_flow_table()
    create_market_sentiment_table()
    create_global_macro_table()


if __name__ == "__main__":
    """命令行执行：创建表并执行迁移。用法：python -m data.schema"""
    from data.mysql import test_connection

    if not test_connection():
        print("数据库连接失败，请检查 config.yaml 中 mysql 配置。")
        exit(1)
    create_all_tables()
    print("表已创建（含 LSTM 训练相关），迁移已执行。")
