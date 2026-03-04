# modules/fund_industry/schema.py
"""
基金行业表定义
"""

from data.mysql import execute


def create_fund_industry_table() -> None:
    """基金行业表：存储基金所属行业分类"""
    sql = """
    CREATE TABLE IF NOT EXISTS fund_industry (
        id INT AUTO_INCREMENT PRIMARY KEY,
        fund_code VARCHAR(20) NOT NULL,
        industry VARCHAR(50) NOT NULL,
        confidence DECIMAL(5,2) DEFAULT 0,
        source VARCHAR(20) DEFAULT 'llm',
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_fund_industry (fund_code, industry),
        INDEX idx_fund_code (fund_code)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    execute(sql)


def drop_fund_industry_table() -> None:
    """删除基金行业表（仅用于测试）"""
    sql = "DROP TABLE IF EXISTS fund_industry"
    execute(sql)


if __name__ == "__main__":
    """命令行执行：创建表"""
    from data.mysql import test_connection

    if not test_connection():
        print("数据库连接失败，请检查 config.yaml 中 mysql 配置。")
        exit(1)
    create_fund_industry_table()
    print("fund_industry 表已创建。")
