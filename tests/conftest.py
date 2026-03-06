import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import os
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        "DB_HOST": os.getenv("DB_HOST", "localhost"),
        "DB_USER": os.getenv("DB_USER", "root"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", "root"),
        "DB_NAME": os.getenv("DB_NAME", "trade_cache"),
        "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
    }


@pytest.fixture(scope="session")
def app():
    from server.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture(scope="function")
def db_connection(test_config):
    """数据库连接"""
    import mysql.connector

    conn = mysql.connector.connect(
        host=test_config["DB_HOST"],
        user=test_config["DB_USER"],
        password=test_config["DB_PASSWORD"],
        database=test_config["DB_NAME"],
    )
    yield conn
    conn.close()


@pytest.fixture(scope="function")
def redis_client(test_config):
    """Redis 客户端"""
    import redis

    client = redis.from_url(test_config["REDIS_URL"])
    yield client
    client.flushdb()


@pytest.fixture(scope="function")
def clean_db(db_connection):
    """清理数据库"""
    yield
    cursor = db_connection.cursor()
    try:
        cursor.execute("DELETE FROM fund_nav WHERE fund_code LIKE 'TEST_%'")
        cursor.execute("DELETE FROM fund_meta WHERE fund_code LIKE 'TEST_%'")
        cursor.execute("DELETE FROM news WHERE title LIKE '测试%'")
        db_connection.commit()
    except Exception as e:
        db_connection.rollback()
    finally:
        cursor.close()


@pytest.fixture
def sample_fund():
    """示例基金数据"""
    return {
        "fund_code": "TEST001",
        "fund_name": "测试基金",
        "fund_type": "混合型",
        "risk_level": "中风险",
    }


@pytest.fixture
def sample_news():
    """示例新闻数据"""
    return {
        "title": "测试新闻",
        "content": "这是一条测试新闻内容",
        "source": "测试来源",
        "url": "http://test.com/news/1",
        "published_at": "2024-01-01 10:00:00",
    }
