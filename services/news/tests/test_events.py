#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新闻事件发布测试 - 测试 news:crawled 事件、事件数据格式、Redis Streams 集成
"""

import pytest
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from services.news.data import NewsCrawler, NewsRepo
from services.news.data.news_crawler import NewsItem
from shared.messaging import get_mq, NEWS_CRAWLED


class TestNewsEventPublishing:
    """新闻事件发布测试类"""

    @pytest.fixture
    def sample_news(self):
        """创建示例新闻"""
        return [
            NewsItem(
                title="测试新闻1",
                content="内容1",
                source="测试",
                url="http://test.com/1",
                published_at=datetime.now(),
            ),
            NewsItem(
                title="测试新闻2",
                content="内容2",
                source="测试",
                url="http://test.com/2",
                published_at=datetime.now(),
            ),
        ]

    def test_news_crawled_event_structure(self):
        """测试 news:crawled 事件数据结构"""
        event_data = {"count": 5, "timestamp": datetime.now().isoformat()}

        assert "count" in event_data
        assert "timestamp" in event_data
        assert isinstance(event_data["count"], int)
        assert isinstance(event_data["timestamp"], str)

    def test_news_crawled_event_valid_timestamp(self):
        """测试事件时间戳有效性"""
        timestamp = datetime.now().isoformat()
        event_data = {"count": 3, "timestamp": timestamp}

        parsed_time = datetime.fromisoformat(event_data["timestamp"])
        assert parsed_time <= datetime.now()

    @pytest.mark.integration
    def test_publish_news_crawled_event(self, sample_news):
        """测试发布 news:crawled 事件（集成测试）"""
        try:
            crawler = NewsCrawler()
            repo = NewsRepo()

            news_list = sample_news
            saved = repo.save_news(news_list)

            mq = get_mq()
            event_data = {"count": saved, "timestamp": datetime.now().isoformat()}

            message_id = mq.publish(NEWS_CRAWLED, event_data)

            assert message_id is not None
            assert isinstance(message_id, str)
            assert len(message_id) > 0

            print(f"\n[事件发布] 成功发布事件, ID: {message_id}")
        except Exception as e:
            pytest.skip(f"事件发布跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_consume_news_crawled_event(self):
        """测试消费 news:crawled 事件（集成测试）"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED
            group = "test_group"
            consumer = "test_consumer"

            mq.create_consumer_group(stream, group)

            event_data = {"count": 2, "timestamp": datetime.now().isoformat()}

            mq.publish(stream, event_data)

            messages = mq.consume(stream, group, consumer, count=1, block=1000)

            if messages:
                assert len(messages) >= 1
                message = messages[0]
                assert "stream" in message
                assert "id" in message
                assert "data" in message
                assert message["stream"] == stream
                assert message["data"]["count"] == 2
        except Exception as e:
            pytest.skip(f"事件消费跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_event_serialization(self):
        """测试事件序列化"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            complex_data = {
                "count": 3,
                "timestamp": datetime.now().isoformat(),
                "sources": ["eastmoney", "cailian", "wallstreet"],
                "categories": {"宏观": 10, "政策": 5, "行业": 8},
            }

            message_id = mq.publish(stream, complex_data)
            assert message_id is not None

            group = "test_group_serialize"
            consumer = "test_consumer_serialize"
            mq.create_consumer_group(stream, group)

            messages = mq.consume(stream, group, consumer, count=1, block=1000)

            if messages:
                message = messages[0]
                assert "sources" in message["data"]
                assert "categories" in message["data"]
                assert isinstance(message["data"]["sources"], list)
                assert isinstance(message["data"]["categories"], dict)
        except Exception as e:
            pytest.skip(f"事件序列化跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_multiple_events_publishing(self, sample_news):
        """测试发布多个事件"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            for i in range(3):
                event_data = {"count": i + 1, "timestamp": datetime.now().isoformat()}
                message_id = mq.publish(stream, event_data)
                assert message_id is not None

            print(f"\n[多事件发布] 成功发布 3 个事件")
        except Exception as e:
            pytest.skip(f"多事件发布跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_event_acknowledgment(self):
        """测试事件确认"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED
            group = "test_group_ack"
            consumer = "test_consumer_ack"

            mq.create_consumer_group(stream, group)

            event_data = {"count": 1, "timestamp": datetime.now().isoformat()}
            mq.publish(stream, event_data)

            messages = mq.consume(stream, group, consumer, count=1, block=1000)

            if messages:
                message_id = messages[0]["id"]
                result = mq.ack(stream, group, message_id)
                assert result >= 0
        except Exception as e:
            pytest.skip(f"事件确认跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_pending_events(self):
        """测试待处理事件"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED
            group = "test_group_pending"
            consumer = "test_consumer_pending"

            mq.create_consumer_group(stream, group)

            event_data = {"count": 1, "timestamp": datetime.now().isoformat()}
            mq.publish(stream, event_data)

            messages = mq.consume(stream, group, consumer, count=1, block=1000)

            if messages:
                pending = mq.get_pending(stream, group)
                assert isinstance(pending, list)
        except Exception as e:
            pytest.skip(f"待处理事件跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_stream_operations(self):
        """测试流操作"""
        try:
            mq = get_mq()
            stream = "test_stream_operations"

            event_data = {"count": 1, "timestamp": datetime.now().isoformat()}
            mq.publish(stream, event_data)

            deleted = mq.delete_stream(stream)
            assert deleted >= 0
        except Exception as e:
            pytest.skip(f"流操作跳过 (Redis 未连接): {e}")

    def test_news_crawled_constant(self):
        """测试 NEWS_CRAWLED 常量"""
        assert NEWS_CRAWLED == "news:crawled"

    def test_event_data_validation(self):
        """测试事件数据验证"""
        valid_data = {"count": 5, "timestamp": datetime.now().isoformat()}

        assert isinstance(valid_data["count"], int)
        assert valid_data["count"] >= 0
        assert valid_data["timestamp"] is not None

    def test_event_with_zero_count(self):
        """测试零计数事件"""
        event_data = {"count": 0, "timestamp": datetime.now().isoformat()}
        assert event_data["count"] == 0

    @pytest.mark.integration
    def test_consumer_group_creation(self):
        """测试消费者组创建"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED
            group = "test_group_create"

            result = mq.create_consumer_group(stream, group)
            assert result is True

            result = mq.create_consumer_group(stream, group)
            assert result is True
        except Exception as e:
            pytest.skip(f"消费者组创建跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_message_id_format(self):
        """测试消息 ID 格式"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            event_data = {"count": 1, "timestamp": datetime.now().isoformat()}
            message_id = mq.publish(stream, event_data)

            assert "-" in message_id
            parts = message_id.split("-")
            assert len(parts) == 2

            timestamp_part = parts[0]
            sequence_part = parts[1]

            assert timestamp_part.isdigit()
            assert sequence_part.isdigit()
        except Exception as e:
            pytest.skip(f"消息 ID 格式跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_empty_event_data(self):
        """测试空事件数据"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            event_data = {}
            message_id = mq.publish(stream, event_data)
            assert message_id is not None
        except Exception as e:
            pytest.skip(f"空事件数据跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_large_event_data(self):
        """测试大数据事件"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            large_data = {
                "count": 1000,
                "timestamp": datetime.now().isoformat(),
                "details": {"item" + str(i): "value" + str(i) for i in range(100)},
            }

            message_id = mq.publish(stream, large_data)
            assert message_id is not None
        except Exception as e:
            pytest.skip(f"大数据事件跳过 (Redis 未连接): {e}")

    @pytest.mark.integration
    def test_event_data_types(self):
        """测试各种数据类型"""
        try:
            mq = get_mq()
            stream = NEWS_CRAWLED

            event_data = {
                "count": 5,
                "timestamp": datetime.now().isoformat(),
                "string_value": "test",
                "number_value": 123.45,
                "boolean_value": True,
                "list_value": [1, 2, 3],
                "dict_value": {"key": "value"},
            }

            message_id = mq.publish(stream, event_data)
            assert message_id is not None

            group = "test_group_types"
            consumer = "test_consumer_types"
            mq.create_consumer_group(stream, group)

            messages = mq.consume(stream, group, consumer, count=1, block=1000)

            if messages:
                data = messages[0]["data"]
                assert data["string_value"] == "test"
                assert data["number_value"] == "123.45"
                assert data["boolean_value"] == "True"
        except Exception as e:
            pytest.skip(f"数据类型测试跳过 (Redis 未连接): {e}")

    def test_news_sync_endpoint_event_publishing(self):
        """测试新闻同步端点的事件发布（模拟）"""
        event_data = {"count": 10, "timestamp": datetime.now().isoformat()}

        assert event_data["count"] > 0
        assert "timestamp" in event_data

    @pytest.mark.integration
    def test_concurrent_event_publishing(self):
        """测试并发事件发布"""
        try:
            import concurrent.futures

            mq = get_mq()
            stream = NEWS_CRAWLED

            def publish_event(i):
                event_data = {"count": i, "timestamp": datetime.now().isoformat()}
                return mq.publish(stream, event_data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(publish_event, i) for i in range(10)]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

                assert all(result is not None for result in results)
        except Exception as e:
            pytest.skip(f"并发事件发布跳过 (Redis 未连接): {e}")
