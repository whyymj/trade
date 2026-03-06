#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Redis Streams 消息队列
"""

import pytest
from shared.messaging import get_mq, NEWS_CRAWLED, FUND_SYNC, LSTM_TRAIN


def test_create_consumer_group():
    """测试创建消费者组"""
    mq = get_mq()
    result = mq.create_consumer_group(NEWS_CRAWLED, "test_group")
    assert result is True


def test_publish_consume():
    """测试消息发布和消费"""
    mq = get_mq()
    mq.create_consumer_group(NEWS_CRAWLED, "test_group_consume")

    message_id = mq.publish(NEWS_CRAWLED, {"test": "data", "number": 123})
    assert message_id is not None

    messages = mq.consume(
        NEWS_CRAWLED, "test_group_consume", "test_consumer", count=1, block=1000
    )
    assert len(messages) > 0
    assert messages[0]["data"]["test"] == "data"
    assert messages[0]["data"]["number"] == 123


def test_ack_message():
    """测试确认消息"""
    mq = get_mq()
    mq.create_consumer_group(FUND_SYNC, "test_group_ack")

    mq.publish(FUND_SYNC, {"action": "sync"})
    messages = mq.consume(
        FUND_SYNC, "test_group_ack", "test_consumer_ack", count=1, block=1000
    )

    if messages:
        result = mq.ack(FUND_SYNC, "test_group_ack", messages[0]["id"])
        assert result >= 0


def test_get_pending():
    """测试获取待处理消息"""
    mq = get_mq()
    mq.create_consumer_group(LSTM_TRAIN, "test_group_pending")

    mq.publish(LSTM_TRAIN, {"action": "train", "model": "lstm"})
    messages = mq.consume(
        LSTM_TRAIN, "test_group_pending", "test_consumer_pending", count=1, block=1000
    )

    pending = mq.get_pending(LSTM_TRAIN, "test_group_pending")
    assert isinstance(pending, list)


def test_multiple_streams():
    """测试多个流"""
    mq = get_mq()

    mq.create_consumer_group(NEWS_CRAWLED, "test_group_multi")
    mq.create_consumer_group(FUND_SYNC, "test_group_multi")
    mq.create_consumer_group(LSTM_TRAIN, "test_group_multi")

    mq.publish(NEWS_CRAWLED, {"type": "news"})
    mq.publish(FUND_SYNC, {"type": "fund"})
    mq.publish(LSTM_TRAIN, {"type": "lstm"})

    messages_news = mq.consume(
        NEWS_CRAWLED, "test_group_multi", "consumer1", count=1, block=1000
    )
    messages_fund = mq.consume(
        FUND_SYNC, "test_group_multi", "consumer2", count=1, block=1000
    )
    messages_lstm = mq.consume(
        LSTM_TRAIN, "test_group_multi", "consumer3", count=1, block=1000
    )

    assert len(messages_news) > 0
    assert len(messages_fund) > 0
    assert len(messages_lstm) > 0


@pytest.mark.parametrize(
    "stream,data",
    [
        (NEWS_CRAWLED, {"title": "test news", "url": "http://example.com"}),
        (FUND_SYNC, {"fund_code": "000001", "action": "sync"}),
        (LSTM_TRAIN, {"model": "lstm", "epochs": 100}),
    ],
)
def test_publish_various_messages(stream, data):
    """测试不同类型的消息"""
    mq = get_mq()
    mq.create_consumer_group(stream, "test_group_various")

    message_id = mq.publish(stream, data)
    assert message_id is not None

    messages = mq.consume(
        stream, "test_group_various", "consumer_various", count=1, block=1000
    )
    assert len(messages) > 0
    for key, value in data.items():
        assert messages[0]["data"][key] == value
