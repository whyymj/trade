#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redis Streams 消息队列实现
"""

import json
import os
from typing import Any, Dict, List, Optional

import redis
from dotenv import load_dotenv

load_dotenv()


class MessageQueue:
    """Redis Streams 消息队列"""

    def __init__(self):
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    def publish(self, stream: str, data: Dict[str, Any]) -> Optional[str]:
        """发布消息到流"""
        serialized_data = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                serialized_data[key] = json.dumps(value, ensure_ascii=False)
            else:
                serialized_data[key] = str(value)
        return self.redis.xadd(stream, serialized_data)

    def consume(
        self, stream: str, group: str, consumer: str, count: int = 1, block: int = 0
    ) -> List[Dict[str, Any]]:
        """从流中消费消息"""
        try:
            messages = self.redis.xreadgroup(
                group, consumer, {stream: ">"}, count=count, block=block
            )
            result = []
            for stream_name, message_list in messages:
                for message_id, message_data in message_list:
                    deserialized_data = {}
                    for key, value in message_data.items():
                        try:
                            deserialized_data[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            deserialized_data[key] = value
                    result.append(
                        {
                            "stream": stream_name,
                            "id": message_id,
                            "data": deserialized_data,
                        }
                    )
            return result
        except redis.exceptions.ResponseError as e:
            if "NOGROUP" in str(e):
                return []
            raise

    def create_consumer_group(self, stream: str, group: str) -> bool:
        """创建消费者组"""
        try:
            self.redis.xgroup_create(stream, group, id="0", mkstream=True)
            return True
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                return True
            raise

    def ack(self, stream: str, group: str, message_id: str) -> int:
        """确认消息已处理"""
        return self.redis.xack(stream, group, message_id)

    def get_pending(self, stream: str, group: str) -> List[Dict[str, Any]]:
        """获取待处理消息"""
        pending = self.redis.xpending_range(stream, group, "-", "+", 10)
        result = []
        for item in pending:
            result.append(
                {
                    "message_id": item[0].decode(),
                    "consumer": item[1].decode(),
                    "idle_time": item[2],
                    "delivery_count": item[3],
                }
            )
        return result

    def delete_stream(self, stream: str) -> int:
        """删除流"""
        return self.redis.delete(stream)


# 消息流定义
NEWS_CRAWLED = "news:crawled"
FUND_SYNC = "fund:sync"
LSTM_TRAIN = "lstm:train"

_mq_instance: Optional[MessageQueue] = None


def get_mq() -> MessageQueue:
    """获取消息队列实例（单例）"""
    global _mq_instance
    if _mq_instance is None:
        _mq_instance = MessageQueue()
    return _mq_instance
