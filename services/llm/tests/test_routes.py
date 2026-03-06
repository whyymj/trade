#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 路由测试
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch


@pytest.fixture
def app():
    """Flask 应用"""
    from services.llm.app import app

    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Flask 测试客户端"""
    return app.test_client()


@pytest.fixture
def mock_cache():
    """Mock 缓存"""
    with patch("services.llm.routes.llm.get_cache") as mock:
        cache = MagicMock()
        cache.get.return_value = None
        mock.return_value = cache
        yield cache


@pytest.fixture
def mock_deepseek():
    """Mock DeepSeek 客户端"""
    with patch("services.llm.routes.llm.deepseek") as mock:
        mock.chat.return_value = "DeepSeek response"
        yield mock


@pytest.fixture
def mock_minimax():
    """Mock MiniMax 客户端"""
    with patch("services.llm.routes.llm.minimax") as mock:
        mock.chat.return_value = "MiniMax response"
        yield mock


class TestHealthEndpoints:
    """健康检查端点测试"""

    def test_health(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "llm-service"

    def test_metrics(self, client):
        """测试指标端点"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["service"] == "llm-service"
        assert "rate_limit_store_size" in data


class TestChatEndpoint:
    """对话端点测试"""

    def test_chat_no_messages(self, client, mock_cache):
        """测试对话 API - 无消息"""
        response = client.post("/api/llm/chat", json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_chat_invalid_provider(self, client, mock_cache):
        """测试对话 API - 无效 provider"""
        response = client.post(
            "/api/llm/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "provider": "invalid",
            },
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_chat_deepseek_success(self, client, mock_cache, mock_deepseek):
        """测试 DeepSeek 对话成功"""
        response = client.post(
            "/api/llm/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "provider": "deepseek",
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"] == "DeepSeek response"
        assert data["cached"] is False
        mock_deepseek.chat.assert_called_once()

    def test_chat_minimax_success(self, client, mock_cache, mock_minimax):
        """测试 MiniMax 对话成功"""
        response = client.post(
            "/api/llm/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "provider": "minimax",
            },
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"] == "MiniMax response"
        assert data["cached"] is False
        mock_minimax.chat.assert_called_once()

    def test_chat_cache_hit(self, client, mock_cache):
        """测试缓存命中"""
        mock_cache.get.return_value = "Cached response"

        response = client.post(
            "/api/llm/chat",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"] == "Cached response"
        assert data["cached"] is True

    def test_chat_default_provider(self, client, mock_cache, mock_deepseek):
        """测试默认提供商"""
        response = client.post(
            "/api/llm/chat",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 200
        mock_deepseek.chat.assert_called_once()

    def test_chat_error_handling(self, client, mock_cache):
        """测试错误处理"""
        with patch("services.llm.routes.llm.deepseek") as mock:
            mock.chat.side_effect = Exception("API error")

            response = client.post(
                "/api/llm/chat",
                json={"messages": [{"role": "user", "content": "test"}]},
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["success"] is False

    def test_chat_rate_limit(self, client):
        """测试速率限制"""
        with patch("services.llm.app.rate_limit_store", {}):
            for _ in range(101):
                response = client.post(
                    "/api/llm/chat",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )
                if _ == 100:
                    assert response.status_code == 429
                else:
                    assert response.status_code in [200, 500, 429]

    def test_chat_with_system_prompt(self, client, mock_cache, mock_deepseek):
        """测试带 system prompt 的对话"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        response = client.post("/api/llm/chat", json={"messages": messages})
        assert response.status_code == 200
        mock_deepseek.chat.assert_called_once()


class TestAnalyzeNewsEndpoint:
    """新闻分析端点测试"""

    def test_analyze_news_no_news(self, client, mock_cache):
        """测试新闻分析 API - 无新闻"""
        response = client.post("/api/llm/analyze-news", json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_analyze_news_success(
        self, client, mock_cache, mock_minimax, mock_deepseek
    ):
        """测试新闻分析成功"""
        mock_minimax.chat.return_value = "Key info extracted"
        mock_deepseek.chat.return_value = "Deep analysis"

        response = client.post(
            "/api/llm/analyze-news",
            json={"news": ["News 1", "News 2"]},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "key_info" in data["data"]
        assert "deep_analysis" in data["data"]
        assert data["data"]["key_info"] == "Key info extracted"
        assert data["data"]["deep_analysis"] == "Deep analysis"

    def test_analyze_news_multiple_news(self, client, mock_cache):
        """测试多条新闻分析"""
        with (
            patch("services.llm.routes.llm.minimax") as mock_mm,
            patch("services.llm.routes.llm.deepseek") as mock_ds,
        ):
            mock_mm.chat.return_value = "Key info"
            mock_ds.chat.return_value = "Analysis"

            news_list = [f"News {i}" for i in range(10)]
            response = client.post("/api/llm/analyze-news", json={"news": news_list})

            assert response.status_code == 200

    def test_analyze_news_error_handling(self, client, mock_cache):
        """测试错误处理"""
        with patch("services.llm.routes.llm.minimax") as mock:
            mock.chat.side_effect = Exception("API error")

            response = client.post(
                "/api/llm/analyze-news",
                json={"news": ["News 1"]},
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["success"] is False

    def test_analyze_news_rate_limit(self, client):
        """测试速率限制"""
        with patch("services.llm.app.rate_limit_store", {}):
            for _ in range(51):
                response = client.post(
                    "/api/llm/analyze-news",
                    json={"news": ["News"]},
                )
                if _ == 50:
                    assert response.status_code == 429


class TestClassifyIndustryEndpoint:
    """行业分类端点测试"""

    def test_classify_industry_no_text(self, client, mock_cache):
        """测试行业分类 API - 无文本"""
        response = client.post("/api/llm/classify-industry", json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data["success"] is False

    def test_classify_industry_success(self, client, mock_cache, mock_deepseek):
        """测试行业分类成功"""
        mock_deepseek.chat.return_value = "行业"

        response = client.post(
            "/api/llm/classify-industry",
            json={"text": "这是一条关于科技行业的新闻"},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["industry"] == "行业"
        assert data["cached"] is False

    def test_classify_industry_cache_hit(self, client, mock_cache):
        """测试缓存命中"""
        mock_cache.get.return_value = "宏观"

        response = client.post(
            "/api/llm/classify-industry",
            json={"text": "测试文本"},
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["industry"] == "宏观"
        assert data["cached"] is True

    def test_classify_industry_different_texts(self, client, mock_cache, mock_deepseek):
        """测试不同文本分类"""
        mock_deepseek.side_effect = ["宏观", "行业", "全球"]

        texts = [
            "央行宣布降息",
            "科技行业蓬勃发展",
            "美联储加息",
        ]

        for i, text in enumerate(texts):
            response = client.post(
                "/api/llm/classify-industry",
                json={"text": text},
            )
            assert response.status_code == 200

    def test_classify_industry_error_handling(self, client, mock_cache):
        """测试错误处理"""
        with patch("services.llm.routes.llm.deepseek") as mock:
            mock.chat.side_effect = Exception("API error")

            response = client.post(
                "/api/llm/classify-industry",
                json={"text": "测试文本"},
            )
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["success"] is False

    def test_classify_industry_rate_limit(self, client):
        """测试速率限制"""
        with patch("services.llm.app.rate_limit_store", {}):
            for _ in range(51):
                response = client.post(
                    "/api/llm/classify-industry",
                    json={"text": "测试文本"},
                )
                if _ == 50:
                    assert response.status_code == 429


class TestRateLimiting:
    """速率限制测试"""

    def test_rate_limit_cleanup(self, client):
        """测试速率限制清理"""
        with patch("services.llm.app.rate_limit_store", {}):
            for _ in range(5):
                client.post(
                    "/api/llm/chat",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )

            from services.llm.app import rate_limit_store

            initial_size = len(rate_limit_store)

            time.sleep(2)
            for _ in range(5):
                client.post(
                    "/api/llm/chat",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )

            final_size = len(rate_limit_store)
            assert final_size <= initial_size

    def test_rate_limit_different_endpoints(self, client):
        """测试不同端点独立限流"""
        with patch("services.llm.app.rate_limit_store", {}):
            chat_count = 0
            analyze_count = 0

            for _ in range(150):
                r = client.post(
                    "/api/llm/chat",
                    json={"messages": [{"role": "user", "content": "test"}]},
                )
                if r.status_code == 429:
                    chat_count += 1

                r = client.post("/api/llm/analyze-news", json={"news": ["test"]})
                if r.status_code == 429:
                    analyze_count += 1

            assert chat_count >= 1
            assert analyze_count >= 1


class TestCacheFunctionality:
    """缓存功能测试"""

    def test_cache_key_generation(self, client, mock_cache):
        """测试缓存键生成"""
        messages = [{"role": "user", "content": "test"}]
        response = client.post("/api/llm/chat", json={"messages": messages})

        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[0][2] == 86400

    def test_cache_ttl_setting(self, client, mock_cache):
        """测试 TTL 设置"""
        client.post(
            "/api/llm/chat", json={"messages": [{"role": "user", "content": "test"}]}
        )
        client.post(
            "/api/llm/classify-industry",
            json={"text": "测试文本"},
        )

        assert mock_cache.set.call_count == 2

    def test_cache_not_hit_on_different_messages(
        self, client, mock_cache, mock_deepseek
    ):
        """测试不同消息不命中缓存"""
        mock_cache.get.return_value = None

        messages_list = [
            [{"role": "user", "content": "test1"}],
            [{"role": "user", "content": "test2"}],
        ]

        for messages in messages_list:
            client.post("/api/llm/chat", json={"messages": messages})

        assert mock_deepseek.chat.call_count == 2


class TestParameterValidation:
    """参数验证测试"""

    def test_empty_messages_array(self, client, mock_cache):
        """测试空消息数组"""
        response = client.post("/api/llm/chat", json={"messages": []})
        assert response.status_code == 400

    def test_invalid_message_format(self, client, mock_cache, mock_deepseek):
        """测试无效消息格式"""
        response = client.post(
            "/api/llm/chat",
            json={"messages": [{"invalid": "format"}]},
        )
        assert response.status_code in [200, 500]

    def test_missing_required_fields(self, client, mock_cache):
        """测试缺少必需字段"""
        response = client.post("/api/llm/analyze-news", json={"invalid": "field"})
        assert response.status_code == 400

    def test_extra_fields_ignored(self, client, mock_cache, mock_deepseek):
        """测试额外字段被忽略"""
        response = client.post(
            "/api/llm/chat",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "extra": "field",
            },
        )
        assert response.status_code == 200
