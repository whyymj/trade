#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 服务集成测试
"""

import pytest
import json
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


class TestChatFlow:
    """完整对话流程测试"""

    def test_multi_turn_conversation(self, client, mock_cache, mock_deepseek):
        """测试多轮对话"""
        mock_deepseek.chat.side_effect = [
            "Hello! How can I help?",
            "I can analyze financial news",
            "Sure, please provide the news",
        ]

        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "What can you do?"},
        ]

        response = client.post("/api/llm/chat", json={"messages": conversation})
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"] == "I can analyze financial news"

    def test_conversation_with_context(self, client, mock_cache, mock_minimax):
        """测试带上下文的对话"""
        mock_minimax.chat.return_value = (
            "Based on previous context, this is a tech stock."
        )

        conversation = [
            {"role": "system", "content": "You are a financial analyst"},
            {"role": "user", "content": "What about Apple stock?"},
            {"role": "assistant", "content": "Apple is a tech company"},
            {"role": "user", "content": "And what about Microsoft?"},
        ]

        response = client.post(
            "/api/llm/chat",
            json={"messages": conversation, "provider": "minimax"},
        )
        assert response.status_code == 200

    def test_conversation_switching_providers(
        self, client, mock_cache, mock_deepseek, mock_minimax
    ):
        """测试切换提供商"""
        mock_deepseek.chat.return_value = "DeepSeek: Hello"
        mock_minimax.chat.return_value = "MiniMax: Hi"

        messages = [{"role": "user", "content": "Hello"}]

        response = client.post(
            "/api/llm/chat",
            json={"messages": messages, "provider": "deepseek"},
        )
        assert json.loads(response.data)["data"] == "DeepSeek: Hello"

        response = client.post(
            "/api/llm/chat",
            json={"messages": messages, "provider": "minimax"},
        )
        assert json.loads(response.data)["data"] == "MiniMax: Hi"

    def test_conversation_with_cache_hit(self, client, mock_cache):
        """测试缓存命中的对话流程"""
        mock_cache.get.return_value = "Cached answer"

        messages = [{"role": "user", "content": "What is GDP?"}]
        response = client.post("/api/llm/chat", json={"messages": messages})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["cached"] is True
        assert data["data"] == "Cached answer"


class TestNewsAnalysisFlow:
    """新闻分析流程测试"""

    def test_single_news_analysis(self, client, mock_cache):
        """测试单条新闻分析"""
        with (
            patch("services.llm.routes.llm.minimax") as mock_mm,
            patch("services.llm.routes.llm.deepseek") as mock_ds,
        ):
            mock_mm.chat.return_value = "央行宣布降息25个基点"
            mock_ds.chat.return_value = "这是宽松货币政策信号"

            news = ["央行宣布下调存款准备金率"]
            response = client.post("/api/llm/analyze-news", json={"news": news})

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True
            assert "key_info" in data["data"]
            assert "deep_analysis" in data["data"]

    def test_multiple_news_batch_analysis(self, client, mock_cache):
        """测试批量新闻分析"""
        with (
            patch("services.llm.routes.llm.minimax") as mock_mm,
            patch("services.llm.routes.llm.deepseek") as mock_ds,
        ):
            mock_mm.chat.return_value = "多条新闻关键信息"
            mock_ds.chat.return_value = "综合分析结果"

            news = [
                "央行降息",
                "GDP增长5%",
                "制造业PMI回升",
            ]
            response = client.post("/api/llm/analyze-news", json={"news": news})

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["success"] is True

    def test_news_analysis_with_cache(self, client, mock_cache):
        """测试带缓存的新闻分析"""
        mock_cache.get.return_value = None

        with (
            patch("services.llm.routes.llm.minimax") as mock_mm,
            patch("services.llm.routes.llm.deepseek") as mock_ds,
        ):
            mock_mm.chat.return_value = "Cached key info"
            mock_ds.chat.return_value = "Cached analysis"

            news = ["测试新闻"]
            response1 = client.post("/api/llm/analyze-news", json={"news": news})
            response2 = client.post("/api/llm/analyze-news", json={"news": news})

            assert response1.status_code == 200
            assert response2.status_code == 200

    def test_news_analysis_error_recovery(self, client, mock_cache):
        """测试新闻分析错误恢复"""
        with patch("services.llm.routes.llm.minimax") as mock:
            mock.chat.side_effect = Exception("MiniMax error")

            response = client.post("/api/llm/analyze-news", json={"news": ["Test"]})
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data["success"] is False


class TestIndustryClassificationFlow:
    """行业分类流程测试"""

    def test_single_text_classification(self, client, mock_cache, mock_deepseek):
        """测试单文本分类"""
        mock_deepseek.chat.return_value = "宏观"

        text = "央行宣布下调存款准备金率"
        response = client.post("/api/llm/classify-industry", json={"text": text})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["data"]["industry"] == "宏观"
        assert data["cached"] is False

    def test_multiple_text_batch_classification(
        self, client, mock_cache, mock_deepseek
    ):
        """测试批量文本分类"""
        mock_deepseek.side_effect = ["宏观", "行业", "全球", "政策", "公司"]

        texts = [
            "央行降息",
            "科技股大涨",
            "美联储加息",
            "新政策出台",
            "公司财报超预期",
        ]

        results = []
        for text in texts:
            response = client.post("/api/llm/classify-industry", json={"text": text})
            assert response.status_code == 200
            results.append(json.loads(response.data)["data"]["industry"])

        expected = ["宏观", "行业", "全球", "政策", "公司"]
        assert results == expected

    def test_classification_with_cache(self, client, mock_cache, mock_deepseek):
        """测试带缓存的分类"""
        mock_cache.get.return_value = None

        mock_deepseek.chat.return_value = "行业"

        text = "科技行业新闻"
        response1 = client.post("/api/llm/classify-industry", json={"text": text})
        response2 = client.post("/api/llm/classify-industry", json={"text": text})

        assert response1.status_code == 200
        assert response2.status_code == 200

    def test_classification_ambiguous_text(self, client, mock_cache, mock_deepseek):
        """测试模糊文本分类"""
        mock_deepseek.chat.return_value = "行业"

        text = "这是一条关于某些事件的新闻"
        response = client.post("/api/llm/classify-industry", json={"text": text})

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["data"]["industry"] == "行业"


class TestCacheHitLogic:
    """缓存命中逻辑测试"""

    def test_same_request_hits_cache(self, client, mock_cache):
        """测试相同请求命中缓存"""
        mock_cache.get.return_value = "Cached result"

        messages = [{"role": "user", "content": "Test"}]

        response1 = client.post("/api/llm/chat", json={"messages": messages})
        response2 = client.post("/api/llm/chat", json={"messages": messages})

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)
        assert data1["cached"] is True
        assert data2["cached"] is True

    def test_different_request_misses_cache(self, client, mock_cache, mock_deepseek):
        """测试不同请求未命中缓存"""
        call_count = [0]

        def get_side_effect(key):
            call_count[0] += 1
            return None

        mock_cache.get.side_effect = get_side_effect

        mock_deepseek.chat.return_value = "Response"

        messages1 = [{"role": "user", "content": "Test1"}]
        messages2 = [{"role": "user", "content": "Test2"}]

        client.post("/api/llm/chat", json={"messages": messages1})
        client.post("/api/llm/chat", json={"messages": messages2})

        assert call_count[0] == 2
        assert mock_deepseek.chat.call_count == 2

    def test_cache_key_different_for_providers(
        self, client, mock_cache, mock_deepseek, mock_minimax
    ):
        """测试不同提供商使用不同缓存键"""
        call_count = [0]

        def get_side_effect(key):
            call_count[0] += 1
            return None

        mock_cache.get.side_effect = get_side_effect

        mock_deepseek.chat.return_value = "DS response"
        mock_minimax.chat.return_value = "MM response"

        messages = [{"role": "user", "content": "Test"}]

        client.post(
            "/api/llm/chat", json={"messages": messages, "provider": "deepseek"}
        )
        client.post("/api/llm/chat", json={"messages": messages, "provider": "minimax"})

        assert call_count[0] == 2

    def test_classification_cache_independent(self, client, mock_cache, mock_deepseek):
        """测试分类缓存独立"""
        call_count = [0]

        def get_side_effect(key):
            call_count[0] += 1
            return None

        mock_cache.get.side_effect = get_side_effect

        mock_deepseek.chat.return_value = "Result"

        client.post("/api/llm/classify-industry", json={"text": "Text 1"})
        client.post("/api/llm/classify-industry", json={"text": "Text 2"})

        assert call_count[0] == 2


class TestEndToEndScenarios:
    """端到端场景测试"""

    def test_complete_workflow_chat_to_analysis(
        self, client, mock_cache, mock_deepseek, mock_minimax
    ):
        """测试从对话到分析的完整工作流"""
        mock_deepseek.chat.side_effect = ["Chat response", "Analysis"]
        mock_minimax.chat.return_value = "Key info"

        chat_response = client.post(
            "/api/llm/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

        analysis_response = client.post(
            "/api/llm/analyze-news",
            json={"news": ["Test news"]},
        )

        assert chat_response.status_code == 200
        assert analysis_response.status_code == 200

    def test_mixed_operations_sequence(
        self, client, mock_cache, mock_deepseek, mock_minimax
    ):
        """测试混合操作序列"""
        mock_deepseek.chat.return_value = "DS result"
        mock_minimax.chat.return_value = "MM result"

        operations = [
            lambda: client.post(
                "/api/llm/chat", json={"messages": [{"role": "user", "content": "Hi"}]}
            ),
            lambda: client.post("/api/llm/analyze-news", json={"news": ["News"]}),
            lambda: client.post("/api/llm/classify-industry", json={"text": "Text"}),
            lambda: client.post(
                "/api/llm/chat", json={"messages": [{"role": "user", "content": "Bye"}]}
            ),
        ]

        for op in operations:
            response = op()
            assert response.status_code == 200

    def test_concurrent_similar_requests(self, client, mock_cache, mock_deepseek):
        """测试并发相似请求"""
        mock_deepseek.chat.return_value = "Response"

        messages = [{"role": "user", "content": "Test"}]

        for _ in range(3):
            response = client.post("/api/llm/chat", json={"messages": messages})
            assert response.status_code == 200

    def test_service_health_during_operations(
        self, client, mock_cache, mock_deepseek, mock_minimax
    ):
        """测试操作期间服务健康状态"""
        health_before = client.get("/health")
        assert health_before.status_code == 200

        client.post(
            "/api/llm/chat", json={"messages": [{"role": "user", "content": "Hi"}]}
        )

        health_after = client.get("/health")
        assert health_after.status_code == 200

        metrics = client.get("/metrics")
        assert metrics.status_code == 200


class TestErrorRecovery:
    """错误恢复测试"""

    def test_provider_failure_fallback(self, client, mock_cache):
        """测试提供商故障恢复"""
        with (
            patch("services.llm.routes.llm.deepseek") as mock_ds,
            patch("services.llm.routes.llm.minimax") as mock_mm,
        ):
            mock_ds.chat.side_effect = Exception("DeepSeek error")

            response = client.post(
                "/api/llm/chat",
                json={
                    "messages": [{"role": "user", "content": "Test"}],
                    "provider": "deepseek",
                },
            )

            assert response.status_code == 500

            response = client.post(
                "/api/llm/chat",
                json={
                    "messages": [{"role": "user", "content": "Test"}],
                    "provider": "minimax",
                },
            )

            assert response.status_code == 200

    def test_cache_failure_graceful(self, client, mock_cache, mock_deepseek):
        """测试缓存故障优雅降级"""
        mock_cache.get.side_effect = Exception("Cache error")

        mock_deepseek.chat.return_value = "Response"

        response = client.post(
            "/api/llm/chat", json={"messages": [{"role": "user", "content": "Test"}]}
        )
        assert response.status_code == 200

    def test_rate_limit_recovery(self, client):
        """测试速率限制恢复"""
        with patch("services.llm.app.rate_limit_store", {}):
            for _ in range(101):
                client.post(
                    "/api/llm/chat",
                    json={"messages": [{"role": "user", "content": "Test"}]},
                )

            response = client.post(
                "/api/llm/chat",
                json={"messages": [{"role": "user", "content": "Test"}]},
            )
            assert response.status_code == 429

            import time

            time.sleep(2)

            response = client.post(
                "/api/llm/chat",
                json={"messages": [{"role": "user", "content": "Test"}]},
            )
            assert response.status_code in [200, 500]


class TestPerformance:
    """性能测试"""

    def test_response_time(self, client, mock_cache, mock_deepseek):
        """测试响应时间"""
        mock_deepseek.chat.return_value = "Response"

        import time

        start = time.time()
        response = client.post(
            "/api/llm/chat", json={"messages": [{"role": "user", "content": "Test"}]}
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 1.0

    def test_cache_performance(self, client, mock_cache):
        """测试缓存性能"""
        mock_cache.get.return_value = "Cached result"

        import time

        start = time.time()
        response = client.post(
            "/api/llm/chat", json={"messages": [{"role": "user", "content": "Test"}]}
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.5
