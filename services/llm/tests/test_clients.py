#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 客户端测试
"""

import pytest
from unittest.mock import MagicMock, patch
from services.llm.llm import DeepSeekClient, MiniMaxClient


class TestDeepSeekClient:
    """DeepSeek 客户端测试"""

    def test_deepseek_client_initialization(self):
        """测试 DeepSeek 客户端初始化"""
        client = DeepSeekClient()
        assert client.model == "deepseek-chat"
        assert client.base_url == "https://api.deepseek.com"

    def test_deepseek_client_custom_model(self):
        """测试自定义模型"""
        client = DeepSeekClient(model="deepseek-coder")
        assert client.model == "deepseek-coder"

    def test_deepseek_client_custom_api_key(self):
        """测试自定义 API Key"""
        client = DeepSeekClient(api_key="test_key")
        assert client.api_key == "test_key"

    def test_deepseek_client_no_api_key(self):
        """测试无 API Key 时的错误"""
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            client = DeepSeekClient()
            with pytest.raises(Exception, match="API key not configured"):
                client.chat([{"role": "user", "content": "test"}])

    def test_deepseek_is_available_without_key(self):
        """测试无 Key 时不可用"""
        with patch.dict("os.environ", {}, clear=True):
            client = DeepSeekClient(api_key="")
            assert client.is_available() is False

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_chat_success(self, mock_post):
        """测试 DeepSeek 对话成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        result = client.chat([{"role": "user", "content": "Hello"}])

        assert result == "Hello!"
        mock_post.assert_called_once()

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_chat_with_system_prompt(self, mock_post):
        """测试带 system prompt 的对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = client.chat(messages)

        assert result == "Response"

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_chat_api_error(self, mock_post):
        """测试 API 错误处理"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        with pytest.raises(Exception, match="API error: 401"):
            client.chat([{"role": "user", "content": "test"}])

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_chat_invalid_response(self, mock_post):
        """测试无效响应"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"no_choices": "error"}
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        with pytest.raises(Exception, match="Invalid response"):
            client.chat([{"role": "user", "content": "test"}])

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_chat_with_kwargs(self, mock_post):
        """测试带额外参数的对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        client.chat(
            [{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=500,
        )

        call_args = mock_post.call_args[1]
        assert call_args["json"]["temperature"] == 0.5
        assert call_args["json"]["max_tokens"] == 500

    def test_deepseek_get_provider_name(self):
        """测试获取提供商名称"""
        client = DeepSeekClient()
        assert client.get_provider_name() == "deepseek"

    @patch("services.llm.llm.deepseek.requests.post")
    def test_deepseek_is_available_with_key(self, mock_post):
        """测试有 Key 时可用"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        mock_post.return_value = mock_response

        client = DeepSeekClient(api_key="test_key")
        assert client.is_available() is True


class TestMiniMaxClient:
    """MiniMax 客户端测试"""

    def test_minimax_client_initialization(self):
        """测试 MiniMax 客户端初始化"""
        client = MiniMaxClient()
        assert client.model == "abab5.5-chat"

    def test_minimax_client_custom_model(self):
        """测试自定义模型"""
        client = MiniMaxClient(model="abab6-chat")
        assert client.model == "abab6-chat"

    def test_minimax_client_custom_api_key(self):
        """测试自定义 API Key"""
        client = MiniMaxClient(api_key="test_key")
        assert client.api_key == "test_key"

    @patch("services.llm.llm.minimax.requests.post")
    def test_minimax_chat_success(self, mock_post):
        """测试 MiniMax 对话成功"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_post.return_value = mock_response

        client = MiniMaxClient(api_key="test_key")
        result = client.chat([{"role": "user", "content": "Hello"}])

        assert result == "Hello!"
        mock_post.assert_called_once()

    def test_minimax_is_available_with_key(self):
        """测试有 Key 时可用"""
        client = MiniMaxClient(api_key="test_key")
        assert client.is_available() is True

    @patch("services.llm.llm.minimax.requests.post")
    def test_minimax_chat_with_system_prompt(self, mock_post):
        """测试带 system prompt 的对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value = mock_response

        client = MiniMaxClient(api_key="test_key")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = client.chat(messages)

        assert result == "Response"

    @patch("services.llm.llm.minimax.requests.post")
    def test_minimax_chat_api_error(self, mock_post):
        """测试 API 错误处理"""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        client = MiniMaxClient(api_key="test_key")
        with pytest.raises(Exception, match="API error: 401"):
            client.chat([{"role": "user", "content": "test"}])

    @patch("services.llm.llm.minimax.requests.post")
    def test_minimax_chat_invalid_response(self, mock_post):
        """测试无效响应"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"no_choices": "error"}
        mock_post.return_value = mock_response

        client = MiniMaxClient(api_key="test_key")
        with pytest.raises(Exception, match="Invalid response"):
            client.chat([{"role": "user", "content": "test"}])

    @patch("services.llm.llm.minimax.requests.post")
    def test_minimax_chat_with_kwargs(self, mock_post):
        """测试带额外参数的对话"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test"}}]
        }
        mock_post.return_value = mock_response

        client = MiniMaxClient(api_key="test_key")
        client.chat(
            [{"role": "user", "content": "test"}],
            temperature=0.5,
            max_tokens=500,
        )

        call_args = mock_post.call_args[1]
        assert call_args["json"]["temperature"] == 0.5
        assert call_args["json"]["max_tokens"] == 500

    def test_minimax_get_provider_name(self):
        """测试获取提供商名称"""
        client = MiniMaxClient()
        assert client.get_provider_name() == "minimax"


class TestClientIntegration:
    """客户端集成测试"""

    @patch("services.llm.llm.minimax.requests.post")
    @patch("services.llm.llm.deepseek.requests.post")
    def test_both_clients_different_providers(self, mock_minimax, mock_deepseek):
        """测试两个客户端使用不同的提供商"""
        mock_deepseek.return_value.status_code = 200
        mock_deepseek.return_value.json.return_value = {
            "choices": [{"message": {"content": "DeepSeek response"}}]
        }
        mock_minimax.return_value.status_code = 200
        mock_minimax.return_value.json.return_value = {
            "choices": [{"message": {"content": "MiniMax response"}}]
        }

        deepseek = DeepSeekClient(api_key="ds_key")
        minimax = MiniMaxClient(api_key="mm_key")

        ds_result = deepseek.chat([{"role": "user", "content": "test"}])
        mm_result = minimax.chat([{"role": "user", "content": "test"}])

        assert ds_result == "DeepSeek response"
        assert mm_result == "MiniMax response"
        assert deepseek.get_provider_name() == "deepseek"
        assert minimax.get_provider_name() == "minimax"
