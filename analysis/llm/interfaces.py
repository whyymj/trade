# analysis/llm/interfaces.py
"""
LLM模块接口定义
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import os


class LLMClientPort(ABC):
    """LLM 客户端端口"""

    @abstractmethod
    def chat(self, messages: List[dict], **kwargs) -> str:
        """发送对话请求"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass


class NewsAnalyzerPort(ABC):
    """新闻分析器端口"""

    @abstractmethod
    def extract_key_info(self, news_list: List[dict]) -> str:
        """提取关键信息 (MiniMax)"""
        pass

    @abstractmethod
    def analyze(self, news_list: List[dict], use_deepseek: bool = False) -> dict:
        """
        综合分析

        Args:
            news_list: 新闻列表
            use_deepseek: 是否使用 DeepSeek (测试=False, 生产=True)
        """
        pass

    @abstractmethod
    def get_available_provider(self) -> str:
        """获取当前可用的 LLM 提供商"""
        pass
