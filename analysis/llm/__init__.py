# analysis/llm/__init__.py
"""LLM Analysis Module - MiniMax M2.5"""

from .client import get_client, is_available, MiniMaxClient

__all__ = ["get_client", "is_available", "MiniMaxClient"]
