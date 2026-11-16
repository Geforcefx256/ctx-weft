"""Default LLM provider implementations, loaded only when explicitly accessed.

``OpenAIAdapter`` uses the official Python SDK for ``openai-official``.
``OpenAICompatibleAdapter`` and ``OpenAICompatibleMultimodalAdapter`` preserve
legacy Chat Completions gateway behavior for ``openai-compatible`` and ``openai``.
Install ``ctx-weft[llm]`` for HTTP/SDK adapters. Mock-only users need neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ctx_weft.protocols.llm import ModelCallConfig, OutputPolicy
from ctx_weft.providers.llm.adapters.config import AnthropicCallConfig
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.llm.config import (
    SUPPORTED_PROTOCOLS,
    LLMAccount,
    LLMModelNotFoundError,
    LLMProtocol,
    ModelCapabilities,
    ModelConfig,
    ModelLimits,
    ReasoningCapabilities,
    ReasoningLevel,
)
from ctx_weft.providers.llm.provider import LLMProvider
from ctx_weft.providers.llm.store import LLMAccountStoreProtocol

if TYPE_CHECKING:  # 让类型检查器/IDE 看得到，但运行时不触发 httpx 导入
    from ctx_weft.providers.llm.adapters.anthropic import (
        AnthropicAdapter,
        AnthropicMultimodalAdapter,
    )
    from ctx_weft.providers.llm.adapters.openai import OpenAIAdapter
    from ctx_weft.providers.llm.adapters.openai_compatible import (
        OpenAICompatibleAdapter,
        OpenAICompatibleMultimodalAdapter,
    )

__all__ = [
    "ModelCallConfig", "AnthropicCallConfig",
    "OutputPolicy",
    "SUPPORTED_PROTOCOLS",
    "AnthropicAdapter",
    "AnthropicMultimodalAdapter",
    "LLMAccount",
    "LLMAccountStoreProtocol",
    "LLMModelNotFoundError",
    "LLMProtocol",
    "LLMProvider",
    "MockLLMAdapter",
    "MockResponse",
    "ModelCapabilities",
    "ModelConfig",
    "ModelLimits",
    "ReasoningCapabilities", "ReasoningLevel",
    "OpenAIAdapter",
    "OpenAICompatibleAdapter",
    "OpenAICompatibleMultimodalAdapter",
]


def __getattr__(name: str):
    """惰性导出 httpx-dependent adapter（仅在被访问时才 import）。"""
    if name in (
        "AnthropicAdapter", "AnthropicMultimodalAdapter",
        "OpenAIAdapter", "OpenAICompatibleAdapter", "OpenAICompatibleMultimodalAdapter",
    ):
        from ctx_weft.providers.llm import adapters

        return getattr(adapters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
