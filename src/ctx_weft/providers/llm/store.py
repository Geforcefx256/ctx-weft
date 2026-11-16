"""LLMAccountStore protocol — persistence interface for LLM account configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ctx_weft.providers.llm.config import LLMAccount


@runtime_checkable
class LLMAccountStoreProtocol(Protocol):
    def save(self, account: LLMAccount) -> None: ...
    def delete(self, resource_id: str, *, namespace: str = "global") -> bool: ...
    def list_all(self, *, namespace: str | None = None) -> list[LLMAccount]: ...
