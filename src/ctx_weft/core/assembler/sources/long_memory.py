"""SemanticRecallSource：MemoryProvider.recall_semantic → summary blocks。

Provider 不支持时返空（core 默认 StructuredBlackboard 即如此）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.ids import generate_id

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextBlock, ContextRequest


class SemanticRecallSource:
    """语义召回。query 默认取 task.user_prompt。"""

    name = "long_memory"

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    async def fetch(
        self,
        request: "ContextRequest",
        deps: "AssemblerDeps",
    ) -> AsyncIterator["ContextBlock"]:
        from ctx_weft.core.assembler.assembler import ContextBlock

        query: str | None = None
        if request.task.user_prompt:
            query = (
                request.task.user_prompt
                if isinstance(request.task.user_prompt, str)
                else content_to_text(request.task.user_prompt)
            )
        if not query:
            return

        records = await deps.memory.recall_semantic(
            query=query,
            scope=request.scope,
            top_k=self._top_k,
            ctx=deps.provider_ctx,
        )

        for record in records:
            yield ContextBlock(
                id=generate_id("blk"),
                source="long_memory",
                kind="summary",
                metadata={
                    "channel": "semantic",
                    "score": record.score,
                    "memory_event_id": record.id,
                },
                value=record,
            )
