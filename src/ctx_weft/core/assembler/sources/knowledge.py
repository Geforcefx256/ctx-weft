"""KnowledgeRetrievalSource：KnowledgeProvider.retrieve → reference blocks。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import KnowledgeQuery

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextBlock, ContextRequest


class KnowledgeRetrievalSource:
    """从所有注册的 KnowledgeProvider 检索参考资料。

    query 取 task.user_prompt 或 task.description。
    """

    name = "knowledge"

    def __init__(self, top_k: int = 5, intent: str | None = None) -> None:
        self._top_k = top_k
        self._intent = intent

    async def fetch(
        self,
        request: "ContextRequest",
        deps: "AssemblerDeps",
    ) -> AsyncIterator["ContextBlock"]:
        from ctx_weft.core.assembler.assembler import ContextBlock

        if not deps.knowledge_providers:
            return

        query_text: str | None = None
        if request.task.user_prompt:
            query_text = (
                request.task.user_prompt
                if isinstance(request.task.user_prompt, str)
                else content_to_text(request.task.user_prompt)
            )
        elif request.task.description:
            query_text = request.task.description
        if not query_text:
            return

        query = KnowledgeQuery(text=query_text, intent=self._intent, top_k=self._top_k)

        for provider in deps.knowledge_providers:
            async for doc in provider.retrieve(query, deps.provider_ctx):
                yield ContextBlock(
                    id=generate_id("blk"),
                    source=f"knowledge:{provider.name}",
                    kind="reference",
                    metadata={
                        "doc_id": doc.id,
                        "score": doc.score,
                        "source": doc.source,
                        "citation": doc.citation,
                    },
                    value=doc,
                )
