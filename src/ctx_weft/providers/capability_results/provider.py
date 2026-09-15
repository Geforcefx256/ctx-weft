"""ResultsCapabilityProvider：一个**可回读的 SpillSink**（spec: tool-result-recovery）。

core 对「工具输出超长怎么办」只有一个契约——`SpillSink.spill()`：把全文交出去、拿回
一个引用。本 provider 是它的一种实现：全文落进进程内 LRU，并额外提供
`read_tool_output(invocation_id, offset | tail, limit)` 让模型按窗口取回来。

宿主二选一（或都不选）：
  - `FilesystemToolsProvider` —— 落盘成文件，宿主自己去读；模型取不回来。
  - 本 provider —— 落进内存，**模型能通过工具取回**；进程退出即失。
两者都靠 `isinstance(p, SpillSink)` 被 gateway 发现，注册一次即生效。**都不注册**则
超长输出硬截断（与改造前无 sink 时同行为）。

刻意**不另立 ToolResultStore 协议**：那与 SpillSink 是同一职责（把全文存到别处）的
第二套说法，只会让宿主面对两个语义重叠的注册点。「能不能回读」是**实现的能力**，
不是**契约的分支**——回读由本 provider 自带的工具提供，而非由 core 的协议规定。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ctx_weft.protocols.capability import (
    Capability,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.filesystem import SpillSink
from ctx_weft.providers._tooldecl import make_tool_registry

logger = logging.getLogger(__name__)

PROVIDER_NAME = "results"
#: 回读工具的 qualified 名。收敛提示里的引用由 `spill()` 生成，与此同源。
READ_TOOL_QUALIFIED_NAME = f"{PROVIDER_NAME}__read_tool_output"

tool, _RESULT_TOOLS, _RESULT_IMPLS = make_tool_registry(PROVIDER_NAME)

# 单次读取的硬上限（字符）：回读自身不回灌全文（spec R3），limit 钳制于此。
_MAX_READ_CHARS = 100_000
# 无窗口参数时的默认首页大小。
_DEFAULT_PAGE_CHARS = 20_000

_UNAVAILABLE = (
    "[no stored output for invocation '{invocation_id}': evicted or written before "
    "this session — the full text is not recoverable via read_tool_output]"
)


@tool(purposes=["act"], side_effects=False, spillable=False, recovery_policy="idempotent")
async def read_tool_output(
    invocation_id: str,
    offset: int | None = None,
    limit: int | None = None,
    tail: int | None = None,
    ctx: ProviderContext = None,  # type: ignore[assignment]
) -> str:
    """Read back the full text of a truncated tool output by its invocation id.

    A truncated tool result says "Full text is retrievable — read it with
    results__read_tool_output(invocation_id='...')" and carries the id to use.
    Use `tail=N` to read the last N chars, or `offset` + `limit` to page from the
    start (0-based). Without window args this returns the first page. The
    invocation_id is per execution attempt — a re-executed tool call has a new id
    (take it from the newest truncated result).
    """
    raise NotImplementedError("declaration only — dispatched via ResultsCapabilityProvider")


class ResultsCapabilityProvider(ToolCapabilityProvider, SpillSink):
    """可回读的 SpillSink：`spill()` 收全文，`read_tool_output` 按窗口取回。

    两个角色住在同一个对象里是有意的——只有存进去的那一方知道怎么取回来；拆成两个
    注册点只会制造「注册了回读工具却没注册对应存储」这类错配。

    ``store`` 默认 `InMemoryToolResultStore`（LRU 双上限）；宿主要跨进程回取就按同一
    形状注入自己的实现——那是**本实现的构造参数**，不是 core 的协议。
    """

    name = PROVIDER_NAME

    def __init__(self, store: Any = None) -> None:
        if store is None:
            from ctx_weft.providers.results import InMemoryToolResultStore
            store = InMemoryToolResultStore()
        self._store = store

    # ── SpillSink ─────────────────────────────────────────────────────────────

    async def spill(self, content: str, ctx: ProviderContext, *, name_hint: str = "") -> str:
        """收下全文，返回**给模型看的取回说明**（gateway 原样嵌进截断提示）。

        ``name_hint`` 是本次执行的 invocation_id——它同时是回读键，所以说明里直接写成
        可照抄的调用形态：**点名工具 + 填好 id**，模型不必从别处推。存不下就抛
        （SpillSink 契约），gateway 据此回退硬截断。
        """
        key = name_hint or "unknown"
        await self._store.put(key, content, ctx=ctx)
        return (f"Full text is retrievable — read it with "
                f"{READ_TOOL_QUALIFIED_NAME}(invocation_id='{key}', tail=N) for the last N "
                f"chars, or (invocation_id='{key}', offset=0, limit=N) to page from the start.")

    async def info(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=PROVIDER_NAME,
            description="Read back the full text of truncated tool outputs by invocation id.",
        )

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return await self.info(ctx)

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        pass  # 纯读取工具，无在途副作用可取消

    async def list(self, ctx: ProviderContext) -> list[Capability]:
        return list(_RESULT_TOOLS.values())

    def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        return self._dispatch(capability_id, arguments, ctx)

    async def _dispatch(
        self, capability_id: str, arguments: dict[str, Any], ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        try:
            out = await self._read_tool_output(
                invocation_id=str(arguments.get("invocation_id", "")),
                offset=arguments.get("offset"),
                limit=arguments.get("limit"),
                tail=arguments.get("tail"),
                ctx=ctx,
            )
            yield CapabilityEvent(kind="result", payload={"content": out})
        except Exception as exc:  # pragma: no cover — 参数已过 schema 校验
            yield CapabilityEvent(kind="error", payload={"code": "ERR", "message": str(exc)})

    async def _read_tool_output(
        self,
        invocation_id: str,
        offset: int | None = None,
        limit: int | None = None,
        tail: int | None = None,
        ctx: ProviderContext = None,  # type: ignore[assignment]
    ) -> str:
        """窗口回读（与 @tool 声明的契约一致）：tail 优先；offset+limit 分页；无窗口
        参数默认首页。limit/tail 钳制硬上限；未命中/异常 → 显式不可用（可区分空输出）。"""
        if not invocation_id:
            return "[read_tool_output requires a non-empty invocation_id]"
        store = self._store
        if limit is not None:
            limit = max(1, min(int(limit), _MAX_READ_CHARS))
        if tail is not None:
            tail = max(1, min(int(tail), _MAX_READ_CHARS))
        elif offset is None:
            offset, limit = 0, _DEFAULT_PAGE_CHARS
        try:
            chunk = await store.get(
                invocation_id, offset=offset, limit=limit, tail=tail, ctx=ctx)
        except Exception:
            logger.exception("read_tool_output: store get failed for %s", invocation_id)
            return _UNAVAILABLE.format(invocation_id=invocation_id)
        if chunk is None:
            return _UNAVAILABLE.format(invocation_id=invocation_id)
        if not chunk:
            return "(empty window)"
        note = ""
        if tail is None and limit is not None and len(chunk) >= limit:
            end = (offset or 0) + len(chunk)
            note = f"\n[window ends at char {end}; continue with offset={end}]"
        return chunk + note
