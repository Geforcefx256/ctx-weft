"""ResultsCapabilityProvider：工具长输出的回读工具（spec: tool-result-recovery）。

`read_tool_output(invocation_id, offset | tail, limit)`——对收敛过的工具输出按窗口
回取（分页 / 末尾直读）。核心回取通路（非 extras）：runtime 构造期自动注册，收敛文本
中的引用指向的 qualified 名由 `protocols.results.READ_TOOL_QUALIFIED_NAME` 同源钉死。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from ctx_weft.protocols.capability import (
    Capability,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.providers._tooldecl import make_tool_registry

logger = logging.getLogger(__name__)

PROVIDER_NAME = "results"

tool, _RESULT_TOOLS, _RESULT_IMPLS = make_tool_registry(PROVIDER_NAME)

# 单次读取的硬上限（字符）：回读自身不回灌全文（spec R3），limit 钳制于此。
_MAX_READ_CHARS = 100_000
# 无窗口参数时的默认首页大小。
_DEFAULT_PAGE_CHARS = 20_000

_UNAVAILABLE = (
    "[no stored output for invocation '{invocation_id}': evicted or written before "
    "this session — the full text is not recoverable via read_tool_output]"
)


@tool(purposes=["act"], side_effects=False, spillable=False)
async def read_tool_output(
    invocation_id: str,
    offset: int | None = None,
    limit: int | None = None,
    tail: int | None = None,
    ctx: ProviderContext = None,  # type: ignore[assignment]
) -> str:
    """Read back the full text of a truncated tool output by its invocation id.

    A truncated tool result includes a line like
    "[Tool output truncated: N chars ...; full text available via "
    "results__read_tool_output(invocation_id='...')]".
    Use `tail=N` to read the last N chars, or `offset` + `limit` to page from the
    start (0-based). Without window args this returns the first page. The
    invocation_id is per execution attempt — a re-executed tool call has a new id
    (take it from the newest truncated result).
    """
    raise NotImplementedError("declaration only — dispatched via ResultsCapabilityProvider")


class ResultsCapabilityProvider(ToolCapabilityProvider):
    """提供 read_tool_output（唯一工具）。store 经构造注入的 getter 惰性解析
    （registry 两级解析在注册后仍可换实现）。"""

    name = PROVIDER_NAME

    def __init__(self, store_getter: Callable[[], Any]) -> None:
        self._store_getter = store_getter

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
        store = self._store_getter()
        if store is None:
            return _UNAVAILABLE.format(invocation_id=invocation_id)
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
