"""ResultsCapabilityProvider 的默认后备存储（spec: tool-result-recovery）。

**不是一个 core 协议的实现**——core 只认 `SpillSink`。这是
`ResultsCapabilityProvider` 内部用来放全文的容器，形状（`put` / 窗口 `get`）由该
provider 自己定义；宿主要跨进程回取，按同一形状注入自己的实现即可。

进程内可用；LRU 双上限（条数 + 总字节）逐出最旧——被逐出的键 get 返回 None，
provider 转显式未命中信息。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class InMemoryToolResultStore:
    """LRU 有界全文存储。窗口切片纯内存操作，无锁竞争热点（put/get 均持同一锁）。"""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        max_total_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._max_bytes = max(1, max_total_bytes)
        self._data: OrderedDict[str, str] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    async def put(self, invocation_id: str, text: str, ctx: Any = None) -> None:
        with self._lock:
            if invocation_id in self._data:
                self._bytes -= len(self._data[invocation_id].encode("utf-8", "replace"))
                del self._data[invocation_id]
            self._data[invocation_id] = text
            self._bytes += len(text.encode("utf-8", "replace"))
            self._evict_locked()

    async def get(
        self,
        invocation_id: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        tail: int | None = None,
        ctx: Any = None,
    ) -> str | None:
        with self._lock:
            text = self._data.get(invocation_id)
            if text is None:
                return None
            self._data.move_to_end(invocation_id)
        return _window(text, offset=offset, limit=limit, tail=tail)

    def _evict_locked(self) -> None:
        while self._data and (
            len(self._data) > self._max_entries or self._bytes > self._max_bytes
        ):
            _, victim = self._data.popitem(last=False)
            self._bytes -= len(victim.encode("utf-8", "replace"))


def _window(
    text: str,
    *,
    offset: int | None,
    limit: int | None,
    tail: int | None,
) -> str:
    """窗口切片：tail 优先（末尾 N 字符）；否则 offset（0 基）+ limit。越界钳制不抛。"""
    if tail is not None:
        n = max(0, tail)
        return text[-n:] if n else ""
    if offset is None:
        offset = 0
    offset = max(0, offset)
    if limit is None:
        return text[offset:]
    return text[offset: offset + max(0, limit)]
