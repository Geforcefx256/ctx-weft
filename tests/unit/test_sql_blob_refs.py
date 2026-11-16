"""SqlMemoryProvider 的 blob **引用边**（字节已不在 RDBMS，见
docs/superpowers/specs/2026-08-29-providers-layout-and-sql-event-store-design.md §5）。

本 provider 只负责两件事：ingest 时把引用边与事件行写进同一个事务；把活引用吐出来
（live_blob_refs）供宿主喂给 FsBlobStore.collect() 做 mark-sweep 的 mark 输入。
字节的存取与回收都归 blob store，本文件不测那些。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from ctx_weft.core.utils.content import collect_blob_refs
from ctx_weft.protocols import (
    EventBlobStore,
    MemoryAddress,
    MemoryBlobStore,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    load_records,
)
from ctx_weft.protocols.content import ImagePart, TextPart
from ctx_weft.providers.memory.sql import (
    MemoryBlobRefModel,
    SqlMemoryProvider,
    open_sqlite_memory,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_REF_A = f"blob:{_SHA_A}"
_REF_B = f"blob:{_SHA_B}"

_ADDR = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")


@asynccontextmanager
async def _provider(tmp_path):
    async with open_sqlite_memory(tmp_path / "mem.db") as m:
        yield m


def _ctx(tenant: str = "default") -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id=tenant)


def _event(ref: str | None = None, *, blob_refs: list[str] | None = None) -> MemoryEvent:
    content = [TextPart(text="hi")]
    if ref is not None:
        content.append(ImagePart(data=ref, media_type="image/png", source_type="ref"))
    return MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN,
        scope=MemoryScope.TASK,
        address=_ADDR,
        content=content,
        timestamp=datetime.now(UTC),
        role="user",
        blob_refs=list(blob_refs or []),
    )


def _multi_ref_event(*refs: str) -> MemoryEvent:
    """一条引用了若干 blob ref 的对话记录（外部化之后的真实形态，多图场景）。"""
    content: list[Any] = [TextPart(text="look at this")]
    content += [
        ImagePart(data=ref, media_type="image/png", source_type="ref", byte_size=64)
        for ref in refs
    ]
    return MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN,
        scope=MemoryScope.TASK,
        address=_ADDR,
        content=content,
        timestamp=datetime.now(UTC),
        role="user",
    )


async def _ref_rows(m: SqlMemoryProvider) -> set[tuple[str, str]]:
    """直读 ``memory_blob_refs`` 表——验的是逐行 (event_id, sha) 配对，不是聚合视图。"""
    async with m._factory() as db:  # noqa: SLF001 - 测试需要越过协议看内部表
        rows = await db.execute(
            select(MemoryBlobRefModel.event_id, MemoryBlobRefModel.sha))
        return {(e, s) for e, s in rows.all()}


async def test_provider_is_no_longer_a_blob_store(tmp_path):
    """字节离开 RDBMS：本 provider 不再实现任何 blob 协议（spec §5.2）。"""
    async with _provider(tmp_path) as m:
        assert not isinstance(m, MemoryBlobStore)
        assert not isinstance(m, EventBlobStore)
        assert not hasattr(m, "put")
        assert not hasattr(m, "get")
        assert not hasattr(m, "collect_blobs")


async def test_live_blob_refs_collects_structured_ref_parts(tmp_path):
    async with _provider(tmp_path) as m:
        await m.ingest(_event(_REF_A), _ctx())
        assert await m.live_blob_refs() == {_REF_A}


async def test_live_blob_refs_collects_declared_refs(tmp_path):
    """L0.5 降级把 ImagePart(ref) 换成文本占位，ref 只能靠 blob_refs 声明传递。"""
    async with _provider(tmp_path) as m:
        await m.ingest(_event(None, blob_refs=[_REF_B]), _ctx())
        assert await m.live_blob_refs() == {_REF_B}


async def test_live_blob_refs_drops_superseded(tmp_path):
    """被 fold 掉的记录不再构成活引用——这正是 GC 的 mark 判据。"""
    async with _provider(tmp_path) as m:
        rid = await m.ingest(_event(_REF_A), _ctx())
        assert await m.live_blob_refs() == {_REF_A}
        await m.fold([rid], [], _ctx())
        assert await m.live_blob_refs() == set()


async def test_live_blob_refs_spans_all_tenants(tmp_path):
    """内容寻址天然跨租户共享一份字节；只看单租户会删掉别人还在用的图。"""
    async with _provider(tmp_path) as m:
        await m.ingest(_event(_REF_A), _ctx("tenant-a"))
        await m.ingest(_event(_REF_B), _ctx("tenant-b"))
        assert await m.live_blob_refs() == {_REF_A, _REF_B}


async def test_live_blob_refs_survives_partial_supersede(tmp_path):
    """同一 sha 被两条记录引用，fold 掉一条 → 仍是活引用。"""
    async with _provider(tmp_path) as m:
        rid1 = await m.ingest(_event(_REF_A), _ctx())
        await m.ingest(_event(_REF_A), _ctx())
        await m.fold([rid1], [], _ctx())
        assert await m.live_blob_refs() == {_REF_A}


async def test_live_blob_refs_empty_on_pure_text(tmp_path):
    async with _provider(tmp_path) as m:
        await m.ingest(_event(None), _ctx())
        assert await m.live_blob_refs() == set()


# ══════════════════════════════════════════════════════════════════════════════
# 引用表：与 ingest 同事务写入 / 读侧回显（本 provider 刻意保留的那一半）
# ══════════════════════════════════════════════════════════════════════════════
#
# 下面三条从被删的 tests/unit/test_sql_blob_store.py（git show 1886e40）搬来，去掉了
# 所有 put/get/collect_blobs 字节存取的部分——它们测的是 `_ingest_in_tx` 的引用边写入
# 与 `_declared_refs` 的去重，即使字节不再落库也必须保留，`live_blob_refs()` 的聚合
# 视图不能替代逐行 (event_id, sha) 配对与「结构化 ref 不重复回显」这两条判据。


async def test_ingest_records_blob_refs_in_same_transaction(tmp_path):
    """ingest 含 ref 的 content → 引用表写入，边与内容里的 ref 一一对应。

    刻意带**两个** ref：单 ref 用例测不出「逐行配对」与「聚合去重」的区别。
    """
    async with _provider(tmp_path) as m:
        eid = await m.ingest(_multi_ref_event(_REF_A, _REF_B), _ctx())
        assert await _ref_rows(m) == {(eid, _SHA_A), (eid, _SHA_B)}


async def test_load_view_restores_blob_refs(tmp_path):
    """声明式 ref 的读侧回显——否则第二次降级时第一次的 ref 无人认领（`_rebuild` 的
    累积逻辑）。"""
    async with _provider(tmp_path) as m:
        await m.ingest(_event(None, blob_refs=[_REF_A]), _ctx())
        (rec,) = await load_records(m,
            _ADDR, MemoryScope.TASK, _ctx(), kinds=[MemoryKind.CONVERSATION_TURN])
        assert rec.blob_refs == [_REF_A]


async def test_structural_refs_are_not_duplicated_in_blob_refs(tmp_path):
    """结构化 ref 已在 content 里，回显时不再重复塞进 blob_refs（否则 `_rebuild`
    的累积逻辑会把结构化 ref 也滚进补偿记录，一轮轮越滚越多）。"""
    async with _provider(tmp_path) as m:
        await m.ingest(_event(_REF_A), _ctx())
        (rec,) = await load_records(m,
            _ADDR, MemoryScope.TASK, _ctx(), kinds=[MemoryKind.CONVERSATION_TURN])
        assert rec.blob_refs == []
        assert collect_blob_refs(rec) == [_REF_A], "仍能从 content 采到"
