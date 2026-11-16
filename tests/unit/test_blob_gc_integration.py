"""mark-sweep 联动：SqlMemoryProvider.live_blob_refs() × FsBlobStore.collect()。

这是**真实的生产组合**——引用边在 SQL（与 ingest 同事务），字节在文件系统。
两者靠 ref 串起来，谁都不知道对方的存在。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ctx_weft.protocols.content import normalize_content, rehydrate_content
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ProviderContext,
)
from ctx_weft.protocols.content import ImagePart, TextPart
from ctx_weft.providers.blob.fs import FsBlobStore
from ctx_weft.providers.memory.sql import open_sqlite_memory

_PNG = b"\x89PNG\r\n\x1a\n" + b"x" * 64
_LATER = datetime.now(UTC) + timedelta(days=2)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _event(content) -> MemoryEvent:
    return MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN,
        scope=MemoryScope.TASK,
        address=MemoryAddress(session_id="s1", task_id="t1", agent_id="a1"),
        content=content,
        timestamp=datetime.now(UTC),
        role="user",
    )


async def test_live_blob_survives_collection(tmp_path):
    blobs = FsBlobStore(tmp_path / "blobs")
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        ref = await blobs.put(_PNG, "image/png", _ctx())
        await mem.ingest(_event([ImagePart(data=ref, media_type="image/png",
                                          source_type="ref")]), _ctx())
        assert await blobs.collect(await mem.live_blob_refs(), now=_LATER) == 0
        assert await blobs.get(ref, _ctx()) is not None


async def test_orphaned_blob_is_collected_after_fold(tmp_path):
    """记录被 fold → 活引用归零 → 过宽限期被回收 → rehydrate 降级成占位。"""
    blobs = FsBlobStore(tmp_path / "blobs")
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        ref = await blobs.put(_PNG, "image/png", _ctx())
        content = [ImagePart(data=ref, media_type="image/png", source_type="ref")]
        rid = await mem.ingest(_event(content), _ctx())

        await mem.fold([rid], [], _ctx())
        assert await mem.live_blob_refs() == set()
        assert await blobs.collect(await mem.live_blob_refs(), now=_LATER) == 1

        out = await rehydrate_content(content, blob_store=blobs, ctx=_ctx())
        assert len(out) == 1
        assert out[0].text == "[image unavailable: image/png]"


async def test_grace_period_protects_the_put_to_ingest_window(tmp_path):
    """已 put、尚未 ingest 的窗口——只按活引用删会把刚上传的图删掉。"""
    blobs = FsBlobStore(tmp_path / "blobs")
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        ref = await blobs.put(_PNG, "image/png", _ctx())
        assert await mem.live_blob_refs() == set()      # 还没有引用边
        assert await blobs.collect(await mem.live_blob_refs()) == 0   # 真实 now：宽限期内
        assert await blobs.get(ref, _ctx()) is not None


async def test_entry_normalize_then_gc_roundtrip(tmp_path):
    """入口外部化 → 落库 → 取回，全链路走真实组合。"""
    import base64

    blobs = FsBlobStore(tmp_path / "blobs")
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        raw = [TextPart(text="看这个"),
               ImagePart(data=base64.b64encode(_PNG).decode(), media_type="image/png")]
        normalized = await normalize_content(raw, blob_store=blobs, ctx=_ctx())
        await mem.ingest(_event(normalized), _ctx())

        assert len(await mem.live_blob_refs()) == 1
        assert await blobs.collect(await mem.live_blob_refs(), now=_LATER) == 0

        back = await rehydrate_content(normalized, blob_store=blobs, ctx=_ctx())
        assert base64.b64decode(back[1].data) == _PNG
