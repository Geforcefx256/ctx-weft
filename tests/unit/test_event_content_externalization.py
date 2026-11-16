"""content_to_event_jsonable 只接受原始 content（blob-store 解耦 Task 2）。

入口双写（Task 1）删除后，``source_type == "ref"`` 的图片 part 不再能被信任为
「event store 也持有这份字节」——那份字节只属于 memory 侧。本文件验证：外来 ref
一律降级为文本占位，绝不透传进事件 payload。
"""

from __future__ import annotations

import base64

import pytest

from ctx_weft.protocols.content import content_to_event_jsonable
from ctx_weft.protocols import BLOB_REF_PREFIX, ImagePart, ProviderContext, TextPart

_PNG = base64.b64encode(b"\x89PNG_fake_bytes").decode()


class _EventStore:
    """event 侧桩。**刻意用与 memory 侧完全不同的 ref 方案**——core 若还偷偷假设
    两边 ref 相同，任何一条链路都会立刻暴露。"""

    can_externalize = True

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data, media_type, ctx):
        ref = f"{BLOB_REF_PREFIX}evt-{len(self.blobs) + 1}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref, ctx):
        return self.blobs.get(ref)


@pytest.mark.asyncio
async def test_base64_part_goes_into_event_store():
    store = _EventStore()
    out = await content_to_event_jsonable(
        [TextPart(text="hi"), ImagePart(data=_PNG, media_type="image/png")],
        event_blob_store=store,
        ctx=ProviderContext(session_id="s1"),
    )
    assert out[0] == {"type": "text", "text": "hi"}
    assert out[1]["data"] == f"{BLOB_REF_PREFIX}evt-1"
    assert out[1]["source_type"] == "ref"
    assert store.blobs[f"{BLOB_REF_PREFIX}evt-1"][0] == base64.b64decode(_PNG)


@pytest.mark.asyncio
async def test_foreign_ref_part_is_demoted_not_passed_through(caplog):
    """喂进来一个 memory ref = 调用方给错了内容。降级成占位，绝不透传。

    透传会让事件 payload 里出现一个 event store 解不开的 ref——正是本次解耦要消灭的
    跨命名空间引用。
    """
    store = _EventStore()
    out = await content_to_event_jsonable(
        [ImagePart(data=f"{BLOB_REF_PREFIX}mem-1", media_type="image/png",
                   source_type="ref")],
        event_blob_store=store,
        ctx=ProviderContext(session_id="s1"),
    )
    assert out == [{"type": "text", "text": "[image image/png]"}]
    assert store.blobs == {}
    assert "ref" in caplog.text.lower()


@pytest.mark.asyncio
async def test_plain_text_is_zero_cost():
    store = _EventStore()
    assert await content_to_event_jsonable(
        "纯文本", event_blob_store=store, ctx=ProviderContext(session_id="s1")) == "纯文本"
    assert store.blobs == {}
