"""SESSION_CREATED / SESSION_RESUMED 的 user_prompt 事件重放：保 ref、不落字节、不拍扁。

裁定（2026-08-27）：`Session.user_prompt` 与两条 session 事件的 payload
**不能拍扁**（图片结构必须留下）、也**不能内联 base64 字节**（事件库体积）——
一律走 ref 形态。

前身是 `test_session_prompt_refs_only.py`，针对过渡实现 `content_to_jsonable_refs_only`
的 6 条纯函数用例已随该函数一起删除（dual-blob-store 设计 §6：`content_to_event_jsonable`
取代了它，纯函数覆盖见 `tests/unit/test_event_blob_store.py`）。本文件保留的 3 条事件
重放用例改挂新函数——它们钉的是「session 事件重放出 parts 与 ref」这个不随实现变的
性质，故文件按新覆盖对象改名。

对照组：`TASK_CREATED` 早已是 `content_to_jsonable`（无损），本文件钉的是 session 侧。
"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone

from ctx_weft.core.utils.content import content_to_event_jsonable
from ctx_weft.core.control.reducers import deserialize_view, rebuild_view, serialize_view
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.protocols import ImagePart, TextPart
from tests.unit.test_event_blob_store import _ctx, _Stub
from tests._event_helpers import append_one

# blob-store 解耦 Task 2 后 content_to_event_jsonable 拒收外来 ref part（降级为占位）；
# 这里改用 base64 图片喂入，走真实的「event 侧独立 put」路径产出 ref，而不是像旧版那样
# 预先伪造一个 ref part 假装「双写已经把字节存进 event store」。
_PNG_BYTES = b"\x89PNG_fake_bytes"
_PNG_REF = f"blob:{hashlib.sha256(_PNG_BYTES).hexdigest()}"


def _b64_image_part() -> ImagePart:
    return ImagePart(data=base64.b64encode(_PNG_BYTES).decode(), media_type="image/png",
                     source_type="base64")


async def _store_with(payload_prompt: object) -> InMemoryEventStore:
    store = InMemoryEventStore()
    await append_one(store, Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id="ses_1",
        tenant_id="default", type=EventType.SESSION_CREATED,
        timestamp=datetime(2026, 8, 27, tzinfo=timezone.utc),
        payload={"user_prompt": payload_prompt, "template_id": "tpl",
                 "root_agent_id": "agt_1"},
    ))
    return store


async def test_rebuild_view_restores_parts_from_session_created() -> None:
    """重放 SESSION_CREATED 还原出 parts 结构与 ref，而不是一个 str。"""
    payload = await content_to_event_jsonable(
        [TextPart(text="看图"), _b64_image_part()], event_blob_store=_Stub(), ctx=_ctx())
    view = await rebuild_view(await _store_with(payload), "ses_1")
    prompt = view.sessions["ses_1"].user_prompt
    assert isinstance(prompt, list)
    assert prompt[0].text == "看图"
    assert prompt[1].source_type == "ref"
    assert prompt[1].data == _PNG_REF


async def test_legacy_str_payload_still_replays() -> None:
    """存量事件的 user_prompt 是裸 str——必须原样还原，零数据迁移。"""
    view = await rebuild_view(await _store_with("旧的纯文本"), "ses_1")
    assert view.sessions["ses_1"].user_prompt == "旧的纯文本"


async def test_view_snapshot_roundtrip_keeps_parts() -> None:
    """serialize_view / deserialize_view 往返不丢 parts。"""
    payload = await content_to_event_jsonable(
        [TextPart(text="看图"), _b64_image_part()], event_blob_store=_Stub(), ctx=_ctx())
    view = await rebuild_view(await _store_with(payload), "ses_1")
    back = deserialize_view(serialize_view(view))
    prompt = back.sessions["ses_1"].user_prompt
    assert isinstance(prompt, list)
    assert prompt[1].data == _PNG_REF
