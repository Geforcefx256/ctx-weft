"""端到端：崩在「结果事件已落盘、memory 记录还没写」之间时，结果里的图不能丢（审查文档 M7）。

那个窗口里，`CapabilityFinished` 的 `result` 是唯一幸存的一份，补写只能照它重建。事件
payload 从前用的是自制短标记（只留 ref 前 12 个字符）——同一个仓库里两套图片标记，而事件
用了残缺的那套，于是补写出来的图取不回来。

现在统一用 L0.5 占位（`media.refs.encode_image_placeholder`）：纯文本、含完整 ref，
`media:get_image` 能把图取回来。补写的那条记录还要**声明**占位里的 ref，否则下一轮 blob GC
会把这些没人认领的字节当孤儿删掉。
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import pytest

from ctx_weft.core.media.fold import placeholder_refs
from ctx_weft.core.utils.content import content_to_text, redact_content_for_event
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols import (
    ImagePart, MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext, TextPart,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _poll
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

pytestmark = pytest.mark.asyncio

SID, TID, AID = "ses_i", "tsk_i", "agt_root"
TS = datetime(2026, 9, 18, tzinfo=timezone.utc)
REF = "blob:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcd"


def test_event_payload_uses_the_retrievable_placeholder() -> None:
    """引用形态的图 → 可取回的占位；内联 base64 → 短标记（没有 ref 可写）；纯文本原样。"""
    with_ref = redact_content_for_event(
        [TextPart(text="here: "), ImagePart(data=REF, media_type="image/png", source_type="ref")])
    assert REF in with_ref and "media:get_image" in with_ref
    assert placeholder_refs(with_ref) == [REF], "占位必须解析得回那个 ref"

    inline = redact_content_for_event(
        [ImagePart(data="AAAABBBBCCCCDDDD", media_type="image/png", source_type="base64")])
    assert "AAAABBBBCCCCDDDD" not in inline, "字节绝不上事件流"
    assert placeholder_refs(inline) == []

    assert redact_content_for_event("plain text") == "plain text"


async def test_crash_between_result_event_and_memory_keeps_the_image_retrievable() -> None:
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(mem)

    tcid = mint_call_id(anchor="asst_i", ordinal=0, raw_id="tc1", turn_seq=0)
    # 工具结果是「一段文字 + 一张已外部化的图」，事件 payload 就是它的脱敏形态。
    result_event_payload = redact_content_for_event(
        [TextPart(text="screenshot taken: "),
         ImagePart(data=REF, media_type="image/png", source_type="ref")])

    def _ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=SID,
                     type=type_, timestamp=TS, task_id=task_id, payload=payload)

    for e in [
        _ev(1, EventType.SESSION_CREATED, user_prompt="shoot", template_id="agent:tpl_echo",
            root_agent_id=AID),
        _ev(2, EventType.TASK_CREATED, task={
            "id": TID, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": AID, "creator_agent_id": AID, "user_prompt": "shoot"}),
        _ev(3, EventType.TASK_STARTED, task_id=TID, assigned_agent_id=AID),
        _ev(4, EventType.CAPABILITY_INVOKED, task_id=TID, tool_call_id=tcid,
            invocation_id="inv_1", capability_name="web__shot", capability_id="web:shot"),
        # 结果事件落盘了，随后崩溃 —— 那条 TOOL_RESULT 从没写成
        _ev(5, EventType.CAPABILITY_FINISHED, task_id=TID, tool_call_id=tcid,
            invocation_id="inv_1", capability_name="web__shot", outcome="success",
            result=result_event_payload, result_length=len(result_event_payload)),
    ]:
        await rt.event_store.append(e)

    scope = MemoryAddress(session_id=SID, task_id=TID, agent_id=AID)
    pctx = ProviderContext(session_id=SID, tenant_id="default", task_id=TID, agent_id=AID)
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="shoot", timestamp=TS, role="user", metadata={"task_id": TID}), pctx)
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=TS, role="assistant",
        metadata={"tool_calls": [{"id": tcid, "name": "web__shot", "input": {}}]}), pctx)

    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await rt.recover_agent(AID)
        await _poll(lambda: llm.act_requests or None, timeout=8.0)

    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    results = [r for r in view if r.role == "tool"]
    assert len(results) == 1, [r.id for r in results]
    text = content_to_text(results[0].content)
    assert REF in text and "media:get_image" in text, (
        f"补写出来的结果必须带可取回的占位，而不是残缺标记：{text!r}")
    assert REF in (results[0].blob_refs or []), (
        "占位里的 ref 必须被声明，否则下一轮 blob GC 会把字节当孤儿回收")
