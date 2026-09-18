"""端到端：提问只进一次、不该开的窗不开、截断的结果不当完整结果用（审查文档 M6 / L6 / M7）。

- **M6**：`task_from_projection` 只能按状态**猜**提问在不在 memory，两个方向都会错。恢复期
  对着 memory 核实并双向修正——猜「已写」却没有要补，猜「没写」其实有要挡住（否则对话末尾
  会凭空多一条原始提问，时间戳还是恢复时刻）。
- **L6**：窗口靠「那一轮的提交点」来关。`NoResumeDelivery`（纯通知 / 取消）与打到已终态 task
  的应答都没有下一轮，所以根本不该开窗——开了就没人关，该 task 之后的事件全被挡在缓冲里。
- **M7**：事件里那份结果被截断过时，不得当作完整结果补写进对话。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest import mock

import pytest

from ctx_weft.core.loop.driver import task_prompt_record_id
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols import (
    MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _poll
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_AFTER_CHUNK, _ask, _next_question, _ScriptedLLM,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

pytestmark = pytest.mark.asyncio

PROMPT = "the original request"
SID, TID, AID = "ses_g", "tsk_g", "agt_root"
TS = datetime(2026, 9, 17, tzinfo=timezone.utc)


def _runtime(llm, *, hitl_timeout_sec=None, memory=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec))
    rt.providers.register_memory(memory or InMemoryMemoryProvider())
    return rt


def _ev(seq, type_, **payload):
    task_id = payload.pop("task_id", None)
    return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=SID,
                 type=type_, timestamp=TS, task_id=task_id, payload=payload)


async def _seed(rt, status_event, *, status="ACTIVE"):
    for e in [
        _ev(1, EventType.SESSION_CREATED, user_prompt=PROMPT, template_id="agent:tpl_echo",
            root_agent_id=AID),
        _ev(2, EventType.TASK_CREATED, task={
            "id": TID, "status": status, "title": "T", "kind": "reasoning",
            "assigned_agent_id": AID, "creator_agent_id": AID, "user_prompt": PROMPT}),
        _ev(3, EventType.TASK_STARTED, task_id=TID, assigned_agent_id=AID),
    ]:
        await rt.event_store.append(e)
    if status_event is not None:
        await rt.event_store.append(_ev(4, status_event, task_id=TID, hitl_id="",
                                        reason="probe", retry_count=0))


def _scope_ctx():
    return (MemoryAddress(session_id=SID, task_id=TID, agent_id=AID),
            ProviderContext(session_id=SID, tenant_id="default", task_id=TID, agent_id=AID))


# ── M6：提问恰好一条 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("status_event,record_id", [
    (EventType.TASK_AWAITING_HUMAN, None),          # 猜「没写」其实有（存量自动 id）
    (EventType.TASK_INTERRUPTED, None),
    (EventType.TASK_AWAITING_HUMAN, "deterministic"),   # 新数据：确定性 id
    (None, "missing"),                                  # 猜「已写」却没有 → 要补
], ids=["awaiting-human", "interrupted", "deterministic-id", "missing"])
async def test_recovery_keeps_exactly_one_prompt(status_event, record_id) -> None:
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    rt = _runtime(llm, memory=mem)
    await _seed(rt, status_event)
    scope, pctx = _scope_ctx()
    if record_id != "missing":
        await mem.ingest(MemoryEvent(
            id=(task_prompt_record_id(TID, PROMPT) if record_id == "deterministic" else None),
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
            content=PROMPT, timestamp=TS, role="user", metadata={"task_id": TID}), pctx)

    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await rt.recover_agent(AID)
        await _poll(lambda: llm.act_requests or None, timeout=8.0)
        await asyncio.sleep(0.2)

    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    prompts = [r for r in view if r.role == "user" and PROMPT in content_to_text(r.content)]
    assert len(prompts) == 1, f"提问必须恰好一条：{[(r.id, content_to_text(r.content)) for r in prompts]}"


async def test_reopened_task_still_writes_its_revised_prompt() -> None:
    """reopen 改写过提问：旧提问还在 memory 里，修订版仍然必须写进去（核对不得误判成已存在）。"""
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    rt = _runtime(llm, memory=mem)
    revised = f"{PROMPT}\n\n[revision] fix the summary"
    for e in [
        _ev(1, EventType.SESSION_CREATED, user_prompt=PROMPT, template_id="agent:tpl_echo",
            root_agent_id=AID),
        _ev(2, EventType.TASK_CREATED, task={
            "id": TID, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": AID, "creator_agent_id": AID, "user_prompt": PROMPT}),
        _ev(3, EventType.TASK_STARTED, task_id=TID, assigned_agent_id=AID),
        _ev(4, EventType.TASK_REQUEUED, task_id=TID, reason="revise",
            user_prompt=revised, original_user_prompt=PROMPT),
    ]:
        await rt.event_store.append(e)
    scope, pctx = _scope_ctx()
    await mem.ingest(MemoryEvent(
        id=task_prompt_record_id(TID, PROMPT),
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content=PROMPT, timestamp=TS, role="user", metadata={"task_id": TID}), pctx)

    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await rt.recover_agent(AID)
        await _poll(lambda: llm.act_requests or None, timeout=8.0)
        await asyncio.sleep(0.2)

    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    texts = [content_to_text(r.content) for r in view if r.role == "user"]
    assert any("[revision]" in t for t in texts), f"修订版提问必须写进对话：{texts}"


# ── L6：不该开的窗不开 ──────────────────────────────────────────────────────────


async def test_reply_to_a_terminal_task_opens_no_window() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)

    tm = rt._task_managers[sid]
    tm.get_task(q1.task_id).status = "CANCELED"          # task 已终态：重排必然 no-op

    view = await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))
    assert view is not None and view.outcome == "accepted"
    assert not tm.open_round_task_ids, "没有下一轮来关的窗，就不该开"
    assert rt.hitl_registry.get(q1.id).resolved, "不开窗 → 一步终局，事实当场落盘"
    assert EventType.HITL_RESOLVED in [
        e.type for e in await rt.event_store.read_by_session(sid)]


async def test_no_resume_delivery_opens_no_window() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)

    from ctx_weft.protocols.hitl import NoResumeDelivery
    rt.hitl_registry.get(q1.id).delivery = NoResumeDelivery()   # 纯通知：没有续跑

    assert await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1")) is not None
    assert not rt._task_managers[sid].open_round_task_ids
    assert rt.hitl_registry.get(q1.id).resolved


# ── M7：截断的结果不当完整结果 ──────────────────────────────────────────────────


async def test_truncated_recorded_result_is_not_backfilled() -> None:
    """`CapabilityFinished` 的 payload 有上限；被截断的那份不得当完整结果补写进对话。"""
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    rt = _runtime(llm, memory=mem)
    tcid = mint_call_id(anchor="asst_x", ordinal=0, raw_id="tc1", turn_seq=0)
    await _seed(rt, None)
    for e in [
        _ev(5, EventType.CAPABILITY_INVOKED, task_id=TID, tool_call_id=tcid,
            invocation_id="inv_1", capability_name="fs__read_file", capability_id="fs:read_file"),
        _ev(6, EventType.CAPABILITY_FINISHED, task_id=TID, tool_call_id=tcid,
            invocation_id="inv_1", capability_name="fs__read_file", outcome="success",
            result="TRUNCATED_HEAD", result_length=99999),
    ]:
        await rt.event_store.append(e)

    scope, pctx = _scope_ctx()
    await mem.ingest(MemoryEvent(
        id=task_prompt_record_id(TID, PROMPT),
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content=PROMPT, timestamp=TS, role="user", metadata={"task_id": TID}), pctx)
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=TS, role="assistant",
        metadata={"tool_calls": [{"id": tcid, "name": "fs__read_file",
                                  "input": {"path": "x"}}]}), pctx)

    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await rt.recover_agent(AID)
        await _poll(lambda: llm.act_requests or None, timeout=8.0)
        await asyncio.sleep(0.2)

    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    results = [content_to_text(r.content) for r in view if r.role == "tool"]
    assert results, "这次调用仍要有一条结果（由 gateway 决定重跑还是作结）"
    assert not any(t.strip() == "TRUNCATED_HEAD" for t in results), (
        f"截断版不得被当成完整结果写入：{results}")
