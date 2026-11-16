"""端到端：未提交窗口 + memory 暂存机制的修复回归（审查文档 docs/follow-ups/2026-09-17-round-staging-window-review.md）。

- H3/H4：提交失败不丢数据——终局事实没落盘就不写 memory、快照保留、下个提交点重试。
- H2：ask_user 的结果就是人的答复，恢复时按 HITL 决定重新生成，不用事件里那份旧的。
- H1：注入消息的正文随 `TaskMessageAppended` 进日志，崩在提交中途时恢复期补回 memory。
- H5：热应答醒来、在工具之间被暂停，同样撤回答复。
- M1：`reply_to_hitl` 自己开的窗，应答没成时自己撤掉；别人开的窗不碰。
- M2：消息注入不替别人的窗口提前提交。
"""

from __future__ import annotations

from tests._recovery import load_pending_fixture

from tests._recovery import recover_fixture_session

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    ToolCall,
    load_records,
)
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    Authorizer,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._event_helpers import all_events, append_one
from tests._initialization import initialize_all
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _all_request_text, _finish_call, _poll
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_AFTER_CHUNK,
    STALL_BEFORE_CHUNK,
    _ask,
    _ask_user_results,
    _next_question,
    _ScriptedLLM,
    _stored_types,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.step_helpers import suppress_observe_start



def _build(llm, *, hitl_timeout_sec, memory=None, event_store=None, register=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    kwargs = {"event_store": event_store} if event_store is not None else {}
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec,
                                           fallback_template_ref="agent:tpl_echo"), **kwargs)
    rt.providers.register_memory(memory or InMemoryMemoryProvider())
    if register is not None:
        register(rt)
    return rt


async def _start(rt):
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go"))
    return handle.session_id


def _fail_hitl_resolved(rt, *, times: int | None):
    """让 `HitlResolved` 的发射失败 `times` 次（None = 一直失败），模拟提交途中存储不可用。"""
    real = rt.event_store.append_batch
    calls = {"n": 0}

    async def flaky(sid, bid, events, *a, **k):
        if any(e.type == EventType.HITL_RESOLVED for e in events):
            calls["n"] += 1
            if times is None or calls["n"] <= times:
                raise OSError("storage unavailable while committing the reply batch")
        return await real(sid, bid, events, *a, **k)

    rt.event_store.append_batch = flaky
    calls["restore"] = lambda: setattr(rt.event_store, "append_batch", real)
    return calls


# ── H3 / H4：提交失败不丢数据 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_retries_after_hitl_resolved_fails_once() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _build(llm, hitl_timeout_sec=0)
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    calls = _fail_hitl_resolved(rt, times=1)

    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="ANSWER"))

    # Required persistence failure isolates the session and stops dispatch.
    # Retry the same input through the actual host recovery entry point.
    await _poll(lambda: calls["n"] == 1 and not rt._task_managers[sid]._dispatch_tasks or None)
    llm.script.append(STALL_AFTER_CHUNK)
    rt.clear_storage_isolation(sid)
    await recover_fixture_session(rt, q1.agent_id, session_id=sid)
    await _poll(lambda: rt.hitl_registry.get(q1.id).resolved or None, timeout=8.0)
    assert EventType.HITL_RESOLVED in await _stored_types(rt, sid)
    results = await _ask_user_results(rt, sid, q1)
    assert len(results) == 1 and "ANSWER" in results[0], results
    assert not rt._task_managers[sid].open_round_task_ids


@pytest.mark.asyncio
async def test_commit_failure_halts_without_losing_the_reply() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _build(llm, hitl_timeout_sec=0)
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    calls = _fail_hitl_resolved(rt, times=None)

    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="ANSWER"))
    await _poll(lambda: calls["n"] >= 1 and not rt._task_managers[sid]._dispatch_tasks or None, timeout=8.0)
    await asyncio.sleep(0.3)

    rec = rt.hitl_registry.get(q1.id)
    assert rec.claim_pending and not rec.resolved, "终局事实没落盘，答复必须退回待终局、不丢"
    assert EventType.HITL_RESOLVED not in await _stored_types(rt, sid)
    assert await _ask_user_results(rt, sid, q1) == [], "终局事实没落盘，答复不得进 memory"
    tm = rt._task_managers[sid]
    assert tm.is_round_open(q1.task_id), "提交没完成，窗口与暂存必须保留以便重试"
    staged = [content_to_text(ev.content) for ev in tm.staged_memory(q1.task_id)]
    assert any("ANSWER" in t for t in staged), staged


# ── H2：崩在提交中途，重启后重答用新答案 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_restart_after_crash_mid_commit_uses_the_new_answer() -> None:
    memory = InMemoryMemoryProvider()
    llm1 = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt1 = _build(llm1, hitl_timeout_sec=0, memory=memory)
    sid = await _start(rt1)
    q1 = await _next_question(rt1, sid)
    await asyncio.sleep(0.2)
    calls = _fail_hitl_resolved(rt1, times=None)
    await initialize_all(rt1)
    await rt1.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="OLD_ANSWER"))
    await _poll(lambda: calls["n"] >= 1 and not rt1._task_managers[sid]._dispatch_tasks or None, timeout=8.0)
    await asyncio.sleep(0.3)
    types = await _stored_types(rt1, sid)
    # The tool result and resolution now share one atomic input batch.
    # A failed commit leaves neither the old answer nor its resolution durable.
    assert EventType.HITL_RESOLVED not in types
    assert not any("OLD_ANSWER" in str(e.payload) for e in await all_events(rt1.event_store, sid))
    calls["restore"]()  # The restarted provider has recovered; only rt1 was faulted.

    # ── 「重启」：新 runtime，同一份事件日志与 memory ────────────────────────────
    llm2 = _ScriptedLLM([STALL_AFTER_CHUNK])
    rt2 = _build(llm2, hitl_timeout_sec=0, memory=memory, event_store=rt1.event_store)
    await initialize_all(rt2)
    await load_pending_fixture(rt2, sid)
    pending = [p for p in rt2.hitl_registry.list_pending(session_id=sid) if p.id == q1.id]
    assert pending, "重启后问题必须回到待答"
    with suppress_observe_start():
        await initialize_all(rt2)
        await rt2.reply_to_hitl(HitlReply(
            hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="NEW_ANSWER"))
        await _poll(lambda: llm2.act_requests or None, timeout=8.0)
        await asyncio.wait_for(llm2.stalled.wait(), timeout=5.0)
    text = _all_request_text(llm2.act_requests[0])
    assert "NEW_ANSWER" in text and "OLD_ANSWER" not in text, (
        "人重答的新答案必须送到模型，不得用事件里崩溃前那份旧结果")
    results = await _ask_user_results(rt2, sid, q1)
    assert len(results) == 1 and "NEW_ANSWER" in results[0], results


# ── H1：注入消息崩在提交中途，恢复期补回 memory ───────────────────────────────────


MESSAGE = "also include the appendix"


@pytest.mark.asyncio
@pytest.mark.parametrize("message_in_memory", [False, True], ids=["missing", "present"])
async def test_recovery_restores_an_appended_message_from_the_event_log(message_in_memory) -> None:
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    rt = _build(llm, hitl_timeout_sec=None, memory=mem)
    sid, tid, aid = "ses_m", "tsk_m", "agt_root"
    ts = datetime(2026, 9, 17, tzinfo=timezone.utc)
    msg_ts = datetime(2026, 9, 17, 0, 0, 5, tzinfo=timezone.utc)

    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=sid,
                     type=type_, timestamp=ts, task_id=task_id, payload=payload)

    for e in [
        ev(1, EventType.SESSION_CREATED, user_prompt="summarize", template_id="agent:tpl_echo",
           root_agent_id=aid),
        ev(2, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid, "user_prompt": "summarize"}),
        ev(3, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
        ev(4, EventType.TASK_MESSAGE_APPENDED, task_id=tid, memory_id="mem_appended",
           agent_id=aid, content=MESSAGE, source="send_message", timestamp=msg_ts.isoformat()),
    ]:
        await append_one(rt.event_store, e)

    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="summarize", timestamp=ts, role="user", metadata={"task_id": tid}), pctx)
    if message_in_memory:
        await mem.ingest(MemoryEvent(
            id="mem_appended", kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=scope, content=MESSAGE, timestamp=msg_ts, role="user",
            metadata={"task_id": tid, "source": "send_message"}), pctx)

    # 装填是调用方的责任（2026-09-21：`recover_agent` 对 registry miss 直接抛
    # `AgentNotLoaded`，按 agent 扫全库的 sweep 已删）。只喂内存，不建 TM、不跑。
    await initialize_all(rt)
    await rt.rebuild_session(sid)
    with suppress_observe_start():
        await initialize_all(rt)
        await recover_fixture_session(rt, aid)
        await _poll(lambda: llm.act_requests or None)

    assert MESSAGE in _all_request_text(llm.act_requests[0])
    view = await load_records(mem, scope, MemoryScope.TASK, pctx, kinds=[MemoryKind.CONVERSATION_TURN])
    msgs = [r for r in view if MESSAGE in content_to_text(r.content)]
    assert len(msgs) == 1 and msgs[0].id == "mem_appended", [r.id for r in msgs]


@pytest.mark.asyncio
async def test_injected_message_is_recorded_in_the_event_log() -> None:
    llm = _ScriptedLLM([MockResponse(text="first reply"), STALL_AFTER_CHUNK])
    rt = _build(llm, hitl_timeout_sec=None)
    sid = await _start(rt)
    wait = await _poll(lambda: [p for p in rt.hitl_registry.list_pending(session_id=sid)
                                if p.form == "wait"] or None)
    await initialize_all(rt)
    await rt.send_message(wait[0].agent_id, MESSAGE, session_id=sid)
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    await asyncio.sleep(0.2)
    appended = [e for e in await all_events(rt.event_store, sid)
                if e.type == EventType.TASK_MESSAGE_APPENDED]
    assert len(appended) == 1 and appended[0].payload["content"] == MESSAGE


# ── H5：热应答后在工具之间暂停，撤回 ─────────────────────────────────────────────


class _SlowTool(ToolCapabilityProvider):
    name = "slow"

    def __init__(self) -> None:
        self.invocations = 0

    async def list(self, ctx):
        return [ToolCapability(id="slow:work", name="work", description="slow work")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, args, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run()

    async def _run(self):
        self.invocations += 1
        yield CapabilityEvent(kind="result", payload={"content": "worked"})

    async def cancel(self, invocation_id, ctx) -> None:
        return None


class _SlowAuthorizer(Authorizer):
    """在 gateway step 5（调工具即关窗）之前卡住，给测试留出按暂停的时机。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.slow = True

    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        if self.slow:
            self.entered.set()
            await asyncio.sleep(3.0)
        return AuthorizationDecision(allowed=True)


@pytest.mark.asyncio
async def test_pause_between_tools_after_a_hot_reply_retracts_it() -> None:
    tool, authz = _SlowTool(), _SlowAuthorizer()
    # 判定 `success`：重答之后那一轮调 `finish_task` 收尾，而它如今也 park + 后台判定（S-b）。
    llm = _ScriptedLLM([
        MockResponse(text="", tool_calls=[
            ToolCall(id="tc1", name="control__ask_user",
                     arguments={"questions": [{"question": "Q1?"}]}),
            ToolCall(id="tc2", name="slow__work", arguments={}),
        ]),
        _finish_call(),
    ], observer_verdicts=["success"])
    rt = _build(llm, hitl_timeout_sec=None, register=lambda r: r.providers.register_capability(
        tool, tool_authorizers={"slow:work": authz}))
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)

    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="RETRACTED"))
    await asyncio.wait_for(authz.entered.wait(), timeout=5.0)
    await initialize_all(rt)
    assert await rt.pause_session(sid) is True
    tm = rt._task_managers[sid]
    await _poll(lambda: tm.get_task(q1.task_id).status == "AWAITING_HUMAN" or None, timeout=8.0)
    await asyncio.sleep(0.3)

    rec = rt.hitl_registry.get(q1.id)
    assert not rec.resolved and not rec.claim_pending
    assert q1.id in [p.id for p in rt.hitl_registry.list_pending(session_id=sid)]
    types = await _stored_types(rt, sid)
    assert EventType.HITL_RESOLVED not in types
    assert await _ask_user_results(rt, sid, q1) == []
    assert tool.invocations == 0
    assert types.count(EventType.RUN_STARTED) == types.count(EventType.RUN_FINISHED)

    # 重答：答复生效，那个没开始的工具被当作首执执行一次。
    authz.slow = False
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="RETYPED"))
    await _poll(lambda: tm.get_task(q1.task_id).status == "FINISHED" or None, timeout=8.0)
    assert tool.invocations == 1
    results = await _ask_user_results(rt, sid, q1)
    assert len(results) == 1 and "RETYPED" in results[0], results


# ── M1：reply_to_hitl 没成时撤掉自己开的窗 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rejected_reply_does_not_leave_a_window_open() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _build(llm, hitl_timeout_sec=0)
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)

    real = rt.hitl._intake.normalize

    async def reject(*a, **k):
        raise ValueError("invalid reply content")

    rt.hitl._intake.normalize = reject
    with pytest.raises(ValueError):
        await initialize_all(rt)
        await rt.reply_to_hitl(HitlReply(
            hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="bad"))
    tm = rt._task_managers[sid]
    assert not tm.open_round_task_ids, "应答没被收下，本次开的窗必须撤掉"
    assert q1.id in [p.id for p in rt.hitl_registry.list_pending(session_id=sid)]

    rt.hitl._intake.normalize = real
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="GOOD"))
    await _poll(lambda: rt.hitl_registry.get(q1.id).resolved or None, timeout=8.0)


@pytest.mark.asyncio
async def test_duplicate_reply_keeps_the_original_window() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _build(llm, hitl_timeout_sec=0)
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    reply = HitlReply(hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1")
    await initialize_all(rt)
    assert await rt.reply_to_hitl(reply) is not None
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)

    await initialize_all(rt)
    assert await rt.reply_to_hitl(reply) is None          # 并发的重复应答：幂等
    tm = rt._task_managers[sid]
    assert tm.is_round_open(q1.task_id), "别人开的窗不得被重复应答撤掉"
    assert rt.hitl_registry.get(q1.id).claim_pending


# ── M2：注入消息不替别人的窗口提前提交 ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_injecting_into_an_open_hot_round_does_not_commit_it() -> None:
    """`send_message` 在 agent 跑着时直接拒（AgentBusyError），所以这一形状从公开入口只在
    「窗开着、task 已排队未开跑」的窄窗里可达；这里直接驱动注入分支钉住判据。"""
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _build(llm, hitl_timeout_sec=None)
    sid = await _start(rt)
    q1 = await _next_question(rt, sid)
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)

    await rt._inject_user_turn(q1.task_id, MESSAGE, session_id=sid)

    tm = rt._task_managers[sid]
    assert tm.is_round_open(q1.task_id), "热应答那一轮还没等到 LLM 开口，不得被注入提前提交"
    assert EventType.HITL_RESOLVED not in await _stored_types(rt, sid)
    assert rt.hitl_registry.get(q1.id).claim_pending
