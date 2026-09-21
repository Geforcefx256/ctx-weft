"""端到端：`ask_user` 的答复——热路径与冷路径——在 LLM 开口之前都能撤回，且不留没人关的窗。

`reply_to_hitl` 在 `resolve` 之前按 task 开未提交窗口（两阶段终局，spec 2026-09-09）。

- **窗口必须有人关。** 冷应答开出新一轮，由那一轮的 act 提交点关窗。热应答没有新 run，
  被叫醒的协程早已过过一次提交点（run 级幂等标志）——曾经窗口一直开到 run 结束，该 task
  的事件全挡在缓冲里。host 上的症状：agent 连问两个问题，第二个问题答了像没答、一直要人
  重答。现在由 gateway 在协程醒来时重新武装提交点。
- **答复要能撤回。** 人答完、LLM 还没开口时按暂停，这条答复当作没说过，问题回到待答，
  重答照常生效。热路径的 run 早已落盘 `RUN_STARTED`，所以它不是整轮丢弃，而是撤回答复
  后照常 park 在那个问题上。
- **终局事实先于答复进 memory。** 这一轮算数之前，答复（及其回灌的工具结果、用户消息）
  只在窗口的暂存区里——memory 里没有，但装配给 LLM 的 prompt 里有；提交时先发
  `HitlResolved`，再落盘。
"""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext, ToolCall
from ctx_weft.protocols.events import EventFilter, EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _all_request_text, _finish_call, _poll
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests._event_helpers import all_events

pytestmark = pytest.mark.asyncio

#: act 回合的剧本项：卡在首个 chunk 之前（TTFT 窗口）/ 吐一个 token 之后卡住。
STALL_BEFORE_CHUNK = "stall_before_chunk"
STALL_AFTER_CHUNK = "stall_after_chunk"
#: 首个 chunk 之前直接抛错（provider 永久错 / outage 耗尽那一类）。
FAIL_BEFORE_CHUNK = "fail_before_chunk"


def _ask(tc_id: str, question: str) -> MockResponse:
    return MockResponse(text="", tool_calls=[ToolCall(
        id=tc_id, name="control__ask_user",
        arguments={"questions": [{"question": question}]})])


class _ScriptedLLM(_ActRouterLLM):
    """act 回合按剧本出；剧本项可以是 `MockResponse` 或上面两个卡住标记。耗尽后一律卡住。"""

    def __init__(self, script: list) -> None:
        super().__init__(act_responses=[], context_limit=100_000)
        self.script = list(script)
        self.stalled = asyncio.Event()

    def complete(self, request, stream: bool = True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return super().complete(request, stream=stream)
        self.act_requests.append(request)
        self.last_request = request
        item = self.script.pop(0) if self.script else STALL_BEFORE_CHUNK
        if isinstance(item, MockResponse):
            return self._stream(item, request)

        async def _gen():
            if item == FAIL_BEFORE_CHUNK:
                raise RuntimeError("provider exploded before the first chunk")
            if item == STALL_AFTER_CHUNK:
                yield LLMChunk(kind="token", text="thinking")
            self.stalled.set()
            await asyncio.Event().wait()
            yield LLMChunk(kind="token", text="unreachable")  # pragma: no cover

        return _gen()


def _runtime(llm, *, hitl_timeout_sec):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _next_question(rt, sid, *, exclude=()):
    pending = await _poll(lambda: [
        p for p in rt.hitl_registry.list_pending(session_id=sid)
        if p.form == "question" and p.id not in exclude] or None)
    return pending[0]


async def _stored_types(rt, sid) -> list[str]:
    return [e.type for e in await all_events(rt.event_store, sid)]


async def _ask_user_results(rt, sid, req) -> list[str]:
    view = await rt.providers.get_memory().load_view(
        MemoryAddress(session_id=sid, task_id=req.task_id, agent_id=req.agent_id),
        MemoryScope.TASK,
        ProviderContext(session_id=sid, tenant_id="default",
                        task_id=req.task_id, agent_id=req.agent_id),
    )
    return [content_to_text(r.content) for r in view
            if r.role == "tool" and r.metadata.get("tool_name") == "control__ask_user"]


async def test_second_hot_question_is_visible_to_live_subscribers() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), _ask("tc2", "Q2?"), STALL_AFTER_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=None)            # 永不驱逐 = 恒走热路径
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    # host 的 SSE 走的就是普通（非 provisional）订阅：窗口里的事件它收不到。
    seen: list[tuple[str, str]] = []

    async def _subscribe() -> None:
        async for ev in handle.event_bus.stream(EventFilter()):
            if ev.session_id == sid:
                seen.append((ev.type, (ev.payload or {}).get("hitl_id", "")))

    sub = asyncio.create_task(_subscribe())
    try:
        q1 = await _next_question(rt, sid)
        await _poll(lambda: (EventType.HITL_OPENED, q1.id) in seen or None)
        await rt.reply_to_hitl(HitlReply(
            hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))

        q2 = await _next_question(rt, sid, exclude={q1.id})
        # 修复前：Q1 的 HitlResolved 与 Q2 的 HitlOpened 都卡在窗口缓冲里，这里超时。
        await _poll(lambda: (EventType.HITL_OPENED, q2.id) in seen or None)
        assert (EventType.HITL_RESOLVED, q1.id) in seen
        await rt.reply_to_hitl(HitlReply(
            hitl_id=q2.id, outcome="accepted", agent_id=q2.agent_id, message="A2"))

        # 第三轮 LLM 开口了（吐了一个 token 后卡住，run 没结束）→ Q2 的答复此刻就该可见。
        await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
        await _poll(lambda: (EventType.HITL_RESOLVED, q2.id) in seen or None)
        assert not rt._task_managers[sid].open_round_task_ids
        text = _all_request_text(llm.act_requests[2])
        assert "A1" in text and "A2" in text
    finally:
        sub.cancel()


@pytest.mark.parametrize("hitl_timeout_sec", [None, 0], ids=["hot", "cold"])
async def test_ask_user_reply_is_retracted_by_pause_before_first_chunk(hitl_timeout_sec) -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK, _finish_call()])
    rt = _runtime(llm, hitl_timeout_sec=hitl_timeout_sec)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)                  # 冷路径：等 park 落定
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="RETRACTED"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    assert q1.id not in [p.id for p in rt.hitl_registry.list_pending(session_id=sid)]
    assert EventType.HITL_RESOLVED not in await _stored_types(rt, sid), (
        "LLM 开口之前，这条答复不得终局落盘")
    assert await _ask_user_results(rt, sid, q1) == [], (
        "答复还没终局，不得进 memory（只能在窗口的暂存区里）")
    assert "RETRACTED" in _all_request_text(llm.act_requests[-1]), (
        "暂存的答复必须叠加进这一轮装配给 LLM 的 prompt")

    # ── 按暂停：答复撤回 ─────────────────────────────────────────────────────
    assert await rt.pause_session(sid) is True
    tm = rt._task_managers[sid]
    await _poll(lambda: tm.get_task(q1.task_id).status == "AWAITING_HUMAN" or None)
    await asyncio.sleep(0.3)

    rec = rt.hitl_registry.get(q1.id)
    assert not rec.resolved and not rec.claim_pending and not rec.claimed
    assert q1.id in [p.id for p in rt.hitl_registry.list_pending(session_id=sid)], (
        "撤回之后问题必须回到待答列表，人才能重答")
    types = await _stored_types(rt, sid)
    assert EventType.HITL_RESOLVED not in types
    assert EventType.HITL_REPLY_RETRACTED in types
    assert not tm.open_round_task_ids
    assert await _ask_user_results(rt, sid, q1) == [], "撤回的答复不得留在对话里"
    # 日志里的 run 必须有始有终：每条 RUN_STARTED 都配一条 RUN_FINISHED。
    assert types.count(EventType.RUN_STARTED) == types.count(EventType.RUN_FINISHED)

    # ── 重答：照常生效，模型只看见新答复 ─────────────────────────────────────
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="RETYPED"))
    await _poll(lambda: tm.get_task(q1.task_id).status == "FINISHED" or None)
    last = _all_request_text(llm.act_requests[-1])
    assert "RETYPED" in last and "RETRACTED" not in last
    results = await _ask_user_results(rt, sid, q1)
    assert len(results) == 1 and "RETYPED" in results[0], results
    assert (await _stored_types(rt, sid)).count(EventType.HITL_RESOLVED) == 1


async def test_replying_to_an_already_resolved_request_opens_no_window() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=None)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    q1 = await _next_question(rt, sid)
    reply = HitlReply(hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1")
    assert await rt.reply_to_hitl(reply) is not None
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    await _poll(lambda: rt.hitl_registry.get(q1.id).resolved or None)
    # 双击 / 陈旧页签重放：幂等 None，且不得开出一扇永远没人关的窗。
    assert await rt.reply_to_hitl(reply) is None
    assert not rt._task_managers[sid].open_round_task_ids


@pytest.mark.parametrize("hitl_timeout_sec", [None, 0], ids=["hot", "cold"])
async def test_reply_is_committed_when_the_round_ends_without_a_chunk(hitl_timeout_sec) -> None:
    """一轮没等到 LLM 开口、也不是被暂停，而是失败收场 → `_run_task` 兜底提交窗口。

    曾经那条兜底只关窗、不终局答复：`HitlResolved` 永远不发，答复停在待终局——进程内
    不在待答列表里，重启后又冒出来让人重答。
    """
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), *([FAIL_BEFORE_CHUNK] * 10)])
    rt = _runtime(llm, hitl_timeout_sec=hitl_timeout_sec)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))

    tm = rt._task_managers[sid]
    await _poll(lambda: not tm.open_round_task_ids
                and len(llm.act_requests) >= 2 or None, timeout=10.0)
    await _poll(lambda: rt.hitl_registry.get(q1.id).resolved or None)
    rec = rt.hitl_registry.get(q1.id)
    assert not rec.claim_pending
    assert EventType.HITL_RESOLVED in await _stored_types(rt, sid)


class _MemoryIngestSpy:
    """包住 memory provider：每次写入时记下「此刻事件日志里有没有那条 HitlResolved」。"""

    def __init__(self, inner, rt, predicate) -> None:
        self._inner, self._rt, self._predicate = inner, rt, predicate
        self.seen: list[tuple[str, bool]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def ingest(self, event, ctx):
        if self._predicate(event):
            stored = [e.type for e in await all_events(self._rt.event_store, ctx.session_id)]
            self.seen.append((content_to_text(event.content), EventType.HITL_RESOLVED in stored))
        return await self._inner.ingest(event, ctx)


@pytest.mark.parametrize("hitl_timeout_sec", [None, 0], ids=["hot", "cold"])
async def test_ask_user_reply_enters_memory_only_after_hitl_resolved(hitl_timeout_sec) -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_AFTER_CHUNK])
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec))
    # 在建会话之前换上间谍：gateway / runtime / act 取到的都是它。
    spy = _MemoryIngestSpy(
        InMemoryMemoryProvider(), rt,
        lambda ev: ev.role == "tool" and ev.metadata.get("tool_name") == "control__ask_user")
    rt.providers.register_memory(spy)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="ANSWER"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    await _poll(lambda: spy.seen or None)
    assert len(spy.seen) == 1 and "ANSWER" in spy.seen[0][0] and spy.seen[0][1], (
        f"ask_user 的答复必须在 HitlResolved 落盘之后才写进 memory：{spy.seen}")
