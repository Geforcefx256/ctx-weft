"""(run_id, sequence) 必须唯一；每个 run_id 必须有起止（总账 A4 / C5）。

**范围裁定（控制方 2026-09-03）**：唯一性断言只针对 background recap 自己的
run_id，不对主 run 的 run_id 做全局唯一性断言——`_run_loop` 与 `driver.run`
两侧的 `LoopState` 在第一个 `state_patch` 落下时分叉成两个独立对象，
`_run_loop` 手里那份 `sequence_counter` 冻结在 prepare 之后的值，末尾
`RunFinished` 把它 +1 后确定性地撞上 prepare 的 `StepCompleted`——这是
`docs/follow-ups/2026-09-03-outstanding-issues.md` A11 记录的**独立、既有**缺陷，
与 A4（后台 recap 蹭主 run 的号）无关，本 task 不修（结构性改动，且与「除两条
命名修复外行为等价」的约束冲突）。「recap 的 run_id ≠ 主 run 的 run_id」+
「recap 那个 run_id 内部无重号」这两条端到端断言，测的正是 A4 修好的那件事，
且不会被 A11 撞到。「每个 run 都有起止」不受 A11 影响（A11 只错了序号，不影响
RunStarted/RunFinished 是否发出、run_id 是否一致），照抄 brief 原样保留、
仍是全局断言。
"""

from __future__ import annotations

import asyncio
import collections
import time
from dataclasses import dataclass

import pytest

from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio


def _assert_no_duplicate_sequence(events):
    seen = collections.defaultdict(set)
    dupes = []
    for e in events:
        if e.run_id is None:
            continue          # run 外的事件恒 sequence=0，不参与唯一性
        if e.sequence in seen[e.run_id]:
            dupes.append(f"{e.run_id}:{e.sequence}:{e.type}")
        seen[e.run_id].add(e.sequence)
    assert dupes == [], f"(run_id, sequence) 撞号: {dupes}"


def _assert_every_run_has_start_and_finish(events):
    starts = {e.run_id for e in events if e.type == EventType.RUN_STARTED}
    finishes = {e.run_id for e in events if e.type == EventType.RUN_FINISHED}
    seen = {e.run_id for e in events if e.run_id is not None}
    orphans = sorted(seen - starts) + sorted(seen - finishes)
    assert orphans == [], f"孤儿 run（无 RunStarted 或无 RunFinished）: {orphans}"


class _RouterLLM(MockLLMAdapter):
    """按 request.tools 路由：recognize_intent（工具集含
    `control__update_task_metadata`）恒回空文本；root task 的 act 首轮回纯文本、
    无 tool_call。

    root task 恒 `interaction_mode="interactive"`（`session_registry.py:355`），
    纯文本首轮触发冷 park（`act.py::_finish_plain_text_turn`），同时并发一次
    `launch_background_observe(boundary="plain_text")`（`act.py:491-493`）——
    这是 A4 的最小复现：该段第一条事件（ACT_TURN_COMPLETED "await_user"）之后，
    主 run 与后台 recap 的快照在同一个 sequence_counter 值上分叉，各自 +=1，
    必然在某个 (run_id, sequence) 上撞号，与调度顺序无关（详见
    task-5-report.md）。is_short_segment 的「段内 ≤1 条 assistant 回复」免折规则
    使这条边界不需要第二次 LLM 调用（`background_observe.py::is_short_segment`），
    所以本路由器只需处理两类工具集，不用另处理
    `control__collect_process_report`。同一次 root task 还会并发跑一次
    recognize_intent（root task 无 title），顺带覆盖 C5 的另一个孤儿 run
    （recognize_intent）。
    """

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)

    def complete(self, request, stream: bool = True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:  # recognize_intent
            return self._stream(MockResponse(text=""), request)
        # act 首轮：纯文本、无 tool_call → root 任务的纯文本冷 park。
        return self._stream(MockResponse(text="Hi! Anything else?"), request)


async def _poll(predicate, *, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError("timed out waiting for background recap to finish")


@dataclass
class _RecapRun:
    events: list
    main_run_id: str


@pytest.fixture
async def bus_after_recap_run() -> _RecapRun:
    """跑一个会触发 background observe 的 run，收集全部发射事件 + 主 run 的 run_id。

    搭台手法照 `tests/unit/test_background_observe_wiring.py`（四个触发点的接线
    单测）与 `tests/integration/test_hitl_e2e_v2.py::
    test_plain_text_pause_injects_reply_once_and_ignores_duplicate`（真实
    `CtxWeftRuntime.start_session` 驱动纯文本冷 park）：这里只保留后者到「冷 park
    落定」为止，不再答复续跑——本测试只关心 park 那一刻并发产生的事件，不关心
    续跑后的第二个 run。
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _RouterLLM()
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    seen: list = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi", context_limit=100_000,
    ))
    state = await handle.wait_for_finish(timeout=5.0)
    assert state is not None and state.task.status == "AWAITING_HUMAN"

    # background recap 与 recognize_intent 都是 fire-and-forget：wait_for_finish
    # 只等主 run 的 RunFinished，还要等它们各自的 RunFinished 落地才算收齐。
    await _poll(lambda: sum(1 for e in seen if e.type == EventType.RUN_FINISHED) >= 3)

    # `TurnHandle` 不再带 `run_id`（Task 5，句柄改轴成 agent+task）——主 run 的 run_id
    # 从事件信封里读：`_run_loop` 发 RunStarted 时把 origin 显式钉成 RUNTIME
    # （runtime.py，`await self._event_bus.emit(make_event(..., origin=EventOrigin.RUNTIME))`），
    # 而 recap（LOOP_BACKGROUND_OBSERVE）与 recognize_intent（LOOP_RECOGNIZE_INTENT）两条
    # 孤儿 run 都不做这个覆盖、各自沿用自己快照的 origin。本 fixture 不触发
    # compact_agent（唯一另一处同样显式钉 RUNTIME 的路径），故 origin=RUNTIME 在这里
    # 无歧义地唯一指向主 run。
    main_run_started = next(
        e for e in seen
        if e.type == EventType.RUN_STARTED and e.origin == EventOrigin.RUNTIME
    )
    return _RecapRun(events=seen, main_run_id=main_run_started.run_id)


async def test_background_observe_does_not_reuse_main_run_sequence(bus_after_recap_run):
    """A4：recap 的 run_id 必须换新号，且它名下的事件序列内部不重号。

    不对主 run 的 run_id 做全局唯一性断言——那会撞上 A11（既有、独立的缺陷，
    见模块 docstring）。这条测的是 A4 实际修的那件事：recap 不再蹭主 run 的号。
    """
    recap_run_ids = {
        e.run_id for e in bus_after_recap_run.events
        if e.type == EventType.TASK_RECAP_STARTED
    }
    assert recap_run_ids, "fixture 没有触发 background observe（TaskRecapStarted 缺失）"
    for rid in recap_run_ids:
        assert rid != bus_after_recap_run.main_run_id, (
            f"recap 复用了主 run 的 run_id: {rid}（A4 未修好）"
        )
    recap_events = [e for e in bus_after_recap_run.events if e.run_id in recap_run_ids]
    _assert_no_duplicate_sequence(recap_events)


async def test_every_run_id_has_start_and_finish(bus_after_recap_run):
    """C5：每个 run_id（含主 run、recognize_intent、background recap）都有起止。

    这条保留为全局断言——A11 只错了 RunFinished 的序号，不影响它「发没发」，
    也不影响 run_id 是否一致，所以不会被 A11 撞到。
    """
    _assert_every_run_has_start_and_finish(bus_after_recap_run.events)


async def test_compact_agent_run_has_start_and_finish():
    """C5 的第三种孤儿 run：`compact_agent` 手动压缩，不经 `_run_loop`。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY")])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)

    seen: list = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    from datetime import datetime, timezone

    from ctx_weft.protocols.events import Event

    sid, aid = "ses_c5", "agt_root"
    ts = datetime(2026, 6, 16, tzinfo=timezone.utc)
    await runtime.event_store.append(Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id=sid,
        type=EventType.SESSION_CREATED, timestamp=ts,
        payload={"template_id": "agent:tpl_echo", "user_prompt": "x",
                 "root_agent_id": aid, "llm_model": "mock", "context_limit": 180000},
    ))
    runtime._agent_lifecycle_manager.register_session(
        sid, tenant_id="default", fallback_template_id="agent:tpl_echo",
    )
    runtime._agent_lifecycle_manager.materialize(
        aid, session_id=sid, tenant_id="default")

    await runtime.compact_agent(aid)

    _assert_no_duplicate_sequence(seen)
    _assert_every_run_has_start_and_finish(seen)
    compact_runs = {
        e.run_id for e in seen
        if e.type == EventType.RUN_STARTED and e.payload.get("initial_step") == "compact"
    }
    assert len(compact_runs) == 1, f"expected exactly one compact run, got {compact_runs}"
