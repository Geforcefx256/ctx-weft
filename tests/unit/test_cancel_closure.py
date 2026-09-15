"""统一取消胶囊闭合（Task 14）：所有 CANCELED 终态 = 父 ack 终态化 + `[outcome=cancelled]` finish 对。

覆盖 task-14-brief.md 「测试覆盖」清单：
- cancel_all 已启动任务 → ack 替换 + finish 对（同 agent 嵌套形态断言 `[outcome=cancelled]`）。
- 未启动 → 零 memory 写（TaskManager 层的 started_at 过滤：从不进 cancel_finalizer 调用列表）。
- born-cancel 无框 → 跳过不补铸（`synthesize_cancel_closure` 的 find-only 分支）。
- 在途取消 funnel → on_task_finished(CANCELED) 后才有 finish 对，信号时刻只有 ack
  （TaskManager 从不在"发信号"处调用 cancel_finalizer，只在终态坐实时调用一次）。
- 抢跑正常收尾 → 无取消 finish 对（cancel_finalizer 只在 status=="CANCELED" 时被调用）。
- root 被 cancel_all → own scope finish 对。
- 熔断路径回归见 tests/unit/test_failure_threshold_trip.py（本文件不重复）。
"""

from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.loop.steps.finalize import (
    _dispatch_running_ack,
    _ensure_dispatch_frame,
    _put_dispatch_result,
    synthesize_cancel_closure,
)
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime
from tests.unit._stub_runner import StubRunner

pytestmark = pytest.mark.asyncio

T = MemoryEventType


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _child(tid: str, parent: str = "root", *, creator: str = "agt_root",
          assigned: str = "agt_root", **kw) -> Task:
    return Task(
        id=tid, session_id="s1", status="ACTIVE", parent_task_id=parent,
        creator_agent_id=creator, assigned_agent_id=assigned, title=kw.pop("title", f"Child {tid}"),
        **kw,
    )


async def _seed_running_ack(mem, child: Task) -> None:
    """先手工 ingest 一个派发框 + running ack（模拟子任务真正 start 后留下的痕迹）。"""
    parent_scope = MemoryAddress(
        session_id="s1", task_id=child.parent_task_id, agent_id=child.creator_agent_id,
    )
    ctx = _ctx()
    ts, tcid = await _ensure_dispatch_frame(mem, parent_scope, child, ctx)
    await _put_dispatch_result(
        mem, parent_scope, child, _dispatch_running_ack(task_ref(child)), ts, ctx, replace=False,
        tool_call_id=tcid,
    )


# ── synthesize_cancel_closure（finalize.py 直测，真实 InMemoryMemoryProvider）───────────


async def test_same_agent_started_task_ack_replaced_and_nested_finish_pair() -> None:
    mem = InMemoryMemoryProvider()
    child = _child("c1", started_at=now_utc(), origin_tool_call_id="call-1")
    await _seed_running_ack(mem, child)

    await synthesize_cancel_closure(mem, "s1", child, _ctx(), "user_cancel")

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    recs = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    tool_recs = [r for r in recs if r.role == "tool"]

    # ack 终态化：旧 running ack 被 supersede，只剩一条取消文案（配对 tool_call_id=call-1）
    ack = [r for r in tool_recs if r.metadata.get("tool_call_id") == "call-1"]
    assert len(ack) == 1
    assert "Sub-task 'Child c1' (c1) was cancelled before completion (user_cancel)" in ack[0].content
    assert "its partial execution below is incomplete" in ack[0].content

    # 嵌套 finish 对（同 agent 共享 scope）：assistant 槽收尾标记 + tool 槽 [outcome=cancelled]
    finish_tool = [r for r in tool_recs if r.metadata.get("origin_task_id") == "c1"]
    assert len(finish_tool) == 1
    assert "[outcome=cancelled]" in finish_tool[0].content
    assert "Cancelled (user_cancel) — no final output was produced" in finish_tool[0].content
    assert "the partial execution above is all that ran" in finish_tool[0].content

    finish_assistant = [
        r for r in recs if r.role == "assistant" and r.metadata.get("origin_task_id") == "c1"
    ]
    assert len(finish_assistant) == 1
    assert finish_assistant[0].content.startswith("Task was cancelled before completion.")
    tool_calls = finish_assistant[0].metadata.get("tool_calls") or []
    assert any(tc.get("name", "").endswith("finish_task") for tc in tool_calls)


async def test_cross_agent_started_task_ack_and_own_scope_finish_pair() -> None:
    mem = InMemoryMemoryProvider()
    child = _child(
        "c1", started_at=now_utc(), origin_tool_call_id="call-1",
        creator="agt_root", assigned="agt_child",
    )
    await _seed_running_ack(mem, child)

    await synthesize_cancel_closure(mem, "s1", child, _ctx(), "user_cancel")

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    parent_recs = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    ack = [r for r in parent_recs if r.role == "tool" and r.metadata.get("tool_call_id") == "call-1"]
    assert len(ack) == 1
    assert "was cancelled before completion (user_cancel)" in ack[0].content
    # 跨 agent：父 scope 只有 ack，没有 finish 对（那在子自己的 scope）
    parent_finish = [
        r for r in parent_recs if r.role == "assistant" and r.metadata.get("origin_task_id") == "c1"
    ]
    assert parent_finish == []

    own_scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="agt_child")
    own_recs = await mem.recall_recent(own_scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    tool = [r for r in own_recs if r.role == "tool"]
    assert len(tool) == 1
    assert "[outcome=cancelled]" in tool[0].content


async def test_root_own_scope_finish_pair() -> None:
    """root（无 parent_task_id）：own-root 形态不需要框，直接在自己 scope 合成 finish 对。"""
    mem = InMemoryMemoryProvider()
    root = Task(
        id="root", session_id="s1", status="CANCELED", parent_task_id=None,
        creator_agent_id="agt_root", assigned_agent_id="agt_root", title="Root Task",
        started_at=now_utc(),
    )

    await synthesize_cancel_closure(mem, "s1", root, _ctx(), "user_cancel")

    scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    recs = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    assistant = [r for r in recs if r.role == "assistant"]
    tool = [r for r in recs if r.role == "tool"]
    assert len(assistant) == 1 and len(tool) == 1
    assert any(
        tc.get("name", "").endswith("finish_task")
        for tc in (assistant[0].metadata.get("tool_calls") or [])
    )
    assert "[outcome=cancelled]" in tool[0].content
    assert "Cancelled (user_cancel)" in tool[0].content


async def test_born_cancel_no_frame_skips_entirely() -> None:
    """子任务从未真正 start（未走 ensure_dispatch_frame_at_start）→ 无框可闭 → 整体跳过。"""
    mem = InMemoryMemoryProvider()
    child = _child("c1", started_at=now_utc(), origin_tool_call_id="call-missing")
    # 故意不调用 _seed_running_ack：没有预先铸框，模拟 born-cancel。

    await synthesize_cancel_closure(mem, "s1", child, _ctx(), "user_cancel")

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    parent_recs = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    assert parent_recs == []
    own_scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="agt_root")
    own_recs = await mem.recall_recent(own_scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    assert own_recs == []


# ── TaskManager 层：调用点的收集/时机语义 ────────────────────────────────────────


class _CapturingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


def _tm(bus: _CapturingBus) -> tuple[TaskManager, Session]:
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm._max_concurrent = 0
    session = Session(id="s1", user_prompt="", status="RUNNING")
    tm.set_session(session)
    tm.set_runner(StubRunner(tm))
    return tm, session


async def test_cancel_all_routes_only_started_tasks_zero_writes_for_unstarted() -> None:
    """cancel_all：已启动任务进 cancel_finalizer；未启动任务从不进入——零 memory 写。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    started = Task(id="a", session_id="s1", status="PENDING", started_at=now_utc(),
                   settings=NormalTaskSettings())
    unstarted = Task(id="b", session_id="s1", status="PENDING", settings=NormalTaskSettings())
    tm.register_task(started)
    tm.register_task(unstarted)
    tm._queue.push(QueueEntry(task_id="a", session_id="s1"))
    tm._queue.push(QueueEntry(task_id="b", session_id="s1"))

    captured: dict = {}

    async def _finalizer(tasks, reason):
        captured["ids"] = sorted(t.id for t in tasks)
        captured["reason"] = reason

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_finalizer))
    await tm.cancel_all(reason="user_cancel")

    assert captured["ids"] == ["a"]  # unstarted "b" never reaches the finalizer
    assert captured["reason"] == "user_cancel"
    assert tm.get_task("a").status == "CANCELED"
    assert tm.get_task("b").status == "CANCELED"


async def test_cancel_all_routes_started_root_too() -> None:
    """root 没有 origin_tool_call_id，但只要 started_at 非空仍应进 cancel_finalizer
    （own-root 形态不需要框）。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    root = Task(id="root", session_id="s1", status="PENDING", started_at=now_utc(),
               parent_task_id=None, settings=NormalTaskSettings())
    tm.register_task(root)
    tm._queue.push(QueueEntry(task_id="root", session_id="s1"))

    captured: dict = {}

    async def _finalizer(tasks, reason):
        captured["ids"] = [t.id for t in tasks]

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_finalizer))
    await tm.cancel_all(reason="user_cancel")

    assert captured["ids"] == ["root"]


async def test_inflight_cancel_funnel_signal_time_no_call_terminal_time_one_call() -> None:
    """在途取消 funnel：发信号时刻（task 仍在 _running_tasks、status 未坐实）不触发
    cancel_finalizer；只有 on_task_finished(CANCELED) 终态坐实后才调用一次。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    calls: list[tuple[list[str], str]] = []

    async def _finalizer(tasks, reason):
        calls.append((sorted(t.id for t in tasks), reason))

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_finalizer))

    inflight = Task(
        id="c1", session_id="s1", status="ACTIVE", parent_task_id="root",
        started_at=now_utc(), origin_tool_call_id="call-1", settings=NormalTaskSettings(),
    )
    tm.register_task(inflight)
    tm._running_tasks.add("c1")

    # 信号阶段：TaskManager 从不在这里调用 cancel_finalizer（只在终态坐实处调用）。
    assert calls == []

    await tm.on_task_finished("c1", status="CANCELED")

    assert len(calls) == 1
    assert calls[0][0] == ["c1"]


async def test_race_normal_finish_does_not_call_cancel_finalizer() -> None:
    """抢跑正常收尾（status 落 FINISHED，不是 CANCELED）→ 绝不调用 cancel_finalizer，
    FinalizeStep 的真实终态覆盖 ack 自愈，取消胶囊 finish 对不会重复写。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    calls: list = []

    async def _finalizer(tasks, reason):
        calls.append(tasks)

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_finalizer))

    task = Task(
        id="c1", session_id="s1", status="ACTIVE", parent_task_id="root",
        started_at=now_utc(), origin_tool_call_id="call-1", settings=NormalTaskSettings(),
    )
    tm.register_task(task)
    tm._running_tasks.add("c1")

    await tm.on_task_finished("c1", status="FINISHED")

    assert calls == []


async def test_unstarted_task_cancel_never_calls_finalizer() -> None:
    """从未 start（started_at 为空）的任务即便终态落 CANCELED，也不触发 cancel_finalizer——
    它从未铸框/从未写过 memory，没有胶囊可闭。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    calls: list = []

    async def _finalizer(tasks, reason):
        calls.append(tasks)

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_finalizer))

    task = Task(id="c1", session_id="s1", status="PENDING", parent_task_id="root",
               settings=NormalTaskSettings())
    tm.register_task(task)

    await tm.on_task_finished("c1", status="CANCELED")

    assert calls == []


# ── runtime 侧接线：真实 memory 内容断言（同 test_threshold_finalizer.py 风格）─────────


def _runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def test_runtime_finalize_cancel_memory_root_own_scope_finish_pair() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = Session(id="s1", user_prompt="do it", status="RUNNING", root_agent_id="agt_root",
                      tenant_id="default")
    root = Task(id="root", session_id="s1", status="CANCELED", parent_task_id=None,
               creator_agent_id="agt_root", assigned_agent_id="agt_root", title="Root Task",
               started_at=now_utc())

    await runtime._finalize_cancel_memory(session, [root], "user_cancel")

    scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    recs = await memory.recall_recent(scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    tool = [r for r in recs if r.role == "tool"]
    assert len(tool) == 1
    assert "[outcome=cancelled]" in tool[0].content


async def test_runtime_finalize_cancel_memory_one_bad_task_does_not_block_others() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = Session(id="s1", user_prompt="do it", status="RUNNING", root_agent_id="agt_root",
                      tenant_id="default")
    good = _child("good", started_at=now_utc(), origin_tool_call_id="call-good")
    bad = _child("bad", started_at=now_utc(), origin_tool_call_id="call-bad")
    await _seed_running_ack(memory, good)
    await _seed_running_ack(memory, bad)

    orig_ingest = memory.ingest

    async def _boom_for_bad(event, ctx):  # noqa: ANN001
        if event.role == "tool" and event.metadata.get("tool_call_id") == "call-bad":
            raise RuntimeError("simulated ingest failure")
        return await orig_ingest(event, ctx)

    memory.ingest = _boom_for_bad  # type: ignore[assignment]

    await runtime._finalize_cancel_memory(session, [good, bad], "user_cancel")

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    recs = await memory.recall_recent(parent_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    good_ack = [r for r in recs if r.role == "tool" and r.metadata.get("tool_call_id") == "call-good"]
    assert len(good_ack) == 1
    assert "was cancelled before completion" in good_ack[0].content
