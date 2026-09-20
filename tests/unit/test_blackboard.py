"""Blackboard / topic 机制的行为测试。

覆盖：
1. Publish 覆盖语义：同 topic 的 BLACKBOARD_PUBLISH 只保留最新一条；非该类型不被覆盖。
2. 订阅按 task 隔离 + session 级可见 + 幂等（保留 cursor）。
3. BlackboardSource 只拉本 task 的订阅，并把 title/outcome 透传到 block。
4. driver 启动钩子（Phase 3 后为 no-op）：不再建立任何订阅——predecessor 经 memory recall 获取，
   子任务句柄经 observe cue（extra["subtasks"]）获取。
5. composer 不再渲染 blackboard 块为 observer 段（Phase 3 后 subtask/predecessor bb 段已移除）——
   子任务句柄经 extra["subtasks"] 渲染；predecessor 不出现于 prompt。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import (
    AssemblerDeps,
    ContextBlock,
    ContextRequest,
)
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.blackboard import BlackboardSource
from ctx_weft.core.loop.driver import StepDriver
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _publish(topic: str, content: str, title: str = "Report", outcome: str = "success") -> MemoryEvent:
    return MemoryEvent(
        type=MemoryEventType.BLACKBOARD_PUBLISH,
        address=MemoryAddress(session_id="s1", task_id=topic, agent_id="a"),
        content=content,
        timestamp=datetime.now(timezone.utc),
        topic=topic,
        metadata={"title": title, "outcome": outcome},
    )


# ── 1. Publish 覆盖语义 ──────────────────────────────────────────────────────────


async def test_publish_overwrites_same_topic() -> None:
    m = InMemoryMemoryProvider()
    await m.ingest(_publish("A", "v1"), _ctx())
    await m.ingest(_publish("A", "v2"), _ctx())
    await m.ingest(_publish("A", "v3"), _ctx())

    recs, cursor = await m.recall_topic("A", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["v3"]
    assert recs[0].metadata.get("title") == "Report"
    assert cursor == recs[0].metadata["topic_seq_no"]


async def test_publish_topics_are_independent() -> None:
    m = InMemoryMemoryProvider()
    await m.ingest(_publish("A", "a-latest"), _ctx())
    await m.ingest(_publish("B", "b-latest"), _ctx())

    a_recs, _ = await m.recall_topic("A", since=0, ctx=_ctx())
    b_recs, _ = await m.recall_topic("B", since=0, ctx=_ctx())
    assert [r.content for r in a_recs] == ["a-latest"]
    assert [r.content for r in b_recs] == ["b-latest"]


async def test_non_publish_topic_events_not_superseded() -> None:
    """只有 BLACKBOARD_PUBLISH 触发覆盖；其它带 topic 的类型应累积保留。"""
    m = InMemoryMemoryProvider()
    log = lambda c: MemoryEvent(  # noqa: E731
        type=MemoryEventType.OBSERVER_SUMMARY,
        address=MemoryAddress(session_id="s1", agent_id="a"),
        content=c, timestamp=datetime.now(timezone.utc), topic="proj_log",
    )
    await m.ingest(log("entry1"), _ctx())
    await m.ingest(log("entry2"), _ctx())

    recs, _ = await m.recall_topic("proj_log", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["entry1", "entry2"]


# ── 2. 订阅隔离 / session 级 / 幂等 ───────────────────────────────────────────────


async def test_subscription_is_task_scoped() -> None:
    m = InMemoryMemoryProvider()
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    await m.subscribe_topic("s1", topic="X", intent="subtask", ctx=_ctx(), task_id="C")

    b = await m.list_subscriptions("s1", ctx=_ctx(), task_id="B")
    c = await m.list_subscriptions("s1", ctx=_ctx(), task_id="C")
    assert sorted(s.topic for s in b) == ["A"]
    assert sorted(s.topic for s in c) == ["X"]


async def test_session_level_subscription_visible_to_all() -> None:
    m = InMemoryMemoryProvider()
    # task_id="" → session 级
    await m.subscribe_topic("s1", topic="G", intent="long_term_background", ctx=_ctx(), task_id="")
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")

    b = {s.topic for s in await m.list_subscriptions("s1", ctx=_ctx(), task_id="B")}
    c = {s.topic for s in await m.list_subscriptions("s1", ctx=_ctx(), task_id="C")}
    assert b == {"G", "A"}     # 自己的 + session 级
    assert c == {"G"}          # 仅 session 级


async def test_subscribe_is_idempotent_and_preserves_cursor() -> None:
    m = InMemoryMemoryProvider()
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    # 模拟游标推进。键含 tenant（Phase 3c Task C1b 起为 (tenant, session, task, topic)——
    # 否则同 session_id 的另一个租户会撞上幂等分支、拿到别人的订阅与游标）；
    # _ctx() 不传 tenant_id，走 ProviderContext 的默认值 "default"。
    m._subscriptions[("default", "s1", "B", "A")].cursor = 7
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")

    subs = await m.list_subscriptions("s1", ctx=_ctx(), task_id="B")
    assert len(subs) == 1
    assert subs[0].cursor == 7


async def test_list_subscriptions_none_returns_all() -> None:
    m = InMemoryMemoryProvider()
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    await m.subscribe_topic("s1", topic="X", intent="subtask", ctx=_ctx(), task_id="C")
    allsubs = await m.list_subscriptions("s1", ctx=_ctx(), task_id=None)
    assert sorted(s.topic for s in allsubs) == ["A", "X"]


# ── 3. BlackboardSource 端到端 ───────────────────────────────────────────────────


def _request_for(task_id: str, m: InMemoryMemoryProvider) -> tuple[ContextRequest, AssemblerDeps]:
    task = Task(id=task_id, session_id="s1", status="ACTIVE")
    agent = Agent(id="a", session_id="s1", template_id="t")
    session = Session(id="s1", user_prompt="go", status="RUNNING")
    req = ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id=task_id, agent_id="a"),
        task=task, agent=agent, session=session, template=None, bound_capabilities=[],
    )
    deps = AssemblerDeps(memory=m, knowledge_providers=[], provider_ctx=_ctx())
    return req, deps


async def test_blackboard_source_pulls_only_current_task_subs() -> None:
    m = InMemoryMemoryProvider()
    await m.ingest(_publish("A", "result-A", title="Build A"), _ctx())
    await m.subscribe_topic("s1", topic="A", intent="subtask", ctx=_ctx(), task_id="B")

    # task B 订阅了 A → 能看到
    req_b, deps = _request_for("B", m)
    blocks_b = [blk async for blk in BlackboardSource().fetch(req_b, deps)]
    assert len(blocks_b) == 1
    assert blocks_b[0].metadata["title"] == "Build A"
    assert blocks_b[0].metadata["outcome"] == "success"
    assert "result-A" in blocks_b[0].content

    # task C 没订阅 → 看不到
    req_c, _ = _request_for("C", m)
    blocks_c = [blk async for blk in BlackboardSource().fetch(req_c, deps)]
    assert blocks_c == []


# ── 4. driver 启动钩子（Phase 3 后为 no-op）─────────────────────────────────────


async def test_driver_hook_subscribes_predecessors_and_children() -> None:
    # Phase 3 (2026-06-30): _ensure_blackboard_subscriptions is a no-op.
    # Predecessors now surface via memory recall (Phase 2); subtask review handles via
    # the observe cue (extra["subtasks"]). The METHOD still exists (mechanism kept),
    # but it must not call subscribe_topic at all.
    m = InMemoryMemoryProvider()
    tm = TaskManager(session_id="s1")
    # task with predecessors and a child — neither should generate a subscription.
    task = Task(id="T", session_id="s1", status="ACTIVE")
    tm.register_task(task)
    tm._children_of["T"] = {"K1"}

    driver = StepDriver(steps={})
    state = SimpleNamespace(task=task)
    ctx = SimpleNamespace(task_manager=tm, memory=m, provider_ctx=_ctx())

    await driver._ensure_blackboard_subscriptions(state, ctx)

    subs = await m.list_subscriptions("s1", ctx=_ctx(), task_id="T")
    assert subs == [], "Phase 3: driver hook is a no-op — no blackboard subscriptions created"
    all_subs = await m.list_subscriptions("s1", ctx=_ctx(), task_id=None)
    assert all_subs == [], "Phase 3: no subscriptions at all"


async def test_driver_hook_idempotent_across_runs() -> None:
    # Phase 3: multiple calls to the no-op still produce zero subscriptions.
    m = InMemoryMemoryProvider()
    tm = TaskManager(session_id="s1")
    task = Task(id="T", session_id="s1", status="ACTIVE")
    tm.register_task(task)
    driver = StepDriver(steps={})
    state = SimpleNamespace(task=task)
    ctx = SimpleNamespace(task_manager=tm, memory=m, provider_ctx=_ctx())

    await driver._ensure_blackboard_subscriptions(state, ctx)
    await driver._ensure_blackboard_subscriptions(state, ctx)  # 第二次 run

    subs = await m.list_subscriptions("s1", ctx=_ctx(), task_id="T")
    assert len(subs) == 0, "Phase 3: no-op across multiple runs"


# ── 5. composer 渲染（Phase 3 后 blackboard 块不再产生 observer 段）──────────────


async def test_composer_renders_subtask_results_with_title() -> None:
    # Phase 3 (2026-06-30): blackboard blocks with intent="subtask" no longer render as the
    # "Your sub-task results" section. Sub-task handles now come via extra["subtasks"]
    # (Task 1). This test verifies: (a) old blackboard rendering is GONE, and (b) the new
    # extra-based path still surfaces the sub-task list.
    comp = DefaultComposer()
    task = Task(id="T", session_id="s1", status="ACTIVE",
                title="Parent", description="do it", user_prompt="please")
    session = Session(id="s1", user_prompt="go", status="RUNNING")
    block = ContextBlock(
        id="b1", source="blackboard:K1", kind="blackboard", target="messages",
        content="Q1 sales: 100", priority=2, token_estimate=5,
        metadata={"title": "Build report", "outcome": "success", "intent": "subtask"},
    )

    # (a) Blackboard block alone — old section must be absent.
    req_no_extra = ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id="T", agent_id="a"),
        task=task, agent=None, session=session, template=None, bound_capabilities=[],    )
    msgs = comp._build_observer_messages([block], req_no_extra)
    text = msgs[-1].content
    assert "Your sub-task results (you may confirm / reopen these):" not in text, (
        "Phase 3: blackboard subtask blocks must not render as a subtask section"
    )

    # (b) New path: extra["subtasks"] still renders the cue.
    req_with_extra = ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id="T", agent_id="a"),
        task=task, agent=None, session=session, template=None, bound_capabilities=[],        extra={"subtasks": [{"task_id": "K1", "title": "Build report", "outcome": "success"}]},
    )
    msgs2 = comp._build_observer_messages([block], req_with_extra)
    text2 = msgs2[-1].content
    assert "## Your sub-tasks" in text2 and "K1" in text2, (
        "Phase 3: sub-task handles must appear via the extra cue"
    )
    # 清单只作信息：指路的是 next_step_hint，不再是已删的 task_reviews 参数
    assert "next_step_hint" in text2
    assert "task_reviews" not in text2


async def test_composer_splits_subtask_and_predecessor_sections() -> None:
    # Phase 3 (2026-06-30): both blackboard observer sections (subtask + predecessor) are removed.
    # Neither "Your sub-task results" nor "Upstream task results" must appear, regardless of
    # what blackboard blocks are present. The judgment cue must still be there.
    comp = DefaultComposer()
    task = Task(id="T", session_id="s1", status="ACTIVE", title="Parent")
    session = Session(id="s1", user_prompt="go", status="RUNNING")
    req = ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id="T", agent_id="a"),
        task=task, agent=None, session=session, template=None, bound_capabilities=[],    )
    sub = ContextBlock(
        id="b1", source="blackboard:K1", kind="blackboard", target="messages",
        content="child output", priority=2, token_estimate=5,
        metadata={"title": "Child", "outcome": "success", "intent": "subtask"},
    )
    pred = ContextBlock(
        id="b2", source="blackboard:P1", kind="blackboard", target="messages",
        content="upstream output", priority=2, token_estimate=5,
        metadata={"title": "Upstream", "outcome": "success", "intent": "predecessor"},
    )
    text = comp._build_observer_messages([sub, pred], req)[-1].content
    # Both old blackboard sections must be absent.
    assert "Your sub-task results" not in text, (
        "Phase 3: blackboard subtask section must not appear in observer prompt"
    )
    assert "Upstream task results (read-only context):" not in text, (
        "Phase 3: blackboard predecessor section must not appear in observer prompt"
    )
    # Judgment cue must still be present.
    assert "report_task_outcome" in text


async def test_composer_no_related_results_when_empty() -> None:
    # Phase 3: "Your sub-task results" (old blackboard heading) never appears — it was removed.
    # The extra["subtasks"] section ("## Your sub-tasks") is also absent when there are none.
    comp = DefaultComposer()
    task = Task(id="T", session_id="s1", status="ACTIVE", title="Parent")
    session = Session(id="s1", user_prompt="go", status="RUNNING")
    req = ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id="T", agent_id="a"),
        task=task, agent=None, session=session, template=None, bound_capabilities=[],    )
    msgs = comp._build_observer_messages([], req)
    joined = " ".join(m.content for m in msgs if isinstance(m.content, str))
    assert "Your sub-task results" not in joined   # Phase 3: old blackboard heading removed
    assert "## Your sub-tasks" not in joined        # no extra["subtasks"] → no cue section
    assert "report_task_outcome" in joined          # 仍带判定提示
