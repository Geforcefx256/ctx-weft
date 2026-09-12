"""spec: task-handoff——集成路径：事件回放恢复、ack-id→reopen、计划断裂、阻塞原因可解释。

驱动方式：真 TaskManager + 真 InProcessEventBus 发射真事件 →（模拟崩溃）事件列表经
reduce_events → converters → 新 TaskManager.restore 重建——这正是 rebuild_view 的
内核链路；控制工具经真 ControlContext 调用。
"""

from __future__ import annotations

import pytest

from ctx_weft.core.assembler.assembler import ContextRequest
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource
from ctx_weft.core.capabilities.control_tools import (
    ControlContext,
    ControlMetaKey,
    delegate_plan,
    delegate_task,
    report_task_outcome,
)
from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus
from types import SimpleNamespace


class _CapturingBus:
    def __init__(self) -> None:
        self.bus = InProcessEventBus()
        self.events: list = []

        async def _sink(e) -> None:
            self.events.append(e)

        self.bus.subscribe(None, _sink)


class _NoopRunner:
    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _tm(bus) -> TaskManager:
    tm = TaskManager(session_id="s1", max_concurrent=0, event_bus=bus.bus)
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    tm.set_runner(_NoopRunner())
    return tm


def _ctx(tm: TaskManager, task: Task) -> ControlContext:
    return ControlContext(
        session_id="s1", task_id=task.id, agent_id="a1", task=task,
        task_manager=tm, session=None, tool_call_id="tc_1",
    )


def _parent() -> Task:
    return Task(id="P", session_id="s1", status="ACTIVE", title="parent")


async def _delegate_plan_ordered(tm: TaskManager, parent: Task, specs: list[dict]):
    """delegate_plan + flush，按 stage 顺序返回 (ack, 子任务列表)——spec 顺序的唯一可靠读法
    （created_at 在微秒内并列，不能作排序键）。"""
    order: list[Task] = []
    orig_stage = tm.stage_task

    def _rec(child: Task, **kw) -> None:
        order.append(child)
        orig_stage(child, **kw)

    tm.stage_task = _rec
    try:
        from ctx_weft.core.capabilities.control_tools import delegate_plan as _dp
        ack = _dp(tasks=specs, ctx=_ctx(tm, parent))
        await tm._flush_staged(parent.id)
    finally:
        tm.stage_task = orig_stage
    return ack, order


# ── 2.4 恢复链：inputs 落盘 → 回放重建 → 执行上下文再投递 ────────────────────


@pytest.mark.asyncio
async def test_inputs_survive_crash_and_reach_reassembled_context():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)

    res = delegate_task(
        title="analyze", task_prompt="analyze the file",
        inputs={"file": "report.csv", "threshold": 3},
        ctx=_ctx(tm, parent),
    )
    assert "(task_id:" in res.content
    child = tm._staged[parent.id][0][0]
    await tm.push_task(child, parent_task_id=parent.id)

    # 模拟崩溃：只留事件日志，从回放重建
    view = reduce_events(bus.events, "run_1")
    recovered_view = view.tasks[child.id]
    assert recovered_view.inputs == {"file": "report.csv", "threshold": 3}
    recovered_task = task_from_projection(recovered_view)
    assert recovered_task.inputs == {"file": "report.csv", "threshold": 3}

    # 恢复后重跑：装配上下文仍含输入区块（走 TaskSpecSource → composer）
    req = ContextRequest(
        purpose="act",
        scope=SimpleNamespace(session_id="s1", task_id=recovered_task.id, agent_id="a1"),
        task=SimpleNamespace(
            id=recovered_task.id, title=recovered_task.title, description="",
            user_prompt=recovered_task.user_prompt, user_prompt_in_memory=False,
            process_report=None, inputs=recovered_task.inputs,
        ),
        agent=SimpleNamespace(id="a1"), session=SimpleNamespace(id="s1"),
        template=None, bound_capabilities=[], extra={},
    )
    blocks = [b async for b in TaskSpecSource().fetch(req, None)]
    text = "\n".join(m.content for m in DefaultComposer()._build_actor_messages(blocks, req)
                     if isinstance(m.content, str))
    assert "## Inputs" in text and "report.csv" in text


# ── 3.3 端到端：派发 ack 的 id → observer 以 task_id 发起 reopen ─────────────


@pytest.mark.asyncio
async def test_delegate_ack_id_drives_review_reopen_by_id():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)
    tm._children_of[parent.id] = set()

    res, children = await _delegate_plan_ordered(
        tm, parent, [{"title": "step one"}, {"title": "step two"}],
    )
    assert len(children) == 2
    ack_ids = [c.id for c in children]
    assert ", ".join(ack_ids) in res.content          # ack 的 id 列表就是真实子任务 id

    # 子任务完成（step one FINISHED），observer 用 ack 里的 id 发起 reopen
    c1 = ack_ids[0]
    c1_task = tm.get_task(c1)
    c1_task.status = "FINISHED"
    c1_task.outputs = "done"
    tm._queue.mark_complete(c1)

    parent.id = parent.id
    review_ctx = ControlContext(
        session_id="s1", task_id=parent.id, agent_id="a1", task=parent,
        task_manager=tm, session=None, tool_call_id="tc_2",
    )
    out = report_task_outcome(
        task_status="success", act_recap="reviewed",
        task_reviews=[{"task_id": c1, "review_status": "reopen", "reasoning": "redo step one"}],
        ctx=review_ctx,
    )
    reopen_map = out.metadata.get(ControlMetaKey.REOPEN_TASK_IDS)
    assert reopen_map == {c1: "redo step one"}

    assert await tm.reopen_chain(c1, "redo step one") is True
    assert tm.get_task(c1).status == "PENDING"        # 按 id 命中、链路重排


# ── 4.4 计划断裂：step2 失败 → step3 取消（带原因）、清理步照常 ──────────────


@pytest.mark.asyncio
async def test_plan_break_cancels_successors_and_runs_cleanup():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)

    _, ordered = await _delegate_plan_ordered(tm, parent, [
        {"title": "produce"},
        {"title": "transform"},
        {"title": "final"},
        {"title": "cleanup", "run_if": "any"},
    ])
    produce, transform, final, cleanup = ordered
    # flush 后 push_task 物化的条件（链序 produce→transform→final→cleanup）
    assert final.dep_conditions == {transform.id: "success"}
    assert cleanup.dep_conditions == {final.id: "any"}

    produce.status = "FINISHED"
    await tm.on_task_finished(produce.id, status="FINISHED")
    transform.status = "FAILED"
    await tm.on_task_finished(transform.id, status="FAILED")

    assert final.status == "CANCELED"
    assert final.error_code == "BLOCKED_BY_FAILED_DEP"
    assert final.error and transform.id in final.error
    assert cleanup.status == "PENDING"               # 清理步不受阻
    assert tm.session.status != "CANCELED"            # 会话不被误标
    assert not any(e.task_id == final.id for e in tm._queue.peek_all())


# ── 5.2 行为变更专项回归 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_title_entries_rejected_with_guidance_through_tool():
    """旧 task_reviews 条目经完整工具链被拒：回执引导改用 task_id，合法条目照常。"""
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)
    _, children = await _delegate_plan_ordered(
        tm, parent, [{"title": "same name"}, {"title": "same name"}],
    )
    c1, c2 = children
    for c in (c1, c2):
        c.status = "FINISHED"
        c.outputs = "done"
        tm._queue.mark_complete(c.id)

    out = report_task_outcome(
        task_status="success", act_recap="reviewed",
        task_reviews=[
            {"task_title": "same name", "review_status": "reopen", "reasoning": "old"},
            {"task_id": c1.id, "review_status": "reopen", "reasoning": "redo first"},
            {"task_id": 123, "review_status": "skip", "reasoning": "bad"},
        ],
        ctx=_ctx(tm, parent),
    )
    # 只有按 id 的合法条目进入 reopen 指令；同名另一子不受影响
    reopen_map = out.metadata.get(ControlMetaKey.REOPEN_TASK_IDS)
    assert reopen_map == {c1.id: "redo first"}
    # 回执携带可引导重发的说明
    assert "task_title" in out.content and "task_id" in out.content
    assert c2.status == "FINISHED"


# ── 4.5 阻塞原因的持久化与恢复后可解释 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_blocked_reason_survives_replay_and_reaches_review_face():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = Task(id="P", session_id="s1", status="SUSPENDED", title="parent")
    tm.register_task(parent)
    _, ordered = await _delegate_plan_ordered(tm, parent, [
        {"title": "produce"},
        {"title": "final"},
    ])
    produce, final = ordered

    produce.status = "FAILED"
    await tm.on_task_finished(produce.id, status="FAILED")
    assert final.status == "CANCELED"

    # 模拟重启：回放全部事件（含阻塞取消的 TASK_CANCELED）→ 投影可解释
    view = reduce_events(bus.events, "run_1")
    fv = view.tasks[final.id]
    assert fv.status == "CANCELED"
    assert fv.error_code == "BLOCKED_BY_FAILED_DEP"
    assert fv.blocked_by_task_id == produce.id
    recovered_final = task_from_projection(fv)
    assert recovered_final.error_code == "BLOCKED_BY_FAILED_DEP"

    # 恢复后的 TM：终态与阻塞原因仍可读（父观察面的 note 数据源）
    restored_tm = _tm(_CapturingBus())
    all_tasks = [task_from_projection(v) for v in view.tasks.values()]
    terminal = {tid for tid, v in view.tasks.items()
                if v.status in ("FINISHED", "FAILED", "CANCELED")}
    restored_tm.restore(all_tasks, terminal)
    child = restored_tm.get_task(final.id)
    assert child.error_code == "BLOCKED_BY_FAILED_DEP"
    assert child.error and produce.id in child.error
