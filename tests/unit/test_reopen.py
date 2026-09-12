"""Observer review 权限范围 + 级联 reopen 的行为测试。

覆盖：
1. _collect_reviews 只接受当前 task 自己派生的子任务；前序 / 其它任务越权被拒并反馈。
2. reopen_chain：reopen 一个 plan 步骤会强制重开其同 plan 后续，重建 blocked_by 链。
3. head 用 "## Revision required"，后续用 "## Upstream task revised" 引导。
4. 当前任务（parent）不会被卷入对其子任务的级联。
5. reopen_chain 对非 FINISHED head 是 no-op。
"""

from __future__ import annotations

from datetime import datetime, timezone

from ctx_weft.core.capabilities.control_tools import ControlContext, _collect_reviews
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.task import Task


def _task(tid: str, title: str, status: str = "FINISHED",
          tracking: list[str] | None = None, created: int = 0) -> Task:
    return Task(
        id=tid, session_id="s1", status=status, title=title,
        tracking_task_ids=tracking or [],
        created_at=datetime(2026, 1, 1, 0, 0, created, tzinfo=timezone.utc),
    )


# ── 1. _collect_reviews 权限范围 ─────────────────────────────────────────────────


def test_collect_reviews_only_accepts_own_children() -> None:
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Build C1")
    c2 = _task("C2", "Build C2")
    pred = _task("P0", "Predecessor")  # 同 session 但非本 task 子任务
    for t in (parent, c1, c2, pred):
        tm.register_task(t)
    tm._children_of["T"] = {"C1", "C2"}

    ctx = ControlContext(session_id="s1", task_id="T", agent_id="a",
                         task=parent, task_manager=tm, session=None)
    reviews = [
        {"task_id": "C1", "review_status": "reopen", "reasoning": "fix it"},
        {"task_id": "C2", "review_status": "confirmed", "reasoning": "ok"},
        {"task_id": "P0", "review_status": "reopen", "reasoning": "nope"},
        {"task_id": "Ghost", "review_status": "reopen", "reasoning": "x"},
    ]
    reopen, summary = _collect_reviews(reviews, ctx)

    assert reopen == {"C1": "fix it"}             # 只有子任务被收集
    assert "out of scope" in summary or "invalid" in summary
    assert "'P0'" in summary and "'Ghost'" in summary
    assert "confirmed 'Build C2' (C2)" in summary


def test_collect_reviews_same_named_children_resolve_by_id() -> None:
    """spec: task-handoff——同名双子任务按 id 定向生效，对象不随登记顺序漂移。"""
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Same Name")
    c2 = _task("C2", "Same Name")  # 同名、后注册
    for t in (parent, c1, c2):
        tm.register_task(t)
    tm._children_of["T"] = {"C1", "C2"}

    ctx = ControlContext(session_id="s1", task_id="T", agent_id="a",
                         task=parent, task_manager=tm, session=None)
    reopen, summary = _collect_reviews(
        [{"task_id": "C1", "review_status": "reopen", "reasoning": "redo first"}], ctx,
    )
    assert reopen == {"C1": "redo first"}
    assert c2.status == "FINISHED"


def test_collect_reviews_survives_child_rename() -> None:
    """spec: task-handoff——审核前子任务改名，按既得 id 操作仍命中。"""
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Build C1")
    for t in (parent, c1):
        tm.register_task(t)
    tm._children_of["T"] = {"C1"}
    c1.title = "Renamed C1"  # 派发后被改名

    ctx = ControlContext(session_id="s1", task_id="T", agent_id="a",
                         task=parent, task_manager=tm, session=None)
    reopen, _ = _collect_reviews(
        [{"task_id": "C1", "review_status": "reopen", "reasoning": "r"}], ctx,
    )
    assert reopen == {"C1": "r"}


def test_collect_reviews_rejects_legacy_title_entries_others_proceed() -> None:
    """spec: task-handoff——旧 task_title 条目被条目级校验拒绝，其余合法条目照常生效。"""
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Build C1")
    c2 = _task("C2", "Build C2")
    for t in (parent, c1, c2):
        tm.register_task(t)
    tm._children_of["T"] = {"C1", "C2"}

    ctx = ControlContext(session_id="s1", task_id="T", agent_id="a",
                         task=parent, task_manager=tm, session=None)
    reviews = [
        {"task_title": "Build C1", "review_status": "reopen", "reasoning": "old style"},
        {"task_id": "C2", "review_status": "confirmed", "reasoning": "ok"},
        {"review_status": "confirmed", "reasoning": "no id"},
        {"task_id": 42, "review_status": "skip", "reasoning": "bad id"},
    ]
    reopen, summary = _collect_reviews(reviews, ctx)

    assert reopen == {}                      # 旧式/缺 id/坏 id 全拒
    assert "confirmed 'Build C2' (C2)" in summary  # 合法条目照常生效
    assert "task_title" in summary           # 回执说明指出旧字段不再接受
    assert "'C1'" not in summary.split("confirmed")[0].replace("Build C1", "")  # C1 未被旧式条目操作


def test_collect_reviews_skips_reopen_of_unfinished_child() -> None:
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Build C1", status="PENDING")
    for t in (parent, c1):
        tm.register_task(t)
    tm._children_of["T"] = {"C1"}

    ctx = ControlContext(session_id="s1", task_id="T", agent_id="a",
                         task=parent, task_manager=tm, session=None)
    reopen, summary = _collect_reviews(
        [{"task_id": "C1", "review_status": "reopen", "reasoning": "r"}], ctx,
    )
    assert reopen == {}
    assert "already active 'Build C1' (C1)" in summary


# ── 2. reopen_chain 级联 ─────────────────────────────────────────────────────────


async def test_reopen_chain_cascades_plan_successors() -> None:
    tm = TaskManager(session_id="s1")
    c1 = _task("C1", "Build C1", tracking=[], created=1)
    c2 = _task("C2", "Build C2", tracking=["C1"], created=2)
    c3 = _task("C3", "Build C3", tracking=["C1", "C2"], created=3)
    c1.outputs, c2.outputs, c3.outputs = "out1", "out2", "out3"
    for t in (c1, c2, c3):
        tm.register_task(t)

    assert await tm.reopen_chain("C1", "C1 is wrong") is True
    assert c1.status == c2.status == c3.status == "PENDING"

    # head：直接修订指令，无 upstream 段
    assert "## Revision required\nC1 is wrong" in c1.user_prompt
    assert "Upstream task revised" not in c1.user_prompt

    # 后续：upstream 段引用 head 的标题与理由
    for cx in (c2, c3):
        assert "## Upstream task revised" in cx.user_prompt
        assert "Build C1" in cx.user_prompt
        assert "C1 is wrong" in cx.user_prompt

    # 重建 blocked_by 链（沿 created_at 顺序，指向链内前驱）
    assert c1.dag_deps == []
    assert c2.dag_deps == ["C1"]
    assert c3.dag_deps == ["C2"]


async def test_reopen_chain_noop_when_head_not_finished() -> None:
    tm = TaskManager(session_id="s1")
    c1 = _task("C1", "Build C1", status="ACTIVE")
    tm.register_task(c1)
    assert await tm.reopen_chain("C1", "x") is False
    assert c1.status == "ACTIVE"


async def test_parent_not_dragged_into_child_cascade() -> None:
    tm = TaskManager(session_id="s1")
    parent = _task("T", "Parent", status="SUSPENDED")
    c1 = _task("C1", "Build C1", tracking=[], created=1)
    c2 = _task("C2", "Build C2", tracking=["C1"], created=2)
    for t in (parent, c1, c2):
        tm.register_task(t)

    await tm.reopen_chain("C1", "fix")
    assert parent.status == "SUSPENDED"   # parent 不是其子任务的后续 → 不被重开
    assert c1.status == "PENDING"
    assert c2.status == "PENDING"
