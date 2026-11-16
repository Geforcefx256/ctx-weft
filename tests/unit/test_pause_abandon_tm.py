"""TaskManager pause 弃子控制面：abandon_pending / set_pause_abandon 守卫 / staged 丢弃。"""

import pytest

from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc



class _StubRunner:
    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _task(tid: str, status: str = "PENDING") -> Task:
    return Task(
        id=tid, session_id="s1", status=status, tenant_id="default",
        assigned_agent_id="", creator_agent_id="agr",
        title=tid, description="", user_prompt="x", created_at=now_utc(),
    )


def _tm_with_session() -> tuple[TaskManager, Session]:
    tm = TaskManager(session_id="s1")
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id="agr")
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    return tm, sess


@pytest.mark.asyncio
async def test_abandon_pending_cancels_queue_without_touching_session():
    tm, sess = _tm_with_session()
    await tm.push_task(_task("t1"))
    await tm.push_task(_task("t2"))
    dropped = await tm.abandon_pending(reason="pause_abandon")
    assert set(dropped) == {"t1", "t2"}
    assert tm.get_task("t1").status == "CANCELED"
    assert tm.get_task("t2").status == "CANCELED"
    assert not hasattr(sess, "status")      # 弃子不动 session 状态
    assert tm.is_done() is True          # 队列已空、无在跑


@pytest.mark.asyncio
async def test_abandon_pending_keeps_root_agent_queued_task():
    # keep_agent：root agent scope 的排队条目保留（不随全清取消），供 pause 补 drain 派发。
    tm, sess = _tm_with_session()   # root_agent_id="agr"
    root_task = _task("t_root")
    root_task.assigned_agent_id = "agr"          # 在 root agent scope
    other = _task("t_other")
    other.assigned_agent_id = "ag_sub"           # 非 root
    await tm.push_task(root_task)
    await tm.push_task(other)
    cancelled = await tm.abandon_pending(keep_agent="agr")
    assert cancelled == ["t_other"]
    assert any(e.task_id == "t_root" for e in tm._queue.peek_all())
    assert tm.get_task("t_root").status != "CANCELED"
    assert tm.get_task("t_other").status == "CANCELED"


@pytest.mark.asyncio
async def test_abandon_pending_resumes_suspended_parent_of_queued_children():
    # 队列弃子不经 on_task_finished，不会自动触发父任务重排：若 SUSPENDED 父任务的
    # 子任务在暂停瞬间**全部**还在排队（无一在途），无人调用 _try_resume_parent →
    # 父任务永不重排、会话滞留 RUNNING 且无气泡。abandon_pending 须补触发重排检查。
    tm = TaskManager(session_id="s1", max_concurrent=0)   # drain 空转，便于观察队列
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id="agr")
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    tm.register_task(_task("tp", status="SUSPENDED"))
    await tm.push_task(_task("tc1"), parent_task_id="tp")
    await tm.push_task(_task("tc2"), parent_task_id="tp")

    cancelled = await tm.abandon_pending()
    assert set(cancelled) == {"tc1", "tc2"}
    # 父任务被重排为唯一续跑点候选（派发后按出生信号 park/级联）。PENDING 不是 ACTIVE
    # （Task 10 / D5）：drain 空转（max_concurrent=0），派发前 ACTIVE 会是抢跑。
    assert tm.get_task("tp").status == "PENDING"
    assert any(e.task_id == "tp" for e in tm._queue.peek_all())


@pytest.mark.asyncio
async def test_abandon_pending_no_resume_while_sibling_still_running():
    # 兄弟仍在途：弃子只清排队的那部分，父任务重排留给在途兄弟的 on_task_finished 接力。
    tm = TaskManager(session_id="s1", max_concurrent=0)
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id="agr")
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    tm.register_task(_task("tp", status="SUSPENDED"))
    running = _task("tc_run", status="ACTIVE")
    tm.register_task(running)
    tm._parent_map["tc_run"] = "tp"
    tm._children_of.setdefault("tp", set()).add("tc_run")
    tm._running_tasks.add("tc_run")
    tm._running_agents["tc_run"] = "ag_sub"
    await tm.push_task(_task("tc_q"), parent_task_id="tp")

    cancelled = await tm.abandon_pending()
    assert cancelled == ["tc_q"]
    assert tm.get_task("tp").status == "SUSPENDED"   # 在途兄弟未终态 → 不得提前重排


@pytest.mark.asyncio
async def test_pause_abandon_guard_keeps_session_status_on_cancel():
    tm, sess = _tm_with_session()
    tm.register_task(_task("t1", status="ACTIVE"))
    tm.set_pause_abandon(True)
    await tm.on_task_finished("t1", status="CANCELED")
    assert tm.get_task("t1").status == "CANCELED"
    assert not hasattr(sess, "status")     # pause 弃子 ≠ 用户取消


@pytest.mark.asyncio
async def test_cancel_without_pause_abandon_still_cancels_session():
    tm, sess = _tm_with_session()
    tm.register_task(_task("t1", status="ACTIVE"))
    await tm.on_task_finished("t1", status="CANCELED")
    assert not hasattr(sess, "status")     # 既有语义不回归


@pytest.mark.asyncio
async def test_flush_staged_dropped_under_pause_abandon():
    tm, _ = _tm_with_session()
    tm.register_task(_task("tp", status="ACTIVE"))
    tm.stage_task(_task("tc"), parent_task_id="tp")
    tm.set_pause_abandon(True)
    await tm._flush_staged("tp")
    assert tm.get_task("tc") is None     # 未入队、未登记（push 时才发 TASK_CREATED，无投影幽灵）
