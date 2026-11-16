"""pause_session 弃子语义：只留 root agent 当前那一轮（pause→park），其余 cancel/放弃。"""

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.runner import AgentBinding
from ctx_weft.core.runtime import _SessionTaskRunner
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime



class _StubRunner:
    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _rt():
    return make_runtime(llm=MockLLMAdapter(responses=[]),
                          agent_provider=InlineAgentTemplateProvider())


def _task(tid: str, status: str = "PENDING") -> Task:
    return Task(
        id=tid, session_id="s1", status=status, tenant_id="default",
        assigned_agent_id="", creator_agent_id="agr",
        title=tid, description="", user_prompt="x", created_at=now_utc(),
    )


def _wire(rt) -> tuple[TaskManager, Session]:
    tm = TaskManager(session_id="s1")
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id="agr")
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    rt._task_managers["s1"] = tm
    return tm, sess


@pytest.mark.asyncio
async def test_pause_session_partitions_root_run_vs_rest():
    rt = _rt()
    tm, sess = _wire(rt)
    # 在跑两轮：root agent 的一轮 + 子 agent 的一轮（真实执行 agent 以 _running_agents 为准）
    for tid, agent in (("t_root", "agr"), ("t_sub", "ag_sub")):
        tm.register_task(_task(tid, status="ACTIVE"))
        tm._running_tasks.add(tid)
        tm._running_agents[tid] = agent
    root_tokens = rt._register_run_tokens("s1", "t_root")
    sub_tokens = rt._register_run_tokens("s1", "t_sub")
    await tm.push_task(_task("t_q"))     # 排队中一个

    await initialize_all(rt)
    assert await rt.pause_session("s1") is True
    # root agent 那一轮：pause → park；不 cancel
    assert root_tokens.pause.is_paused and not root_tokens.cancel.is_cancelled
    # 其余在途 run：cancel → 终态；不 pause
    assert sub_tokens.cancel.is_cancelled and not sub_tokens.pause.is_paused
    # 排队任务被放弃
    assert tm.get_task("t_q").status == "CANCELED"
    # 闩锁置位（等 root park 后 _on_idle 清除）、session 状态未被弃子污染
    assert "s1" in rt._pausing
    assert not hasattr(sess, "status")


@pytest.mark.asyncio
async def test_pause_session_inflight_root_run_claims_resume_point():
    # 在途 root run 被 pause → 名额已认领：闩锁窗口内此后派发的 root scope 任务（如
    # 子任务死光被 _try_resume_parent 重排的 SUSPENDED root 任务）born-cancel，不出第二气泡。
    rt = _rt()
    tm, _ = _wire(rt)
    tm.register_task(_task("t_root", status="ACTIVE"))
    tm._running_tasks.add("t_root")
    tm._running_agents["t_root"] = "agr"
    root_tokens = rt._register_run_tokens("s1", "t_root")

    await initialize_all(rt)
    assert await rt.pause_session("s1") is True
    assert root_tokens.pause.is_paused is True           # 在途那一轮 = 唯一续跑点
    late = rt._register_run_tokens("s1", "t_requeued", root_run=True)
    assert late.cancel.is_cancelled is True
    assert late.pause.is_paused is False


@pytest.mark.asyncio
async def test_pause_session_unknown_task_runs_are_cancelled():
    # 旧 TM inflight：registry 在册但当前 TM 的 _running_agents 不认识 → 按"非 root 那一轮"cancel
    rt = _rt()
    tm, _ = _wire(rt)
    tm.register_task(_task("t_keepalive", status="ACTIVE"))
    tm._running_tasks.add("t_keepalive")
    tm._running_agents["t_keepalive"] = "agr"
    rt._register_run_tokens("s1", "t_keepalive")
    stale = rt._register_run_tokens("s1", "t_stale")   # 旧代 run，当前 TM 不认识
    await initialize_all(rt)
    assert await rt.pause_session("s1") is True
    assert stale.cancel.is_cancelled is True


@pytest.mark.asyncio
async def test_pause_session_idle_session_is_noop_false():
    rt = _rt()
    _wire(rt)   # TM 存在但无在跑、无排队 → is_done
    await initialize_all(rt)
    assert await rt.pause_session("s1") is False
    assert "s1" not in rt._pausing


# ── I-1：assemble 窗口内到达的信号，在 execute 派发点补投 ────────────────────────


class _StubExecTM:
    def is_task_tree_canceled(self, task_id):
        return False

    """最小 TaskManager 替身：只提供 execute 补偿逻辑要用的 get_task / is_cancelled。"""

    def __init__(self, *, cancelled: bool = False):
        self._cancelled = cancelled
        self._task = _task("t1", status="ACTIVE")

    def get_task(self, tid):
        return self._task

    def is_cancelled(self):
        return self._cancelled


def _exec_runner(rt, tm, *, root_agent_id="agr") -> _SessionTaskRunner:
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id=root_agent_id)
    return _SessionTaskRunner(
        runtime=rt, session=sess,
        lm=None, memory=None,
        task_manager=tm, handle=None,
    )


async def _run_execute(rt, runner, *, agent_id):
    """经真实 execute 路径跑一次派发；拦下 _execute_task（loop）只记录收到的令牌。"""
    captured: dict = {}

    async def _fake_exec(**kw):
        captured["cancel"] = kw.get("cancel_token")
        captured["pause"] = kw.get("pause_token")
        return (None, None)

    rt._execute_task = _fake_exec
    await runner.execute(AgentBinding(agent_id=agent_id), "t1")
    return captured


@pytest.mark.asyncio
async def test_execute_born_cancel_when_tm_cancelled():
    # TM 已整体取消（cancel_all 置 _cancelled）→ 之后经 execute 派发的 run 出生即取消
    rt = _rt()
    runner = _exec_runner(rt, _StubExecTM(cancelled=True))
    cap = await _run_execute(rt, runner, agent_id="agr")
    assert cap["cancel"].is_cancelled is True


@pytest.mark.asyncio
async def test_execute_born_cancel_for_non_root_run_during_pause():
    # _pausing 中、派发 agent 非 root → 弃子窗口内的迟到 run born-cancel（不额外 park 第二气泡）。
    # M-1：且**不得** born-pause——act 检查点 pause 先于 cancel，双信号会让多级委派中被
    # 重排的中间 agent 父任务 park 出气泡、气泡归属中间 agent 而非 root。
    rt = _rt()
    rt._pausing.add("s1")
    runner = _exec_runner(rt, _StubExecTM(), root_agent_id="agr")
    cap = await _run_execute(rt, runner, agent_id="ag_sub")
    assert cap["cancel"].is_cancelled is True
    assert cap["pause"].is_paused is False


@pytest.mark.asyncio
async def test_execute_root_run_stays_born_paused_during_pause():
    # _pausing 中、派发 agent == root → 保持 born-paused（补偿逻辑不得把它 cancel）
    rt = _rt()
    rt._pausing.add("s1")
    runner = _exec_runner(rt, _StubExecTM(), root_agent_id="agr")
    cap = await _run_execute(rt, runner, agent_id="agr")
    assert cap["cancel"].is_cancelled is False
    assert cap["pause"].is_paused is True


# ── I-2（runtime 级）：pause 恰逢 root 任务已入队未派发 → 保留、不随全清取消 ───────────


@pytest.mark.asyncio
async def test_pause_session_keeps_queued_root_task():
    rt = _rt()
    tm = TaskManager(session_id="s1", max_concurrent=1)
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id="agr")
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    rt._task_managers["s1"] = tm
    # 子 agent run 在跑，占满单并发槽（drain 补 no-op）
    tm.register_task(_task("t_sub", status="ACTIVE"))
    tm._running_tasks.add("t_sub")
    tm._running_agents["t_sub"] = "ag_sub"
    sub_tokens = rt._register_run_tokens("s1", "t_sub")
    # root agent 任务仅入队、未派发
    root_task = _task("t_root")
    root_task.assigned_agent_id = "agr"
    await tm.push_task(root_task)

    await initialize_all(rt)
    assert await rt.pause_session("s1") is True
    # root 未被弃子取消、仍在队列待派发
    assert tm.get_task("t_root").status != "CANCELED"
    assert any(e.task_id == "t_root" for e in tm._queue.peek_all())
    # 子 run 被 cancel
    assert sub_tokens.cancel.is_cancelled is True


@pytest.mark.asyncio
async def test_pause_task_targets_single_run():
    # 定向暂停单个在途 task，其他 task 不受影响；task 不在跑→False
    # `_pause_task` 是内部原语（`pause_agent`/`pause_session` 仅有的两个调用方），
    # 这里直接调用它测的就是这个原语本身。
    rt = _rt()
    a = rt._register_run_tokens("s1", "ta")
    b = rt._register_run_tokens("s1", "tb")
    assert rt._pause_task("s1", "ta") is True
    assert a.pause.is_paused and not b.pause.is_paused
    assert rt._pause_task("s1", "nope") is False    # 不在跑 → False
