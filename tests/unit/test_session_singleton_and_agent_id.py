"""session 的 TaskManager 单例 + agent_id 全局唯一。

两条不变量合起来才保证「同一个 agent_id 同一时刻至多一个 run」：

- 同 agent 串行靠 TaskManager 的 drain（`busy_agents`），但它只看得见**自己**派发的
  run——一个 session 若并存两个 TM，这条保证就破了；
- 运行期按 agent_id 存的状态（`CapabilityCache` 的工具面，run 收尾按 agent_id
  `evict`）要求一个 id 只对应一个 agent、一个 session。

两者任一破了，都会出现「发给模型的工具列表里明明有、gateway 却报 unknown tool」：
另一个同 agent_id 的 run 收尾时把这个 run 的工具面清掉了。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.control.types import AgentView
from ctx_weft.core.models.errors import SessionAlreadyExistsError, UnfinishedTasksError
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.orchestrator.lifecycle.agent_manager import (
    AgentLifecycleManager,
    DuplicateAgentId,
    _AgentRecord,
)
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import (
    _FinishEveryTaskLLM,
    _wait_until,
)
from tests.unit._stub_runner import StubRunner

pytestmark = pytest.mark.asyncio


def _task(tid: str, status: str, **kw) -> Task:
    return Task(id=tid, session_id="s1", status=status, assigned_agent_id="a",
                creator_agent_id="a", settings=NormalTaskSettings(), **kw)


def _echo_runtime(llm=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm or _FinishEveryTaskLLM(), agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _run_first_round(rt) -> tuple[str, str, str]:
    """起一个 session 并等第一轮真正收尾（含 `_fire_session_done` 的 gather）。"""
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="first question", context_limit=100_000,
    ))
    sid, first = handle.session_id, handle.task_id

    async def _done() -> bool:
        t = (await rebuild_view(rt.event_store, sid)).tasks.get(first)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_done)
    await asyncio.sleep(0.3)
    return sid, handle.agent_id, first


# ── TaskManager.requeue_resumable：/resume 在活 owner 上就地重排 ───────────────


async def test_requeue_resumable_picks_exactly_what_restore_would():
    tm = TaskManager("s1", event_bus=InProcessEventBus(), max_concurrent=1)
    for t in (
        _task("interrupted", "INTERRUPTED"),
        _task("blocked", "INTERRUPTED", dag_deps=["interrupted", "done"]),
        _task("running", "ACTIVE"),
        _task("done", "FINISHED"),
        _task("parked", "AWAITING_HUMAN"),          # 仍有未决 HITL
        _task("answered", "AWAITING_HUMAN"),        # HITL 已终局、没人驱动
        _task("parent_live", "SUSPENDED"),
        _task("live_child", "PENDING"),             # 已在队列
        _task("parent_done", "SUSPENDED"),
        _task("done_child", "FINISHED"),
    ):
        tm.register_task(t)
    tm._running_tasks.add("running")
    tm._children_of.update({"parent_live": {"live_child"}, "parent_done": {"done_child"}})
    tm._queue.push(QueueEntry(task_id="live_child", session_id="s1"))

    requeued = tm.requeue_resumable({"parked"})

    assert set(requeued) == {"interrupted", "blocked", "answered", "parent_done"}
    entries = {e.task_id: e for e in tm._queue.peek_all()}
    assert set(entries) == {"live_child", *requeued}
    assert entries["blocked"].blocked_by == {"interrupted"}, "未终态的 dag 依赖仍作阻塞"
    assert tm.get_task("interrupted").status == "PENDING"
    assert tm.get_task("parent_live").status == "SUSPENDED", \
        "挂在活子任务上的父任务要留给 _try_resume_parent 唤醒"
    assert tm.get_task("parked").status == "AWAITING_HUMAN"
    assert tm.requeue_resumable({"parked"}) == [], "重复 /resume 必须幂等：不重复入队"


# ── recover_agent：活 owner 在世就绝不重建 ─────────────────────────────────────


def _live_owner(rt, ran: list[str]) -> TaskManager:
    """经 runtime 真实 wiring 登记的活 owner，带一个 INTERRUPTED 的 task。"""
    rt._agent_lifecycle_manager._agents["agt"] = _AgentRecord(
        session_id="ses_1", tenant_id="default", template_id="tpl",
        parent_agent_id=None, spawn_depth=0, memory_config=None, loop_config=None,
    )
    session = Session(id="ses_1", tenant_id="default", user_prompt="x", status="RUNNING",
                      root_agent_id="agt", token_budget=0)
    tm = TaskManager(session_id="ses_1", event_bus=rt.event_bus, max_concurrent=1)
    tm.set_session(session)

    async def _runner(_sid, tid):
        ran.append(tid)

    tm.set_runner(StubRunner(tm, _runner, session_id="ses_1"))
    tm.register_task(Task(id="tsk_A", session_id="ses_1", status="INTERRUPTED",
                          assigned_agent_id="agt", creator_agent_id="agt",
                          settings=NormalTaskSettings()))
    rt._register_and_drain(session, tm)
    return tm


async def test_resume_on_a_live_owner_requeues_in_place_without_rebuilding(monkeypatch):
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                      agent_provider=InlineAgentTemplateProvider())
    rebuilds: list[int] = []
    import ctx_weft.core.control.reducers as _reducers
    _orig = _reducers.rebuild_view

    async def _spy(*a, **k):
        rebuilds.append(1)
        return await _orig(*a, **k)

    monkeypatch.setattr(_reducers, "rebuild_view", _spy)
    ran: list[str] = []
    tm = _live_owner(rt, ran)

    await rt.recover_agent("agt")                 # /resume：不带 resumed_task_id
    for _ in range(10):
        await asyncio.sleep(0)
    await rt.recover_agent("agt")                 # 再按一次：幂等
    for _ in range(10):
        await asyncio.sleep(0)

    assert rebuilds == [], "活 owner 在世时不得走事件日志重建"
    assert rt._task_managers["ses_1"] is tm, "活 owner 不得被顶替"
    assert ran == ["tsk_A"], "INTERRUPTED 的 task 应在活 owner 上就地重跑，且只跑一次"


async def test_cold_answer_for_a_task_the_live_owner_does_not_hold_is_refused():
    """活 owner 持有本 session 的全部 task；不持有 = 不变量破了——抛，而不是另建一个
    TM 去兜（另建正是单例要消灭的东西）。"""
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                      agent_provider=InlineAgentTemplateProvider())
    tm = _live_owner(rt, [])

    with pytest.raises(RuntimeError, match="does not own task"):
        await rt.recover_agent("agt", resumed_task_id="tsk_unknown")
    assert rt._task_managers["ses_1"] is tm


# ── start_session ─────────────────────────────────────────────────────────────


async def test_start_session_refuses_to_create_a_session_id_that_already_exists():
    rt = _echo_runtime()
    sid, _aid, _first = await _run_first_round(rt)
    owner = rt._task_managers[sid]

    with pytest.raises(SessionAlreadyExistsError):
        await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="again", context_limit=100_000,
            session_id=sid,
        ))

    assert rt._task_managers[sid] is owner
    created = [e for e in await rt.event_store.read_by_session(sid)
               if e.type == EventType.SESSION_CREATED]
    assert len(created) == 1, "入口即拒：不得落第二条 SESSION_CREATED"


async def test_start_session_resume_pushes_into_the_live_task_manager():
    rt = _echo_runtime()
    sid, aid, first = await _run_first_round(rt)
    owner = rt._task_managers[sid]

    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="second question", context_limit=100_000,
        session_id=sid, resume=True,
    ))

    assert rt._task_managers[sid] is owner, "续聊复用活 owner，不另建 TM"
    assert handle.agent_id == aid and handle.task_id != first
    assert owner.get_task(handle.task_id) is not None

    async def _second_done() -> bool:
        t = (await rebuild_view(rt.event_store, sid)).tasks.get(handle.task_id)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_second_done)


async def test_start_session_resume_sees_unfinished_work_that_is_only_in_memory():
    """未提交窗口里的 task 还没落盘，事件视图看不见——活 owner 的内存要一并参与判定。"""
    rt = _echo_runtime()
    sid, aid, _first = await _run_first_round(rt)
    rt._task_managers[sid].register_task(Task(
        id="tsk_uncommitted", session_id=sid, status="PENDING", assigned_agent_id=aid,
        creator_agent_id=aid, settings=NormalTaskSettings()))

    with pytest.raises(UnfinishedTasksError):
        await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="second question",
            context_limit=100_000, session_id=sid, resume=True,
        ))


# ── run_single_task ───────────────────────────────────────────────────────────


async def test_run_single_task_refuses_a_session_that_already_has_an_owner():
    rt = _echo_runtime(MockLLMAdapter(responses=[MockResponse(text="hi")]))
    rt._task_managers["ses_owned"] = TaskManager("ses_owned", event_bus=rt.event_bus)

    with pytest.raises(SessionAlreadyExistsError):
        await rt.run_single_task(
            session_id="ses_owned", template_id="agent:tpl_echo", user_prompt="hi")


async def test_run_single_task_owns_the_session_only_while_it_runs(monkeypatch):
    from ctx_weft.core.runtime import CtxWeftRuntime

    rt = _echo_runtime(MockLLMAdapter(responses=[MockResponse(text="hi")]))
    seen: list[bool] = []
    orig = CtxWeftRuntime._execute_task

    async def _spy(self, *, session, task_manager, **kw):
        seen.append(self._task_managers.get(session.id) is task_manager)
        return await orig(self, session=session, task_manager=task_manager, **kw)

    monkeypatch.setattr(CtxWeftRuntime, "_execute_task", _spy)

    handle, _state = await rt.run_single_task(template_id="agent:tpl_echo", user_prompt="hi")

    assert seen == [True], "跑的这段时间里它的 TM 就是这个 session 的 owner"
    assert handle.session_id not in rt._task_managers, \
        "跑完摘掉：没有 runner 的 TM 留在册里会让之后的续跑拿到一个派发不了的 TM"


# ── agent_id 全局唯一 ─────────────────────────────────────────────────────────


def _alm() -> AgentLifecycleManager:
    class _Bus:
        async def emit(self, ev) -> None:
            pass

        def subscribe(self, *_a, **_k) -> None:
            pass

    return AgentLifecycleManager(template_lookup=None, event_bus=_Bus(),
                                 model_resolver=lambda a, m: None)


async def test_load_refuses_an_agent_id_owned_by_another_session():
    reg = _alm()
    await reg.load({"a1": AgentView(id="a1")}, session_id="s1", tenant_id="default",
                   fallback_template_id="tpl")

    with pytest.raises(DuplicateAgentId):
        await reg.load({"a2": AgentView(id="a2"), "a1": AgentView(id="a1")},
                       session_id="s2", tenant_id="default", fallback_template_id="tpl")

    assert reg.record_of("a1").session_id == "s1", "冲突时不得把 agent 搬进另一个 session"
    assert not reg.has("a2"), "整批拒绝：检查先于任何改动"


async def test_load_reloading_the_same_session_is_still_idempotent():
    reg = _alm()
    views = {"a1": AgentView(id="a1")}
    await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")
    await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")
    assert reg.record_of("a1").session_id == "s1"


async def test_load_replaces_a_fallback_placeholder_from_another_session():
    """`materialize` 撞上还没装填的 id 会就地补一条占位；真记录随后装填时要能覆盖它
    ——占位不代表已确认的归属，不参与唯一性判定。

    占位记录的 session 按**调用方给的**归属（2026-09-19 起 `_register_fallback` 必传，不
    再猜「最近一次 register_session 的会话」）。这里刻意让占位落在 `s_placeholder`、真记录
    来自 `s_real`：两者不同，才测得到「覆盖」而非「恰好同名」。若占位参与唯一性判定，
    下面那次 `load` 会抛 `DuplicateAgentId` 而不是修正它。"""
    reg = _alm()
    reg.register_session("s_placeholder", tenant_id="default", fallback_template_id="tpl")
    reg._register_fallback("a1", session_id="s_placeholder", tenant_id="default")
    assert reg.record_of("a1").session_id == "s_placeholder"

    await reg.load({"a1": AgentView(id="a1")}, session_id="s_real", tenant_id="default",
                   fallback_template_id="tpl")

    assert reg.record_of("a1").session_id == "s_real"
