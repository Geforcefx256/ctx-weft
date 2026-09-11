"""上一轮迟到的收尾不得冲掉下一轮；session 的 TaskManager 是单例。

复现的 bug：后台 observe 尚未跑完时用户开启下一轮对话 → 上一轮的 `_fire_session_done`
迟发 `SessionFinished` + 触发 `_release_session`，冲掉新一轮已进入的 HITL 挂起态，
任务卡在 ACTIVE、会话显示已结束（详见 session ses_01KWGR112XHFYQYR3HES8D4MYK）。

当时的形状是「下一轮新建一个 TM 顶替上一轮的」。现在 session 的 TM 是单例
（`CtxWeftRuntime._bind_task_manager`）：下一轮复用同一个 TM，第二个 TM 登记不上；
旧 TM 变成「非 owner」只剩一种途径——被逐出（forget/purge）之后又为这个 session
建了新 TM。两种形状下迟到的收尾都必须 no-op，判定都放在 gather **之后**：
下一轮/新 TM 可能恰好在等待后台任务期间到来。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.providers.llm.mock import MockLLMAdapter
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit._stub_runner import StubRunner

pytestmark = pytest.mark.asyncio


async def _noop_runner(_sid, _tid):
    return None


def _running_tm(rt: CtxWeftRuntime, session: Session, task_id: str) -> TaskManager:
    """构造一个「有一个在跑任务」的 TaskManager，并经 runtime 真实 wiring 注册。"""
    tm = TaskManager(session_id=session.id, event_bus=rt._event_bus, max_concurrent=1)
    tm.set_runner(StubRunner(tm, _noop_runner))
    tm.set_session(session)
    tm.register_task(Task(id=task_id, session_id=session.id, status="ACTIVE",
                          assigned_agent_id="a", creator_agent_id="a",
                          settings=NormalTaskSettings()))
    tm._running_tasks.add(task_id)
    rt._register_and_drain(session, tm)
    return tm


def _tm(bus: InProcessEventBus) -> TaskManager:
    tm = TaskManager(session_id="s1", event_bus=bus, max_concurrent=1)
    tm.set_session(Session(id="s1", tenant_id="default", user_prompt="x",
                           status="SUCCEEDED", token_budget=0))
    return tm


async def test_superseded_tm_skips_finish_when_replaced_during_gather() -> None:
    """顶替发生在 `_fire_session_done` 的 gather 期间：旧 TM 恢复后必须发现自己已非
    owner，no-op（不触发 `on_session_done` → 不会误 `_release_session` 掉新 TM）。

    2026-09-04（Task 12）起不再额外断言「不报队列状态」——`announce_queue_state`
    已停发（events-v2 §5），`done_called` 才是这条不变量唯一还在的观测点。
    """
    bus = InProcessEventBus()
    tm = _tm(bus)
    done_called: list = []

    async def _on_done():
        done_called.append(True)

    current = {"v": True}
    tm.set_hooks(TaskManagerHooks(
        on_session_done=_on_done, is_current=lambda: current["v"]))

    # 后台任务：卡住 gather 直到我们放行（模拟慢的 background observe）
    release = asyncio.Event()

    async def _blocker():
        await release.wait()

    bg = asyncio.create_task(_blocker())
    tm.track_background(bg)

    fire = asyncio.create_task(tm._fire_session_done())
    await asyncio.sleep(0)  # 让 _fire_session_done 进入 gather 阻塞
    await asyncio.sleep(0)

    # 等待期间被新一轮 TaskManager 顶替
    current["v"] = False
    release.set()
    await fire

    assert done_called == [], "被顶替的旧 TM 不应触发终结回调（_release_session）"


async def test_current_tm_still_fires_session_finished() -> None:
    """对照：current TM（未被顶替）的 `_fire_session_done` 照常触发终结回调。

    Task 6 起：TM 曾报 TaskQueueDrained（SessionFinished 由 SM 据它发）。2026-09-04
    （Task 12，events-v2 §5）起 `announce_queue_state`/`TaskQueueDrained` 已停发——
    它唯一的消费者（会话状态机）早已随 SessionRegistry 降格退役——`on_session_done`
    回调是否触发因此是这条不变量唯一还在的观测点。
    """
    bus = InProcessEventBus()
    tm = _tm(bus)
    done_called: list = []

    async def _on_done():
        done_called.append(True)

    tm.set_hooks(TaskManagerHooks(on_session_done=_on_done, is_current=lambda: True))

    await tm._fire_session_done()

    assert done_called == [True]


# 2026-09-04（Task 12，events-v2 §5）删除说明：这里原有两条测试——
# `test_superseded_tm_park_does_not_announce_queue_state` 与
# `test_superseded_tm_crash_suspend_does_not_announce_queue_state`——分别钉着
# `announce_queue_state` 内部的 is_current 守卫在 idle 路径（`_fire_session_idle`）
# 与崩溃挂起路径（`_suspend_task_interrupted`/`_handle_task_failure`）上不让被顶替的
# 旧 TM 补发一句会话级 TaskQueueBlocked/TaskQueueInterrupted。`announce_queue_state`
# 本身已随本任务删除，这两条守卫的**对象**不复存在——不是「弱化验证强度」，是
# 「被验证的行为已经不存在」：
#   - `_fire_session_idle`（删除 announce_queue_state 调用后）不再检查 is_current，
#     旧 TM 与 current TM 现在行为完全一致（这一点在删除前也成立——is_current 守卫
#     原本只长在 announce_queue_state 内部，只挡事件发射，从不挡 `_fire_session_idle`
#     里紧随其后的 `on_session_idle` 回调调用）；
#   - `_suspend_task_interrupted` 从来没有自己的 is_current 守卫，TASK_INTERRUPTED
#     （task 级事实）无论是否被顶替都无条件发出，这一点也不因本任务而改变。
# idle 路径真正生产环境下的顶替防护活在 `runtime.py::_on_idle` 闭包自己的
# compare-and-check（`self._task_managers.get(session.id) is task_manager`），
# 与本文件测的 TaskManager 层 is_current 守卫是两回事，不受本次改动影响、也不需要
# 在这里补一条新测试替代——它已经是一段独立、稳定的既有代码路径。
# done 路径的孪生守卫仍然真实存在（`_fire_session_done` 里的 is_current 检查），
# 由 `test_superseded_tm_skips_finish_when_replaced_during_gather` /
# `test_current_tm_still_fires_session_finished` 继续覆盖。


async def test_register_and_drain_refuses_a_second_tm() -> None:
    """单例：session 已有活 owner 时，第二个 TM 登记不上——不覆盖、不顶替。

    被覆盖的旧 TM 已派发的 run 会照样跑完，而新 TM 的同 agent 串行判定看不见它们，
    同一个 agent 上就会并发出两个 run。"""
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                   status="RUNNING", token_budget=0)

    tm_old = TaskManager(session_id="s1", event_bus=rt._event_bus, max_concurrent=1)
    tm_old.set_runner(StubRunner(tm_old, _noop_runner))
    tm_new = TaskManager(session_id="s1", event_bus=rt._event_bus, max_concurrent=1)
    tm_new.set_runner(StubRunner(tm_new, _noop_runner))

    rt._register_and_drain(sess, tm_old)
    with pytest.raises(RuntimeError, match="already has a live TaskManager"):
        rt._register_and_drain(sess, tm_new)

    assert rt._task_managers["s1"] is tm_old
    assert tm_old.is_alive() is True
    assert tm_new._hooks.is_current is None, "被拒的 TM 不应被接线"
    # 同一个 TM 重入是幂等的（复用活 owner 的几条路径都靠这一点）。
    rt._register_and_drain(sess, tm_old)
    assert rt._task_managers["s1"] is tm_old


async def test_evicted_tm_is_no_longer_current() -> None:
    """活 owner 不会被顶替；被逐出之后，为同一 session 新建的 TM 才能登记，旧的随之
    不再是 owner（它迟到的收尾因此变 no-op，见下面两条端到端用例）。"""
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                   status="RUNNING", token_budget=0)

    tm_old = TaskManager(session_id="s1", event_bus=rt._event_bus, max_concurrent=1)
    tm_old.set_runner(StubRunner(tm_old, _noop_runner))
    tm_new = TaskManager(session_id="s1", event_bus=rt._event_bus, max_concurrent=1)
    tm_new.set_runner(StubRunner(tm_new, _noop_runner))

    rt._register_and_drain(sess, tm_old)
    rt._evict_session_memory("s1")
    rt._register_and_drain(sess, tm_new)

    assert tm_old._hooks.is_current is not None and tm_old._hooks.is_current() is False
    assert tm_new._hooks.is_current is not None and tm_new._hooks.is_current() is True


async def test_superseded_tm_drain_does_not_dispatch() -> None:
    """被顶替的 TM（_is_current()==False）的 drain 不派发任何任务——防重叠 resume 下两套 drain。"""
    tm = _tm(InProcessEventBus())
    started: list = []

    async def runner(_sid, tid):
        started.append(tid)

    tm.set_runner(StubRunner(tm, runner))
    tm.set_hooks(TaskManagerHooks(is_current=lambda: False))  # 已被顶替
    await tm.push_task(Task(id="A", session_id="s1", status="PENDING",
                            assigned_agent_id="a", creator_agent_id="a",
                            settings=NormalTaskSettings()))
    await tm.drain()
    await asyncio.sleep(0)  # 若误 create_task(_run_task) 给它跑的机会
    assert started == [], "被顶替的 TM 的 drain 不应派发任务"


async def test_current_tm_drain_dispatches() -> None:
    """对照：current TM（_is_current()==True）的 drain 正常派发。"""
    tm = _tm(InProcessEventBus())
    started: list = []
    release = asyncio.Event()

    async def runner(_sid, tid):
        started.append(tid)
        await release.wait()  # park 住，避免收尾级联干扰断言

    tm.set_runner(StubRunner(tm, runner))
    tm.set_hooks(TaskManagerHooks(is_current=lambda: True))
    await tm.push_task(Task(id="A", session_id="s1", status="PENDING",
                            assigned_agent_id="a", creator_agent_id="a",
                            settings=NormalTaskSettings()))
    await tm.drain()
    await asyncio.sleep(0)
    assert started == ["A"]
    release.set()
    await asyncio.sleep(0)


def _plant_agent(rt: CtxWeftRuntime, agent_id: str, session_id: str) -> None:
    """直接种一条最小 ALM 记录，让 `record_of` 立即命中——这两条用例要测的是
    `_recover_session_locked` 本身的 per-session 锁，不是 `recover_agent` 的
    registry-miss 自愈路径（自愈会去扫事件日志，而这两个 session 压根没有事件）。
    """
    from ctx_weft.core.orchestrator.lifecycle.agent_manager import _AgentRecord
    rt._agent_lifecycle_manager._agents[agent_id] = _AgentRecord(
        session_id=session_id, tenant_id="default", template_id="tpl",
        parent_agent_id=None, spawn_depth=0, memory_config=None, loop_config=None,
    )


async def test_recover_agent_serialized_per_session(monkeypatch) -> None:
    """per-session resume 锁：同一 session 的两次并发 recover_agent 串行执行（不并发建两套 drain）。

    用一个在闸门处阻塞的 fake rebuild_view 观测并发度：有锁 → 峰值并发 1；无锁 → 2。
    """
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    _plant_agent(rt, "a1", "s1")
    active = {"n": 0, "max": 0}
    gate = asyncio.Event()

    async def fake_rebuild_view(_store, _sid):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await gate.wait()
        active["n"] -= 1
        raise RuntimeError("stop after gate")  # 短路 recover 余下步骤，释放锁

    monkeypatch.setattr("ctx_weft.core.control.reducers.rebuild_view", fake_rebuild_view)

    t1 = asyncio.create_task(rt.recover_agent("a1"))
    t2 = asyncio.create_task(rt.recover_agent("a1"))
    await asyncio.sleep(0.02)  # 两个都尝试进入锁
    assert active["max"] == 1, "per-session 锁应串行化两次 recover（峰值并发=1）"
    gate.set()
    res = await asyncio.gather(t1, t2, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in res)


async def test_recover_agent_different_sessions_not_serialized(monkeypatch) -> None:
    """不同 session 之间不应被 resume 锁串行化（各自独立锁，可并发）。"""
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    _plant_agent(rt, "aA", "sA")
    _plant_agent(rt, "aB", "sB")
    active = {"n": 0, "max": 0}
    gate = asyncio.Event()

    async def fake_rebuild_view(_store, _sid):
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await gate.wait()
        active["n"] -= 1
        raise RuntimeError("stop after gate")

    monkeypatch.setattr("ctx_weft.core.control.reducers.rebuild_view", fake_rebuild_view)

    t1 = asyncio.create_task(rt.recover_agent("aA"))
    t2 = asyncio.create_task(rt.recover_agent("aB"))
    await asyncio.sleep(0.02)
    assert active["max"] == 2, "不同 session 的 recover 应能并发（峰值并发=2）"
    gate.set()
    await asyncio.gather(t1, t2, return_exceptions=True)


async def test_slow_prior_turn_bg_observe_does_not_clobber_next_turn() -> None:
    """端到端复现：第 N 轮 finish_task 后其 background observe 拖久了，第 N+1 轮已在
    **同一个 TM** 上开跑（单例：下一轮复用活 owner）；第 N 轮迟到的收尾必须 no-op——
    不触发 on_session_done（不清新一轮的 pause 闩锁）、不发 SessionFinished。"""
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    sid = "s1"
    finished: list = []

    async def _capture(ev):
        finished.append(ev)

    rt._event_bus.subscribe(EventType.SESSION_FINISHED, _capture)

    # ── 第 N 轮 "6"：任务完成，但其 background observe 慢 ──
    sess = Session(id=sid, tenant_id="default", user_prompt="6", status="RUNNING", token_budget=0)
    tm = _running_tm(rt, sess, "t6")
    release = asyncio.Event()

    async def _slow_bg():
        await release.wait()

    tm.track_background(asyncio.create_task(_slow_bg()))

    # 任务完成 → is_done → _fire_session_done → gather 阻塞
    done6 = asyncio.create_task(tm.on_task_finished("t6", status="FINISHED"))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # ── 第 N+1 轮 "7"：同一个 TM 接着跑；模拟其在跑 + runtime 侧保留 pause 闩锁 ──
    tm.register_task(Task(id="t7", session_id=sid, status="ACTIVE",
                          assigned_agent_id="a", creator_agent_id="a",
                          settings=NormalTaskSettings()))
    tm._running_tasks.add("t7")
    rt._pausing.add(sid)

    # ── 旧轮的慢 bg observe 现在才跑完 → _fire_session_done 从 gather 恢复 ──
    release.set()
    await done6

    assert rt._task_managers[sid] is tm
    assert sid in rt._pausing, "新一轮的 pause 闩锁不得被上一轮迟到的收尾清除"
    assert finished == [], "上一轮迟到的收尾不得发 SessionFinished"


async def test_probe_prior_turn_finishing_after_supersession() -> None:
    """探针：强制让旧 TM 的 on_task_finished 在它被逐出、新 TM 接管**之后**才跑
    （forced ordering）。单例下「非 owner 的旧 TM」只可能这样出现。

    验证关键不变量仍成立：不释放新 TM、不发 SessionFinished、旧轮的 Session 对象
    不被误落定终态。2026-09-04（Task 12，events-v2 §5）起不再断言「连 stale 的
    TaskQueueDrained 也不发出」——`announce_queue_state` 本身已停发，没有信号可断言；
    `_settle` 里的归属权守卫（`is_current` 检查）现在真正防的是
    ``sess6.status = self._final_status()`` 这一步被越权执行，所以改断言这个。
    """
    rt = make_runtime(llm=MockLLMAdapter(responses=[]),
                        agent_provider=InlineAgentTemplateProvider())
    sid = "s1"
    finished: list = []

    async def _cap(ev):
        if ev.type == EventType.SESSION_FINISHED:
            finished.append(ev)

    rt._event_bus.subscribe(None, _cap)

    sess6 = Session(id=sid, tenant_id="default", user_prompt="6", status="RUNNING", token_budget=0)
    tm6 = _running_tm(rt, sess6, "t6")

    # 旧 TM 被逐出、新 TM 接管，旧 TM 的任务此后才结束
    rt._evict_session_memory(sid)
    sess7 = Session(id=sid, tenant_id="default", user_prompt="7", status="RUNNING", token_budget=0)
    tm7 = _running_tm(rt, sess7, "t7")
    rt._pausing.add(sid)

    await tm6.on_task_finished("t6", status="FINISHED")

    # 关键不变量：迟到的旧轮收尾不得造成不可恢复的破坏——不释放新 TM、不发 SessionFinished。
    assert rt._task_managers[sid] is tm7
    assert sid in rt._pausing, "新一轮的 pause 闩锁不得被旧 TM 迟到收尾清除"
    assert finished == [], "被顶替的旧 TM 不得发 SessionFinished"
    # is_done 分支的归属权守卫在 `self._session.status = self._final_status()` 之前
    # 就 return 掉了——旧轮的 Session 对象因此不被越权落定终态，仍停在 RUNNING。
    assert sess6.status == "RUNNING", "被顶替的旧 TM 不得越权落定 session.status"
