"""2026-09-08 生命周期改造的契约：内存里的 session/agent 是**只增不删的缓存**，
回收只由显式 `forget_session` / `forget_agent` 触发，`rebuild_session` 是它的逆操作。

改造前：`TaskManager` 一判定会话走完（`is_done()` 且无 parked task）就经
`_fire_session_done` → `on_session_done` → `_release_session` 把 TaskManager 和该
session 下**全部** agent record 一并拆掉。而 agent 概念上只是回到 `idle`（spec 3.1：
task 终态不是 agent 终态），于是「正常结束的会话」在内存里等于不存在——`send_message`
抛 `AgentNotFound`、`list_agents` 返回空、`get_agent` 抛错。回收时机由 core 替调用方
决定，而 core 没有判据：一条已终态的会话，用户可能正开着看历史下一秒就接着聊，也可能
三天前就关了，只有持有方知道。

改造后 core 不再替谁做这个决定，持有方按自己的策略收。真相源始终是事件日志，逐出之后
任何入口都能 `rebuild_session` 装填回来。

两个动词分得很清，别混：

  forget_session  忘记。纯缓存逐出，不终结任何东西、不发事件、不碰事件日志，会话一点
                  没少。**还在跑 / 还有人等着就拒绝**（自己把关，不要求调用方先背不
                  变量），所以随便调都安全。
  purge_session   销毁。先 cancel_session 把在跑的一切终结、未决 HITL 收口，再无条件
                  逐出内存。给"删除这条会话"这类不可逆操作用。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.errors import AgentNotFound
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

pytestmark = pytest.mark.asyncio


class _FinishLLM(MockLLMAdapter):
    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "done",
                                    "task_summary": "done"}),
            ]), request)
        if "control__collect_process_report" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__collect_process_report",
                         arguments={"act_recap": "seg"}),
            ]), request)
        # 交付物 = 收尾回合正文（finish_task 不带未声明参数，spec: capability-gateway）
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=self._id("fin"), name="control__finish_task",
                     arguments={}),
        ]), request)


def _make_runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=_FinishLLM(), agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _wait_until(predicate, timeout=10.0, interval=0.02):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met within timeout")


async def _run_one_round(rt: CtxWeftRuntime):
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi", context_limit=100_000,
    ))
    sid, aid, tid = handle.session_id, handle.agent_id, handle.task_id

    async def _done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(tid)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_done)
    await asyncio.sleep(0.3)   # 让 _fire_session_done 的 gather + 回调跑完
    return sid, aid, tid


async def test_finishing_a_session_keeps_it_in_memory():
    """核心契约：跑完不逐出。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    assert sid in rt._task_managers, "TaskManager 留着"
    assert rt._agent_lifecycle_manager.has(aid), "agent record 留着"
    assert rt.get_agent(aid).status == "idle", "agent 回 idle，不是消失"
    assert [a.agent_id for a in rt.list_agents(session_id=sid)] == [aid]


async def test_forget_session_evicts_everything():
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    rt.forget_session(sid)

    assert sid not in rt._task_managers
    assert sid not in rt._run_tokens
    assert not rt._agent_lifecycle_manager.has(aid)
    assert rt.list_agents(session_id=sid) == []
    with pytest.raises(AgentNotFound):
        rt.get_agent(aid)
    assert rt._session_registry.agent_ids_of(sid) == set(), "成员登记表也一并收（此前只增不减）"

    rt.forget_session(sid)   # 幂等


async def test_rebuild_session_brings_it_back_without_running_anything():
    """`rebuild_session` 是 `forget_session` 的逆操作，且**只装填不跑**。"""
    rt = _make_runtime()
    sid, aid, tid = await _run_one_round(rt)
    rt.forget_session(sid)

    n = await rt.rebuild_session(sid)

    assert n == 1
    assert rt._agent_lifecycle_manager.has(aid)
    assert rt.get_agent(aid).status == "idle"
    assert [a.agent_id for a in rt.list_agents(session_id=sid)] == [aid]
    # 只装填：不建 TaskManager、不派发。要续跑得另外走 recover_agent。
    assert sid not in rt._task_managers, "rebuild_* 不建 TM、不 drain"

    # 装填回来的 task 状态确实是终态（说明读的是事件日志，不是凭空造的）
    view = await rebuild_view(rt.event_store, sid)
    assert view.tasks[tid].status == "FINISHED"


async def test_rebuild_session_on_an_unknown_session_returns_zero_and_does_not_raise():
    rt = _make_runtime()
    assert await rt.rebuild_session("ses_never_existed") == 0


async def test_forget_agent_evicts_one_without_touching_siblings():
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    assert rt.forget_agent(aid) is True
    assert not rt._agent_lifecycle_manager.has(aid)
    # session 级的东西不受单点逐出影响
    assert sid in rt._task_managers
    assert rt.forget_agent(aid) is False, "已经不在 → False，幂等"


async def test_forget_agent_refuses_a_running_agent():
    """在跑的 agent 不许逐出：删掉 record 之后它的 TASK_* 会被 `handle_event` 静默
    丢弃，状态机停在 running 再也不会有 AGENT_IDLE。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    rt._agent_lifecycle_manager._agents[aid].status = "running"

    assert rt.forget_agent(aid) is False
    assert rt._agent_lifecycle_manager.has(aid)


# ── forget 的准入判据：还活着就不许忘 ─────────────────────────────────────────

async def test_forget_session_refuses_while_a_task_is_still_queued():
    """光看 agent 状态不够：一个 PENDING 还没派发的 task，其 agent 在 ALM 里仍是
    `idle`——此时逐出会把 TM 连同那个排着队的 task 一起丢掉，它永远不会跑。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    assert rt.session_is_quiescent(sid)

    rt._task_managers[sid]._queue.push(
        QueueEntry(task_id="tsk_queued", session_id=sid, priority=5, blocked_by=set()))

    assert rt.session_is_quiescent(sid) is False
    assert rt.forget_session(sid) is False, "队列里还排着活，不许忘"
    assert sid in rt._task_managers
    assert rt._agent_lifecycle_manager.has(aid)


async def test_forget_session_refuses_while_someone_is_waiting_on_a_hitl(monkeypatch):
    """有人正等着回答 → 那条 pending 记录还要用，不许忘。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    monkeypatch.setattr(rt.hitl_registry, "list_pending",
                        lambda session_id=None, **kw: ["<一条未决请求>"])

    assert rt.session_is_quiescent(sid) is False
    assert rt.forget_session(sid) is False
    assert rt._agent_lifecycle_manager.has(aid)


@pytest.mark.parametrize("status", ["running", "waiting_human", "interrupted"])
async def test_forget_refuses_agents_that_are_still_live(status):
    """五态里只有 idle / terminated 可以忘：其余三态各自还有人或事在等它。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    rt._agent_lifecycle_manager._agents[aid].status = status

    assert rt.session_is_quiescent(sid) is False
    assert rt.forget_session(sid) is False
    assert rt.forget_agent(aid) is False
    assert rt._agent_lifecycle_manager.has(aid)


async def test_forget_agent_does_not_look_at_the_rest_of_the_session():
    """单点逐出只对这一个 agent 负责：同 session 的队列/别的 agent 不是它的判据。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    # 同 session 塞一个还活着的 agent —— forget_session 会因它拒绝，forget_agent 不该。
    rt._agent_lifecycle_manager._register_fallback("agt_busy", session_id=sid)
    rt._agent_lifecycle_manager._agents["agt_busy"].status = "running"

    assert rt.forget_session(sid) is False
    assert rt.forget_agent(aid) is True, "它自己是 idle，就该能忘"
    assert rt._agent_lifecycle_manager.has("agt_busy")


# ── purge：销毁 ──────────────────────────────────────────────────────────────

async def test_purge_session_terminates_then_evicts_unconditionally():
    """`purge_session` 对**还活着**的会话也必须收干净——那正是它与 forget 的分野。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    # 造一个 forget 会拒绝的状态
    rt._agent_lifecycle_manager._agents[aid].status = "waiting_human"
    assert rt.forget_session(sid) is False

    await rt.purge_session(sid)

    assert sid not in rt._task_managers
    assert not rt._agent_lifecycle_manager.has(aid)
    assert rt._session_registry.agent_ids_of(sid) == set()
    with pytest.raises(AgentNotFound):
        rt.get_agent(aid)


async def test_purge_session_is_idempotent_and_safe_on_an_unknown_session():
    rt = _make_runtime()
    await rt.purge_session("ses_never_existed")     # 不抛
    sid, _, _ = await _run_one_round(rt)
    await rt.purge_session(sid)
    await rt.purge_session(sid)                      # 幂等


async def test_purge_does_not_touch_the_event_log():
    """core 不管持久化：purge 停掉并遗忘运行时那一半，事实仍在事件日志里，
    因此 `rebuild_session` 还能把它装回来（删事件是调用方自己的步骤）。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    await rt.purge_session(sid)
    assert not rt._agent_lifecycle_manager.has(aid)

    assert await rt.rebuild_session(sid) == 1
    assert rt._agent_lifecycle_manager.has(aid)


async def test_purge_rebuilds_a_cold_session_before_terminating_it():
    """已被逐出内存的会话再 purge：必须先装填再终结，否则收口一次都不进。

    `cancel_session` 读的是 `hitl_registry` 与 ALM 的内存现状。冷会话那两处都是空的，
    「收口未决 HITL」「逐个 cancel_agent」全部静默跳过——表现是删掉的会话在重启后又冒出
    一条未决提问（`rebuild_hitl` 按「有 HitlOpened 无终局事件」把它当未决恢复出来）。
    """
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    assert rt.forget_session(sid) is True          # 先逐出，模拟空闲淘汰过
    assert not rt._agent_lifecycle_manager.has(aid)

    canceled: list[str] = []
    orig = rt.cancel_agent

    async def _spy(agent_id, **kw):
        canceled.append(agent_id)
        return await orig(agent_id, **kw)

    rt.cancel_agent = _spy   # type: ignore[method-assign]
    await rt.purge_session(sid)

    assert canceled == [aid], "冷会话必须先装填回来，终结才落得到实处"
    assert not rt._agent_lifecycle_manager.has(aid)


# ── HITL 记录：forget 留着，purge 摘掉 ────────────────────────────────────────

def _plant_hitl(rt, session_id: str, agent_id: str, *, resolved: bool):
    """直接往 registry 里种一条记录。不走 `hitl.open()`：那条路要 `HitlAsk` + 一整套
    delivery 语境，而这里要验的只是"摘不摘"，与请求长什么样无关。"""
    from datetime import UTC, datetime

    from ctx_weft.core.hitl.registry import PendingHitl
    from ctx_weft.protocols.hitl import HitlDecision, NoResumeDelivery

    hid = f"hit_test_{session_id}_{'r' if resolved else 'p'}"
    req = PendingHitl(
        id=hid, form="question", session_id=session_id, task_id="tsk_x",
        agent_id=agent_id, delivery=NoResumeDelivery(), created_at=datetime.now(UTC),
    )
    if resolved:
        req.decision = HitlDecision(outcome="cancelled", message="")
        req.resolved_at = datetime.now(UTC)
    rt.hitl_registry._requests[hid] = req
    return hid


async def test_forget_session_keeps_resolved_hitl_records():
    """`forget_session` 只是忘记，会话随时能装填回来——已终局记录还要给
    `resolved_for_session()` 的崩溃窗口兜底用（决定已落盘、进程在续跑前就死了；对
    UserTurn 那一类，不补的话人说的那句话静默消失）。摘了它兜底就失效。
    """
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    hid = _plant_hitl(rt, sid, aid, resolved=True)

    assert rt.forget_session(sid) is True
    assert rt.hitl_registry.get(hid) is not None, "forget 不该摘 HITL 记录"
    assert [r.id for r in rt.hitl_registry.resolved_for_session(sid)] == [hid]


async def test_purge_session_drops_the_hitl_records():
    """purge 之后会话不会回来了，留着就是指向一个再也重建不出来的 session 的孤儿——
    而 `gc()` 只按 `max_resolved` 裁剪最旧的，要等它被后来的挤出去才消失。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    hid = _plant_hitl(rt, sid, aid, resolved=True)

    await rt.purge_session(sid)

    assert rt.hitl_registry.get(hid) is None
    assert rt.hitl_registry.resolved_for_session(sid) == []


async def test_purge_does_not_touch_other_sessions_hitl_records():
    rt = _make_runtime()
    sid_a, aid_a, _ = await _run_one_round(rt)
    sid_b, aid_b, _ = await _run_one_round(rt)
    hid_a = _plant_hitl(rt, sid_a, aid_a, resolved=True)
    hid_b = _plant_hitl(rt, sid_b, aid_b, resolved=True)

    await rt.purge_session(sid_a)

    assert rt.hitl_registry.get(hid_a) is None
    assert rt.hitl_registry.get(hid_b) is not None, "只摘这一条会话的"


async def test_registry_forget_session_warns_when_it_drops_an_unresolved_one(caplog):
    """摘到未决的说明上一步没做干净（热等待者会就此永远挂着）——要响亮，不能忍。"""
    import logging

    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    _plant_hitl(rt, sid, aid, resolved=False)

    with caplog.at_level(logging.WARNING, logger="ctx_weft.core.hitl.registry"):
        n = rt.hitl_registry.forget_session(sid)

    assert n == 1
    assert any("未决" in r.getMessage() for r in caplog.records), caplog.text


# ── forget 之后重新启用：不只是"句柄回来了"，得真的跑完 ───────────────────────

async def test_a_forgotten_session_runs_a_full_new_round_end_to_end():
    """逐出 → 再发一条 → 新 task 必须**真的跑到 FINISHED**。

    「句柄回来了」只证明路由和装填没崩。真正要验的是自愈把整条执行链都接回来了：
    ALM record（派发时 `materialize()` 要按它解模型、造 Agent）、TaskManager（队列 +
    runner + 8 个 hooks）、以及 `_register_and_drain` 重新挂上的 ControlCapabilityProvider
    注册——任何一环没接上，task 都会卡在队列里不动，而句柄照样是好的。
    """
    rt = _make_runtime()
    sid, aid, first_task = await _run_one_round(rt)

    assert rt.forget_session(sid) is True
    assert not rt._agent_lifecycle_manager.has(aid)
    assert sid not in rt._task_managers

    handle = await rt.send_message(aid, "第二轮", session_id=sid)

    async def _second_done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(handle.task_id)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_second_done)

    # 跑完之后一切照旧：agent 回 idle、两轮的 task 都在、还能再来一轮。
    assert rt.get_agent(aid).status == "idle"
    view = await rebuild_view(rt.event_store, sid)
    assert {t.status for t in view.tasks.values()} == {"FINISHED"}
    assert len(view.tasks) == 2


async def test_forget_then_rebuild_then_send_also_runs():
    """host 可能先 `rebuild_session`（比如 `_root_agent_id` 那一步）再发消息——
    装填过一次之后，`send_message` 走的是热路径（`record_of` 命中，不再自愈），
    TM 那一半仍要由 `_start_task_for_agent` 的探测接住。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    assert rt.forget_session(sid) is True

    assert await rt.rebuild_session(sid) == 1
    assert rt._agent_lifecycle_manager.has(aid)
    assert sid not in rt._task_managers, "rebuild 不建 TM"

    handle = await rt.send_message(aid, "第二轮", session_id=sid)

    async def _done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(handle.task_id)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_done)
    assert sid in rt._task_managers


async def test_two_forget_send_cycles_in_a_row():
    """逐出→跑→再逐出→再跑。装填不是一次性的，得能反复。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)

    for i in range(2):
        assert rt.forget_session(sid) is True, f"第 {i+1} 轮逐出失败"
        handle = await rt.send_message(aid, f"第 {i+2} 轮", session_id=sid)

        async def _done(tid=handle.task_id) -> bool:
            view = await rebuild_view(rt.event_store, sid)
            t = view.tasks.get(tid)
            return t is not None and t.status == "FINISHED"

        await _wait_until(_done)

    view = await rebuild_view(rt.event_store, sid)
    assert len(view.tasks) == 3
    assert {t.status for t in view.tasks.values()} == {"FINISHED"}


# ── 恢复期裁定：崩溃时在跑的 agent 由 core 判成 interrupted ────────────────────

async def _seed_crashed_running(rt, sid: str, aid: str) -> None:
    """写一条「崩溃时正在跑」的事件流：折出来 agent 是 running，而进程里没有任何 run。

    id 前缀用 `evt_0000…`：`InMemoryEventStore.read_by_session` 按**事件 id**（ULID 字典序）
    排序，手造 id 若排在真 ULID（`evt_01M2…`）之后，后续由 runtime 真发出来的事件会被折在
    种子事件**之前**，折叠顺序颠倒。
    """
    from datetime import UTC, datetime

    from ctx_weft.protocols.events import Event, EventType

    ts = datetime(2026, 6, 13, tzinfo=UTC)

    def ev(seq, type_, **p):
        return Event(id=f"evt_0000{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                     type=type_, timestamp=ts, task_id="tsk_crashed",
                     agent_id=p.pop("agent_id", None), payload=p)

    await rt.event_store.append(ev(1, EventType.SESSION_CREATED,
                                   template_id="agent:tpl_echo", root_agent_id=aid))
    await rt.event_store.append(ev(2, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                   template_id="agent:tpl_echo"))
    await rt.event_store.append(ev(3, EventType.AGENT_RUNNING, agent_id=aid))


async def test_rebuild_settles_a_crashed_running_agent_to_interrupted():
    """core 自己裁定，不再让宿主去折事件流替它下判决。

    `ALM.load()` 把 `AgentView.status` 照实读回来，崩溃时在跑的 agent 因此回到内存里仍是
    `running`；而 `_RECOVERY_BROADCAST_BY_STATUS` 刻意不广播 running。从前 core 就在这里
    沉默了——宿主只能自己折一遍事件流去补，那等于替五态机下判决。现在装填之后由
    `_settle_crashed_agents` 走 `apply_input` 判成 `interrupted`。
    """
    rt = _make_runtime()
    sid, aid = "S_crash", "agt_crashed"
    await _seed_crashed_running(rt, sid, aid)

    seen: list[str] = []

    async def recorder(ev):
        if getattr(ev, "agent_id", None) == aid:
            seen.append(ev.type)

    rt.event_bus.subscribe(None, recorder)

    n = await rt.rebuild_session(sid)

    assert n == 1
    assert rt.get_agent(aid).status == "interrupted", "崩溃残影的 running 必须被判掉"
    assert seen == ["AgentInterrupted"], f"该发且只发一条真转移事件，实得 {seen}"


async def test_settling_is_idempotent_across_repeated_rebuilds():
    """`AgentInterrupted` 落库之后，再装填折出来就是 `interrupted`，同态输入不再发事件。"""
    rt = _make_runtime()
    sid, aid = "S_crash2", "agt_crashed"
    await _seed_crashed_running(rt, sid, aid)
    await rt.rebuild_session(sid)

    seen: list[str] = []

    async def recorder(ev):
        if getattr(ev, "agent_id", None) == aid:
            seen.append(ev.type)

    rt.event_bus.subscribe(None, recorder)

    rt.forget_session(sid)          # 逐出，逼下一次走完整装填
    await rt.rebuild_session(sid)

    assert rt.get_agent(aid).status == "interrupted"
    assert seen == [], f"重复装填不该重复发转移事件，实得 {seen}"


async def test_a_finished_session_is_not_settled():
    """正常跑完的会话装填回来是 `idle`，不该被判成中断。"""
    rt = _make_runtime()
    sid, aid, _ = await _run_one_round(rt)
    rt.forget_session(sid)

    seen: list[str] = []

    async def recorder(ev):
        if getattr(ev, "agent_id", None) == aid:
            seen.append(ev.type)

    rt.event_bus.subscribe(None, recorder)
    await rt.rebuild_session(sid)

    assert rt.get_agent(aid).status == "idle"
    assert "AgentInterrupted" not in seen


async def test_recover_agent_does_not_settle_before_running_it():
    """分工：`rebuild_session` 装填 + 裁定（不跑）；`recover_agent` 装填 + 跑（不裁定）。

    后者装填完立刻 restore + drain，agent 马上拿到真的 `AGENT_RUNNING`；在那之前插一条
    `AgentInterrupted` 只会让宿主界面闪一下中断态。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._recover_session_locked)
    assert "_settle_crashed_agents" not in src, "续跑路径不该裁定"
    assert "_settle_crashed_agents" in inspect.getsource(CtxWeftRuntime.rebuild_session)
