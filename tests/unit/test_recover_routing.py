"""recover() 启动恢复（spec/07 §9）——core 内闭环,无回调,启动不 drain。

2026-09-04（Task 12）起：`TaskManager.announce_queue_state` / `_announce_queue_state_as_tm_proxy`
——recover() 曾经代 TaskManager 发的那条会话级 TaskQueueBlocked/TaskQueueInterrupted 队列聚合
信号——已停发（其唯一消费者、会话状态机，早在 2026-09-03 就已降格；枚举成员与 L 档登记见
events-v2 §5）。恢复期的可观测性现在由（2026-09-04 spec §6.3/§6.4）补上的一步接管：recover()
把每个 session 的 agent 记录装填进 `AgentLifecycleManager`，装填完按折出来的现状发
AGENT_IDLE / AGENT_WAITING_HUMAN / AGENT_INTERRUPTED（`_load_agents_of` →
`AgentLifecycleManager.load`）。本文件因此改钉这一条广播本身：
`_capture_waiting_human_broadcast` / `_capture_interrupted_broadcast` 捕获的是恢复期 ALM
的现状广播，不再是已经停发的 TM 聚合信号。

- 崩溃前 agent 停在 waiting_human（有未决 HITL 等人答）→ 恢复后重发 AGENT_WAITING_HUMAN
- 崩溃前 agent 停在 interrupted（没有未决 HITL，进程重启打断）→ 恢复后重发 AGENT_INTERRUPTED

**这曾经不是最强观测点，现在是**（旧 docstring 在此的论点已被推翻，不是漏补）：旧论点是
recover() 本身不 drain、不派发任何 task/agent，ALM 的五态机这时候根本没跑起来，没有 AGENT_*
可捕。Task 11 起这个前提不再成立——ALM 的装填 + 广播这一步被直接搬进了 recover() 的调用链
本身（`_load_agents_of` 就在这个方法里被调用），所以 AGENT_* 现状广播现在**正是**这个调用点
能拿到的观测点，不是退而求其次的替代品。

「等的是审批面板还是一句话」不上升到任何状态事件——那是 delivery 的性质,2026-09-04
Task 14 起由 host 直接从 `list_pending_hitl(...)` 的 `delivery` 字段自判,core 不再
提供派生的 session 级状态串。决策只折叠 HITL 类事件,不全量回放。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlAsk, HitlReply, ToolResultDelivery
from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 6, 13, tzinfo=timezone.utc)


def _ev(seq: int, sid: str, type_: EventType, *, agent_id: str | None = None, **payload) -> Event:
    return Event(id=f"evt_{sid}_{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                 type=type_, timestamp=_TS, task_id="t1", agent_id=agent_id, payload=payload)


def _capture_waiting_human_broadcast(runtime) -> list[str]:
    """记下恢复期被 ALM 现状广播成 waiting_human 的 session（该 agent 崩溃前正等人答复）。

    2026-09-04（Task 12）起这是 `AgentLifecycleManager.load()` 装填完发的
    `AGENT_WAITING_HUMAN`——不再是已停发的会话级 `TaskQueueBlocked`。
    """
    seen: list[str] = []
    async def recorder(ev: Event) -> None:
        if ev.type == EventType.AGENT_WAITING_HUMAN:
            seen.append(ev.session_id)
    runtime.event_bus.subscribe(None, recorder)
    return seen


def _capture_interrupted_broadcast(runtime) -> list[str]:
    """记下恢复期被 ALM 现状广播成 interrupted 的 session（该 agent 崩溃前没有未决 HITL）。

    2026-09-04（Task 12）起这是 `AgentLifecycleManager.load()` 装填完发的
    `AGENT_INTERRUPTED`——不再是已停发的会话级 `TaskQueueInterrupted`。
    """
    seen: list[str] = []
    async def recorder(ev: Event) -> None:
        if ev.type == EventType.AGENT_INTERRUPTED:
            seen.append(ev.session_id)
    runtime.event_bus.subscribe(None, recorder)
    return seen


async def test_recover_routes_by_pending_hitl(monkeypatch) -> None:
    runtime = make_runtime(agent_provider=InlineAgentTemplateProvider())
    store = runtime.event_store

    # A: 有未解决 pending HITL，agent 崩溃前停在 waiting_human → 只装填 HitlRegistry，
    #    恢复后重发 AGENT_WAITING_HUMAN。
    await append_one(store, _ev(1, "A", EventType.SESSION_CREATED, template_id="t",
                            root_agent_id="agtA"))
    await append_one(store, _ev(2, "A", EventType.AGENT_INSTANTIATED, agent_id="agtA",
                            template_id="t"))
    await append_one(store, _ev(3, "A", EventType.HITL_REQUIRED, hitl_id="hA", form="question",
                            tool_call_id="tcA"))
    await append_one(store, _ev(4, "A", EventType.AGENT_WAITING_HUMAN, agent_id="agtA"))
    # B: HITL 已答复 → 无 pending，agent 应完之后又被进程重启打断 → 恢复后重发
    #    AGENT_INTERRUPTED。
    await append_one(store, _ev(1, "B", EventType.SESSION_CREATED, template_id="t",
                            root_agent_id="agtB"))
    await append_one(store, _ev(2, "B", EventType.AGENT_INSTANTIATED, agent_id="agtB",
                            template_id="t"))
    await append_one(store, _ev(3, "B", EventType.HITL_REQUIRED, hitl_id="hB", form="question"))
    await append_one(store, _ev(4, "B", EventType.HITL_ANSWERED, hitl_id="hB"))
    await append_one(store, _ev(5, "B", EventType.AGENT_INTERRUPTED, agent_id="agtB"))
    # C: 从无 HITL、直接被进程重启打断 → 恢复后重发 AGENT_INTERRUPTED。
    await append_one(store, _ev(1, "C", EventType.SESSION_CREATED, template_id="t",
                            root_agent_id="agtC"))
    await append_one(store, _ev(2, "C", EventType.AGENT_INSTANTIATED, agent_id="agtC",
                            template_id="t"))
    await append_one(store, _ev(3, "C", EventType.AGENT_INTERRUPTED, agent_id="agtC"))

    # 启动不应调 recover_agent（task 重建推迟到应答）
    called: list[str] = []
    async def fail_recover_agent(aid, **kw):
        called.append(aid)
    monkeypatch.setattr(runtime, "recover_agent", fail_recover_agent)
    waiting_human = _capture_waiting_human_broadcast(runtime)
    interrupted = _capture_interrupted_broadcast(runtime)

    n = await rebuild_all_active(runtime)

    # Task 11 起 recover() 的返回值语义换成「恢复的 agent 数」，不再是 session 数
    # （2026-09-04 spec §6.2）。三个 session 各自发过一条 AGENT_INSTANTIATED、
    # SESSION_CREATED 也各带了 root_agent_id——reducer 能折出恰好 3 个 AgentView
    # （`core/control/reducers.py` 的 `_rebuild_agents`/`AGENT_INSTANTIATED` 分支）。
    assert n == 3
    assert called == []                                          # 启动不 drain/不重建 task
    assert [r.id for r in runtime.hitl_registry.list_pending(session_id="A")] == ["hA"]
    # 决定缓存键是三维的 (session, tool_call, stage)——只按 tool_call_id 查会让 A 会话的
    # 批准替 B 会话里同名 id 的调用开门，那正是本次重设计关掉的跨会话授权洞。
    assert runtime.hitl_registry.find_for_tool_call("A", "tcA", HITL_STAGE_TOOL) is not None
    assert runtime.hitl_registry.find_for_tool_call("B", "tcA", HITL_STAGE_TOOL) is None
    assert runtime.hitl_registry.list_pending(session_id="B") == []
    # 无 pending 的两个 → 被进程重启打断，等 /resume。
    assert set(interrupted) == {"B", "C"}
    # 有人在等回话的那个 → AGENT_WAITING_HUMAN，不是 interrupted（绝不把 parked 任务孤立）。
    assert waiting_human == ["A"]
    # 不再断言 `runtime._session_registry.status_of(...)`（该方法本身已在 Task 15 被
    # 摘除）——现在的观测点换成了 AGENT_* 现状广播本身，见模块 docstring。


async def test_recover_multi_hitl_partial_resolve_still_pending() -> None:
    """两个 pending、只解决一个 → 仍 pending → 重建剩余、重发 AGENT_WAITING_HUMAN 而非 interrupted。"""
    runtime = make_runtime(agent_provider=InlineAgentTemplateProvider())
    store = runtime.event_store
    await append_one(store, _ev(1, "M", EventType.SESSION_CREATED, template_id="t",
                            root_agent_id="agtM"))
    await append_one(store, _ev(2, "M", EventType.AGENT_INSTANTIATED, agent_id="agtM",
                            template_id="t"))
    await append_one(store, _ev(3, "M", EventType.HITL_REQUIRED, hitl_id="h1", form="question"))
    await append_one(store, _ev(4, "M", EventType.HITL_REQUIRED, hitl_id="h2", form="approval"))
    await append_one(store, _ev(5, "M", EventType.HITL_ANSWERED, hitl_id="h1"))
    await append_one(store, _ev(6, "M", EventType.AGENT_WAITING_HUMAN, agent_id="agtM"))

    waiting_human = _capture_waiting_human_broadcast(runtime)
    interrupted = _capture_interrupted_broadcast(runtime)
    n = await rebuild_all_active(runtime)

    assert {r.id for r in runtime.hitl_registry.list_pending(session_id="M")} == {"h2"}
    assert n == 1                                                 # 一个 session、一个 agent
    assert interrupted == []
    assert waiting_human == ["M"]


# ── 2026-09-04 spec §6.2：恢复入口换轴 ─────────────────────────────────────


def test_recover_session_is_gone():
    assert not hasattr(CtxWeftRuntime, "recover_session")


async def _crashed_session(rt: CtxWeftRuntime) -> tuple[str, str]:
    """在 `rt` 的事件库里种一个「崩溃前挂着未决 HITL」的 session，返回 (session_id, root_agent_id)。

    复用 `test_hitl_recovery.py::test_recover_agent_rebuilds_pending_hitl_and_parks`
    验证过的种子形状：task 停在未决 HITL 上 → 续跑只装填 registry、保持 parked，不
    真的驱动 LLM——这让「续跑准确定位到了正确的 session」这件事可以脱离一整条
    LLM 驱动的 loop 单独断言，也不需要 llm/echo template 之外的任何装配。
    """
    ts = _TS
    sid = "ses_recover_agent"
    aid = "agt_root"

    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{sid}_{seq:04d}", run_id="run_1", sequence=seq, session_id=sid,
                     type=type_, timestamp=ts, task_id=task_id, payload=payload)

    seed = [
        ev(1, EventType.SESSION_CREATED, user_prompt="do it", template_id="agent:tpl_echo",
           root_agent_id=aid),
        ev(2, EventType.RUN_STARTED),
        ev(3, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "PENDING", "title": "T1",
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        ev(4, EventType.TASK_STARTED, task_id="tsk_1", assigned_agent_id=aid),
        ev(5, EventType.HITL_REQUIRED, task_id="tsk_1", hitl_id="hit_1", form="question",
           capability_id="control:ask_user", tool_call_id="tcA", question="Which DB?"),
        ev(6, EventType.TASK_SUSPENDED, task_id="tsk_1"),
    ]
    for e in seed:
        await append_one(rt.event_store, e)
    return sid, aid


def _runtime_for_recover_agent() -> CtxWeftRuntime:
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="should not run")])
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def test_recover_agent_resolves_session_from_the_record() -> None:
    """调用方只给 agent_id，session 由 ALM 记录反查——这就是「换轴」的全部含义。"""
    rt = _runtime_for_recover_agent()
    session_id, root_agent_id = await _crashed_session(rt)
    await rebuild_all_active(rt)
    await rt.recover_agent(root_agent_id)          # 不传 session_id 也能跑通
    assert rt.get_agent(root_agent_id).session_id == session_id


async def test_recover_agent_unknown_raises() -> None:
    """事件库里也确实不存在的 agent_id → 抛 `AgentNotFound`。

    今天走的是 `AgentNotLoaded` 那条（它是子类，断言照样成立）：core 不再区分「没装填」
    与「真不存在」——要区分就得扫全库，而那正是 2026-09-21 删掉的东西。调用方先
    `rebuild_session` 再撞上这个错，才说明是真的查无此 agent。"""
    from ctx_weft.core.models.errors import AgentNotFound

    rt = _runtime_for_recover_agent()
    with pytest.raises(AgentNotFound):
        await rt.recover_agent("agt_nope")


async def test_recover_agent_refuses_a_cold_registry_then_works_after_rebuild_session() -> None:
    """`recover_agent` 对未装填的 registry **抛 `AgentNotLoaded`，不自己去找**
    （2026-09-21）。

    从前它对 miss 会先扫全部 active session 把记录找出来（`rebuild_agent` →
    `rebuild_all_agents`）。那条 sweep 已删：它拿 O(会话数) 换一个调用方本来就知道的
    值，而且扫的集合是坏的（`list_active_session_ids` 的 discard 依据从不 emit，判据
    恒真 = 历史全部会话）。装填改由调用方负责，与 `rebuild_session` 定下的用户驱动
    模型一致（「用到哪条装哪条」）。

    **冷 HITL 应答不受影响**，那是从前留着自愈的唯一理由（`reply_to_hitl` 把 HITL 判成
    终局后没有第二次机会，续跑摔了会话就永久卡住）：那条路由 `_resume_after_hitl` 在调
    `recover_agent` **之前**的 `_hydrate_agent_for_cold_resume(req)` 用 `req.session_id`
    精确装填接住，由下面那条 `test_cold_hitl_reply_hydrates_only_its_own_session_not_a_sweep`
    走一次真实的 `reply_to_hitl` 钉住。

    本条钉两件事：① 冷 registry 直接抛，错误类型可诊断；② 先 `rebuild_session` 再调就
    照常跑通——即「装填是前提」而不是「这条路没了」。
    """
    from ctx_weft.core.models.errors import AgentNotFound, AgentNotLoaded

    rt = _runtime_for_recover_agent()
    session_id, root_agent_id = await _crashed_session(rt)

    # 冷启动断言：这个进程从没装填过，registry 应确实是空的。
    assert rt._agent_lifecycle_manager.record_of(root_agent_id) is None

    with pytest.raises(AgentNotLoaded) as exc:
        await rt.recover_agent(root_agent_id)
    # 宿主既有的 `except AgentNotFound` 必须照样接住（子类，不是新的顶层类型）。
    assert isinstance(exc.value, AgentNotFound)
    assert "rebuild_session" in str(exc.value), "报错要告诉调用方下一步做什么"

    # 装填之后同一句调用照常跑通——被推翻的是「core 替你找」，不是这条路径本身。
    await rt.rebuild_session(session_id)
    await rt.recover_agent(root_agent_id)

    assert rt.get_agent(root_agent_id).session_id == session_id
    assert [r.id for r in rt.hitl_registry.list_pending(session_id=session_id)] == ["hit_1"]


async def test_cold_hitl_reply_hydrates_only_its_own_session_not_a_sweep() -> None:
    """冷 HITL 应答只装填**它自己那一个** session：`_resume_after_hitl` 用
    `req.session_id`（`PendingHitl` 本来就带）精确装填，让随后 `recover_agent` 里的
    `record_of` 直接命中。

    这条原本守的是「别退化成 `rebuild_all_agents` 的全库 sweep」。那条 sweep 已于
    2026-09-21 删除，所以今天它守的是**更强的一条**：`recover_agent` 对 miss 直接抛
    `AgentNotLoaded`，前面那次精确装填因此不是优化而是**前提**——漏掉它，人刚答完的
    这次续跑就会摔掉，而 `reply_to_hitl` 已把 HITL 判成终局、没有第二次机会。

    种两个都在事件库里「活着」的 session：S1 是这次真正要续跑的，S2 是旁观者。全程
    冷启动（不预先装填），只走一次真实的 `reply_to_hitl`。两条断言各守一半：S1 跑通
    说明精确装填确实接住了；S2 的 agent 事后仍不在 registry 里说明没有任何人在扫全库。
    """
    rt = _runtime_for_recover_agent()

    # S1：真正要续跑的 session。tsk_1 上开两个 ToolResultDelivery 请求——只答其中
    # 一个（hit_1），另一个（hit_2）留着不答，让 task 恢复后仍保持 parked（不触发
    # 真实 LLM 派发），断言可以只盯 registry 内容，不必陪一整条 run 走完。
    sid1, aid1, tid1 = "ses_1", "agt_1", "tsk_1"

    def ev1(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_s1_{seq:04d}", run_id="r1", sequence=seq, session_id=sid1,
                     type=type_, timestamp=_TS, task_id=task_id, payload=payload)

    await append_one(rt.event_store, ev1(1, EventType.SESSION_CREATED, user_prompt="do it",
                                     template_id="agent:tpl_echo", root_agent_id=aid1))
    await append_one(rt.event_store, ev1(2, EventType.TASK_CREATED, task={
        "id": tid1, "status": "PENDING", "title": "T1",
        "assigned_agent_id": aid1, "creator_agent_id": aid1}))

    req1 = await rt.hitl.open(
        HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id="tcA")),
        session_id=sid1, task_id=tid1, agent_id=aid1, tool_call_id="tcA", stage=HITL_STAGE_TOOL, unattended=False,
    )
    await rt.hitl.open(
        HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id="tcB")),
        session_id=sid1, task_id=tid1, agent_id=aid1, tool_call_id="tcB", stage=HITL_STAGE_TOOL, unattended=False,
    )

    # S2：另一个真实 active 的 session——事件库里有它自己的 agent。任何「按 agent 找
    # session 就扫全库」的写法回来，它都会被顺带扫进 ALM，下面那条断言就红。
    sid2, aid2 = "ses_2", "agt_2"
    await append_one(rt.event_store, Event(
        id="evt_s2_0001", run_id="r2", sequence=1, session_id=sid2,
        type=EventType.SESSION_CREATED, timestamp=_TS,
        payload={"user_prompt": "hi", "template_id": "agent:tpl_echo", "root_agent_id": aid2},
    ))

    # 冷启动断言：全程不调 `rt.recover()`，ALM 对两个 session 都还没装填。
    assert rt._agent_lifecycle_manager.record_of(aid1) is None
    assert rt._agent_lifecycle_manager.record_of(aid2) is None

    await rt.reply_to_hitl(HitlReply(hitl_id=req1.id, outcome="accepted", agent_id=aid1))

    # S1 被精确装填——这次续跑真正需要的那个 session。
    assert rt.get_agent(aid1).session_id == sid1
    # tsk_1 仍有一个未决请求（hit_2）挂着，保持 parked：断言不必陪一整条 LLM run 走完。
    assert [r.id for r in rt.hitl_registry.list_pending(session_id=sid1)] == \
        [r.id for r in rt.hitl_registry.list_pending(session_id=sid1) if r.tool_call_id == "tcB"]

    # S2 一个字节都没被碰——sweep 没有发生，只有 S1 被喂进 ALM。
    assert rt._agent_lifecycle_manager.record_of(aid2) is None
    assert rt.list_agents(session_id=sid2) == []
