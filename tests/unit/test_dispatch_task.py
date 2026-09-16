"""`dispatch_task`：宿主侧任务派发入口（spec/09）。

覆盖 spec/09 §9 的十一条不变式。三条最值钱的：

- **重放等价**（§3.1，本设计的地基）：`instantiate(parent_agent_id=X)` 的 X 恒等于
  `task.creator_agent_id or None`，因为 `_rebuild_agents` 重放时正是按 `creator or None`
  推 parent/depth。两边任一侧漂了，崩溃重建折出来的就是另一棵森林。
- **两道守卫分岔**（§7.2）：要 agent 亲自执行则 `running` 拒；只当血缘父则 `running`
  放行。后者正是本轮明确要放开的那一条。
- **派生不污染父对话**（§2.4）：顶层 task 的 `parent_task_id is None` 使 finalize 的
  派发框/bubble 两处准入判据都不成立——A 是血缘上的父，不是对话上的父。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.errors import (
    AgentBusyError,
    AgentNotFound,
    AgentTerminatedError,
    SessionAlreadyExistsError,
    UnfinishedTasksError,
)
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import MemoryAddress, MemoryEventType, ProviderContext, ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio


def _finish(n: int = 12) -> list[MockResponse]:
    """n 个「回一句就 finish_task」的回合，够 root + 派发出去的那几条用。"""
    return [
        MockResponse(text="done", tool_calls=[
            ToolCall(id=f"tc{i}", name="control__finish_task", arguments={})])
        for i in range(n)
    ]


async def _runtime_with_session(*, responses: list[MockResponse] | None = None):
    """起一个跑完 root task 的真实会话，返回 (runtime, session_id, root_agent_id, events)。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=MockLLMAdapter(responses=responses or _finish()),
                      agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())

    events: list = []

    async def _cap(ev):
        events.append(ev)

    rt.event_bus.subscribe(None, _cap)

    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="root turn", context_limit=180_000,
    ))
    await handle.wait_for_finish(timeout=10.0)
    return rt, handle.session_id, handle.agent_id, events


def _instantiated(events) -> list[str]:
    return [e.agent_id for e in events if e.type == EventType.AGENT_INSTANTIATED]


def _spawned(events) -> list:
    return [e for e in events if e.type == EventType.AGENT_SPAWNED]


# ── §9.1 全新树 ───────────────────────────────────────────────────────────────


async def test_fresh_tree_has_no_parent_and_does_not_touch_root_agent():
    rt, sid, root, events = await _runtime_with_session()
    before = len(events)

    h = await rt.dispatch_task(
        sid, "background job",
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)

    new_events = events[before:]
    assert _instantiated(new_events) == [h.agent_id], "恰好一条 AgentInstantiated"
    assert _spawned(new_events) == [], "无父 → 不发 AgentSpawned（那条的主语是父的一次 spawn 动作）"
    assert h.agent_id != root

    rec = rt._agent_lifecycle_manager.record_of(h.agent_id)
    assert rec.parent_agent_id is None
    assert rec.spawn_depth == 0
    # root agent 一个字节都不改——森林只是 ALM 里多一条无父 record
    assert rt._task_managers[sid].session.root_agent_id == root


# ── §9.2 派生 ─────────────────────────────────────────────────────────────────


async def test_derived_subtree_records_parent_and_depth():
    rt, sid, root, events = await _runtime_with_session()
    before = len(events)

    h = await rt.dispatch_task(
        sid, "derived job", agent_id=root,
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)

    new_events = events[before:]
    spawned = _spawned(new_events)
    assert len(spawned) == 1
    assert spawned[0].payload["parent_agent_id"] == root
    # 因果序：先记「这次 spawn 被准了」，再记「诞生的 agent 长这样」
    assert new_events.index(spawned[0]) < next(
        i for i, e in enumerate(new_events)
        if e.type == EventType.AGENT_INSTANTIATED and e.agent_id == h.agent_id
    )

    rec = rt._agent_lifecycle_manager.record_of(h.agent_id)
    root_rec = rt._agent_lifecycle_manager.record_of(root)
    assert rec.parent_agent_id == root
    assert rec.spawn_depth == root_rec.spawn_depth + 1


# ── §9.3 重放等价（地基）──────────────────────────────────────────────────────


@pytest.mark.parametrize("derive", [False, True], ids=["fresh_tree", "derived"])
async def test_replay_rebuilds_the_same_forest(derive: bool):
    """内存里的 `_AgentRecord` 与事件重放折出的 `AgentView` 必须逐字段相等。

    这是 spec/09 §3.1 的地基：`instantiate(parent_agent_id=X)` 与
    `_rebuild_agents` 的 `creator or None` 是同一个口径，任一侧漂了这条就红。
    """
    rt, sid, root, _ = await _runtime_with_session()

    h = await rt.dispatch_task(
        sid, "job", agent_id=(root if derive else None),
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)

    rec = rt._agent_lifecycle_manager.record_of(h.agent_id)
    view = await rebuild_view(rt.event_store, sid)
    av = view.agents[h.agent_id]

    assert av.parent_agent_id == rec.parent_agent_id
    assert av.spawn_depth == rec.spawn_depth
    assert av.template_id == rec.template_id
    # root 不被这次派发改写
    assert view.agents[root].parent_agent_id is None
    assert view.agents[root].spawn_depth == 0


# ── §9.7 两道守卫分岔（本轮的核心决定）────────────────────────────────────────


async def test_busy_agent_rejects_execution_but_allows_parenting():
    """A 在跑时：让它**执行** → AgentBusyError；只让它当**血缘父** → 放行。"""
    rt, sid, root, events = await _runtime_with_session()
    # 把 root 摆成 running（真实场景是它正跑着一条 task）
    rt._agent_lifecycle_manager._agents[root].status = "running"

    before = len(events)
    with pytest.raises(AgentBusyError):
        await rt.dispatch_task(sid, "run it on A", agent_id=root)
    assert events[before:] == [], "拒绝路径零副作用：不发事件"
    assert len(rt._task_managers[sid].all_tasks()) == 1, "也不入队"
    assert len(rt._agent_lifecycle_manager.agent_ids_of_session(sid)) == 1, "也不建 agent"

    # 同一个 running 的 A，当血缘父放行
    h = await rt.dispatch_task(
        sid, "graft a subtree onto busy A", agent_id=root,
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    assert h.agent_id != root
    assert rt._agent_lifecycle_manager.record_of(h.agent_id).parent_agent_id == root
    # A 自己的状态不被这次派发碰
    assert rt._agent_lifecycle_manager.status_of(root) == "running"


async def test_terminated_agent_rejected_in_both_roles():
    rt, sid, root, _ = await _runtime_with_session()
    rt._agent_lifecycle_manager._agents[root].status = "terminated"

    with pytest.raises(AgentTerminatedError):
        await rt.dispatch_task(sid, "x", agent_id=root)
    with pytest.raises(AgentTerminatedError):
        await rt.dispatch_task(
            sid, "x", agent_id=root,
            settings=NormalTaskSettings(use_subagent=True,
                                        subagent_template="agent:tpl_echo"))


async def test_unknown_agent_rejected():
    rt, sid, _root, _ = await _runtime_with_session()
    with pytest.raises(AgentNotFound):
        await rt.dispatch_task(sid, "x", agent_id="agt_nope")


# ── §9.8 派生不污染父对话 ─────────────────────────────────────────────────────


async def test_derived_subtree_leaves_no_dispatch_frame_in_parent_scope():
    """顶层 task 的 `parent_task_id is None` → finalize 不往 A 的 scope 铸派发框。"""
    rt, sid, root, _ = await _runtime_with_session()
    mem = rt.providers.get_memory()
    ctx = ProviderContext(session_id=sid)
    root_scope = MemoryAddress(session_id=sid, agent_id=root)
    before = len(await mem.recall_recent(
        root_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 100, ctx))

    h = await rt.dispatch_task(
        sid, "derived job", agent_id=root,
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo",
                                    inherit_memory=False),
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)

    after = await mem.recall_recent(
        root_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 100, ctx)
    assert before > 0, "前置：root 跑完后它的 agent scope 本就有 finish 对，否则本测试空转"
    assert len(after) == before, \
        f"A 的 agent scope 不该因为一次外部派发多出记录，got {[r.content for r in after]}"


# ── §9.5 跨树继承 ─────────────────────────────────────────────────────────────


async def test_fresh_tree_can_inherit_from_an_unrelated_agent():
    """血缘与记忆来源是正交的两个轴：无父的新树照样能拿 root 的上下文。"""
    rt, sid, root, _ = await _runtime_with_session()

    h = await rt.dispatch_task(
        sid, "continue from root's context",
        settings=NormalTaskSettings(
            use_subagent=True, subagent_template="agent:tpl_echo",
            inherit_memory=True, inherit_from_agent_id=root,
        ),
        unattended=True,
    )
    mem = rt.providers.get_memory()
    ctx = ProviderContext(session_id=sid)
    child_scope = MemoryAddress(session_id=sid, task_id=h.task_id, agent_id=h.agent_id)
    # 继承发生在 assemble（派发那一刻），等 task 跑完再读
    await h.wait_for_finish(timeout=10.0)
    turns = await mem.recall_recent(
        child_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 100, ctx)

    inherited = [t for t in turns if t.metadata.get("inherited_from_agent_id")]
    assert inherited, "显式 inherit_from_agent_id 必须真的复制出东西"
    assert all(t.metadata["inherited_from_agent_id"] == root for t in inherited)
    # 血缘仍是无父——继承不改变树形
    assert rt._agent_lifecycle_manager.record_of(h.agent_id).parent_agent_id is None


# ── §9.9 入口即拒（四条纯参数校验，零事件）────────────────────────────────────


@pytest.mark.parametrize("kwargs, why", [
    (dict(settings=NormalTaskSettings(use_subagent=False)),
     "V1: agent_id=None 且 use_subagent=False"),
    (dict(agent_id="__ROOT__", unattended=True, interaction_mode="interactive"),
     "V2: unattended ⟹ auto 的不变式冲突"),
    (dict(agent_id="__ROOT__",
          settings=NormalTaskSettings(use_subagent=False, inherit_from_agent_id="agt_x")),
     "V3: 显式继承源在非 subagent 下无处安放"),
    (dict(settings=NormalTaskSettings(use_subagent=True)),
     "V5: use_subagent=True 必须给 subagent_template"),
])
async def test_entry_validation_rejects_with_zero_side_effects(kwargs, why):
    rt, sid, root, events = await _runtime_with_session()
    if kwargs.get("agent_id") == "__ROOT__":
        kwargs = {**kwargs, "agent_id": root}
    before = len(events)

    with pytest.raises(ValueError):
        await rt.dispatch_task(sid, "x", **kwargs)

    assert events[before:] == [], f"{why} —— 拒绝路径不得发任何事件"


async def test_settings_accepts_raw_dict():
    """host 从 JSON 配置直接喂 —— 与 SessionStartParams.create(initial_task=dict) 同口径。"""
    rt, sid, _root, _ = await _runtime_with_session()
    h = await rt.dispatch_task(
        sid, "job",
        settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)
    assert rt._agent_lifecycle_manager.record_of(h.agent_id) is not None


# ── §9.10 不发 SessionResumed / 不受弃轮禁止 ─────────────────────────────────


async def test_dispatch_works_where_resume_is_blocked_by_unfinished_tasks():
    """同一个会话状态下：`start_session(resume=True)` 抛 `UnfinishedTasksError`，派发照跑。

    这是 `dispatch_task` 相对 resume 的核心差别，两边都在同一状态下跑一遍当场对照——
    只断言「派发成功」证明不了「resume 在这里本来会失败」。

    非终态任务由一条 interactive 的纯文本回合造出：actor 不调 `finish_task` → HITL
    input 冷 park → `AWAITING_HUMAN`，非终态且不会自己走掉。
    """
    # 全程纯文本的 LLM——**不按调用次序排响应**：后台步骤（recognize_intent /
    # background observe）也从同一个队列抽，位置假设靠不住。纯文本对两种 task 的
    # 效果是确定的：interactive → HITL input 冷 park（非终态）；auto → 文本即产出、
    # 路由 observe 正常收尾。
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(
        llm=MockLLMAdapter(responses=[MockResponse(text="I need a human decision.")] * 40),
        agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    events: list = []

    async def _cap(ev):
        events.append(ev)

    rt.event_bus.subscribe(None, _cap)

    # root task 是 interactive 的（`_make_root_task_manager` 的口径）→ 纯文本即 park
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="root turn", context_limit=180_000,
    ))
    sid, parked = handle.session_id, handle
    for _ in range(300):        # park 发生在 act 内部，等它落定
        t = rt._task_managers[sid].get_task(parked.task_id)
        if t is not None and t.status not in ("PENDING", "ACTIVE"):
            break
        await asyncio.sleep(0.01)
    t = rt._task_managers[sid].get_task(parked.task_id)
    assert t.status == "AWAITING_HUMAN", f"前置没成立，task 停在 {t.status}"

    # resume 那条路在此状态下是关着的
    with pytest.raises(UnfinishedTasksError):
        await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="another turn",
            context_limit=180_000, session_id=sid, resume=True,
        ))

    # 派发这条路开着，且不发 SessionResumed
    before = len(events)
    h = await rt.dispatch_task(
        sid, "an independent tree, while another task is parked",
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    assert h.task_id != parked.task_id
    await h.wait_for_finish(timeout=10.0)
    assert not [e for e in events[before:] if e.type == EventType.SESSION_RESUMED]


# ── §9.11 句柄可寻址 ─────────────────────────────────────────────────────────


async def test_handle_agent_is_addressable_afterwards():
    rt, sid, _root, _ = await _runtime_with_session()
    h = await rt.dispatch_task(
        sid, "job",
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"),
        unattended=True,
    )
    await h.wait_for_finish(timeout=10.0)

    assert h.agent_id
    assert rt.get_agent(h.agent_id) is not None
    # 跑完之后还能对这个 id 继续说话（开出下一轮）
    nxt = await rt.send_message(h.agent_id, "and one more thing", session_id=sid)
    assert nxt.task_id != h.task_id
    assert nxt.agent_id == h.agent_id


# ── 非 subagent：挂已有 agent 直接跑 ──────────────────────────────────────────


async def test_dispatch_onto_existing_idle_agent_creates_no_new_agent():
    rt, sid, root, events = await _runtime_with_session()
    before = len(events)

    h = await rt.dispatch_task(sid, "run on root", agent_id=root, unattended=True)
    await h.wait_for_finish(timeout=10.0)

    assert h.agent_id == root
    assert _instantiated(events[before:]) == [], "非 subagent 派发不建任何 agent"
    assert _spawned(events[before:]) == []


# ── §11 容器会话（create_session）────────────────────────────────────────────


async def _container(**kw):
    """建一条容器会话，返回 (runtime, SessionHandle, events)。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=MockLLMAdapter(responses=_finish(20)), agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    events: list = []

    async def _cap(ev):
        events.append(ev)

    rt.event_bus.subscribe(None, _cap)
    sh = await rt.create_session(
        template_id="agent:tpl_echo", context_limit=180_000, **kw)
    return rt, sh, events


async def test_container_session_has_root_agent_but_no_task():
    rt, sh, events = await _container()

    types = [str(e.type) for e in events]
    assert "SessionCreated" in types
    assert "AgentInstantiated" in types
    assert "TaskCreated" not in types, "容器会话不推 root task"
    # 因果序 Session → Agent（host 投影对 session_id 有 FK）
    assert types.index("SessionCreated") < types.index("AgentInstantiated")

    assert sh.root_agent_id
    assert rt._agent_lifecycle_manager.status_of(sh.root_agent_id) == "idle"
    assert rt._task_managers[sh.session_id].all_tasks() == []


async def test_container_session_does_not_announce_itself_finished():
    """空队列上的 drain 只是空转——不得因为「一件活都没有」就发会话终结信号。"""
    rt, sh, events = await _container()
    await asyncio.sleep(0.05)   # 给 _register_and_drain 那次 fire-and-forget drain 机会跑完
    assert not [e for e in events if str(e.type) in ("SessionFinished", "SessionStatusChanged")]
    assert rt._task_managers[sh.session_id].is_alive()


@pytest.mark.parametrize("onto_root", [True, False], ids=["onto_root_agent", "fresh_tree"])
async def test_dispatch_into_a_container_session_runs(onto_root: bool):
    """容器会话的 TM 必须连 runner 一起接好，否则派进来的 task 静默搁浅。"""
    rt, sh, _ = await _container()

    kwargs = ({"agent_id": sh.root_agent_id} if onto_root else
              {"settings": NormalTaskSettings(use_subagent=True,
                                              subagent_template="agent:tpl_echo")})
    h = await rt.dispatch_task(sh.session_id, "first job", unattended=True, **kwargs)
    await h.wait_for_finish(timeout=10.0)

    # 断 task 状态而不是 `wait_for_finish` 的返回值：句柄的 `_state` 只由 runner 回填
    # 给**它构造时拿到的那一个**句柄（会话级），`send_message` / `dispatch_task` 另铸
    # 的句柄恒返回 None——等待本身是真的（判据是事件流上的 task 终态 + recap），
    # 只是拿不到 LoopState。
    task = rt._task_managers[sh.session_id].get_task(h.task_id)
    assert task.status == "FINISHED", "接好了 runner 才跑得起来；没接就会静默搁浅在 PENDING"
    assert (h.agent_id == sh.root_agent_id) is onto_root


async def test_container_session_rejects_duplicate_session_id():
    rt, sh, _ = await _container(session_id="ses_fixed")
    assert sh.session_id == "ses_fixed"
    with pytest.raises(SessionAlreadyExistsError):
        await rt.create_session(template_id="agent:tpl_echo", context_limit=180_000,
                                session_id="ses_fixed")


async def test_container_session_replays_with_a_root_agent_and_no_tasks():
    """重放：root_agent_id 非空（resume 链的假设不破），task 表是空的。"""
    rt, sh, _ = await _container()
    view = await rebuild_view(rt.event_store, sh.session_id)

    assert view.sessions[sh.session_id].root_agent_id == sh.root_agent_id
    assert view.tasks == {}
    assert view.agents[sh.root_agent_id].parent_agent_id is None
    assert view.agents[sh.root_agent_id].spawn_depth == 0
