from __future__ import annotations

import pytest

from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
from ctx_weft.core.models.errors import AgentNotFound, AgentNotRunningError
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import (
    HITL_FORM_QUESTION,
    HITL_FORM_WAIT,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    HitlAsk,
    ToolResultDelivery,
    UserTurnDelivery,
)
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit.test_runtime_agent_api import (
    _plant,
    _plant_live_task,
    _reach_commit_point,
)



def _rt():
    return make_runtime(agent_provider=InlineAgentTemplateProvider())


@pytest.mark.asyncio
async def test_cancel_does_not_terminate_descendant_agents():
    rt = _rt()
    _plant(rt, "root", None)
    _plant(rt, "kid1", "root")
    _plant(rt, "kid2", "root")
    _plant(rt, "grandkid", "kid1")

    await initialize_all(rt)
    killed = await rt.cancel_agent("root", reason="user")
    assert killed == ["root"]
    assert rt._agent_lifecycle_manager.status_of("root") == "terminated"
    for aid in ("kid1", "kid2", "grandkid"):
        assert rt._agent_lifecycle_manager.status_of(aid) == "idle"


@pytest.mark.asyncio
async def test_cancel_emits_termination_only_for_the_target_agent():
    rt = _rt()
    _plant(rt, "root", None)
    _plant(rt, "kid", "root")
    seen = []
    rt._event_bus.subscribe(None, lambda ev: seen.append(ev) or _noop())

    await initialize_all(rt)
    await rt.cancel_agent("root", reason="user")
    term = {e.agent_id: e.payload for e in seen if e.type == EventType.AGENT_TERMINATED}
    assert term["root"]["cascaded_from"] is None
    assert set(term) == {"root"}


async def _noop():
    return None


@pytest.mark.asyncio
async def test_cancel_finalizes_pending_hitl(monkeypatch):
    """waiting_human 的 agent：先终局未决 ask_user，再终态化。"""
    rt = _rt()
    _plant(rt, "a1", None, status="waiting_human")

    canceled: list[str] = []

    def _fake_list(session_id=None):
        class _V:
            id = "h1"
            agent_id = "a1"
            resolved = False
        return [_V()]

    async def _fake_cancel(hitl_id, **_kw):
        canceled.append(hitl_id)

    monkeypatch.setattr(rt, "_list_pending_hitl_memory", _fake_list, raising=False)
    monkeypatch.setattr(rt.hitl, "cancel", _fake_cancel, raising=False)

    await initialize_all(rt)
    await rt.cancel_agent("a1")
    assert canceled == ["h1"]


@pytest.mark.asyncio
async def test_cancel_unknown_agent_is_noop():
    rt = _rt()
    await initialize_all(rt)
    assert await rt.cancel_agent("ghost") == []


@pytest.mark.asyncio
async def test_cancel_only_finalizes_hitl_of_target_agent(monkeypatch):
    """同 session 下 a1、a2 各自都有未决 HITL；只 cancel a1 时，a2 的未决提问不受影响。"""
    rt = _rt()
    _plant(rt, "a1", None, status="waiting_human", session_id="s1")
    _plant(rt, "a2", None, status="waiting_human", session_id="s1")

    class _V:
        def __init__(self, id_, agent_id):
            self.id = id_
            self.agent_id = agent_id
            self.resolved = False

    pending = [_V("h1", "a1"), _V("h2", "a2")]
    canceled: list[str] = []

    def _fake_list(session_id=None):
        return list(pending)

    async def _fake_cancel(hitl_id, **_kw):
        canceled.append(hitl_id)

    monkeypatch.setattr(rt, "_list_pending_hitl_memory", _fake_list, raising=False)
    monkeypatch.setattr(rt.hitl, "cancel", _fake_cancel, raising=False)

    await initialize_all(rt)
    await rt.cancel_agent("a1")

    assert canceled == ["h1"]
    assert rt._agent_lifecycle_manager.status_of("a1") == "terminated"
    assert rt._agent_lifecycle_manager.status_of("a2") == "waiting_human"


@pytest.mark.asyncio
async def test_cancel_finalizes_hitl_before_agent_terminated_event(monkeypatch):
    """事件流顺序：未决 HITL 的终局必须先于该 agent 的 AgentTerminated。"""
    rt = _rt()
    _plant(rt, "a1", None, status="waiting_human")

    order: list[str] = []

    def _fake_list(session_id=None):
        class _V:
            id = "h1"
            agent_id = "a1"
            resolved = False
        return [_V()]

    async def _fake_cancel(hitl_id, **_kw):
        order.append(f"hitl_canceled:{hitl_id}")

    def _on_event(ev):
        if ev.type == EventType.AGENT_TERMINATED:
            order.append(f"agent_terminated:{ev.agent_id}")
        return _noop()

    monkeypatch.setattr(rt, "_list_pending_hitl_memory", _fake_list, raising=False)
    monkeypatch.setattr(rt.hitl, "cancel", _fake_cancel, raising=False)
    rt._event_bus.subscribe(None, _on_event)

    await initialize_all(rt)
    await rt.cancel_agent("a1")

    assert order == ["hitl_canceled:h1", "agent_terminated:a1"]


# ── pause_agent（Task 20, R22：建在既有的 _pause_task 之上）──────────────────────


@pytest.mark.asyncio
async def test_pause_agent_signals_running_descendants_only():
    """running 的目标 + running 的子孙都收到暂停信号；idle 的子孙原样不动。

    暂停是异步生效的（真正落 waiting_human 要等各自的 run 跑到检查点）——这里只
    断言"信号已经递送到对应的 run token"（`_pause_task` 命中），不断言状态已经翻转。
    """
    rt = _rt()
    _plant(rt, "root", None, status="running", session_id="s1")
    _plant(rt, "busy_kid", "root", status="running", session_id="s1")
    _plant(rt, "idle_kid", "root", status="idle", session_id="s1")
    rt._agent_lifecycle_manager._agents["root"].current_task_id = "t_root"
    rt._agent_lifecycle_manager._agents["busy_kid"].current_task_id = "t_kid"
    root_tokens = rt._register_run_tokens("s1", "t_root")
    kid_tokens = rt._register_run_tokens("s1", "t_kid")

    await initialize_all(rt)
    paused = await rt.pause_agent("root")

    assert set(paused) == {"root", "busy_kid"}, "idle 的子孙不动"
    assert root_tokens.pause.is_paused is True
    assert kid_tokens.pause.is_paused is True
    # 信号只是递送了，状态转移要等真实的 TASK_AWAITING_HUMAN 事件（R24）——此刻仍是 running。
    assert rt._agent_lifecycle_manager.status_of("root") == "running"
    assert rt._agent_lifecycle_manager.status_of("idle_kid") == "idle"


@pytest.mark.asyncio
async def test_pause_agent_skips_running_agent_with_no_live_run_token():
    """running 但没有在册 run token（race / 陈旧 record）——`_pause_task` 命不中,不计入返回值。"""
    rt = _rt()
    _plant(rt, "root", None, status="running", session_id="s1")
    rt._agent_lifecycle_manager._agents["root"].current_task_id = "t_root"
    # 故意不注册 run token

    await initialize_all(rt)
    paused = await rt.pause_agent("root")

    assert paused == []


@pytest.mark.asyncio
async def test_pause_agent_non_running_raises():
    """spec 7：只对 running 生效，其余状态直接报错。"""
    rt = _rt()
    _plant(rt, "a1", None, status="idle")
    with pytest.raises(AgentNotRunningError):
        await initialize_all(rt)
        await rt.pause_agent("a1")


@pytest.mark.asyncio
async def test_pause_agent_unknown_raises():
    rt = _rt()
    with pytest.raises(AgentNotFound):
        await initialize_all(rt)
        await rt.pause_agent("ghost")


# ── resume_agent（Task 20, R24：只解析暂停产生的气泡，不误答 ask_user）───────────


def _open_pause_bubble(rt, *, agent_id, task_id, session_id="s1", edit=False):
    """与 `act._park_for_interrupt(...)` 完全同形的构造
    （act.py 656-671 行：`form=HITL_FORM_WAIT`，`delivery=UserTurnDelivery(task_id=...,
    preface=PREFACE_AFTER_INTERRUPT[_EDIT])`）——`pause_agent` 递送信号后，run 在
    检查点自己 park 出的正是这一种。"""
    preface = PREFACE_AFTER_INTERRUPT_EDIT if edit else PREFACE_AFTER_INTERRUPT
    return rt.hitl.open(
        HitlAsk(form=HITL_FORM_WAIT, delivery=UserTurnDelivery(task_id=task_id, preface=preface)),
        session_id=session_id, task_id=task_id, agent_id=agent_id, stage=HITL_STAGE_TOOL, unattended=False,
    )


def _open_ask_user_bubble(rt, *, agent_id, task_id, session_id="s1", tool_call_id="tc1"):
    """与 `control_capability.py` 的 `ask_user` 构造完全同形：`form=HITL_FORM_QUESTION`，
    `delivery=ToolResultDelivery(...)`，`reply_as_result=True`。"""
    return rt.hitl.open(
        HitlAsk(form=HITL_FORM_QUESTION, delivery=ToolResultDelivery(tool_call_id=tool_call_id),
                reply_as_result=True),
        session_id=session_id, task_id=task_id, agent_id=agent_id, stage=HITL_STAGE_TOOL, unattended=False,
    )


def _open_plain_text_wait_bubble(rt, *, agent_id, task_id, session_id="s1"):
    """与 `act._park_await_user`（纯文本让位）完全同形：同为
    `UserTurnDelivery`，但 `preface=PREFACE_NORMAL`——不是暂停产生的，是正常一轮
    说完话后的自然等待。"""
    return rt.hitl.open(
        HitlAsk(form=HITL_FORM_WAIT,
                delivery=UserTurnDelivery(task_id=task_id, preface=PREFACE_NORMAL)),
        session_id=session_id, task_id=task_id, agent_id=agent_id, stage=HITL_STAGE_TOOL, unattended=False,
    )


@pytest.mark.asyncio
async def test_resume_agent_resolves_the_pause_bubble_via_cold_resume(monkeypatch):
    """resume_agent 命中暂停气泡：真走 `reply_to_hitl` → `hitl.resolve`（真实组件）
    → 未被热投递消费 → `_resume_after_hitl` → `recover_agent`（这里桩掉，只记调用
    参数——它自己的行为由别处的 recover_agent 测试覆盖，不是本测试要盯的东西）。"""
    rt = _rt()
    _plant(rt, "root", None, status="waiting_human", session_id="s1")
    recovered: list[tuple] = []

    async def _fake_recover_agent(agent_id, **kw):
        recovered.append((agent_id, kw))

    monkeypatch.setattr(rt, "_recover_session_locked", _fake_recover_agent)
    req = await _open_pause_bubble(rt, agent_id="root", task_id="t_root")

    await initialize_all(rt)
    resumed = await rt.resume_agent("root")

    assert resumed == ["root"]
    assert rt.hitl_registry.get(req.id).resolved is True
    assert recovered and recovered[0][0] == "s1"
    assert recovered[0][1]["hitl_id"] == req.id
    assert recovered[0][1]["resumed_task_id"] == "t_root"


@pytest.mark.asyncio
async def test_resume_agent_does_not_answer_a_real_ask_user_question(monkeypatch):
    """R24 的核心止损点：ask_user 的真实提问必须原样悬着，不能被 resume_agent 顺手答了。"""
    rt = _rt()
    _plant(rt, "root", None, status="waiting_human", session_id="s1")
    monkeypatch.setattr(rt, "_recover_session_locked", _unreachable_recover_agent)
    req = await _open_ask_user_bubble(rt, agent_id="root", task_id="t_root")

    await initialize_all(rt)
    resumed = await rt.resume_agent("root")

    assert resumed == []
    assert rt.hitl_registry.get(req.id).resolved is False


@pytest.mark.asyncio
async def test_resume_agent_does_not_touch_plain_text_wait_bubble(monkeypatch):
    """同为 UserTurnDelivery 的软待命（正常一轮结束后的自然等待，非暂停产生）不该被续跑。"""
    rt = _rt()
    _plant(rt, "root", None, status="waiting_human", session_id="s1")
    monkeypatch.setattr(rt, "_recover_session_locked", _unreachable_recover_agent)
    req = await _open_plain_text_wait_bubble(rt, agent_id="root", task_id="t_root")

    await initialize_all(rt)
    resumed = await rt.resume_agent("root")

    assert resumed == []
    assert rt.hitl_registry.get(req.id).resolved is False


async def _unreachable_recover_agent(*_a, **_kw):
    raise AssertionError("resume_agent 不该对这种气泡触发冷续跑")


@pytest.mark.asyncio
async def test_resume_agent_cascades_to_waiting_human_descendants_only(monkeypatch):
    rt = _rt()
    _plant(rt, "root", None, status="waiting_human", session_id="s1")
    _plant(rt, "kid", "root", status="waiting_human", session_id="s1")
    _plant(rt, "other", "root", status="idle", session_id="s1")

    async def _fake_recover_agent(agent_id, **kw):
        return None

    monkeypatch.setattr(rt, "_recover_session_locked", _fake_recover_agent)
    r1 = await _open_pause_bubble(rt, agent_id="root", task_id="t_root")
    r2 = await _open_pause_bubble(rt, agent_id="kid", task_id="t_kid", edit=True)

    await initialize_all(rt)
    resumed = await rt.resume_agent("root")

    assert set(resumed) == {"root", "kid"}
    assert rt._agent_lifecycle_manager.status_of("other") == "idle"
    assert rt.hitl_registry.get(r1.id).resolved is True
    assert rt.hitl_registry.get(r2.id).resolved is True


@pytest.mark.asyncio
async def test_resume_agent_unknown_raises():
    rt = _rt()
    with pytest.raises(AgentNotFound):
        await initialize_all(rt)
        await rt.resume_agent("ghost")


# ── send_message 收口未决 HITL（最终审查修复波 #1）───────────────────────────
#
# 缺口：`send_message` 对 `waiting_human` 的 agent 放行（`assert_can_receive`
# 允许），但 `_inject_user_turn` 全程不碰 HITL registry——留下一条永久孤儿化的
# 未决提问：`list_pending_hitl` 一直挂着它；重启后 `rebuild_hitl` 会把它当未决
# 复活；`resume_agent` 的 `_pause_bubble_of` 还可能命中它、误放行冷续跑（见下面
# 的组合回归）。


@pytest.mark.asyncio
async def test_send_message_finalizes_all_pending_hitl_of_target_agent():
    """钉住核心修复：`send_message` 之后，该 agent 名下不该再有任何未决 HITL。"""
    rt = _rt()
    _plant_live_task(rt, "a1", "t1", task_status="AWAITING_HUMAN", agent_status="waiting_human")
    req = await _open_ask_user_bubble(rt, agent_id="a1", task_id="t1")

    await initialize_all(rt)
    tid = (await rt.send_message("a1", "please continue without answering that")).task_id

    assert tid == "t1"
    # 两阶段（spec 2026-09-09）：收口先落成「待终局」——对调用方而言这个气泡已经不是
    # 未决的（`list_pending` 立刻就不列它了），只是要等这一轮的 LLM 真的开口才终局。
    assert rt.hitl_registry.get(req.id).claim_pending is True
    await initialize_all(rt)
    assert [v for v in [item.data for item in await rt.list_pending_hitl(session_id="s1")] if v.agent_id == "a1"] == []

    await _reach_commit_point(rt)
    assert rt.hitl_registry.get(req.id).resolved is True
    await initialize_all(rt)
    assert [v for v in [item.data for item in await rt.list_pending_hitl(session_id="s1")] if v.agent_id == "a1"] == []


@pytest.mark.asyncio
async def test_send_message_does_not_touch_other_agents_pending_hitl():
    """不误伤：同 session 下另一个 agent 的未决提问不受影响（按 agent_id 过滤）。"""
    rt = _rt()
    _plant_live_task(rt, "a1", "t1", task_status="AWAITING_HUMAN", agent_status="waiting_human")
    _plant(rt, "a2", None, status="waiting_human", session_id="s1")
    req_a1 = await _open_ask_user_bubble(rt, agent_id="a1", task_id="t1")
    req_a2 = await _open_ask_user_bubble(rt, agent_id="a2", task_id="t2")

    await initialize_all(rt)
    await rt.send_message("a1", "hello")

    await _reach_commit_point(rt)
    assert rt.hitl_registry.get(req_a1.id).resolved is True
    # 不误伤：另一个 agent 的提问既没被终局、也没被拉进这一轮的待终局。
    assert rt.hitl_registry.get(req_a2.id).resolved is False
    assert rt.hitl_registry.get(req_a2.id).claim_pending is False


@pytest.mark.asyncio
async def test_send_message_resolves_stale_pause_bubble_before_new_real_question_arrives(
    monkeypatch,
):
    """组合路径回归（本次修复的核心危害场景）：`pause_agent` 产生的暂停气泡若不被
    `send_message` 收口，之后一次真实 `ask_user` 再把该 agent 落 `waiting_human`
    时，`resume_agent` 的 `_pause_bubble_of` 会先命中那条陈旧气泡、误放行一次冷
    续跑——用一句陈旧的"继续吧"回复顶替了用户还没来得及回答的真问题，正是 R24
    专门设的止损点（"真问题悬而未决时不能替用户放行"）要防的场景。

    没有本次修复时：第 3 步之后 `stale_bubble` 仍未终局，第 5 步 `resume_agent`
    会命中它、触发 `recover_agent`（这里桩成必炸），断言失败，复现该缺口。
    """
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    rt = _rt()
    rt.providers.register_memory(InMemoryMemoryProvider())
    _plant(rt, "root", None, status="running", session_id="s1")
    rt._agent_lifecycle_manager._agents["root"].current_task_id = "t_root"
    root_tokens = rt._register_run_tokens("s1", "t_root")

    tm = TaskManager(session_id="s1", event_bus=rt._event_bus)
    session = Session(
        id="s1", user_prompt="hi",  tenant_id="default",
        root_agent_id="root", created_at=None,
    )
    tm.set_session(session)
    task = Task(
        id="t_root", session_id="s1", status="RUNNING", tenant_id="default",
        assigned_agent_id="root", creator_agent_id="root",
    )
    tm.register_task(task)
    rt._task_managers["s1"] = tm

    async def _noop_drain():
        return None

    tm.drain = _noop_drain  # type: ignore[method-assign]

    # 1. 真实暂停信号递送（root 此刻仍是 running，暂停异步生效）。
    await initialize_all(rt)
    paused = await rt.pause_agent("root")
    assert paused == ["root"]
    assert root_tokens.pause.is_paused is True

    # 2. 模拟 run 到下一个检查点真正 park（`act._park_for_interrupt(...)` 的产物）
    #    ——与 test_resume_agent_* 系列同一构造，不跑真实 run loop。
    stale_bubble = await _open_pause_bubble(rt, agent_id="root", task_id="t_root")
    task.status = "AWAITING_HUMAN"
    rt._agent_lifecycle_manager._agents["root"].status = "waiting_human"

    # 3. 用户此刻改口，发了条新消息（没有专门回答那条暂停气泡）——这正是本次修复
    #    要收口的缺口：send_message 必须把这条陈旧气泡终局掉。
    await initialize_all(rt)
    tid = (await rt.send_message("root", "actually let's change the plan")).task_id
    assert tid == "t_root"
    # 两阶段（spec 2026-09-09）：收口先落成待终局。**本用例要防的危害与终不终局无关**
    # ——它防的是「这条陈旧气泡还挂在 pending 列表里被 `_pause_bubble_of` 命中」，而
    # 待终局的请求同样已经不在那份列表里了（`list_pending` 排除 `claim_pending`）。
    assert rt.hitl_registry.get(stale_bubble.id).claim_pending is True, (
        "send_message 必须收口这条陈旧暂停气泡；否则它会一直挂在 pending 列表里"
    )
    await initialize_all(rt)
    assert stale_bubble.id not in [v.id for v in [item.data for item in await rt.list_pending_hitl(session_id="s1")]]
    await _reach_commit_point(rt, task_id="t_root")
    assert rt.hitl_registry.get(stale_bubble.id).resolved is True

    # 4. task 被重排、agent 回 idle（真正的 running 要等 drain 派发——这里 drain
    #    是 no-op），之后模拟它再跑一轮、真的问了一个新问题（真实 ask_user），再次
    #    落 waiting_human。
    assert rt._agent_lifecycle_manager.status_of("root") == "idle"
    real_question = await _open_ask_user_bubble(rt, agent_id="root", task_id="t_root")
    task.status = "AWAITING_HUMAN"
    rt._agent_lifecycle_manager._agents["root"].status = "waiting_human"

    # 5. 核心断言：resume_agent 不能被那条早该终局的陈旧气泡误导去冷续跑——陈旧
    #    气泡已经被第 3 步收口，此刻 pending 列表里只有真问题（ToolResultDelivery），
    #    `_pause_bubble_of` 的类型过滤天然不会命中它。
    monkeypatch.setattr(rt, "_recover_session_locked", _unreachable_recover_agent)
    await initialize_all(rt)
    resumed = await rt.resume_agent("root")

    assert resumed == [], "不该有任何气泡被当成暂停续跑气泡放行"
    assert rt.hitl_registry.get(real_question.id).resolved is False, (
        "真实 ask_user 问题必须原样悬着，不能被 resume_agent 顺手答了"
    )
