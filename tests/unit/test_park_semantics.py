"""park 语义拆分：「人按了暂停键」与「agent 想让位」是两件事。

前者有人保证会回来续接——运维暂停一个后台无人值守作业完全合法，那时 park 正是对的
行为，所以走 `_park_for_interrupt`（对无人值守守卫**豁免**）。后者没有任何人保证会
发下一条消息，所以走 `_park_await_user`：判断前置于一切副作用，无人值守时零副作用
返回，让调用方照常发它那条 `stop`。

本文件钉的核心事实：无人值守的纯文本回合**只发一条** `ACT_TURN_COMPLETED`。改动前
`await_user` 事件与 background observe 已经发出去了才在 `hitl.open()` 里撞上守卫，
同一回合会出现「在等用户」+「已停止」两条收尾——前端看到的是自相矛盾的一对。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

import inspect
from types import SimpleNamespace

import pytest

import ctx_weft.core.loop.steps.act as act
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import (
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    UserTurnDelivery,
)
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs
from tests.hitl_env import make_hitl
from tests.step_helpers import patch_observe_start



class _RecordingBus:
    """只记不派发的 bus：本文件断言的是「发了哪几条」，不需要真的扇出给订阅者。"""

    def __init__(self) -> None:
        self.events: list = []

    async def emit_batch(self, events, *, on_committed=None):
        if on_committed is not None:
            on_committed()
        for event in events:
            await self.emit(event)

    async def emit(self, event) -> None:
        self.events.append(event)


def _env(*, unattended: bool, llm=None):
    """最小 act 环境：真 Task / 真 HitlService（守卫住在 `open()` 里，必须是真的）。

    只给 `unattended`：纯文本回合的归宿判据 2026-09-22 起只剩它一个（`interaction_mode`
    已删）。设置点的不变式（曾经的 unattended ⟹ auto）在
    这里**故意不施加**，好让测试能手工摆出「无人值守却 interactive」这个越过不变式的
    组合——那正是本次改动要堵的口子。
    """
    bus = _RecordingBus()
    hitl_service, registry = make_hitl(bus)
    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = Task(
        id="t1", session_id="s1", status="ACTIVE",
        title="Greet", description="say hi",
        unattended=unattended,
        settings=NormalTaskSettings(),
    )
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    state = LoopState(
        run_id="r1", session=session, task=task, agent=agent, scope=scope,
        resolved_model=model_snapshot(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=llm, memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1"),
        hitl=hitl_service,
    )
    return state, ctx, task, registry, bus


def _spy_observe(monkeypatch) -> list[str]:
    """拦下 background observe：act 是在函数体内 import 的，patch 模块属性即可命中。"""
    import asyncio


    launched: list[str] = []

    def _fake(state, ctx, *, boundary=""):
        launched.append(boundary)
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, _fake)
    return launched


def _completed(bus) -> list[str]:
    return [e.payload.get("reason") for e in bus.events
            if e.type == EventType.ACT_TURN_COMPLETED]


# ── ① 无人值守 + 纯文本回合：零副作用，且只发一条 stop ──────────────────────

@pytest.mark.asyncio
async def test_unattended_plain_text_turn_emits_exactly_one_completed_event(monkeypatch):
    launched = _spy_observe(monkeypatch)
    state, ctx, task, reg, bus = _env(unattended=True)

    await act._finish_plain_text_turn(state, ctx, turn_num=1, transcript=[])   # 不抛 HitlPark

    assert _completed(bus) == ["stop"], (
        "无人值守回合必须只有一条收尾事件；出现 await_user 说明判断没有前置到副作用之前")
    assert launched == [], "不让位就不该折叠：background observe 一次也不该起"
    assert reg.list_pending() == [], "不让位就不该有任何等人回答的登记"


@pytest.mark.asyncio
async def test_unattended_park_await_user_returns_with_no_side_effects(monkeypatch):
    """函数级：`_park_await_user` 自己就是零副作用返回，不靠调用方兜底。"""
    launched = _spy_observe(monkeypatch)
    state, ctx, task, reg, bus = _env(unattended=True)

    assert await act._park_await_user(state, ctx, turn_num=1) is None
    assert bus.events == []
    assert launched == []
    assert reg.list_pending() == []


# ── ② 有人值守 + 纯文本回合：照常让位 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_attended_plain_text_turn_parks_and_never_reaches_stop(monkeypatch):
    launched = _spy_observe(monkeypatch)
    state, ctx, task, reg, bus = _env(unattended=False)

    with pytest.raises(HitlPark):
        await act._finish_plain_text_turn(state, ctx, turn_num=1, transcript=[])

    assert _completed(bus) == ["await_user"], "让位则 stop 那条根本到不了"
    assert launched == ["plain_text"]
    pend = reg.list_pending()
    assert len(pend) == 1
    assert pend[0].delivery == UserTurnDelivery(task_id="t1", preface=PREFACE_NORMAL)


# ── ③ 无人值守 + 人按暂停：豁免生效，照常 park ─────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("edit", "preface"),
    [(False, PREFACE_AFTER_INTERRUPT), (True, PREFACE_AFTER_INTERRUPT_EDIT)],
)
async def test_unattended_interrupt_still_parks(edit, preface):
    """守卫挡的是「没有人可问」，不是「没有人在场」：按下暂停键的就是一个人。"""
    state, ctx, task, reg, bus = _env(unattended=True)

    with pytest.raises(HitlPark):
        await act._park_for_interrupt(state, ctx, edit=edit)

    pend = reg.list_pending()
    assert len(pend) == 1
    assert pend[0].delivery == UserTurnDelivery(task_id="t1", preface=preface)


# ── ④ 手工越过设置点不变式：unattended=True 且 interactive ─────────────────

@pytest.mark.asyncio
async def test_unattended_interactive_task_runs_to_stop_end_to_end(monkeypatch):
    """整条 act 跑下来不崩、不 park——上一个提交遗留的那个口子。

    设置点强制 `unattended ⟹ auto`，但那是设置点的不变式，不是类型系统的；这里手工
    摆出被绕过的组合，验证 act 自己也站得住：`UnattendedHitl` 不会逸出打挂整个 run。
    """
    launched = _spy_observe(monkeypatch)
    llm = MockLLMAdapter(responses=[MockResponse(text="Hi! Anything else?")])
    state, ctx, task, reg, bus = _env(
        unattended=True, llm=llm)
    from ctx_weft.core.assembler.assembler import AssembledPrompt
    from ctx_weft.protocols import LLMMessage
    state.prompt_inputs = scripted_inputs(AssembledPrompt(
        system="", messages=[LLMMessage(role="user", content="hi")], tools=[], token_count=1))

    outcome = await act.ActStep().execute(state, ctx)

    assert outcome.next_step == "observe"
    assert task.outputs == "Hi! Anything else?"   # 不让位 → 这段文本就是本轮产出
    assert _completed(bus) == ["stop"]
    assert launched == []
    assert reg.list_pending() == []


# ── ⑤ 旧的双语义函数与 source 开关已消失 ───────────────────────────────────

def test_the_two_semantics_no_longer_share_one_function():
    src = inspect.getsource(act)
    assert "_park_wait_for_user" not in src, "双语义的旧函数应已拆掉"
    assert 'source="interrupt"' not in src and 'source="plain_text"' not in src, (
        "`source` 这个既选 preface 又管豁免的字符串开关应已消失")
    # 四个 interrupt 调用点 + 两个让位调用点（纯文本 / finish_task），各自走具名函数。
    # 让位从 1 个变 2 个：S-b（2026-09-27）把 `actor_done` 那条出口也并进了让位分支——
    # root 上「说了段话」与「宣布做完了」不该一个被复核、一个不被。
    assert src.count("_park_for_interrupt(state, ctx") == 4
    # 让位那两个按**调用点**数，不按 "(state, ctx" 这个字面数：`finish_task` 那一处是多行写法
    # （轮号取 `state.extra[ACT_TURNS_USED_KEY]`，不借 for 循环泄漏出来的变量）。
    assert src.count("_park_await_user(") - src.count("async def _park_await_user(") == 1
