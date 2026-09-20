"""act 的纯文本暂停与软打断续接改用 UserTurn delivery（段 2 · Task 7）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.hitl.registry import HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.park import HitlPark
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.hitl import (
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    UserTurnDelivery,
)
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


def _act_env(*, interactive: bool = True):
    """搭一套最小的 LoopState/LoopContext + 真实 HitlService，供直接调用 act 的私有 park
    函数。风格与 ``tests/unit/test_gateway_authz_hitl.py`` 的直接构造惯例一致。
    """
    session = SimpleNamespace(id="s1", tenant_id="default", status="RUNNING")
    task = SimpleNamespace(
        id="tsk_1",
        status="ACTIVE",
        interaction_mode="interactive" if interactive else "auto",
        unattended=False,
        settings=NormalTaskSettings(),
        creator_agent_id="agt_1",
        assigned_agent_id="agt_1",
        parent_task_id=None,
    )
    agent = SimpleNamespace(id="agt_1", template_id="tmpl_a", session_id="s1")
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope, resolved_model=SimpleNamespace(model="mock", account=""))

    registry = HitlRegistry()
    bus = InProcessEventBus()
    service = HitlService(
        registry=registry, event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(),
        event_bus=bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1",
        ),
        hitl=service,
    )
    return state, ctx, registry


async def _run_plain_text_pause(state, ctx):
    import ctx_weft.core.loop.steps.act as act

    await act._finish_plain_text_turn(state, ctx, turn_num=1)


async def _run_interrupt(state, ctx, *, edit: bool):
    import ctx_weft.core.loop.steps.act as act

    await act._park_for_interrupt(state, ctx, edit=edit)


async def test_plain_text_pause_opens_a_user_turn_request():
    state, ctx, reg = _act_env(interactive=True)
    with pytest.raises(HitlPark):
        await _run_plain_text_pause(state, ctx)
    req = reg.list_pending()[0]
    assert req.delivery == UserTurnDelivery(task_id=state.task.id, preface=PREFACE_NORMAL)
    assert req.tool_call_id == ""                      # 纯文本暂停没有 tool_call


async def test_interrupt_uses_the_after_interrupt_preface():
    state, ctx, reg = _act_env(interactive=True)
    with pytest.raises(HitlPark):
        await _run_interrupt(state, ctx, edit=False)
    assert reg.list_pending()[0].delivery.preface == PREFACE_AFTER_INTERRUPT


async def test_interrupt_edit_uses_its_own_preface():
    state, ctx, reg = _act_env(interactive=True)
    with pytest.raises(HitlPark):
        await _run_interrupt(state, ctx, edit=True)
    assert reg.list_pending()[0].delivery.preface == PREFACE_AFTER_INTERRUPT_EDIT


async def test_park_leaves_no_wait_slot_so_the_reply_goes_cold():
    """冷 park：不建槽，应答必然走冷续跑（等价于旧的 request_parked）。"""
    state, ctx, reg = _act_env(interactive=True)
    with pytest.raises(HitlPark):
        await _run_plain_text_pause(state, ctx)
    assert reg.list_pending()[0].slot is None


async def test_act_no_longer_references_the_wait_for_user_sentinel():
    import inspect

    import ctx_weft.core.loop.steps.act as mod

    assert "WAIT_FOR_USER_CAPABILITY_ID" not in inspect.getsource(mod)
