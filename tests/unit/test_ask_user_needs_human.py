"""ask_user 改走 needs_human + reply_as_result（段 2 · Task 6）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.control_tools import ControlCapabilityProvider
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import ProviderContext
from ctx_weft.protocols.hitl import HITL_FORM_QUESTION, ToolResultDelivery


def _register_session(provider: ControlCapabilityProvider) -> Session:
    session = Session(id="s1", tenant_id="default", user_prompt="do it", status="RUNNING")
    task = Task(id="tsk_1", session_id="s1", status="ACTIVE", title="T1")
    tm = SimpleNamespace(get_task=lambda tid: task)
    provider.register_session("s1", tm, session)
    return session


def _ctx(tool_call_id: str) -> ProviderContext:
    return ProviderContext(
        session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1",
        extra={"tool_call_id": tool_call_id},
    )


@pytest.mark.asyncio
async def test_provider_no_longer_takes_a_hitl_manager():
    ControlCapabilityProvider()                       # 不再需要任何 HITL 依赖
    assert "hitl_manager" not in ControlCapabilityProvider.__init__.__code__.co_varnames


@pytest.mark.asyncio
async def test_ask_user_yields_a_needs_human_event_with_reply_as_result():
    provider = ControlCapabilityProvider()
    _register_session(provider)
    events = [ev async for ev in provider.invoke(
        "control:ask_user", {"questions": [{"text": "你的名字？"}]}, _ctx("call_1"))]
    kinds = [ev.kind for ev in events]
    assert kinds[-1] == "needs_human"                  # 且是最后一个
    ask = events[-1].payload["ask"]
    assert ask.form == HITL_FORM_QUESTION
    assert ask.reply_as_result is True                 # 答复即结果，不需要 resume
    assert ask.delivery == ToolResultDelivery(tool_call_id="call_1")
    assert ask.fields == [{"text": "你的名字？"}]


@pytest.mark.asyncio
async def test_ask_user_does_not_implement_human_resumable():
    """reply_as_result 的 provider 不必实现重入接口（spec §2.1 的判定表）。"""
    from ctx_weft.protocols.capability import HumanResumable

    assert not isinstance(ControlCapabilityProvider(), HumanResumable)


@pytest.mark.asyncio
async def test_ask_user_no_longer_sets_session_status_itself():
    """会话暂停态由 pending 集合推导，不再由工具函数体直接改（spec §7.1）。"""
    provider = ControlCapabilityProvider()
    session = _register_session(provider)
    before = session.status
    _ = [ev async for ev in provider.invoke(
        "control:ask_user", {"questions": []}, _ctx("call_1"))]
    assert session.status == before
