"""SessionRegistry 降格为会话内 agent 登记表（Task 15）。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry, _SessionState
from ctx_weft.protocols.events import Event, EventType

pytestmark = pytest.mark.asyncio


class _SpyBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)

    def subscribe(self, _flt, _handler, *, provisional: bool = False,
                  required: bool = False) -> None:
        pass


def _sm() -> SessionRegistry:
    return SessionRegistry(agent_lifecycle_manager=None, event_bus=_SpyBus())


def _agent_ev(t: str, agent_id: str, payload: dict | None = None) -> Event:
    return Event(
        id="evt_x", run_id=None, sequence=0, session_id="s1", type=t,
        timestamp=datetime.now(UTC), agent_id=agent_id, payload=payload or {},
    )


def test_session_state_has_no_status_field():
    """状态整体挪到 agent 身上（spec 2）。"""
    st = _SessionState()
    assert not hasattr(st, "status")
    assert st.agent_ids == set()


async def test_agent_instantiated_joins_member_set():
    sm = _sm()
    sm.register_session("s1")
    await sm.handle_event(_agent_ev(EventType.AGENT_INSTANTIATED, "root", {"template_id": "t"}))
    assert sm.agent_ids_of("s1") == {"root"}


async def test_agent_spawned_joins_member_set():
    sm = _sm()
    sm.register_session("s1")
    await sm.handle_event(_agent_ev(EventType.AGENT_INSTANTIATED, "root", {}))
    await sm.handle_event(
        _agent_ev(EventType.AGENT_SPAWNED, "kid", {"parent_agent_id": "root"})
    )
    assert sm.agent_ids_of("s1") == {"root", "kid"}


async def test_session_registry_no_longer_consumes_queue_signals():
    """三条 TaskQueue* 原是 SM 唯一输入，现在不再消费。

    2026-09-04（Task 12）起发射点已删除，2026-09-05 起枚举成员也删了。这里用**裸
    字符串**构造事件喂给 `handle_event`，验证 `_MEMBER_EVENTS` 白名单是按「在不在册」
    挡的——任何不认识的 `type` 都进不来，不依赖这三个名字还在不在枚举里。
    """
    sm = _sm()
    sm.register_session("s1")
    for t in (
        "TaskQueueBlocked",
        "TaskQueueInterrupted",
        "TaskQueueDrained",
        EventType.TASK_STARTED,
    ):
        await sm.handle_event(_agent_ev(t, "root", {"count": 1}))
    assert sm.event_bus.events == [], "SM 不应再因队列信号发任何事件"


def test_input_by_event_table_is_gone():
    assert not hasattr(SessionRegistry, "_INPUT_BY_EVENT")


async def test_attach_to_bus_registers_one_handler():
    """迁自 `test_session_registry_inputs.py`（Task 16，随会话状态机一并退役该文件）。

    `_SpyBus.subscribe` 是空实现，验证不了订阅是否真的发生——换成会记录 handler 的
    `RecordingBus`，直接断言 `attach_to_bus()` 确实调用了一次 `subscribe`。这条与
    `AgentLifecycleManager` 的同名测试（`test_agent_lifecycle.py`）互为对等先例，各自钉住
    一个组件的接线，不是重复覆盖。
    """
    from tests.unit._session_helpers import RecordingBus

    bus = RecordingBus()
    SessionRegistry(agent_lifecycle_manager=None, event_bus=bus).attach_to_bus()
    assert len(bus.handlers) == 1
