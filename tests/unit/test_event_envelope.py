"""core.util：全仓事件封套的唯一构造点 + 唯一一份白名单校验。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctx_weft.core.utils.event import emit_event, new_event
from ctx_weft.protocols.events import EventOrigin, EventType


class _Bus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)


def test_defaults_are_the_no_run_context_shape():
    """run_id=None / sequence=0 是 session/agent/task 级事实的定义特征。"""
    ev = new_event(
        EventType.TASK_CREATED, session_id="ses_1", tenant_id="acme",
        origin=EventOrigin.ORCHESTRATOR_TASK_MANAGER,
    )
    assert ev.run_id is None and ev.sequence == 0
    assert ev.payload == {} and ev.metadata == {}
    assert ev.id.startswith("evt")


def test_run_scoped_fields_passthrough():
    """make_event 委托本函数时显式传 run_id/sequence。"""
    ev = new_event(
        EventType.STEP_STARTED, session_id="s", tenant_id="default",
        origin=EventOrigin.LOOP_DRIVER, run_id="run_1", sequence=7,
        task_id="tsk_1", agent_id="agt_1", metadata={"m": 1}, causation_id="cau_1",
    )
    assert (ev.run_id, ev.sequence) == ("run_1", 7)
    assert ev.task_id == "tsk_1" and ev.agent_id == "agt_1"
    assert ev.metadata == {"m": 1} and ev.causation_id == "cau_1"


def test_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="Unknown event type"):
        new_event("NotARealEvent", session_id="s", tenant_id="d", origin="x")


def test_explicit_timestamp_wins():
    """HitlService 的可注入时钟依赖这一点。"""
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ev = new_event(EventType.SESSION_CREATED, session_id="s", tenant_id="d",
                   origin=EventOrigin.ORCHESTRATOR_SESSION_REGISTRY, timestamp=ts)
    assert ev.timestamp == ts


async def test_emit_event_puts_it_on_the_bus():
    bus = _Bus()
    await emit_event(bus, EventType.TASK_RESUMED, session_id="s", tenant_id="d",
                     origin=EventOrigin.ORCHESTRATOR_TASK_MANAGER, task_id="t")
    (ev,) = bus.events
    assert ev.type == EventType.TASK_RESUMED and ev.task_id == "t"


async def test_none_bus_is_noop():
    """TaskManager 的既有语义：event_bus 未注入时静默跳过（大量单测依赖）。"""
    await emit_event(None, EventType.TASK_CREATED, session_id="s",
                     tenant_id="d", origin=EventOrigin.ORCHESTRATOR_TASK_MANAGER)


async def test_validation_precedes_emission():
    """坏类型不该先落一半再报错。"""
    bus = _Bus()
    with pytest.raises(ValueError):
        await emit_event(bus, "Nope", session_id="s", tenant_id="d", origin="x")
    assert bus.events == []


def test_make_event_does_not_bump_counter_on_bad_type():
    """行为等价守卫：`llm_gateway` / `act` 在 emit 前读 sequence_counter 拼
    request_id，一次校验失败不该跳号。make_event 因此「先算后提交」。"""
    from ctx_weft.core.loop.driver import make_event

    class _S:
        sequence_counter = 3
        run_id = "run_1"
        origin = ""
        session = type("X", (), {"id": "s", "tenant_id": "d"})()
        task = type("X", (), {"id": "t"})()
        agent = type("X", (), {"id": "a"})()

    st = _S()
    with pytest.raises(ValueError):
        make_event(st, "NotARealEvent")
    assert st.sequence_counter == 3

    ev = make_event(st, EventType.STEP_STARTED)
    assert st.sequence_counter == 4 and ev.sequence == 4


@pytest.mark.parametrize("kind", list(EventType))
def test_all_event_types_capture_opaque_metadata(kind):
    metadata = {"host": {"values": [None, False, 0, "中文"]}}
    event = new_event(kind, session_id="s", tenant_id="d", origin="test", metadata=metadata)
    metadata["host"]["values"].append("mutated")
    assert event.metadata == {"host": {"values": [None, False, 0, "中文"]}}
    assert event.payload == {}


@pytest.mark.parametrize("metadata", [{"x": object()}, {"x": float("nan")}, {"x": {1: "bad key"}}])
def test_event_metadata_must_be_json_data(metadata):
    with pytest.raises((TypeError, ValueError)):
        new_event(EventType.RUN_STARTED, session_id="s", tenant_id="d", origin="test", metadata=metadata)


async def test_session_and_hitl_emitters_forward_metadata():
    from types import SimpleNamespace as NS
    from ctx_weft.core.hitl.service import HitlService
    from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry

    bus = _Bus()
    extensions = {"host": {"trace": ["x"]}}
    await SessionRegistry._emit(NS(event_bus=bus), EventType.SESSION_RESUMED,
                               "s", "d", {}, metadata=extensions)
    service = NS(_now=lambda: datetime.now(timezone.utc))
    request = NS(session_id="s", tenant_id="d", task_id="t", agent_id="a")
    hitl = HitlService._make_event(service, EventType.HITL_CLOSED, request,
                                   {"hitl_id": "h"}, metadata=extensions)
    extensions["host"]["trace"].clear()
    assert bus.events[0].metadata == hitl.metadata == {"host": {"trace": ["x"]}}


async def test_task_emitter_keeps_original_metadata_while_retrying():
    from types import SimpleNamespace as NS
    from unittest.mock import AsyncMock
    from ctx_weft.core.orchestrator.task.manager import TaskManager

    owner = NS(_session_id="s", _session=NS(tenant_id="d"), _pending_emissions={},
               _agent_id_of=lambda tid: "a", _publish_emission=AsyncMock())
    extensions = {"host": {"trace": ["original"]}}
    first = await TaskManager._emit(owner, EventType.TASK_STARTED, "t", {}, metadata=extensions)
    extensions["host"]["trace"].clear()
    retried = await TaskManager._emit(owner, EventType.TASK_STARTED, "t", {}, metadata={"host": "changed"})
    assert retried is first
    assert retried.metadata == {"host": {"trace": ["original"]}}
