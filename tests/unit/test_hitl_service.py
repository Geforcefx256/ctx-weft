"""HitlService：唯一漏斗——open / resolve / cancel，只发事实、不认识 Runtime。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ as STAGE_AUTHZ
from ctx_weft.core.hitl.registry import HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import (
    HitlAsk,
    HitlDecision,
    HitlReply,
    ToolResultDelivery,
    UserTurnDelivery,
)

T0 = datetime(2026, 9, 1, tzinfo=UTC)


class RecordingBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def payload_of(self, event_type: str) -> dict:
        return next(e.payload for e in self.events if e.type == event_type)


class PassthroughNormalizer:
    """内容原样透传的测试替身；记录被调用次数。"""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_tenants: list[str] = []

    async def __call__(self, content, session_id, tenant_id):
        self.calls += 1
        self.seen_tenants.append(tenant_id)
        return content, content


class RejectingNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        raise ValueError("unsupported media type")


class FakeSlot:
    def __init__(self, accepts: bool = True) -> None:
        self.accepts = accepts
        self.delivered = None

    def deliver(self, decision: HitlDecision) -> bool:
        self.delivered = decision
        return self.accepts


class RaisingSlot:
    """模拟一个已经完成的 future 再被 deliver：InvalidStateError 之类。"""

    def deliver(self, decision: HitlDecision) -> bool:
        raise RuntimeError("future already done")


def _service(bus: RecordingBus, normalizer=None) -> HitlService:
    ids = iter(f"hit_{i}" for i in range(1, 100))
    return HitlService(
        registry=HitlRegistry(),
        event_bus=bus,
        reply_intake=ReplyIntake(normalizer or PassthroughNormalizer()),
        id_factory=lambda: next(ids),
        clock=lambda: T0,
    )


def _ask(tool_call_id: str = "call_1") -> HitlAsk:
    return HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id=tool_call_id),
                   prompt="Allow bash?", subject_id="fs:bash_exec",
                   proposal={"command": "ls"}, resume_state={"plan": "p1"})


async def test_open_emits_exactly_one_hitl_opened():
    bus = RecordingBus()
    svc = _service(bus)
    req = await svc.open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                         tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    assert req.id == "hit_1"
    assert bus.types() == [EventType.HITL_OPENED]   # 不再发 SessionPausedHitl


async def test_hitl_opened_payload_carries_delivery_and_resume_state():
    bus = RecordingBus()
    await _service(bus).open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                             tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    p = bus.payload_of(EventType.HITL_OPENED)
    assert p["form"] == "approval"
    assert p["delivery"] == {"kind": "tool_result", "tool_call_id": "call_1"}
    assert p["resume_state"] == {"plan": "p1"}
    assert p["subject_id"] == "fs:bash_exec" and p["proposal"] == {"command": "ls"}
    assert p["stage"] == STAGE_AUTHZ


async def test_user_turn_delivery_serialises_its_preface():
    bus = RecordingBus()
    ask = HitlAsk(form="wait",
                  delivery=UserTurnDelivery(task_id="t1", preface="interrupt_edit"))
    await _service(bus).open(ask, session_id="s1", task_id="t1", stage=STAGE_AUTHZ, unattended=False)
    assert bus.payload_of(EventType.HITL_OPENED)["delivery"] == {
        "kind": "user_turn", "task_id": "t1", "preface": "interrupt_edit"}


async def test_reopen_same_tool_call_reuses_request_and_emits_nothing_new():
    bus = RecordingBus()
    svc = _service(bus)
    a = await svc.open(_ask(), session_id="s1", task_id="t1",
                       tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    b = await svc.open(_ask(), session_id="s1", task_id="t1",
                       tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    assert a is b
    assert bus.types() == [EventType.HITL_OPENED]


async def test_resolve_emits_hitl_resolved_with_outcome_and_message():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    req = await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id="",
                                      message="go"))
    assert req is not None and req.decision.outcome == "accepted"
    assert bus.types() == [EventType.HITL_OPENED, EventType.HITL_RESOLVED]
    p = bus.payload_of(EventType.HITL_RESOLVED)
    assert p["outcome"] == "accepted" and p["message"] == "go"


async def test_resolve_marks_claimed_true_when_a_hot_slot_takes_it():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    slot = FakeSlot()
    svc.registry.attach_slot("hit_1", slot)
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id="", message="go"))
    assert slot.delivered.message == "go"
    assert bus.payload_of(EventType.HITL_RESOLVED)["claimed"] is True


async def test_resolve_marks_claimed_false_when_no_slot():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id=""))
    assert bus.payload_of(EventType.HITL_RESOLVED)["claimed"] is False


async def test_resolve_marks_claimed_false_when_slot_refuses():
    """等待方已放弃（超时驱逐与应答擦肩）→ 必须走冷续跑，不能算已消费。"""
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    svc.registry.attach_slot("hit_1", FakeSlot(accepts=False))
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id=""))
    assert bus.payload_of(EventType.HITL_RESOLVED)["claimed"] is False


async def test_resolve_still_emits_hitl_resolved_when_slot_deliver_raises():
    """deliver 声明 -> bool 不该抛，但一旦真抛（如对已完成 future 再 set），resolve() 已
    不可逆——发事实的义务优先于让异常传播：必须仍然发出 HitlResolved(claimed=False)，
    否则请求停在「已终局」却没有可跨重启恢复的事实（Minor B）。"""
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    svc.registry.attach_slot("hit_1", RaisingSlot())
    req = await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id=""))
    assert req is not None
    assert bus.types() == [EventType.HITL_OPENED, EventType.HITL_RESOLVED]
    assert bus.payload_of(EventType.HITL_RESOLVED)["claimed"] is False


async def test_second_resolve_is_a_noop_and_emits_nothing():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id=""))
    assert await svc.resolve(HitlReply(hitl_id="hit_1", outcome="rejected", agent_id="")) is None
    assert bus.types().count(EventType.HITL_RESOLVED) == 1


async def test_resolve_unknown_id_raises_keyerror():
    svc = _service(RecordingBus())
    with pytest.raises(KeyError):
        await svc.resolve(HitlReply(hitl_id="nope", outcome="accepted", agent_id=""))


async def test_modified_arguments_ride_along_in_the_payload():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id="",
                                modified_arguments={"command": "ls -l"}))
    assert bus.payload_of(EventType.HITL_RESOLVED)["modified_arguments"] == {
        "command": "ls -l"}


async def test_validation_failure_leaves_the_request_pending_and_emits_nothing():
    """校验先于任何状态改动：被拒的内容不得写进 decision、不得发事实（spec §7.4）。"""
    bus = RecordingBus()
    svc = _service(bus, normalizer=RejectingNormalizer())
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    with pytest.raises(ValueError):
        await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id="",
                                    message="bad"))
    assert svc.registry.get("hit_1").resolved is False
    assert bus.types() == [EventType.HITL_OPENED]


async def test_cancel_resolves_with_cancelled_and_carries_its_reason():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    req = await svc.cancel("hit_1", message="failure_threshold")
    assert req is not None and req.decision.outcome == "cancelled"
    p = bus.payload_of(EventType.HITL_RESOLVED)
    assert p["outcome"] == "cancelled" and p["message"] == "failure_threshold"


async def test_cancel_on_resolved_request_is_a_noop():
    bus = RecordingBus()
    svc = _service(bus)
    await svc.open(_ask(), session_id="s1", task_id="t1", tool_call_id="call_1", stage=STAGE_AUTHZ, unattended=False)
    await svc.resolve(HitlReply(hitl_id="hit_1", outcome="accepted", agent_id=""))
    assert await svc.cancel("hit_1", message="too late") is None
    assert bus.types().count(EventType.HITL_RESOLVED) == 1


async def test_service_never_imports_runtime_or_loop():
    """分层不变式：core/hitl 不认识协程栈，也不认识 Runtime（spec §3）。"""
    import inspect

    import ctx_weft.core.hitl.service as mod

    src = inspect.getsource(mod)
    assert "core.loop" not in src and "core.runtime" not in src
