"""决定缓存的键必须是 (session_id, tool_call_id, stage)（段 2 · Task 4.5）。"""

from __future__ import annotations

from datetime import UTC, datetime

from ctx_weft.core.hitl.registry import HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlAsk, HitlDecision, ToolResultDelivery

T0 = datetime(2026, 9, 1, tzinfo=UTC)
STAGE_AUTHZ = "authz"
STAGE_TOOL = "tool"


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


def _service():
    bus = _RecordingBus()
    ids = iter(f"hit_{i}" for i in range(1, 100))
    svc = HitlService(
        registry=HitlRegistry(),
        event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
        id_factory=lambda: next(ids),
        clock=lambda: T0,
    )
    return svc, bus


def _open(reg, hitl_id, session_id, tool_call_id, stage):
    return reg.open(
        HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id=tool_call_id)),
        hitl_id=hitl_id, session_id=session_id, task_id="t1",
        tool_call_id=tool_call_id, stage=stage, created_at=T0)


def test_a_decision_in_one_session_is_invisible_to_another():
    """跨会话授权绕过：LLM 的 tool_call id 常是 call_1 这类短值，会碰撞。"""
    reg = HitlRegistry()
    _open(reg, "hit_1", "s1", "call_1", STAGE_AUTHZ)
    reg.resolve("hit_1", HitlDecision(outcome="accepted"), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ) is not None
    assert reg.decision_for("s2", "call_1", STAGE_AUTHZ) is None      # ← 洞在这里


def test_an_authz_decision_is_not_consumed_by_the_tool_stage():
    reg = HitlRegistry()
    _open(reg, "hit_1", "s1", "call_1", STAGE_AUTHZ)
    reg.resolve("hit_1", HitlDecision(outcome="accepted"), T0)
    assert reg.decision_for("s1", "call_1", STAGE_TOOL) is None


def test_a_tool_stage_decision_is_not_consumed_by_the_authz_stage():
    """最危险的一条：工具阶段答复的 modified_arguments 会被当成工具参数送进 provider。"""
    reg = HitlRegistry()
    _open(reg, "hit_2", "s1", "call_1", STAGE_TOOL)
    reg.resolve("hit_2", HitlDecision(outcome="accepted",
                                      modified_arguments={"command": "rm -rf /"}), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ) is None


def test_both_stages_can_coexist_under_one_tool_call_id():
    reg = HitlRegistry()
    _open(reg, "hit_1", "s1", "call_1", STAGE_AUTHZ)
    _open(reg, "hit_2", "s1", "call_1", STAGE_TOOL)
    reg.resolve("hit_1", HitlDecision(outcome="accepted", message="authz"), T0)
    reg.resolve("hit_2", HitlDecision(outcome="accepted", message="tool"), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ)[0].message == "authz"
    assert reg.decision_for("s1", "call_1", STAGE_TOOL)[0].message == "tool"


def test_open_is_still_idempotent_within_one_session_and_stage():
    reg = HitlRegistry()
    a = _open(reg, "hit_1", "s1", "call_1", STAGE_AUTHZ)
    b = _open(reg, "hit_2", "s1", "call_1", STAGE_AUTHZ)
    assert a is b


def test_open_is_not_idempotent_across_stages():
    reg = HitlRegistry()
    a = _open(reg, "hit_1", "s1", "call_1", STAGE_AUTHZ)
    b = _open(reg, "hit_2", "s1", "call_1", STAGE_TOOL)
    assert a is not b


async def test_stage_survives_a_restart_via_the_event_payload():
    """stage 不进事件就等于重启后丢失，两个洞立刻复活。"""
    from ctx_weft.core.control.reducers import fold_hitl_snapshot

    svc, bus = _service()
    await svc.open(HitlAsk(form="approval",
                           delivery=ToolResultDelivery(tool_call_id="call_1")),
                   session_id="s1", task_id="t1", tool_call_id="call_1",
                   stage=STAGE_TOOL, unattended=False)
    snap = fold_hitl_snapshot(bus.events)
    assert next(iter(snap.pending.values())).stage == STAGE_TOOL


def test_legacy_events_infer_their_stage_from_form():
    """旧数据没有 stage：approval 出自授权步，其余出自工具步。"""
    from ctx_weft.core.control.reducers import fold_hitl_snapshot

    def _legacy_required(form: str) -> Event:
        return Event(id="evt_1", run_id=None, sequence=0, session_id="s1",
                     type=EventType.HITL_REQUIRED, timestamp=T0, task_id="t1", agent_id="a1",
                     payload={"hitl_id": "hit_1", "form": form, "capability_id": "fs:bash_exec",
                              "tool_call_id": "call_1", "agent_id": "a1",
                              "question": "Allow?", "context": "",
                              "arguments": {}, "questions": []})

    approval = fold_hitl_snapshot([_legacy_required(form="approval")])
    question = fold_hitl_snapshot([_legacy_required(form="question")])
    assert next(iter(approval.pending.values())).stage == STAGE_AUTHZ
    assert next(iter(question.pending.values())).stage == STAGE_TOOL


# ── 第四维：invocation_key（复审 I3）──────────────────────────────────────────


def _open_keyed(reg, hitl_id, tool_call_id, key):
    return reg.open(
        HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id=tool_call_id)),
        hitl_id=hitl_id, session_id="s1", task_id="t1",
        tool_call_id=tool_call_id, stage=STAGE_AUTHZ, created_at=T0,
        invocation_key=key)


def test_same_invocation_key_still_hits_the_cache():
    """合法重放：同一次调用重入（reconcile）用的是同一份参数 → 同 key → 照旧短路。"""
    reg = HitlRegistry()
    _open_keyed(reg, "hit_1", "call_1", "bash:AAA")
    reg.resolve("hit_1", HitlDecision(outcome="accepted"), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ, "bash:AAA") is not None


def test_a_different_invocation_key_under_the_same_tool_call_id_misses():
    """模型复用 `call_1`：第 3 轮的批准不得替第 9 轮的**另一次**调用开门。"""
    reg = HitlRegistry()
    _open_keyed(reg, "hit_1", "call_1", "bash:AAA")
    reg.resolve("hit_1", HitlDecision(outcome="accepted",
                                      modified_arguments={"command": "ls -l"}), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ, "bash:BBB") is None
    # 且 `open()` 的幂等复用也不得把新调用挂到那条已终局的旧记录上（否则 waiter 见
    # resolved 判为驱逐 → 永久 park）。
    fresh = _open_keyed(reg, "hit_2", "call_1", "bash:BBB")
    assert fresh.id == "hit_2" and fresh.resolved is False


def test_an_unkeyed_record_is_a_wildcard():
    """旧模型事件 / host 直接喂的快照没有这一维（key=""）→ 通配，迁移期行为逐条同构。"""
    reg = HitlRegistry()
    _open_keyed(reg, "hit_1", "call_1", "")
    reg.resolve("hit_1", HitlDecision(outcome="accepted"), T0)
    assert reg.decision_for("s1", "call_1", STAGE_AUTHZ, "bash:ANY") is not None


def test_open_emits_and_folds_the_invocation_key():
    """key 必须跨重启存活——不落事件就等于重启后这一维消失、洞重新打开。"""
    import asyncio

    from ctx_weft.core.control.reducers import fold_hitl_snapshot

    svc, bus = _service()

    async def _go():
        return await svc.open(
            HitlAsk(form="approval",
                    delivery=ToolResultDelivery(tool_call_id="call_1")),
            session_id="s1", task_id="t1", tool_call_id="call_1",
            stage=STAGE_AUTHZ, unattended=False, invocation_key="bash:AAA")

    req = asyncio.run(_go())
    opened = [e for e in bus.events if e.type is EventType.HITL_OPENED][0]
    assert opened.payload["invocation_key"] == "bash:AAA"
    snap = fold_hitl_snapshot([opened])
    assert snap.pending[req.id].invocation_key == "bash:AAA"
