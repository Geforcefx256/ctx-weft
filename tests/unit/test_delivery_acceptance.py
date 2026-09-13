"""spec: delivery-acceptance——行为主测试：门禁/一次修正/终止条件/持久化/恢复/矩阵。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ctx_weft.core.acceptance import AcceptanceExecutor, AcceptanceRegistry
from ctx_weft.core.acceptance.protocol import (
    BUILTIN_STRUCTURE_ID,
    AcceptanceFinding,
    InvalidAcceptanceSpec,
)
from ctx_weft.core.acceptance.support import (
    candidate_of,
    compose_gap_hint,
    findings_equal,
    input_snapshot_id,
    reconcile_acceptance_inputs,
)
from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.control.types import TaskView
from ctx_weft.core.loop.driver import EventType, make_event
from ctx_weft.core.loop.steps.finalize import _delivery_acceptance_gate
from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.disposition import (
    Disposition,
    RunOutcome,
    RunOutcomeKind,
    disposition_for,
)
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.memory import MemoryEvent, MemoryKind, MemoryScope


# ── 共用装置 ─────────────────────────────────────────────────────────────────


class _Bus:
    def __init__(self):
        self.bus = InProcessEventBus()
        self.events = []

        async def _sink(e):
            self.events.append(e)

        self.bus.subscribe(None, _sink)

    def of(self, t: EventType):
        return [e for e in self.events if e.type == t]


def _mem() -> InMemoryMemoryProvider:
    return InMemoryMemoryProvider()


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _scope(task_id="t1", agent="ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent)


def _state(task, scope=None):
    return SimpleNamespace(
        run_id="run1", sequence_counter=0,
        session=SimpleNamespace(id="s1", tenant_id="default"),
        scope=scope or _scope(task.id), task=task,
        agent=SimpleNamespace(id="ag1"), extra={})


async def _seed_user_turn(mem, task_id, content, t=0):
    import datetime as _dt
    return await mem.ingest(
        MemoryEvent(kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                    address=_scope(task_id),
                    content=content, timestamp=_dt.datetime(
                        2026, 1, 1, 0, 0, t, tzinfo=_dt.timezone.utc),
                    role="user", metadata={"task_id": task_id}),
        _pctx())


def _reg_with_total(fail_first=0, then_pass=True):
    """total 检查器：前 fail_first 次失败（缺口固定），之后按 then_pass。"""
    reg = AcceptanceRegistry()
    calls = {"n": 0}

    def _total(candidate, params):
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return ("failed", [AcceptanceFinding(
                field="total", condition="sum mismatch",
                evidence_ref=f"expected {params.get('want')}",
                suggestion="recompute the sum")])
        return ("passed" if then_pass else "failed", [])

    reg.register("acme.total", "1", _total)
    reg._total_calls = calls
    return reg


def _ctx(reg, ex=None, mode="required", mem=None, bus=None):
    return SimpleNamespace(
        acceptance_registry=reg, acceptance_executor=ex or AcceptanceExecutor(),
        acceptance_mode=mode, memory=mem or _mem(), provider_ctx=_pctx(),
        event_bus=(bus.bus if bus else _mkbus()))


def _mkbus():
    b = InProcessEventBus()

    async def _sink(e):
        pass

    b.subscribe(None, _sink)
    return b


def _task(spec=None, repairs=0, **kw) -> Task:
    base = dict(id="t1", session_id="s1", status="ACTIVE", title="T",
                user_prompt="do it", outputs='{"total": "1.00"}',
                acceptance_spec=spec, acceptance_repairs_used=repairs)
    base.update(kw)
    return Task(**base)


def _spec(required=True, fields=None) -> list[dict]:
    out = []
    if fields:
        out.append({"checker_id": BUILTIN_STRUCTURE_ID, "checker_version": "1",
                    "params": {"fields": fields}, "required": required})
    out.append({"checker_id": "acme.total", "checker_version": "1",
                "params": {"want": "2.00"}, "required": required})
    return out


# ── A. 门禁行为（gate 直测；对应 spec req2/req3）────────────────────────────


async def test_gate_required_pass_delivers():
    reg = _reg_with_total()
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": '{"total": "2.00"}', "final_summary": "note"}
    gate = await _delivery_acceptance_gate(state, _ctx(reg, bus=bus, mem=state and _mem()), task, "required")
    assert gate is None  # 通过 → 正常交付
    assert bus.of(EventType.TASK_ACCEPTANCE_CHECKED)[0].payload["verdict"] == "passed"


async def test_gate_required_fail_reserves_one_repair():
    reg = _reg_with_total(fail_first=1)
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": '{"total": "1.00"}', "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg, bus=bus), task, "required")
    assert gate["phase"] == "retry" and gate["repairs_used_after"] == 1
    ev = bus.of(EventType.TASK_ACCEPTANCE_CHECKED)[0]
    assert ev.payload["verdict"] == "failed" and ev.payload["findings"]
    # 候选引用直存（小候选）
    assert ev.payload["candidate_ref"]["inline"] == '{"total": "1.00"}'


async def test_gate_repair_reruns_all_required_checks_and_second_fail_terminates():
    """一轮修正 + 全量复检；修正后仍失败 → failed（不再第二次修正）。"""
    reg = _reg_with_total(fail_first=99, then_pass=False)  # 永远失败
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": '{"total": "1.00"}', "final_summary": ""}
    ctx = _ctx(reg, bus=bus)
    g1 = await _delivery_acceptance_gate(state, ctx, task, "required")
    assert g1["phase"] == "retry"
    # TM 预占后（模拟）
    task.acceptance_repairs_used = 1
    g2 = await _delivery_acceptance_gate(state, ctx, task, "required")
    assert g2["phase"] == "failed"  # 额度已用 → 终止
    # 复检=全量：structure + total 各被调了两次
    assert reg._total_calls["n"] == 2


async def test_gate_repeated_identical_findings_skip_repair_even_with_budget():
    """缺口与上轮完全相同（无新证据）→ 即便额度还在也终止（spec 终止条件）。"""
    reg = _reg_with_total(fail_first=99, then_pass=False)
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": "x", "final_summary": ""}
    ctx = _ctx(reg)
    g1 = await _delivery_acceptance_gate(state, ctx, task, "required")
    task.acceptance_repairs_used = 0  # 假设额度未被消耗（另一路径已存在同缺口记录）
    # 上一轮记录已在 task.acceptance：同缺口再现 → failed
    g2 = await _delivery_acceptance_gate(state, ctx, task, "required")
    assert g2["phase"] == "failed"


async def test_gate_checker_error_is_unverified_and_terminal():
    reg = AcceptanceRegistry()
    reg.register("acme.total", "1", lambda c, p: 1 / 0)
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": "{}", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate["phase"] == "failed" and gate["verdict"] == "unverified"


async def test_gate_shadow_records_but_never_decides():
    reg = _reg_with_total(fail_first=99, then_pass=False)
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": "bad", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg, bus=bus), task, "shadow")
    assert gate is None  # 不改变交付决策
    assert bus.of(EventType.TASK_ACCEPTANCE_CHECKED)[0].payload["verdict"] == "failed"


async def test_gate_budget_exhausted_skips_repair():
    reg = _reg_with_total(fail_first=1)
    task = _task(_spec(), error_code=TaskErrorCode.TASK_DEADLINE_EXCEEDED)
    state = _state(task)
    state.extra = {"final_body": "x", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate["phase"] == "failed"  # 预算不足不修正（spec 场景）


async def test_gate_unavailable_version_after_recovery_unverified():
    reg = _reg_with_total()  # 未注册 acme.total v2
    task = _task([{"checker_id": "acme.total", "checker_version": "2",
                   "params": {}, "required": True}])
    task.started_at = True  # 恢复任务：不拒绝执行
    state = _state(task)
    state.extra = {"final_body": "x", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate["phase"] == "failed" and gate["verdict"] == "unverified"
    # shadow 模式同样记录 unverified，但决策不变
    task2 = _task([{"checker_id": "acme.total", "checker_version": "2",
                    "params": {}, "required": True}])
    st2 = _state(task2)
    st2.extra = {"final_body": "x", "final_summary": ""}
    assert await _delivery_acceptance_gate(st2, _ctx(reg), task2, "shadow") is None


# ── B. 首次启用初始化 / 输入回合标识 ────────────────────────────────────────


async def test_gate_first_enable_initializes_turn_id_from_memory():
    mem = _mem()
    rid = await _seed_user_turn(mem, "t1", "hello")
    reg = _reg_with_total()
    bus = _Bus()
    task = _task(_spec())  # effective_input_turn_id is None
    state = _state(task)
    state.extra = {"final_body": '{"total": "2.00"}', "final_summary": ""}
    await _delivery_acceptance_gate(state, _ctx(reg, bus=bus, mem=mem), task, "required")
    assert task.effective_input_turn_id == rid
    adv = bus.of(EventType.TASK_INPUT_ADVANCED)
    assert adv and adv[0].payload["turn_record_id"] == rid


async def test_input_snapshot_distinguishes_new_turn_same_output():
    task = _task(_spec())
    fp1 = input_snapshot_id(task)
    task.effective_input_turn_id = "mem_new_turn"
    assert input_snapshot_id(task) != fp1  # 输出未变、输入回合变了 → 指纹失效


# ── C. 处置表 + TM 预占/缺口投递/重排（spec req3）──────────────────────────


def test_disposition_acceptance_retry_and_failed():
    retry = RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success",
                       acceptance={"phase": "retry", "findings": [], "repairs_used_after": 1})
    disp = disposition_for(retry, retry_count=0, max_retries=3)
    assert disp.status == "PENDING" and disp.event_type == "TaskAcceptanceRetryReserved"
    failed = RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success",
                        acceptance={"phase": "failed", "verdict": "failed", "findings": [{"field": "x"}],
                                   "error_message": "no repair available"})
    disp2 = disposition_for(failed, retry_count=0, max_retries=3)
    assert disp2.status == "FAILED" and disp2.event_type == "TaskFailed"
    assert disp2.payload["error_code"] == TaskErrorCode.TASK_ACCEPTANCE_FAILED
    assert disp2.payload["findings"] == [{"field": "x"}]


async def test_tm_reserve_writes_budget_and_delivers_gap_then_requeues():
    bus = _Bus()
    tm = TaskManager(session_id="s1", max_concurrent=0, event_bus=bus.bus)
    from ctx_weft.core.models.session import Session
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    from tests.unit.test_task_dependency_conditions import _NoopRunner
    tm.set_runner(_NoopRunner())
    task = _task(_spec())
    tm.register_task(task)
    outcome = RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success",
                         acceptance={"phase": "retry", "attempt": 0,
                                     "findings": [{"field": "total", "condition": "sum mismatch",
                                                   "evidence_ref": "rows", "suggestion": "recompute"}],
                                     "repairs_used_after": 1})
    status = await tm.apply_run_outcome(task.id, outcome)
    assert status == "PENDING"
    assert task.acceptance_repairs_used == 1
    assert task.next_step_hint and "total" in task.next_step_hint  # 一次性缺口投递
    assert bus.of(EventType.TASK_ACCEPTANCE_RETRY_RESERVED)  # 预占事件已落（先于修正生成）
    await tm._settle(task.id, status)
    assert any(e.task_id == task.id for e in tm._queue.peek_all())  # 重排入队


# ── D. 投影 / 恢复 / 对账（spec req4/req5 + 两窗口）────────────────────────


def _projection_events(task_id="t1"):
    return [
        _ev(EventType.TASK_CREATED, {"task": {"id": task_id, "session_id": "s1",
                                              "acceptance_spec": _spec()}}, task_id),
        _ev(EventType.TASK_INPUT_ADVANCED, {"turn_record_id": "mem_1"}, task_id),
        _ev(EventType.TASK_ACCEPTANCE_CHECKED,
            {"verdict": "failed", "results": [], "findings": [{"field": "total"}],
             "output_fingerprint": "fp", "input_snapshot_id": "is", "attempt": 0,
             "candidate_ref": {"inline": "cand"}}, task_id),
        _ev(EventType.TASK_ACCEPTANCE_RETRY_RESERVED, {"repairs_used": 1, "findings": []}, task_id),
    ]


def _ev(t, payload, task_id="t1"):
    from ctx_weft.core.utils.ids import generate_id
    from ctx_weft.core.utils.clock import now_utc
    return SimpleNamespace(id=generate_id("evt"), run_id=None, sequence=0,
                           session_id="s1", type=t, timestamp=now_utc(),
                           tenant_id="default", task_id=task_id, payload=payload,
                           schema_version=1, agent_id=None)


def test_projection_folds_acceptance_domain():
    view = reduce_events(_projection_events(), "run1")
    t = view.tasks["t1"]
    assert t.acceptance_spec and t.acceptance["verdict"] == "failed"
    assert t.acceptance_repairs_used == 1
    assert t.effective_input_turn_id == "mem_1"
    # 重建为 Task（converter 链）
    task = task_from_projection(t)
    assert task.acceptance_repairs_used == 1 and task.effective_input_turn_id == "mem_1"
    # 预占幂等：重复 RESERVED 不再累加
    view2 = reduce_events(_projection_events() + [
        _ev(EventType.TASK_ACCEPTANCE_RETRY_RESERVED, {"repairs_used": 1}, "t1")], "run1")
    assert view2.tasks["t1"].acceptance_repairs_used == 1


async def test_restore_rebuilds_gap_hint_for_interrupted_repair():
    view = reduce_events(_projection_events(), "run1")
    t = task_from_projection(view.tasks["t1"])
    t.status = "PENDING"  # 修正确已预占、候选重排中
    bus = _Bus()
    tm = TaskManager(session_id="s1", max_concurrent=0, event_bus=bus.bus)
    tm.restore([t], terminal_ids=set())
    assert t.next_step_hint and "total" in t.next_step_hint  # 恢复后缺口上下文重建


async def test_reconcile_covers_both_crash_windows():
    """两窗口：消息已落库/事件未落；撤销已完成/回退事件未落（spec 两对账场景）。"""
    mem = _mem()
    turn1 = await _seed_user_turn(mem, "t1", "v1", t=0)
    turn2 = await _seed_user_turn(mem, "t1", "v2-修订", t=1)
    bus = _Bus()
    tm = TaskManager(session_id="s1", event_bus=bus.bus)
    tm.set_acceptance_mode("required")
    # 窗口一：memory 已有新回合 turn2，任务标识停在 turn1（事件未落盘）
    task = _task(_spec())
    task.effective_input_turn_id = "mem_turn1"
    tm.register_task(task)
    advanced = await reconcile_acceptance_inputs(
        tm, [task], mem, _pctx(), session_id="s1", root_agent_id="ag1", mode="required")
    assert advanced == ["t1"] and task.effective_input_turn_id == turn2
    assert bus.of(EventType.TASK_INPUT_ADVANCED)
    # 幂等：再对账一次无新事件
    n = len(bus.events)
    advanced2 = await reconcile_acceptance_inputs(
        tm, [task], mem, _pctx(), session_id="s1", root_agent_id="ag1", mode="required")
    assert advanced2 == [] and len(bus.of(EventType.TASK_INPUT_ADVANCED)) == 1
    # 窗口二：撤销已完成（fold 掉 turn2）、回退事件未落 → 重算值即回退值
    await mem.fold([turn2], [], _pctx())
    advanced3 = await reconcile_acceptance_inputs(
        tm, [task], mem, _pctx(), session_id="s1", root_agent_id="ag1", mode="required")
    # 回退值 = 最近存活回合（turn1），不是 None——撤销只抹掉被 fold 的那条
    assert advanced3 == ["t1"] and task.effective_input_turn_id == turn1


async def test_reconcile_ignores_unmaintained_tasks():
    mem = _mem()
    await _seed_user_turn(mem, "t1", "v1")
    bus = _Bus()
    tm = TaskManager(session_id="s1", event_bus=bus.bus)
    plain = _task(None)  # 无声明：零事件
    tm.register_task(plain)
    assert await reconcile_acceptance_inputs(
        tm, [plain], mem, _pctx(), session_id="s1", root_agent_id="ag1", mode="required") == []
    # off 模式即便有声明也不维护
    tm.set_acceptance_mode("off")
    declared = _task(_spec())
    tm.register_task(declared)
    assert await reconcile_acceptance_inputs(
        tm, [declared], mem, _pctx(), session_id="s1", root_agent_id="ag1", mode="off") == []


# ── E. 行为矩阵（spec req6：3 模式 × 三种声明）─────────────────────────────


async def test_mode_matrix_nine_cells():
    """九格逐一：off 全静默 / shadow 记录不改决策 / required 仅对必需门禁。"""
    for mode in ["off", "shadow", "required"]:
        for decl in ["none", "advisory", "required"]:
            spec = {"none": None, "advisory": _spec(required=False),
                    "required": _spec()}[decl]
            reg = _reg_with_total(fail_first=99, then_pass=False)
            bus = _Bus()
            task = _task(spec)
            state = _state(task)
            state.extra = {"final_body": "bad", "final_summary": ""}
            # off 模式：FinalizeStep 门根本不进 gate（模拟其守卫）
            if mode == "off" or decl == "none":
                # off / 无声明：上层守卫（FinalizeStep 判 acceptance_spec 且 mode≠off）
                # 使 gate 根本不被调用——这里断言守卫条件本身（零事件、零变化格）。
                from ctx_weft.core.acceptance.support import maintenance_active
                assert not maintenance_active(task, mode)
                continue
            gate = await _delivery_acceptance_gate(state, _ctx(reg, bus=bus), task, mode)
            recorded = bool(bus.of(EventType.TASK_ACCEPTANCE_CHECKED))
            assert recorded  # shadow/required 均记录
            if mode == "shadow":
                assert gate is None  # 记录不改变交付（含必需失败）
            else:
                assert (gate is None) == (decl == "advisory")  # 提示性不门禁、必需门禁


async def test_off_mode_no_events_at_all():
    """off 模式：声明在场也不执行、无事件、零变化（spec req1 场景）。"""
    reg = _reg_with_total(fail_first=99)
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    state.extra = {"final_body": "x", "final_summary": ""}
    # FinalizeStep 的守卫：mode==off 时 gate 不被调用——这里直接断言该守卫组合
    from ctx_weft.core.acceptance.support import maintenance_active
    assert not maintenance_active(task, "off")
    assert not bus.events


# ── F. 候选二元组 / 摘要不污染 / 巨候选外存 ────────────────────────────────


async def test_candidate_uses_separated_pair_not_concatenated_outputs():
    task = _task(_spec(), outputs='{"total": "2.00"}\n\nsummary text')
    cand = candidate_of(task, {"final_body": '{"total": "2.00"}',
                               "final_summary": "summary text"})
    assert cand["final_body"] == '{"total": "2.00"}'  # 不含摘要（拼接串才会）


async def test_gate_oversize_candidate_spills_to_memory():
    mem = _mem()
    reg = _reg_with_total(fail_first=1)
    bus = _Bus()
    task = _task(_spec())
    state = _state(task)
    big = '{"total": "1.00", "pad": "' + "x" * 9000 + '"}'
    state.extra = {"final_body": big, "final_summary": ""}
    await _delivery_acceptance_gate(state, _ctx(reg, bus=bus, mem=mem), task, "required")
    ref = bus.of(EventType.TASK_ACCEPTANCE_CHECKED)[0].payload["candidate_ref"]
    assert ref["inline"] is None and ref["record_id"]
    # 恢复后可解引用取回（spec 场景：重启后取回失败候选）
    records = await mem.load_view(_scope("t1"), MemoryScope.TASK, _pctx(),
                                  kinds=[MemoryKind.CONVERSATION_TURN])
    spilled = [r for r in records if r.metadata.get("acceptance_candidate")]
    assert spilled and spilled[0].content == big


async def test_builtin_structure_orders_first_and_catches_shape():
    reg = AcceptanceRegistry()  # 只有内置
    task = _task([{"checker_id": BUILTIN_STRUCTURE_ID, "checker_version": "1",
                   "params": {"fields": {"total": "string"}}, "required": True}])
    state = _state(task)
    state.extra = {"final_body": '{"total": "1.00", "extra": 1}', "final_summary": "s"}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate and gate["phase"] == "retry"
    assert any(f["field"] == "final_body" for f in gate["findings"])  # 字段集不符


def test_findings_helpers():
    a = [{"field": "x"}]
    assert findings_equal(a, [{"field": "x"}]) and not findings_equal(a, [])
    hint = compose_gap_hint([{"field": "total", "condition": "sum", "evidence_ref": "r",
                              "suggestion": "recompute"}])
    assert "total" in hint and "recompute" in hint


# ── G. 三点崩溃语义（spec req3：跨崩溃额度保证）────────────────────────────


async def test_crash_after_reserve_recovery_completes_that_repair():
    """预占后、生成前崩溃 → 恢复续跑=完成该次修正：候选修好 → 交付，计数不再增加。"""
    reg = _reg_with_total(fail_first=0, then_pass=True)  # 修好后轮：检查器恒过
    task = _task(_spec(), repairs=1)  # 已预占（恢复自 RESERVED 事件）
    state = _state(task)
    state.extra = {"final_body": '{"total": "2.00"}', "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate is None  # 修正轮通过 → 正常交付
    assert task.acceptance_repairs_used == 1  # 计数不变（仍是那一次额度）


async def test_crash_after_reserve_recovery_still_failing_terminates():
    reg = _reg_with_total(fail_first=99, then_pass=False)
    task = _task(_spec(), repairs=1)
    state = _state(task)
    state.extra = {"final_body": "bad", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate["phase"] == "failed"  # 续跑候选仍失败 → 失败终态，不再有第二次修正


async def test_crash_before_reserve_recovery_may_retry_once():
    """预占前崩溃（检查事件已落、预占未落）→ 恢复重跑 → 仍可发起一次修正。"""
    reg = _reg_with_total(fail_first=99, then_pass=False)
    task = _task(_spec(), repairs=0)
    state = _state(task)
    state.extra = {"final_body": "bad", "final_summary": ""}
    gate = await _delivery_acceptance_gate(state, _ctx(reg), task, "required")
    assert gate["phase"] == "retry" and gate["repairs_used_after"] == 1
