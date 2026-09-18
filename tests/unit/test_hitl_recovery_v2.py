"""恢复期装填与暂停态推导（段 2 · Task 9）。

两个方向必须**同时**成立：
- 挂在**未决** HITL 上的 task → 保持 parked、绝不重排（人还没答，任务不能自己跑）；
- 挂在**已终局** HITL 上的 task → 必须重排（决定已落盘、进程在续跑前死了的崩溃窗口）。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from datetime import UTC, datetime

import pytest

from ctx_weft.protocols import BLOB_REF_PREFIX, ImagePart, ProviderContext
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import ToolResultDelivery, UserTurnDelivery

pytestmark = pytest.mark.asyncio

SID = "ses_1"
AID = "agt_root"          # root agent id planted by `_session_prelude()`（`recover_agent` 主键）
TID = "tsk_1"
TS = datetime(2026, 9, 1, tzinfo=UTC)
PNG = b"\x89PNG\r\n\x1a\nfake-bytes"
PNG_REF = f"{BLOB_REF_PREFIX}{hashlib.sha256(PNG).hexdigest()}"


class _StubEventBlobStore:
    """最小可外部化 event blob 桩——`get` 只认自己 put 过的 ref。"""

    can_externalize = True

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        ref = f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


class _ExplodingEventBlobStore(_StubEventBlobStore):
    """`get` 直接抛——用来证明装填路径吞得住异常（best-effort，绝不抛）。"""

    async def get(self, ref: str, ctx: ProviderContext):
        raise RuntimeError("event blob store is down")


# ── 事件构造 ────────────────────────────────────────────────────────────────


def _ev(seq: int, type_: EventType, **payload) -> Event:
    task_id = payload.pop("task_id", None)
    return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=SID,
                 type=type_, timestamp=TS, task_id=task_id, payload=payload)


def _session_prelude() -> list[Event]:
    return [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id="agt_root"),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": TID, "status": "PENDING", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}),
        _ev(4, EventType.TASK_STARTED, task_id=TID, assigned_agent_id="agt_root"),
    ]


def _legacy_pending_approval_events() -> list[Event]:
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_REQUIRED, task_id=TID, hitl_id="hit_1", form="approval",
            capability_id="tool:deploy", tool_call_id="call_1", question="ok?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
    ]


def _image_ref_payload(ref: str) -> list[dict]:
    return [{"type": "image", "data": ref, "media_type": "image/png", "source_type": "ref"}]


def _legacy_answered_with_event_blob_ref() -> list[Event]:
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_REQUIRED, task_id=TID, hitl_id="hit_1", form="question",
            capability_id="control:ask_user", tool_call_id="call_1", question="pic?"),
        _ev(6, EventType.HITL_ANSWERED, task_id=TID, hitl_id="hit_1",
            message=_image_ref_payload(PNG_REF)),
        _ev(7, EventType.TASK_SUSPENDED, task_id=TID),
    ]


def _legacy_answered_with_broken_ref() -> list[Event]:
    evs = _legacy_answered_with_event_blob_ref()
    evs[5] = _ev(6, EventType.HITL_ANSWERED, task_id=TID, hitl_id="hit_1",
                 message=_image_ref_payload(f"{BLOB_REF_PREFIX}deadbeef"))
    return evs


def _pending_with_tool_result_delivery() -> list[Event]:
    return _legacy_pending_approval_events()


def _pending_with_user_turn_delivery() -> list[Event]:
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_REQUIRED, task_id=TID, hitl_id="hit_1", form="wait",
            capability_id="control:wait_for_user", context="plain_text", question="?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
    ]


def _pending_with_custom_form_and_user_turn() -> list[Event]:
    """host 自定义 form + user_turn delivery（新两事件模型）。"""
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="host:triage",
            delivery={"kind": "user_turn", "task_id": TID, "preface": "normal"},
            prompt="?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
    ]


def _resolved_but_never_resumed_tool_result() -> list[Event]:
    """决定已落盘，但进程在续跑之前就死了：HITL 已终局、task 仍 SUSPENDED。"""
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_REQUIRED, task_id=TID, hitl_id="hit_1", form="approval",
            capability_id="tool:deploy", tool_call_id="call_1", question="ok?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
        _ev(7, EventType.HITL_APPROVED, task_id=TID, hitl_id="hit_1", message="go"),
    ]


def _new_model_resolved_tool_result() -> list[Event]:
    """**新两事件模型**的已终局 tool_result：HITL_OPENED + HITL_RESOLVED，task 仍 SUSPENDED。

    旧投影 reducer 根本没有 HITL_OPENED 分支，这条流对它是透明的。
    """
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="approval",
            delivery={"kind": "tool_result", "tool_call_id": "call_1"},
            tool_call_id="call_1", stage="authz", prompt="ok?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
        _ev(7, EventType.HITL_RESOLVED, task_id=TID, hitl_id="hit_1",
            outcome="accepted", message="go"),
    ]


def _new_model_unresolved_tool_result() -> list[Event]:
    """新两事件模型的**未决** HITL：旧投影看不见它 → 旧实现会把任务重排（Critical）。"""
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="approval",
            delivery={"kind": "tool_result", "tool_call_id": "call_1"},
            tool_call_id="call_1", stage="authz", prompt="ok?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
    ]


def _resolved_user_turn_never_injected() -> list[Event]:
    """`wait_for_user` 已被应答，但进程在注入之前就死了。

    关键形状：`UserTurn` 的 park **没有 tool_call_id**（`act.py:_cold_park`）。
    """
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="wait",
            delivery={"kind": "user_turn", "task_id": TID, "preface": "normal"},
            stage="tool", agent_id="agt_root", prompt=""),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
        _ev(7, EventType.HITL_RESOLVED, task_id=TID, hitl_id="hit_1",
            outcome="accepted", message="use postgres"),
    ]


def _resolved_user_turn_on_a_parent_with_a_live_child() -> list[Event]:
    """父任务挂在活子任务上，且有一条已终局的 `UserTurn`。

    子任务带一个永不终态的 dag_dep，因此 restore 会入队但永不派发——父任务的
    「children 全终态」闸门确定性地关着，不受 drain 时序影响。
    """
    return [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id="agt_root"),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": TID, "status": "PENDING", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}),
        _ev(4, EventType.TASK_STARTED, task_id=TID, assigned_agent_id="agt_root"),
        _ev(5, EventType.TASK_CREATED, task={
            "id": "tsk_child", "status": "PENDING", "title": "C1",
            "parent_task_id": TID, "dag_deps": ["tsk_never_done"],
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}),
        _ev(6, EventType.TASK_STARTED, task_id="tsk_child", assigned_agent_id="agt_root"),
        _ev(7, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="wait",
            delivery={"kind": "user_turn", "task_id": TID, "preface": "normal"},
            stage="tool", agent_id="agt_root", prompt=""),
        _ev(8, EventType.TASK_SUSPENDED, task_id=TID),
        _ev(9, EventType.HITL_RESOLVED, task_id=TID, hitl_id="hit_1",
            outcome="accepted", message="use postgres"),
    ]


def _legacy_answered_user_turn() -> list[Event]:
    """**旧模型**的 wait 应答：旧路径注入的记忆记录不带幂等键，补写会重复。"""
    return [
        *_session_prelude(),
        _ev(5, EventType.HITL_REQUIRED, task_id=TID, hitl_id="hit_1", form="wait",
            capability_id="control:wait_for_user", context="plain_text", question="?"),
        _ev(6, EventType.TASK_SUSPENDED, task_id=TID),
        _ev(7, EventType.HITL_ANSWERED, task_id=TID, hitl_id="hit_1",
            message="use postgres"),
    ]


# ── runtime 装配 ────────────────────────────────────────────────────────────


async def _runtime_with_events(events, *, event_blob_store=None):
    from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="resumed")] * 8)
    llm.last_request = None
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    store = event_blob_store if event_blob_store is not None else _StubEventBlobStore()
    if isinstance(store, _StubEventBlobStore):
        store.blobs[PNG_REF] = (PNG, "image/png")
    rt.providers.register_event_blob_store(store)
    for e in events:
        await rt.event_store.append(e)
    rt._test_llm = llm
    return rt


def _task_status(rt, task_id: str) -> str:
    tm = rt._task_managers.get(SID)
    assert tm is not None, "recover_agent should have registered a TaskManager"
    task = tm.get_task(task_id)
    assert task is not None
    return task.status


def _tm(rt):
    tm = rt._task_managers.get(SID)
    assert tm is not None, "recover_agent should have registered a TaskManager"
    return tm


async def _drained_tasks(rt) -> list[str]:
    """实际被派发跑起来的 task —— 以「LLM 被调用过」为证（与既有恢复测试同口径）。

    **只用于肯定断言**：轮询等一个必然发生的事件是可靠的；等一个「不该发生」的事件
    则是天然 flaky 的，那一侧改用 `restore` 之后的确定性状态（见
    `test_a_task_with_an_unresolved_hitl_stays_parked_and_is_not_requeued`）。
    """
    for _ in range(40):
        if rt._test_llm.last_request is not None:
            return [TID]
        await asyncio.sleep(0.02)
    return []


async def _hitl_reply_prompts(rt, task_id: str = TID, agent_id: str = "agt_root") -> list:
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope
    scope = MemoryAddress(session_id=SID, task_id=task_id, agent_id=agent_id)
    pctx = ProviderContext(session_id=SID, tenant_id="default", task_id=task_id,
                           agent_id=agent_id)
    view = await rt.providers.get_memory().load_view(
        scope, MemoryScope.TASK, pctx, kinds=[MemoryKind.CONVERSATION_TURN])
    return [r for r in view if r.metadata.get("source") == "hitl_reply"]


def _is_event_side_ref(message) -> bool:
    return any(getattr(p, "source_type", "") == "ref" for p in (message or [])
               if isinstance(p, ImagePart))


def _text_of(message) -> str:
    from ctx_weft.core.utils.content import content_to_text
    return content_to_text(message)


# ── 装填 ────────────────────────────────────────────────────────────────────


async def test_recovery_fills_the_registry_from_folded_events():
    rt = await _runtime_with_events(_legacy_pending_approval_events())
    n = await rt.rebuild_hitl(SID)
    assert n == 1
    assert rt.hitl_registry.list_pending(SID)[0].tool_call_id == "call_1"


async def test_filled_requests_have_no_wait_slot():
    """重启后一切皆冷。"""
    rt = await _runtime_with_events(_legacy_pending_approval_events())
    await rt.rebuild_hitl(SID)
    assert rt.hitl_registry.list_pending(SID)[0].slot is None


async def test_event_side_refs_are_hydrated_before_filling():
    """spec §12.3.3：event 侧 ref 直接进 memory 会写一个永远打不开的引用。"""
    rt = await _runtime_with_events(_legacy_answered_with_event_blob_ref())
    await rt.rebuild_hitl(SID)
    got = rt.hitl_registry.decision_for(SID, "call_1", "tool")
    assert got is not None
    decision, _ = got
    assert not _is_event_side_ref(decision.message)
    assert decision.message[0].data == base64.b64encode(PNG).decode()


async def test_hydration_failure_degrades_to_text_and_never_raises():
    rt = await _runtime_with_events(_legacy_answered_with_broken_ref())
    await rt.rebuild_hitl(SID)                       # 不得抛
    decision, _ = rt.hitl_registry.decision_for(SID, "call_1", "tool")
    assert "[image" in _text_of(decision.message)
    assert not _is_event_side_ref(decision.message)


async def test_hydration_never_raises_even_when_the_blob_store_explodes():
    """降级本身也不能再抛：异常会卡住整条恢复路径。"""
    rt = await _runtime_with_events(_legacy_answered_with_event_blob_ref(),
                                    event_blob_store=_ExplodingEventBlobStore())
    await rt.rebuild_hitl(SID)                       # 不得抛
    decision, _ = rt.hitl_registry.decision_for(SID, "call_1", "tool")
    assert not _is_event_side_ref(decision.message)
    assert "[image" in _text_of(decision.message)


# ── delivery：面板提示的原始事实（2026-09-04 Task 14 起派生串已删，host 自判） ──


async def test_legacy_approval_pending_has_tool_result_delivery():
    """旧 approval 请求（有 tool_call_id、form != wait）折成 `ToolResultDelivery`——
    host 据此知道「有面板可答」（旧串是 PAUSED_HITL）。"""
    rt = await _runtime_with_events(_pending_with_tool_result_delivery())
    await rt.rebuild_hitl(SID)
    assert isinstance(rt.hitl_registry.list_pending(SID)[0].delivery, ToolResultDelivery)


async def test_legacy_wait_only_pending_has_user_turn_delivery():
    """UserTurn = 软待命，没有面板要答（旧串是 PAUSED）—— 误标会让前端等一个
    不存在的面板。"""
    rt = await _runtime_with_events(_pending_with_user_turn_delivery())
    await rt.rebuild_hitl(SID)
    assert isinstance(rt.hitl_registry.list_pending(SID)[0].delivery, UserTurnDelivery)


async def test_delivery_is_carried_by_the_explicit_field_not_by_form():
    """host 自定义 form 也能拿到正确的 delivery（旧的 PAUSED/PAUSED_HITL 判据曾经
    按 `form == "wait"` 字面量判定，那条路子对 host 自定义 form 会判错；新两事件
    模型下 `delivery` 是 `HITL_OPENED` 携带的显式字段，与 `form` 无关）。"""
    rt = await _runtime_with_events(_pending_with_custom_form_and_user_turn())
    pend = await rt.rebuild_hitl(SID)
    assert pend == 1 and rt.hitl_registry.list_pending(SID)[0].form == "host:triage"
    assert isinstance(rt.hitl_registry.list_pending(SID)[0].delivery, UserTurnDelivery)


async def test_nothing_is_pending_once_the_hitl_is_resolved_but_never_resumed():
    """决定已落盘、进程在续跑前死了：这条 HITL 已终局，恢复期装填后不应再出现在
    未决列表里（旧串在这种情况下是 `""`，即「无未决，不该暂停」）。"""
    rt = await _runtime_with_events(_resolved_but_never_resumed_tool_result())
    await rt.rebuild_hitl(SID)
    assert rt.hitl_registry.list_pending(SID) == []


# ── 两个方向 ────────────────────────────────────────────────────────────────


async def test_a_task_with_an_unresolved_hitl_stays_parked_and_is_not_requeued():
    """最高风险的一条：人还没答，任务绝不能自己跑起来。

    断言是**确定性**的：`restore` 在 `recover_agent` 返回之前同步跑完，被重排的 task
    在那一刻就已经被改成 `PENDING` 并入队。所以「仍是 SUSPENDED 且队列里没有它」是一个
    不依赖时序的判据——不必去轮询一个「不该发生」的事件。
    """
    rt = await _runtime_with_events(_pending_with_tool_result_delivery())
    await rt.recover_agent(AID)
    tm = _tm(rt)
    assert tm.get_task(TID).status == "SUSPENDED"
    assert not tm._queue.has_pending()
    assert rt._test_llm.last_request is None


async def test_a_new_model_unresolved_hitl_also_keeps_its_task_parked():
    """基线的 Critical：旧投影 reducer 没有 HITL_OPENED 分支 → parked 集合为空 → 人还
    没答，任务就被重排跑起来了。parked 真相源改成 registry 之后这条才成立。"""
    rt = await _runtime_with_events(_new_model_unresolved_tool_result())
    await rt.recover_agent(AID)
    tm = _tm(rt)
    assert [r.tool_call_id for r in rt.hitl_registry.list_pending(SID)] == ["call_1"]
    assert tm.get_task(TID).status == "SUSPENDED"
    assert not tm._queue.has_pending()
    assert rt._test_llm.last_request is None


async def test_a_task_parked_on_an_already_resolved_hitl_is_requeued():
    """崩溃窗口：决定已落盘、但进程在续跑之前死了 —— 任务必须重新跑起来。"""
    rt = await _runtime_with_events(_resolved_but_never_resumed_tool_result())
    await rt.recover_agent(AID)
    assert TID in await _drained_tasks(rt)


async def test_a_task_on_a_resolved_new_model_hitl_is_requeued():
    """新两事件模型（HITL_OPENED + HITL_RESOLVED）的同一条路。旧投影对这条流是瞎的。"""
    rt = await _runtime_with_events(_new_model_resolved_tool_result())
    await rt.recover_agent(AID)
    assert rt.hitl_registry.list_pending(SID) == []          # 已终局 → 不 park
    assert rt.hitl_registry.decision_for(SID, "call_1", "authz")[0].outcome == "accepted"
    assert TID in await _drained_tasks(rt)


async def test_resolved_for_session_sees_the_task_id_of_a_filled_decision():
    """承重点：装填出来的已终局记录必须带 task_id，否则恢复期认不出该唤醒哪个 task。"""
    rt = await _runtime_with_events(_resolved_but_never_resumed_tool_result())
    await rt.rebuild_hitl(SID)
    resolved = rt.hitl_registry.resolved_for_session(SID)
    assert [r.task_id for r in resolved] == [TID]
    assert rt.hitl_registry.resolved_for_session("other") == []


async def test_a_resolved_user_turn_without_a_tool_call_id_is_still_filled():
    """`UserTurn` 的 park 没有 tool_call_id —— 按 tool_call 过滤会把整整一类请求丢掉。"""
    rt = await _runtime_with_events(_resolved_user_turn_never_injected())
    await rt.rebuild_hitl(SID)
    resolved = rt.hitl_registry.resolved_for_session(SID)
    assert len(resolved) == 1
    assert resolved[0].tool_call_id == "" and resolved[0].task_id == TID
    assert isinstance(resolved[0].delivery, UserTurnDelivery)


async def test_recovery_injects_the_answer_of_a_resolved_user_turn():
    """崩溃窗口里最容易静默丢的一条：人答了 `wait_for_user`，进程在注入之前就死了。

    任务照样会被重排（children 闸门放行），但没人把人的答复写进对话——不补这一步，
    那句话就彻底消失。
    """
    rt = await _runtime_with_events(_resolved_user_turn_never_injected())
    await rt.recover_agent(AID)
    prompts = await _hitl_reply_prompts(rt)
    assert len(prompts) == 1
    from ctx_weft.core.utils.content import content_to_text
    assert "use postgres" in content_to_text(prompts[0].content)


async def test_requeue_of_a_resolved_user_turn_does_not_duplicate_the_injection():
    """UserTurn 侧的幂等靠 MemoryEvent.id = f"hitlreply:{hitl_id}"（§7.3/§12.2）。

    恢复可以反复跑（重启、/resume、冷应答各来一遍），注入必须只留一条。
    """
    rt = await _runtime_with_events(_resolved_user_turn_never_injected())
    await rt.recover_agent(AID)
    await rt.recover_agent(AID)
    assert len(await _hitl_reply_prompts(rt)) == 1


async def test_a_task_parked_on_another_hitl_still_gets_the_reply_injected():
    """该 task 还挂着别的**未决** HITL：状态一动不动，但那条已终局的答复照样要补进对话。

    「有未决 HITL 就整个跳过」曾经是这里的判据（从 `restore` 的重排判据借来的）。它错在
    把两件事混成一件：**要不要重排**看未决问题，**这条答复进没进过对话**不看。代价实测
    过——同一个 task 上前一条已终局的答复被跳过，而人答掉那个未决问题时走的是活 TM 那条
    路（它当时不做补写），那条答复就此永久缺失（审查文档 M8）。

    补写本身仍然是**只写记忆**：下面第一条断言钉住 task 状态不被碰。
    """
    events = [
        *_resolved_user_turn_never_injected(),
        _ev(8, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_2", form="approval",
            delivery={"kind": "tool_result", "tool_call_id": "call_9"},
            tool_call_id="call_9", stage="authz", prompt="ok?"),
    ]
    rt = await _runtime_with_events(events)
    await rt.recover_agent(AID)
    tm = _tm(rt)
    assert tm.get_task(TID).status == "SUSPENDED"      # 只写记忆，绝不碰状态
    injected = await _hitl_reply_prompts(rt)
    assert [str(r.content) for r in injected] == ["use postgres"]


async def test_restore_leaves_a_parent_with_live_children_suspended():
    """删掉 `resumable_task_ids` 之后的守卫：父任务提前置 PENDING 会让
    `_try_resume_parent`（以 status == "SUSPENDED" 为门）在子任务收尾时静默失效。"""
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task

    tm = TaskManager(session_id=SID)
    parent = Task(id="p", session_id=SID, status="SUSPENDED")
    child = Task(id="c", session_id=SID, status="ACTIVE", parent_task_id="p")
    tm.restore([parent, child], terminal_ids=set(), parked_task_ids=set())
    assert tm.get_task("p").status == "SUSPENDED"   # 等 children，靠 _try_resume_parent 唤醒
    assert tm.get_task("c").status == "PENDING"     # 子任务本身被这一趟 restore 重排了


async def test_recovery_does_not_touch_a_parent_suspended_on_a_live_child():
    """复审 Critical：补写必须是**只写**的。

    父任务 SUSPENDED 在活子任务上、且有一条已终局 UserTurn。若补写沿用
    `_inject_user_reply`（尾部会 `status = "PENDING"`），父任务会被翻成
    PENDING **却没人入队**，`_try_resume_parent` 的 SUSPENDED 门随之失效 → 永久停摆。
    这件事必须不发生，而人的答复仍要落进对话。

    注意：这里不断言 outputs——`TaskView.outputs` 唯一的非空写入点是
    `TaskFinished`（reducers.py:581），一个从未 FINISHED 过、只是 SUSPENDED
    在活子任务上的父任务，outputs 本就是 None（`TASK_CREATED` 的
    `t.get("outputs")` 在 payload 不带这个键时也是 None）。断言它「非空」
    曾经能过，是因为这份 fixture 用一条伪造的 `TaskFinalized{outputs:...}`
    在任意时刻硬注入了 outputs——那正是本 batch Task 1 在修的洞。见
    `docs/follow-ups/2026-09-03-outstanding-issues.md` A8：非终态任务的
    中途产出目前确实无法从事件流恢复，这不是本测试要保护的东西。
    """
    rt = await _runtime_with_events(_resolved_user_turn_on_a_parent_with_a_live_child())
    await rt.recover_agent(AID)
    parent = _tm(rt).get_task(TID)
    assert parent.status == "SUSPENDED"                  # 留给 _try_resume_parent
    assert parent.outputs is None                        # 从未 FINISHED 过，本就没有 outputs——实测确认，非猜测
    assert parent.process_report is None                 # 没被 _inject_user_reply 尾部清过（同一行代码路径）
    assert len(await _hitl_reply_prompts(rt)) == 1       # 答复照样补上了


async def test_recovery_backfill_survives_repeated_recovery_without_state_drift():
    """反复恢复：状态不漂移、注入不重复。"""
    rt = await _runtime_with_events(_resolved_user_turn_on_a_parent_with_a_live_child())
    await rt.recover_agent(AID)
    await rt.recover_agent(AID)
    parent = _tm(rt).get_task(TID)
    assert parent.status == "SUSPENDED"
    assert parent.outputs is None                        # 见上一个测试的注释：本就没有，不因重复恢复而变
    assert len(await _hitl_reply_prompts(rt)) == 1


async def test_legacy_origin_user_turns_are_not_backfilled():
    """旧模型的应答不补写：旧路径的记忆记录没有 `hitlreply:` 幂等键，补一次就多一轮。"""
    rt = await _runtime_with_events(_legacy_answered_user_turn())
    await rt.rebuild_hitl(SID)
    resolved = rt.hitl_registry.resolved_for_session(SID)
    assert len(resolved) == 1 and resolved[0].legacy_origin is True
    await rt.recover_agent(AID)
    assert await _hitl_reply_prompts(rt) == []


async def test_new_model_user_turns_are_not_flagged_legacy():
    rt = await _runtime_with_events(_resolved_user_turn_never_injected())
    await rt.rebuild_hitl(SID)
    assert rt.hitl_registry.resolved_for_session(SID)[0].legacy_origin is False


async def test_resume_on_a_live_owner_neither_rebuilds_nor_backfills():
    """单例：内存里有活 owner 时，`/resume` 就地复用它——不重建一个 TM 去顶替它，也
    不跑重建路径那套崩溃善后（补注入已终局的 UserTurn）。

    owner 活着就说明进程没崩。它还在跑的 task 既不能被重排，更不能被补写一轮用户
    发言——从前这里靠「补写的跳过集合与 restore 共用 inflight」护着（复审
    Important），那道过滤正是为「新 TM 顶替活 TM」而设；单例之后顶替本身不再发生。"""
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import NormalTaskSettings, Task
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from tests.unit._stub_runner import StubRunner

    rt = await _runtime_with_events(_resolved_user_turn_never_injected())
    live = TaskManager(session_id=SID, event_bus=rt.event_bus, max_concurrent=1)
    live.set_session(Session(id=SID, tenant_id="default", user_prompt="do it",
                             status="RUNNING", root_agent_id=AID, token_budget=0))
    live.set_runner(StubRunner(live, session_id=SID))
    live.register_task(Task(id=TID, session_id=SID, status="ACTIVE", assigned_agent_id=AID,
                            creator_agent_id=AID, settings=NormalTaskSettings()))
    live._running_tasks.add(TID)                # 活 owner 还在跑 TID
    rt._register_and_drain(live.session, live)

    await rt.recover_agent(AID)               # 不带 resumed_task_id → /resume
    await asyncio.sleep(0)

    assert rt._task_managers[SID] is live, "活 owner 不得被重建出来的 TM 顶替"
    assert live.running_task_ids() == {TID} and not live._queue.peek_all(), \
        "在跑的 task 不得被重排"
    assert await _hitl_reply_prompts(rt) == []
