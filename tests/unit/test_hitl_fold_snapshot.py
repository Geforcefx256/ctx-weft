"""双读折叠：新旧两套 HITL 事件 → HitlSnapshot（spec §12.3）。

对旧数据的判据必须与升级前**逐条同构**——迁移的正确性标准是「行为不变」，
不是「更符合新设计的意图」。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ctx_weft.core.control.reducers import fold_hitl_snapshot
from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL
from ctx_weft.protocols import ImagePart, TextPart
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import (
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)

T0 = datetime(2026, 9, 1, tzinfo=UTC)


def _ev(event_type: str, payload: dict, *, seq: int = 0, task_id: str = "t1",
        agent_id: str = "a1") -> Event:
    return Event(id=f"evt_{seq}", run_id=None, sequence=seq, session_id="s1",
                 type=event_type, timestamp=T0 + timedelta(seconds=seq),
                 task_id=task_id, agent_id=agent_id, payload=payload)


def _key(tool_call_id: str, stage: str) -> tuple[str, str, str]:
    """`decisions_for` 的键——所有测试事件都落在 session "s1"（第 2 段 · Task 4.5）。"""
    return "s1", tool_call_id, stage


# ── 旧事件（legacy）─────────────────────────────────────────────────────────────

def _legacy_required(hitl_id="hit_1", form="approval", capability_id="fs:bash_exec",
                     tool_call_id="call_1", context="", seq=0) -> Event:
    return _ev(EventType.HITL_REQUIRED, {
        "hitl_id": hitl_id, "form": form, "capability_id": capability_id,
        "tool_call_id": tool_call_id, "agent_id": "a1",
        "question": "Allow bash?", "context": context,
        "arguments": {"command": "ls"}, "questions": [],
    }, seq=seq)


def test_legacy_required_becomes_pending_with_tool_result_delivery():
    snap = fold_hitl_snapshot([_legacy_required()])
    req = snap.pending["hit_1"]
    assert req.form == "approval" and req.subject_id == "fs:bash_exec"
    assert req.prompt == "Allow bash?" and req.proposal == {"command": "ls"}
    assert req.delivery == ToolResultDelivery(tool_call_id="call_1")
    assert req.agent_id == "a1"
    assert req.stage == HITL_STAGE_AUTHZ


def test_legacy_wait_form_becomes_user_turn_delivery():
    """反推用 form == 'wait'（今天 runtime 的实际判据），不是 sentinel capability_id。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="wait", capability_id="control:wait_for_user",
                         tool_call_id="", context="plain_text")])
    assert snap.pending["hit_1"].delivery == UserTurnDelivery(task_id="t1",
                                                              preface="normal")


def test_legacy_wait_form_without_the_sentinel_still_becomes_user_turn():
    """判据是 form，不是 capability_id——否则迁移会改变这条在途请求的行为。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="wait", capability_id="", tool_call_id="",
                         context="interrupt")])
    assert snap.pending["hit_1"].delivery == UserTurnDelivery(task_id="t1",
                                                              preface="interrupt")


def test_legacy_wait_context_interrupt_edit_maps_to_its_preface():
    snap = fold_hitl_snapshot([
        _legacy_required(form="wait", tool_call_id="", context="interrupt:edit")])
    assert snap.pending["hit_1"].delivery == UserTurnDelivery(
        task_id="t1", preface="interrupt_edit")


def test_legacy_non_wait_without_tool_call_id_falls_back_to_no_resume():
    """既非 wait、又无 tool_call 可补 → 显式「只可取消」，不静默丢。"""
    snap = fold_hitl_snapshot([_legacy_required(form="question", tool_call_id="")])
    assert snap.pending["hit_1"].delivery == NoResumeDelivery()


def test_legacy_approved_resolves_to_accepted():
    snap = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.HITL_APPROVED, {"hitl_id": "hit_1"}, seq=1)])
    assert snap.pending == {}
    decision, resume_state = snap.decisions_for[_key("call_1", HITL_STAGE_AUTHZ)]
    assert decision.outcome == "accepted" and decision.modified_arguments is None
    assert resume_state is None


def test_legacy_modified_resolves_to_accepted_with_arguments():
    snap = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.HITL_MODIFIED,
            {"hitl_id": "hit_1", "modified_arguments": {"command": "ls -l"}}, seq=1)])
    decision, _ = snap.decisions_for[_key("call_1", HITL_STAGE_AUTHZ)]
    assert decision.outcome == "accepted"
    assert decision.modified_arguments == {"command": "ls -l"}


def test_legacy_answered_carries_its_message():
    snap = fold_hitl_snapshot([
        _legacy_required(form="question"),
        _ev(EventType.HITL_ANSWERED, {"hitl_id": "hit_1", "message": "yes"}, seq=1)])
    decision, _ = snap.decisions_for[_key("call_1", HITL_STAGE_TOOL)]
    assert decision.outcome == "accepted" and decision.message == "yes"


def test_legacy_rejected_and_cancelled_map_to_their_outcomes():
    rejected = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.HITL_REJECTED, {"hitl_id": "hit_1", "message": "no"}, seq=1)])
    assert rejected.decisions_for[_key("call_1", HITL_STAGE_AUTHZ)][0].outcome == "rejected"

    cancelled = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.HITL_CANCELLED, {"hitl_id": "hit_1"}, seq=1)])
    assert cancelled.pending == {}
    # cancelled 不是可用决定
    assert _key("call_1", HITL_STAGE_AUTHZ) not in cancelled.decisions_for


def test_answered_without_message_is_not_a_usable_decision():
    """旧事件只有 hitl_id → 还原不出答案。按未决重问，**绝不臆造**（spec §12.3.3）。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="question"),
        _ev(EventType.HITL_ANSWERED, {"hitl_id": "hit_1"}, seq=1)])
    assert _key("call_1", HITL_STAGE_TOOL) not in snap.decisions_for
    assert snap.pending == {}          # 已终局，故不在 pending；但也不可用作决定


def test_modified_without_arguments_is_not_a_usable_decision():
    """缺改参会拿原参执行，违背改参意图 → 不可用。"""
    snap = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.HITL_MODIFIED, {"hitl_id": "hit_1"}, seq=1)])
    assert _key("call_1", HITL_STAGE_AUTHZ) not in snap.decisions_for


def test_session_paused_hitl_is_ignored():
    """会话暂停态在新模型里由 pending 集合推导，旧事件不再是真相（spec §7.1）。"""
    snap = fold_hitl_snapshot([
        _legacy_required(),
        _ev(EventType.SESSION_PAUSED_HITL, {"capability_id": "fs:bash_exec"}, seq=1)])
    assert set(snap.pending) == {"hit_1"}


# ── 新事件 ──────────────────────────────────────────────────────────────────────

def _opened(hitl_id="hit_9", delivery=None, resume_state=None, seq=0,
           stage=HITL_STAGE_TOOL) -> Event:
    return _ev(EventType.HITL_OPENED, {
        "hitl_id": hitl_id, "form": "question",
        "delivery": delivery or {"kind": "tool_result", "tool_call_id": "call_9"},
        "subject_id": "deploy:apply", "prompt": "确认部署？", "detail": "",
        "fields": [], "proposal": None, "tool_call_id": "call_9", "agent_id": "a1",
        "resume_state": resume_state, "reply_as_result": False, "stage": stage,
    }, seq=seq)


def test_new_opened_folds_into_pending():
    snap = fold_hitl_snapshot([_opened()])
    req = snap.pending["hit_9"]
    assert req.delivery == ToolResultDelivery(tool_call_id="call_9")
    assert req.subject_id == "deploy:apply" and req.prompt == "确认部署？"
    assert req.stage == HITL_STAGE_TOOL


def test_new_user_turn_delivery_round_trips():
    snap = fold_hitl_snapshot([_opened(
        delivery={"kind": "user_turn", "task_id": "t1", "preface": "interrupt_edit"})])
    assert snap.pending["hit_9"].delivery == UserTurnDelivery(
        task_id="t1", preface="interrupt_edit")


def test_new_resolved_pairs_the_decision_with_its_resume_state():
    """决定必须与 resume_state 成对——只给决定就要重做让出前的工作（spec §7.2）。"""
    snap = fold_hitl_snapshot([
        _opened(resume_state={"plan": "deploy-7"}),
        _ev(EventType.HITL_RESOLVED,
            {"hitl_id": "hit_9", "outcome": "accepted", "message": "go",
             "claimed": False}, seq=1)])
    assert snap.pending == {}
    decision, resume_state = snap.decisions_for[_key("call_9", HITL_STAGE_TOOL)]
    assert decision.outcome == "accepted" and decision.message == "go"
    assert resume_state == {"plan": "deploy-7"}


def test_new_resolved_with_host_custom_outcome_is_passed_through():
    """outcome 是开放值域，core 只判「非空即终局」，不解释语义（spec §9.4）。"""
    snap = fold_hitl_snapshot([
        _opened(),
        _ev(EventType.HITL_RESOLVED,
            {"hitl_id": "hit_9", "outcome": "escalated", "claimed": False}, seq=1)])
    assert snap.decisions_for[_key("call_9", HITL_STAGE_TOOL)][0].outcome == "escalated"


def test_last_usable_decision_wins_for_the_same_tool_call():
    """同 tool_call 重问副本：最后一条可用决定胜出。"""
    snap = fold_hitl_snapshot([
        _legacy_required(hitl_id="hit_1", seq=0),
        _ev(EventType.HITL_REJECTED, {"hitl_id": "hit_1", "message": "no"}, seq=1),
        _legacy_required(hitl_id="hit_2", seq=2),
        _ev(EventType.HITL_APPROVED, {"hitl_id": "hit_2"}, seq=3)])
    assert snap.decisions_for[_key("call_1", HITL_STAGE_AUTHZ)][0].outcome == "accepted"


def test_empty_event_list_yields_an_empty_snapshot():
    snap = fold_hitl_snapshot([])
    assert snap.pending == {} and snap.decisions_for == {}


# ── 回归修复（review Finding 1/2）────────────────────────────────────────────────

def test_legacy_answered_multimodal_message_round_trips_to_content_parts():
    """事件载荷存 jsonable 形态；decision.message 须经 content_from_jsonable 转回
    ContentPart，否则图片答复在 split_for_tool_result 里因无 .text 被拆成空文本。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="question"),
        _ev(EventType.HITL_ANSWERED, {
            "hitl_id": "hit_1",
            "message": [
                {"type": "text", "text": "here"},
                {"type": "image", "data": "abc", "media_type": "image/png"},
            ],
        }, seq=1)])
    decision, _ = snap.decisions_for[_key("call_1", HITL_STAGE_TOOL)]
    assert decision.message == [
        TextPart(text="here"),
        ImagePart(data="abc", media_type="image/png", source_type="base64"),
    ]


def test_new_resolved_multimodal_message_round_trips_to_content_parts():
    snap = fold_hitl_snapshot([
        _opened(),
        _ev(EventType.HITL_RESOLVED, {
            "hitl_id": "hit_9", "outcome": "accepted", "claimed": False,
            "message": [{"type": "text", "text": "go"}],
        }, seq=1)])
    decision, _ = snap.decisions_for[_key("call_9", HITL_STAGE_TOOL)]
    assert decision.message == [TextPart(text="go")]


def test_legacy_answered_with_empty_message_is_not_a_usable_decision():
    """判据是真值而非 is None——空串同样还原不出答案，与「缺 message」同样不可用。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="question"),
        _ev(EventType.HITL_ANSWERED, {"hitl_id": "hit_1", "message": ""}, seq=1)])
    assert _key("call_1", HITL_STAGE_TOOL) not in snap.decisions_for


# ── 回归修复（final review：reply_as_result / detail / pop 顺序）───────────────────

def test_legacy_question_form_folds_to_reply_as_result_true():
    """form == "question" 的唯一生产者是 ask_user（control_capability.py:714），其契约
    就是 reply_as_result=True——答案直接当工具结果，不再入 provider。"""
    snap = fold_hitl_snapshot([_legacy_required(form="question")])
    assert snap.pending["hit_1"].reply_as_result is True


def test_legacy_approval_form_folds_to_reply_as_result_false():
    snap = fold_hitl_snapshot([_legacy_required(form="approval")])
    assert snap.pending["hit_1"].reply_as_result is False


def test_legacy_wait_form_context_mode_marker_is_not_leaked_into_detail():
    """wait 表单的旧 context 是模式标记（preface 已承接其语义），不是人类可读文案——
    塞进 detail 会把私有语义泄给人看（spec §4）。"""
    snap = fold_hitl_snapshot([
        _legacy_required(form="wait", tool_call_id="", context="interrupt:edit")])
    req = snap.pending["hit_1"]
    assert req.detail == ""
    assert req.delivery == UserTurnDelivery(task_id="t1", preface="interrupt_edit")


def test_malformed_hitl_resolved_with_empty_outcome_leaves_request_pending():
    """popped-before-validated 会让畸形事件把请求既不留在 pending、也不留下决定——凭空
    消失。正确方向是留在 pending：大不了被重新问一遍（Minor C）。"""
    snap = fold_hitl_snapshot([
        _opened(),
        _ev(EventType.HITL_RESOLVED, {"hitl_id": "hit_9", "outcome": ""}, seq=1)])
    assert "hit_9" in snap.pending
    assert _key("call_9", HITL_STAGE_TOOL) not in snap.decisions_for


# ── HitlReplyInjected：把「已终局」与「已了结」分开 ─────────────────────────────


def _injected(hitl_id="hit_u", seq=2) -> Event:
    return _ev(EventType.HITL_REPLY_INJECTED, {"hitl_id": hitl_id}, seq=seq)


def _user_turn_pair(hitl_id="hit_u", message="我的答复"):
    """一条 UserTurn park 的开+终局。UserTurn 没有 tool_call_id（`_cold_park`）。"""
    return [
        _ev(EventType.HITL_OPENED, {
            "hitl_id": hitl_id, "form": "wait",
            "delivery": {"kind": "user_turn", "task_id": "t1", "preface": "normal"},
            "subject_id": "", "prompt": "", "detail": "", "fields": [], "proposal": None,
            "tool_call_id": "", "agent_id": "a1", "resume_state": None,
            "reply_as_result": False, "stage": HITL_STAGE_TOOL,
        }, seq=0),
        _ev(EventType.HITL_RESOLVED, {
            "hitl_id": hitl_id, "outcome": "accepted", "claimed": False,
            "message": message,
        }, seq=1),
    ]


def test_resolved_user_turn_stays_in_resolved_until_it_is_injected():
    """没有 injected 事件 → 留在 `resolved`，等恢复期补注入。这是旧行为，不能变。"""
    snap = fold_hitl_snapshot(_user_turn_pair())
    assert "hit_u" in snap.resolved
    assert "hit_u" not in snap.pending


def test_injected_retires_it_from_resolved():
    """有了 injected → 从 `resolved` 销账。

    这是这条事件存在的全部理由：`resolved` 是恢复期补注入的清单，而它从前留的是**全部**
    已终局请求——交互式会话里每条用户消息都是一次 UserTurn HITL，所以那个清单随对话轮数
    线性增长。销账之后它只剩「答了但还没落进对话」的那几条。
    """
    snap = fold_hitl_snapshot([*_user_turn_pair(), _injected()])
    assert "hit_u" not in snap.resolved
    assert "hit_u" not in snap.pending, "销账不等于回到未决——人已经答过了"


def test_injected_does_not_touch_the_decision_cache():
    """**不动 `decisions_for`**：那一半的消费信号是 `CapabilityFinished`，不是「进对话」。

    在这里顺手 pop，会让一条仍需短路的冷决定消失 → 同一个工具重新求批一遍。
    """
    snap = fold_hitl_snapshot([
        _opened(),                                        # 带 tool_call_id="call_9"
        _ev(EventType.HITL_RESOLVED, {
            "hitl_id": "hit_9", "outcome": "accepted", "claimed": False,
            "message": "go"}, seq=1),
        _injected(hitl_id="hit_9", seq=2),
    ])
    assert "hit_9" not in snap.resolved
    assert _key("call_9", HITL_STAGE_TOOL) in snap.decisions_for


def test_retract_after_injected_still_puts_the_bubble_back_to_pending():
    """injected **不得**把请求从折叠的 `opened` 工作集里摘掉。

    摘了，`HitlReplyRetracted` 的 `req is None` 分支就 `continue`，气泡永远回不到未决——
    撤销之后那条请求既不在 pending、也没有决定，凭空消失。
    """
    snap = fold_hitl_snapshot([
        *_user_turn_pair(),
        _injected(),
        _ev(EventType.HITL_REPLY_RETRACTED, {"hitl_id": "hit_u"}, seq=3),
    ])
    assert "hit_u" in snap.pending
    assert "hit_u" not in snap.resolved
    assert snap.pending["hit_u"].reply_attempt == 1, "撤销次数仍要数出来（memory 幂等键第二维）"


def test_injected_for_an_unknown_hitl_id_is_a_no_op():
    """存量/外部流里孤立的一条 injected：不炸、不凭空造记录。"""
    snap = fold_hitl_snapshot([_injected(hitl_id="hit_never_seen")])
    assert snap.pending == {} and snap.resolved == {}


def test_injected_is_in_the_fold_type_set():
    """折叠读的类型集必须含它——漏了就等于这条事件不存在，清单继续线性增长。"""
    from ctx_weft.core.control.reducers import HITL_FOLD_EVENT_TYPES

    assert EventType.HITL_REPLY_INJECTED in HITL_FOLD_EVENT_TYPES
