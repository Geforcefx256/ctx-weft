"""后台判决交给 TaskManager 仲裁，带外收尾发送真实事件并关闭本任务的 park 气泡。

机械判决已随 ObserveRequested 固定时，后台只生成报告；需要 LLM 判决的路径才提交新判决。
状态转换的真实 TaskManager 契约由 test_out_of_band_verdict.py 覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.core.capabilities.control_tools import ControlMetaKey as K
from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.loop.steps.observe import judges, submit_verdict
from ctx_weft.core.models.task import Task
from ctx_weft.protocols.events import EventType


class _RecordingTM:
    """带外入口的替身：**只记下收到了什么，不代替真品去调 `finalize`**。

    分层是刻意的。「什么时候调收尾」是带外入口自己的契约（只在判决被接受且是 `success`
    时，且跑在「状态已写定」与「转移已宣布」之间），那三条约束钉在
    `test_out_of_band_verdict.py` 里、打的是**真** `TaskManager`——替身复述一遍只会让两处
    对同一件事各有一份说法。本文件管的是另一半：传了什么进去、那个回调自己做了什么。

    曾经让这个替身照契约调过一次 `finalize`，于是它必须背上 memory / llm / event_bus 一整套
    替身，而气泡规则本身反倒读不清了。
    """

    def __init__(self, accepted: bool = True) -> None:
        self.calls: list[tuple] = []
        self._accepted = accepted

    async def apply_out_of_band_verdict(self, task_id, outcome, **kw):
        self.calls.append((task_id, outcome, kw))
        return self._accepted


def _state():
    return SimpleNamespace(task=Task(id="t1", session_id="s1", status="ACTIVE", outputs="交付物"), extra={})


def _ctx(tm):
    return SimpleNamespace(task_manager=tm)


_ctx_with_tm = _ctx


def _meta(outcome: str = "success", **kw) -> dict:
    base = {
        K.OBSERVER_OUTCOME: outcome,
        K.OBSERVER_ACT_RECAP: "做了甲乙丙",
        K.OBSERVER_TASK_SUMMARY: "全程小结",
        K.OBSERVER_NEXT_STEP_HINT: "",
        K.OBSERVER_FAILURE_REASON: "",
    }
    base.update(kw)
    return base


# ── 只有 plain_text 判定 ──────────────────────────────────────────────────────

def test_the_two_yielding_boundaries_judge() -> None:
    """产 verdict 的恰是「让位给人」的那两个：说了段话（`plain_text`）与宣布做完了
    （`finish_park`，S-b 2026-09-27）。root 上这两种收尾不该一个被复核、一个不被。

    `finish` / `normal` 是**不让位**的那条 finish_task（无人值守 / 子任务 / 无 hitl），判定由
    前台给出；`mechanical` 是机械判决刚判过；`interrupt` / `dispatch` 压根不是一个结局。后台
    再判一次只会把那份判决覆盖掉。
    """
    assert judges("plain_text") is True
    assert judges("finish_park") is True
    for boundary in ("mechanical", "finish", "normal", "interrupt", "dispatch"):
        assert judges(boundary) is False, f"{boundary} 不该产 verdict"


# ── 提交 ──────────────────────────────────────────────────────────────────────

async def test_success_verdict_is_submitted_with_every_field() -> None:
    tm = _RecordingTM()
    await submit_verdict(_state(), _ctx(tm), _meta(
        **{K.OBSERVER_NEXT_STEP_HINT: "Next Step Hint: 下一步"}))

    assert len(tm.calls) == 1
    task_id, outcome, kw = tm.calls[0]
    assert task_id == "t1"
    assert outcome.verdict == "success"
    assert outcome.outputs == "交付物"
    assert kw["process_report"] == "做了甲乙丙"
    assert kw["task_summary"] == "全程小结"
    assert kw["next_step_hint"] == "Next Step Hint: 下一步"


async def test_retry_verdict_is_submitted_too() -> None:
    """retry 也提交——hint 要落地供下一轮用；「维持 park」是带外入口那边的分支。"""
    tm = _RecordingTM()
    await submit_verdict(_state(), _ctx(tm), _meta(
        "continue", **{K.OBSERVER_FAILURE_REASON: "缺凭据"}))

    _tid, outcome, _kw = tm.calls[0]
    assert outcome.verdict == "continue"
    assert outcome.error == "缺凭据"


# ── verdict 缺失 ≡ retry（什么都不提交，维持 park）────────────────────────────

async def test_missing_verdict_submits_nothing() -> None:
    tm = _RecordingTM()
    await submit_verdict(_state(), _ctx(tm), _meta(outcome=""))
    assert tm.calls == []


async def test_empty_metadata_submits_nothing() -> None:
    """LLM 没调 terminal tool → metadata 空 → 绝不静默放行后继。"""
    tm = _RecordingTM()
    await submit_verdict(_state(), _ctx(tm), {})
    assert tm.calls == []


async def test_missing_task_manager_is_tolerated() -> None:
    """拿不到 TaskManager（手构 ctx / 已拆会话）：记日志，不炸。"""
    await submit_verdict(_state(), SimpleNamespace(task_manager=None), _meta())


# ── 被拒绝不是错误 ────────────────────────────────────────────────────────────

async def test_rejected_verdict_is_not_an_error() -> None:
    """人先开口 → 仲裁拒绝。后台只记一笔，不抛、不重试。"""
    tm = _RecordingTM(accepted=False)
    await submit_verdict(_state(), _ctx(tm), _meta())
    assert len(tm.calls) == 1


# ── park 气泡：判 success 就收，不分 root 与子任务（2026-09-27）──────────────
#
# task 被判完成了，那个让位入口就用掉了。人后面还想说话就再发一条消息——`send_message`
# 看到 `current_task_id` 已终态，自己会开新的一轮（`_start_task_for_agent`），不需要留着
# 这个气泡当入口。
#
# 留着的代价两头都疼：宿主折会话状态时气泡**优先于** task 终态（`SessionStatusFold.status`
# 里 `pending_hitl` 排在 `terminal` 之前），于是会话永久停在 PAUSED，SSE 的终态收口不触发、
# `session_is_quiescent` 一律拒绝逐出；子任务更糟——线已经交回 parent，再没有人会来答它，
# 每个这样的子任务留一个、会累积。
#
# `faedd25` 曾为 root 保下这个气泡。两条理由只有一条成立（「会话会显示已完成」——而判决说它
# 完成了，显示已完成是实话），另一条「投递路径跟着状态分叉」本就与气泡无关：`send_message`
# 的路由只看 `current_task_id` 是否终态，从不看气泡。
#
# 这一节直接测 `_close_park_bubble`：它是气泡规则的所在，而它的兄弟（带外 finalize 的
# memory 那一半）要一整套 memory/llm 替身，混在一个用例里只会让规则本身读不清。两者怎么
# 组装、由谁调，钉在下一节。

from ctx_weft.core.loop.steps.observe import (  # noqa: E402
    _close_park_bubble,
    _out_of_band_finalize,
)
from ctx_weft.protocols.hitl import ToolResultDelivery, UserTurnDelivery  # noqa: E402


class _Bubble:
    def __init__(self, hid: str, task_id: str, delivery=None) -> None:
        self.id, self.task_id = hid, task_id
        self.delivery = delivery if delivery is not None else UserTurnDelivery(task_id=task_id)


class _Registry:
    def __init__(self, bubbles: list) -> None:
        self._bubbles = bubbles

    def list_pending(self, session_id=None, **kw):
        return list(self._bubbles)


class _Hitl:
    def __init__(self, bubbles: list) -> None:
        self.registry = _Registry(bubbles)
        self.cancelled: list[str] = []

    async def cancel(self, hitl_id, *, message="", defer=False):
        self.cancelled.append(hitl_id)
        return None


def _ctx_with_hitl(hitl):
    return SimpleNamespace(hitl=hitl)


def _state_with_session(*, parent: str | None = None, creator="ag_1", assigned="ag_1"):
    return SimpleNamespace(
        task=Task(id="t1", session_id="s1", status="FINISHED", outputs="交付物", parent_task_id=parent,
                             creator_agent_id=creator, assigned_agent_id=assigned),
        session=SimpleNamespace(id="s1"),
        extra={},
    )


# ── 收，不看 parent ───────────────────────────────────────────────────────────

async def test_a_subtask_bubble_is_closed() -> None:
    """线交回 parent 了，这个气泡再没有人会来答——留着会把会话钉死在 PAUSED。"""
    hitl = _Hitl([_Bubble("hit_1", "t1")])
    await _close_park_bubble(_state_with_session(parent="t0"), _ctx_with_hitl(hitl))
    assert hitl.cancelled == ["hit_1"]


async def test_a_root_bubble_is_closed_too() -> None:
    """**要害**：root 也收。人后面还想说话就再发一条消息，`send_message` 看到
    `current_task_id` 已终态自己会开新的一轮——不需要留着这个气泡当入口。

    留着它的后果是会话永久停在 PAUSED（气泡在折叠里优先于 task 终态），连
    `forget_session` 都一律被拒。
    """
    hitl = _Hitl([_Bubble("hit_1", "t1")])
    await _close_park_bubble(_state_with_session(parent=None), _ctx_with_hitl(hitl))
    assert hitl.cancelled == ["hit_1"]


async def test_the_parent_task_id_is_not_part_of_the_predicate() -> None:
    """判据里没有 `parent_task_id`，也没有 `_is_own_root`——三种形态一视同仁。

    曾经按「这条线交回给谁了」分过 root 与子任务（2026-09-27 上午），当天下午取消：判完成
    就是判完成，下一句话本来就该开新一轮。
    """
    for parent, creator, assigned in (("t0", "ag_1", "ag_1"),      # 同 agent 子任务
                                      ("t0", "ag_1", "ag_2"),      # 跨 agent 子任务
                                      (None, "ag_1", "ag_1")):     # root
        hitl = _Hitl([_Bubble("hit_1", "t1")])
        await _close_park_bubble(
            _state_with_session(parent=parent, creator=creator, assigned=assigned),
            _ctx_with_hitl(hitl))
        assert hitl.cancelled == ["hit_1"], f"parent={parent} assigned={assigned} 漏收了"


# ── 过滤 ──────────────────────────────────────────────────────────────────────

async def test_only_this_task_and_only_user_turn_bubbles() -> None:
    """双重过滤：兄弟任务的气泡不归这次判决管；同一 task 上「等人拍板」那一档也不归——
    它有自己的处置路径，本函数只认「等人开口」。"""
    hitl = _Hitl([
        _Bubble("hit_mine", "t1"),
        _Bubble("hit_sibling", "t2"),
        _Bubble("hit_approval", "t1", delivery=ToolResultDelivery(tool_call_id="tc_1")),
    ])
    await _close_park_bubble(_state_with_session(parent="t0"), _ctx_with_hitl(hitl))
    assert hitl.cancelled == ["hit_mine"]


async def test_a_missing_hitl_service_is_tolerated() -> None:
    """手构 ctx 里没有 hitl（单测替身很常见）：静默跳过，不炸。"""
    await _close_park_bubble(_state_with_session(parent="t0"), SimpleNamespace(hitl=None))


# ── 收尾怎么组装、由谁调（2026-09-27）────────────────────────────────────────
#
# park 之后 `FinalizeStep` 压根没跑过（`HitlPark` 在 ActStep 里就把 run 结束了），而它是
# close/bubble、blackboard、TASK_FINALIZED 三者的唯一发生地。带外收尾把同一份实现照样跑
# 一遍——这一节只钉「调了什么、事件发出去了、谁来调它」，memory 那一半的语义归 finalize
# 自己的用例。

class _Bus:
    def __init__(self) -> None:
        self.emitted: list = []

    async def emit(self, event) -> None:
        self.emitted.append(event)

    async def emit_batch(self, events, *, on_committed=None) -> None:
        self.emitted.extend(events)
        if on_committed is not None:
            on_committed()


async def test_out_of_band_finalize_runs_close_and_emits_its_events(monkeypatch, fake_state_ctx) -> None:
    """**要害**：带外收尾把 `apply_task_close` 的产物发出去，并补上 `TASK_FINALIZED`。

    不发 TASK_FINALIZED 的后果是 host 的 `tasks.outputs_json` 恒空——S5 起一直如此。
    """
    import ctx_weft.core.loop.steps.finalize as fin

    calls: dict = {}

    async def _fake_close(state, task, ctx, **kw):
        calls["close"] = kw
        return [make_event(state, EventType.MEMORY_INGESTED, {"source": "root_finish_pair"})]

    monkeypatch.setattr(fin, "apply_task_close", _fake_close)

    bus = _Bus()
    state, ctx = fake_state_ctx
    state.task = _state_with_session(parent="t0").task
    state.extra["observe_id"] = "obs-test"
    ctx.hitl = _Hitl([])
    ctx.event_bus = bus
    await _out_of_band_finalize(state, ctx, _meta())

    assert [e.type for e in bus.emitted] == [EventType.MEMORY_INGESTED, EventType.TASK_FINALIZED]
    assert all(e.causation_id == "obs-test" for e in bus.emitted)
    await _out_of_band_finalize(state, ctx, _meta())
    assert len(bus.emitted) == 2
    assert calls["close"]["outcome"] == "success"
    assert calls["close"]["act_recap"] == "做了甲乙丙"
    assert calls["close"]["task_summary"] == "全程小结"
    # 判决方就是 LLM，摘要是真的 → close 即折末段 raw，不走占位-替换那条延迟路径。
    assert calls["close"]["has_llm_summary"] is True


async def test_out_of_band_finalize_also_closes_the_bubble(monkeypatch, fake_state_ctx) -> None:
    """收尾与收气泡是同一次带外动作的两半，别让将来有人只接一半。"""
    import ctx_weft.core.loop.steps.finalize as fin

    async def _fake_close(state, task, ctx, **kw):
        return []

    monkeypatch.setattr(fin, "apply_task_close", _fake_close)

    hitl = _Hitl([_Bubble("hit_1", "t1")])
    state, ctx = fake_state_ctx
    state.task = _state_with_session(parent="t0").task
    ctx.hitl = hitl
    ctx.event_bus = _Bus()
    await _out_of_band_finalize(state, ctx, _meta())
    assert hitl.cancelled == ["hit_1"]


async def test_the_verdict_entry_is_handed_the_finalize_callback() -> None:
    """`_submit_verdict` 必须把收尾交给带外入口，而不是自己就地跑。

    位置是要害：带外入口在「状态已写定」与「转移已宣布」之间调它（见
    `apply_out_of_band_verdict` 的三条约束）。自己就地跑 = 跑在 `_settle` 之后，parent 可能
    已经醒过来装配、而子任务的产出还没落地。
    """
    tm = _RecordingTM()
    await submit_verdict(_state_with_session(parent="t0"), _ctx_with_tm(tm), _meta())
    _tid, _outcome, kw = tm.calls[0]
    assert callable(kw["finalize"])
