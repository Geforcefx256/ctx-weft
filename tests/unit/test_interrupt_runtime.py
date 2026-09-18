"""Runtime: pause_session = 弃子（只留 root agent 那一轮继续跑，其余在途 run 与排队子任务全部弃）；
cancel_session = 硬取消（cancel 全部在途 run + cancel-all，会话终态 CANCELED）。"""

from types import SimpleNamespace

import pytest

from datetime import UTC, datetime

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.hitl.registry import PendingHitl
from ctx_weft.core.models.session import Session
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    HitlDecision,
    UserTurnDelivery,
)
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.protocols.memory import MemoryEvent, MemoryEventType
from ctx_weft.providers.llm.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit.test_runtime_agent_api import _plant

pytestmark = pytest.mark.asyncio


def _runtime():
    return make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=InlineAgentTemplateProvider())

def _user_turn_req(
    *, hitl_id="hit1", session_id="s1", task_id="t1", agent_id="ag1",
    outcome="accepted", message="ship it", preface=PREFACE_NORMAL,
):
    """一条**已终局**的 `UserTurn` 请求——`_inject_user_reply` / `_write_hitl_reply_turn`
    收的就是这个形态（core 内部的活记录，不是 host 视图）。"""
    req = PendingHitl(
        id=hitl_id, form=HITL_FORM_WAIT, session_id=session_id, task_id=task_id,
        agent_id=agent_id,
        delivery=UserTurnDelivery(task_id=task_id, preface=preface),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    req.decision = HitlDecision(outcome=outcome, message=message)
    req.resolved_at = datetime(2026, 1, 1, tzinfo=UTC)
    return req


async def test_pause_session_without_tm_cancels_all_runs():
    # 无 TM（纯 registry 残留）：无法辨认 root agent → 全部按"其余"cancel，返回 True
    rt = _runtime()
    a = rt._register_run_tokens("s1", "t1")
    b = rt._register_run_tokens("s1", "t2")
    assert await rt.pause_session("s1") is True
    assert a.cancel.is_cancelled and b.cancel.is_cancelled


async def test_cancel_session_cancels_all_run_tokens_and_drains_queue():
    """2026-09-04 spec §7.2 起，在途 run 的取消经 `cancel_agent`（对 `running` 目标
    内部调 `_cancel_run_token`）覆盖，runtime 不再自己遍历 `_run_tokens`——因此这里
    要有一个 `running` 且 `current_task_id="t1"` 的 agent record，`cancel_agent` 才会
    找到并取消这个 run token（与 `_plant_live_task` 同一手法）。"""
    rt = _runtime()
    tokens = rt._register_run_tokens("s1", "t1")
    _plant(rt, "ag1", None, session_id="s1", status="running")
    rt._agent_lifecycle_manager._agents["ag1"].current_task_id = "t1"
    drained = {"called": False}

    class _TM:
        def is_done(self):
            return False          # active drain in flight → _on_done reclaims after cancel completes

        async def cancel_all(self, *, reason=""):
            drained["called"] = True

    rt._task_managers["s1"] = _TM()
    assert await rt.cancel_session("s1") is True
    assert tokens.cancel.is_cancelled is True
    assert drained["called"] is True


async def test_unknown_session_returns_false():
    rt = _runtime()
    assert await rt.pause_session("nope") is False
    assert await rt.cancel_session("nope") is False


@pytest.mark.asyncio
async def test_inject_user_reply_phase1_adds_edit_note():
    rt = _runtime()
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)
    session = Session(id="s1", tenant_id="default", user_prompt="X", status="WAITING", token_budget=0)
    task = SimpleNamespace(id="t1", status="SUSPENDED", outputs=None, process_report=None)
    # children_of：`_inject_user_reply` 现在据它跳过「SUSPENDED 在活子任务上」的父任务
    # （复审 I7）。这里的 fake 无子任务。
    tm = SimpleNamespace(get_task=lambda tid: task, children_of=lambda tid: set(),
                         mark_human_resolved=_fake_mark_human_resolved({"t1": task}))
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1")
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.USER_PROMPT, address=scope, content="原始请求X",
        timestamp=now_utc(), role="user",
    ), pctx)

    req = _user_turn_req(hitl_id="h1", message="新请求Y",
                         preface=PREFACE_AFTER_INTERRUPT_EDIT)
    await rt._inject_user_reply(req, session, tm)

    recs = await mem.recall_recent(scope, [MemoryEventType.USER_PROMPT], 10, pctx)
    assert any(
        "原始请求X" in (r.content or "") and "新请求Y" in (r.content or "") and "cancelled" in (r.content or "")
        for r in recs
    )


@pytest.mark.asyncio
async def test_inject_user_reply_non_edit_has_no_note():
    rt = _runtime()
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)
    session = Session(id="s1", tenant_id="default", user_prompt="X", status="WAITING", token_budget=0)
    task = SimpleNamespace(id="t1", status="SUSPENDED", outputs=None, process_report=None)
    # children_of：`_inject_user_reply` 现在据它跳过「SUSPENDED 在活子任务上」的父任务
    # （复审 I7）。这里的 fake 无子任务。
    tm = SimpleNamespace(get_task=lambda tid: task, children_of=lambda tid: set(),
                         mark_human_resolved=_fake_mark_human_resolved({"t1": task}))
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1")

    req = _user_turn_req(hitl_id="h1", message="just continue",
                         preface=PREFACE_AFTER_INTERRUPT)  # ② not edit
    await rt._inject_user_reply(req, session, tm)
    recs = await mem.recall_recent(scope, [MemoryEventType.USER_PROMPT], 10, pctx)
    assert any(r.content == "just continue" for r in recs)
    assert all("cancelled" not in (r.content or "") for r in recs)


# ── I7：reply 路径上的 task 状态重置必须是有条件的 ────────────────────────────


def _fake_mark_human_resolved(by_id: dict):
    """`TaskManager.mark_human_resolved` 的轻量替身：不看当前状态，非终态即置 PENDING。

    与生产实现（`task_manager.py`）同形，只是不发事件——这些测试只关心 `_inject_user_reply`
    交出的状态转移，事件真实发射由 `test_task_human_resolved_e2e.py` 走真实 TM 覆盖。
    """
    async def _mark_human_resolved(tid, *, hitl_id):
        t = by_id.get(tid)
        if t is not None and t.status not in ("FINISHED", "FAILED", "CANCELED"):
            t.status = "PENDING"
    return _mark_human_resolved


def _tm_with_children(parent, children):
    """真实 `TaskManager` 的三个被用到的读/写接口：`get_task` / `children_of` / `mark_human_resolved`。"""
    by_id = {t.id: t for t in [parent, *children]}
    return SimpleNamespace(
        get_task=lambda tid: by_id.get(tid),
        children_of=lambda tid: {c.id for c in children} if tid == parent.id else set(),
        mark_human_resolved=_fake_mark_human_resolved(by_id),
    )


async def test_inject_user_reply_leaves_a_parent_suspended_on_live_children_alone():
    """SUSPENDED + 尚有活子任务 = `restore` 刻意不重排的那一种形状。

    在这里无条件翻成 PENDING 却没有人入队，`_try_resume_parent` 的
    `status == "SUSPENDED"` 门随之失效 → 子任务收尾时那次合法唤醒被静默吞掉，
    父任务永久停摆（复审 I7）。答复照写，状态不碰。
    """
    rt = _runtime()
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)
    session = Session(id="s1", tenant_id="default", user_prompt="X",
                      status="WAITING", token_budget=0)
    parent = SimpleNamespace(id="t1", status="SUSPENDED", outputs="old",
                             process_report="rep", process_report_at="then")
    child = SimpleNamespace(id="t2", status="ACTIVE")
    tm = _tm_with_children(parent, [child])

    await rt._inject_user_reply(_user_turn_req(), session, tm)

    assert parent.status == "SUSPENDED"         # 门还在，唤醒不会丢
    assert parent.outputs == "old"              # 与本窗口无关的进度不被清掉
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default",
                           task_id="t1", agent_id="ag1")
    view = await mem.load_view(scope, MemoryScope.TASK, pctx)
    assert any(r.role == "user" and "ship it" in str(r.content) for r in view), (
        "答复仍然必须被写进对话——跳过的只是状态重置"
    )


async def test_inject_user_reply_still_resets_a_parent_whose_children_are_done():
    """子任务全终态 ⟹ 不是那种形状 ⟹ 照旧掰回 PENDING（`restore` 也会重排它）。"""
    rt = _runtime()
    rt.providers.register_memory(InMemoryMemoryProvider())
    session = Session(id="s1", tenant_id="default", user_prompt="X",
                      status="WAITING", token_budget=0)
    parent = SimpleNamespace(id="t1", status="SUSPENDED", outputs="old",
                             process_report="rep", process_report_at="then")
    child = SimpleNamespace(id="t2", status="FINISHED")
    await rt._inject_user_reply(_user_turn_req(), session, _tm_with_children(parent, [child]))
    assert parent.status == "PENDING"
    assert parent.outputs is None


# ── I6：恢复期补写的界 ────────────────────────────────────────────────────────


async def test_inject_resolved_user_turns_skips_replies_already_in_the_conversation():
    """已经注入过的答复不再重走一次写入（复审 I6 的界）。

    判据不是启发式：`hitlreply:{hitl_id}` 这条记忆记录只可能由
    `_write_hitl_reply_turn` 自己写下，「在对话里」**就是**「已经注入过」。
    """
    rt = _runtime()
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)
    session = Session(id="s1", tenant_id="default", user_prompt="X",
                      status="WAITING", token_budget=0)
    task = SimpleNamespace(id="t1", status="ACTIVE", outputs=None,
                           process_report=None, process_report_at=None)
    tm = _tm_with_children(task, [])

    done = _user_turn_req(hitl_id="h_done", message="already said")
    fresh = _user_turn_req(hitl_id="h_fresh", message="not yet said")
    rt.hitl_registry._requests[done.id] = done
    rt.hitl_registry._requests[fresh.id] = fresh

    # h_done 的答复已经在对话里（上一次恢复写下的）
    await rt._write_hitl_reply_turn(done, session, task)

    writes: list[str] = []
    original = rt._write_hitl_reply_turn

    async def _spy(req, sess, target, **kw):
        writes.append(req.id)
        return await original(req, sess, target, **kw)

    rt._write_hitl_reply_turn = _spy  # type: ignore[method-assign]
    await rt._inject_resolved_user_turns(session, tm)

    assert writes == ["h_fresh"], "已注入过的那条不该再写一次；没注入过的那条必须补上"
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default",
                           task_id="t1", agent_id="ag1")
    view = await mem.load_view(scope, MemoryScope.TASK, pctx)
    texts = [str(r.content) for r in view if r.role == "user"]
    assert any("already said" in t for t in texts)
    assert any("not yet said" in t for t in texts)
