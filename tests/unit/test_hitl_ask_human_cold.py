"""act 纯文本暂停（wait_for_user）的冷路径——**注入侧**。

热路径：actor 纯文本 → HITL park → 用户回复经冷 resume 注入 task 层 → 重入 act。
冷路径：超时驱逐 / 重启使 reconcile 覆盖不到（无 dangling tool_call）→ recover_agent 收到
user_reply 时在 drain 前注入回复。本文件钉死 `_inject_user_reply` 的注入逻辑。

「哪种请求走注入、哪种走 reconcile」的分流原先也在这里（按 `form` 判），现在由
**delivery** 决定、且只在 `reply_to_hitl` 这一个入口分流：见
`test_runtime_hitl_wiring.py` 的 `test_reply_returns_the_view_and_drives_resume_by_delivery`
/ `test_user_turn_delivery_injects_instead_of_reconciling` / `test_no_resume_delivery_triggers_nothing`。
"""

from __future__ import annotations

import pytest

from datetime import UTC, datetime

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.hitl.registry import PendingHitl
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_NORMAL,
    HitlDecision,
    UserTurnDelivery,
)
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime



def _runtime_with_memory() -> tuple[CtxWeftRuntime, InMemoryMemoryProvider]:
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)
    return rt, mem


def _tm_with_task(status: str = "SUSPENDED") -> tuple[TaskManager, Task]:
    tm = TaskManager(session_id="s1")
    task = Task(id="t1", session_id="s1", status=status, title="T1")
    tm._tasks["t1"] = task                    # restore 在真实路径已登记;这里直接放
    return tm, task


async def _recall_user_prompts(mem: InMemoryMemoryProvider) -> list:
    return await mem.recall_recent(
        scope=MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1"),
        types=[MemoryEventType.USER_PROMPT],
        limit=10,
        ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="t1"),
    )


def _wait_for_user_req(outcome: str = "accepted", message: str = "ship it") -> PendingHitl:
    return _user_turn_req(outcome=outcome, message=message)


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


@pytest.mark.asyncio
async def test_inject_writes_user_prompt_and_requeues() -> None:
    rt, mem = _runtime_with_memory()
    tm, task = _tm_with_task()
    session = Session(id="s1", tenant_id="default", user_prompt="x")

    await rt._inject_user_reply(_wait_for_user_req(message="ship it"), session, tm)

    assert task.status == "PENDING"
    assert task.process_report is None
    recs = await _recall_user_prompts(mem)
    assert any(r.content == "ship it" and r.metadata.get("source") == "hitl_reply" for r in recs)


@pytest.mark.asyncio
async def test_inject_reject_writes_declined() -> None:
    rt, mem = _runtime_with_memory()
    tm, _ = _tm_with_task()
    session = Session(id="s1", tenant_id="default", user_prompt="x")

    await rt._inject_user_reply(_wait_for_user_req(outcome="rejected", message="not now"), session, tm)

    recs = await _recall_user_prompts(mem)
    assert any("Human declined: not now" in r.content for r in recs)


@pytest.mark.asyncio
async def test_inject_missing_task_is_noop() -> None:
    rt, mem = _runtime_with_memory()
    tm = TaskManager(session_id="s1")             # 无 t1
    session = Session(id="s1", tenant_id="default", user_prompt="x")

    await rt._inject_user_reply(_wait_for_user_req(), session, tm)   # 不抛

    assert await _recall_user_prompts(mem) == []
