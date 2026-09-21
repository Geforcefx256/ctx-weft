"""`TASK_HUMAN_RESOLVED` 焊死：两条 HITL 解除路径各走一次真实调用 + 真实 bus。

`tests/unit/test_task_unblock_events.py` 的 6 个用例全是 reducer 层合成事件
（`_ev(...)` 手工捏造），没有一个走真实发射——这份文件补上真实路径。

`TASK_HUMAN_RESOLVED` 有两条触发路径，互斥（同一次 HITL 解决只会走其中一条）：
  - `TaskManager.resume_task`（approval 分支，`was_blocked` 判据）；
  - `CtxWeftRuntime._inject_user_reply` → `TaskManager.mark_human_resolved`
    （wait_for_user 分支）。

**两者现在都只由 `TaskManager._emit` 发出**（Task 6：把 `_inject_user_reply` 里
绕开 TM、就地构造 `Event` 的越界发射收拢进 `mark_human_resolved`）——不再是「一条
在 TM 内、一条在 TM 外」的两个真实发射点，而是同一发射者的两条触发路径，合并成
一个测试用例，分别驱动两条冷路径后断言事件。

两条都借用 `tests/integration/test_hitl_e2e_v2.py` 已经搭好的真实端到端夹具
（真实 `CtxWeftRuntime.start_session` → 真实 `TaskManager` / `HitlService` /
`HitlRegistry`，只在 LLM 与工具 provider 两处打桩），驱动到同样的冷路径，
额外去读 `event_store` 断言 `TASK_HUMAN_RESOLVED` 真的发了、`hitl_id` 对得上。
"""
from __future__ import annotations

import asyncio
import time

import pytest

from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HitlReply, ToolResultDelivery, UserTurnDelivery
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import (
    _BASH_CALL,
    _ActRouterLLM,
    _finish_call,
    _make_runtime_with_bash_tool,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests._event_helpers import all_events

pytestmark = pytest.mark.asyncio


async def _poll(predicate, *, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError("timed out waiting for condition")


def _human_resolved_events(events, task_id: str):
    return [
        e for e in events
        if e.type == EventType.TASK_HUMAN_RESOLVED and e.task_id == task_id
    ]


async def test_task_human_resolved_emitted_via_task_manager() -> None:
    """两条触发路径都只由 `TaskManager._emit` 发出 `TASK_HUMAN_RESOLVED`（Task 6）。

    路径 1：`TaskManager.resume_task` 的 approval 分支（`was_blocked` 判据）——
    驱动与 `test_hitl_e2e_v2.test_cold_approval_reconciles_and_invokes_the_tool_exactly_once`
    完全相同的冷审批链路（零热窗强制驱逐 → AWAITING_HUMAN → `reply_to_hitl` accepted →
    `recover_agent` → `resume_task`），额外去读 `event_store` 断言
    `TASK_HUMAN_RESOLVED` 真的发出、`hitl_id` 与那次 HITL 请求一致。

    路径 2：`CtxWeftRuntime._inject_user_reply` → `TaskManager.mark_human_resolved`
    的 wait_for_user 分支——驱动与
    `test_hitl_e2e_v2.test_plain_text_pause_injects_reply_once_and_ignores_duplicate`
    完全相同的纯文本暂停冷路径（无 tool_call 的 act 回合 → 冷 park → `reply_to_hitl` →
    `recover_agent` → `_inject_user_reply`），同样断言事件真的发出，并额外验证重复
    应答（no-op）不会催生第二条。
    """
    # ── 路径 1：resume_task 的 approval 分支 ──────────────────────────────
    llm = _ActRouterLLM(act_responses=[_BASH_CALL, _finish_call()])
    runtime, tool = _make_runtime_with_bash_tool(llm, hitl_timeout_sec=0)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="please run ls", context_limit=100_000,
    ))
    sid = handle.session_id

    state = await handle.wait_for_finish(timeout=5.0)
    assert state is not None and state.task.status == "AWAITING_HUMAN"
    assert tool.invocations == 0

    pending = runtime.hitl_registry.list_pending(session_id=sid)
    assert len(pending) == 1
    req = pending[0]
    assert req.form == "approval"
    assert isinstance(req.delivery, ToolResultDelivery)

    view = await runtime.reply_to_hitl(HitlReply(
        hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
        modified_arguments={"command": "ls -l"},
    ))
    assert view is not None and view.outcome == "accepted"

    tm = runtime._task_managers[sid]

    def _final_task():
        t = tm.get_task(req.task_id)
        return t if (t is not None and t.status in ("FINISHED", "FAILED", "CANCELED")) else None

    task = await _poll(_final_task)
    assert task.status == "FINISHED"

    events = await all_events(runtime.event_store, sid)
    resolved = _human_resolved_events(events, req.task_id)
    assert len(resolved) == 1, (
        f"expected exactly one TaskHumanResolved for {req.task_id}, got {resolved!r}"
    )
    assert resolved[0].payload.get("hitl_id") == req.id

    # ── 路径 2：_inject_user_reply → mark_human_resolved 的 wait_for_user 分支 ──
    llm = _ActRouterLLM(act_responses=[
        MockResponse(text="Hi! Anything else?"),  # 纯文本、无 tool_call → 冷 park
        _finish_call(),
    ])
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi", context_limit=100_000,
    ))
    sid = handle.session_id

    state = await handle.wait_for_finish(timeout=5.0)
    assert state is not None and state.task.status == "AWAITING_HUMAN"

    pending = runtime.hitl_registry.list_pending(session_id=sid)
    assert len(pending) == 1
    req = pending[0]
    assert req.form == "wait"
    assert isinstance(req.delivery, UserTurnDelivery)

    reply = HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
                      message="use postgres too")
    first_view = await runtime.reply_to_hitl(reply)
    assert first_view is not None and first_view.outcome == "accepted"

    tm = runtime._task_managers[sid]

    def _final_task():
        t = tm.get_task(req.task_id)
        return t if (t is not None and t.status in ("FINISHED", "FAILED", "CANCELED")) else None

    task = await _poll(_final_task)
    assert task.status == "FINISHED"

    events = await all_events(runtime.event_store, sid)
    resolved = _human_resolved_events(events, req.task_id)
    assert len(resolved) == 1, (
        f"expected exactly one TaskHumanResolved for {req.task_id}, got {resolved!r}"
    )
    assert resolved[0].payload.get("hitl_id") == req.id

    # 重复应答是 no-op（resolve() 对已终局请求幂等返回 None），不该催生第二条。
    second_view = await runtime.reply_to_hitl(reply)
    assert second_view is None
    await asyncio.sleep(0.05)
    events_after = await all_events(runtime.event_store, sid)
    resolved_after = _human_resolved_events(events_after, req.task_id)
    assert len(resolved_after) == 1, (
        f"duplicate reply produced a second TaskHumanResolved: {resolved_after!r}"
    )
