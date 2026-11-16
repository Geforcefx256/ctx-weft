"""TASK_FAILED.error_message 须携带真死因（task.error = observer 的 task_failure_reason）。

此前 payload 填的是 verdict.act_recap（过程复述）——host 把它当 session_notice.reason_text
展示成「死因」，而 observer 专门写的 task_failure_reason（→ task.error）从未出核。
error_message = task.error；无死因（如规则 observe 判死）置空——过程复述不冒充死因，
固定提示文案由 host 合成 notice 时按 error_code 补。

Task 4 起 **TaskFailed 由 TaskManager 发**：FinalizeStep 只产出 `RunOutcome`
（带 observer 的原值 verdict + task.error），payload 由 `disposition_for` 算。
本文件因此改成「跑完 FinalizeStep → 把它产出的 RunOutcome 喂处置表 → 断 payload」，
continue 不因异常重试预算耗尽而生成失败收尾。
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import FinalizeStep
from ctx_weft.core.orchestrator.task.disposition import disposition_for
from ctx_weft.core.loop.steps.observe import Verdict
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


class _FakeTM:
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _loop_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_ctx(), task_manager=_FakeTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


async def _failed_state(mem, *, task_error: str | None, act_recap: str):
    scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=scope, content="do it",
                                 timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                                 role="user"), _ctx())
    task = Task(id="c1", session_id="s1", status="FAILED", tenant_id="default",
                assigned_agent_id="ag2", creator_agent_id="ag1", parent_task_id="p1",
                title="Doomed Task", user_prompt="do it", settings=NormalTaskSettings())
    task.error = task_error
    verdict = Verdict(task_outcome="fail", act_recap=act_recap, task_summary="")
    agent = SimpleNamespace(id="ag2", loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="r1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent, verdict=verdict, extra={})


def _task_failed_payload(step_outcome, task) -> dict:
    """FinalizeStep 的 RunOutcome → 处置表 → TaskManager 会发的那条 TaskFailed 的 payload。"""
    run_outcome = step_outcome.state_patch["run_outcome"]
    disp = disposition_for(run_outcome, retry_count=task.retry_count,
                           max_retries=task.max_retries)
    assert disp.event_type == "TaskFailed" and disp.status == "FAILED"
    return disp.payload


async def _continuing_state(mem, *, task_error: str | None):
    """observer 判 continue，异常重试计数已到上限。"""
    scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=scope, content="do it",
                                 timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                                 role="user"), _ctx())
    task = Task(id="c1", session_id="s1", status="PENDING", tenant_id="default",
                assigned_agent_id="ag2", creator_agent_id="ag1", parent_task_id="p1",
                title="Doomed Task", user_prompt="do it", settings=NormalTaskSettings())
    task.retry_count = task.max_retries
    task.error = task_error
    verdict = Verdict(task_outcome="continue", act_recap="第 3 轮尝试仍未通过校验", task_summary="")
    agent = SimpleNamespace(id="ag2", loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="r1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent, verdict=verdict, extra={})


@pytest.mark.asyncio
@pytest.mark.parametrize("task_error", ["登录页有人机校验，自动化被拦", None])
@pytest.mark.parametrize("has_verdict", [True, False])
async def test_continue_at_retry_limit_does_not_close_task(monkeypatch, task_error, has_verdict) -> None:
    mem = InMemoryMemoryProvider()
    state = await _continuing_state(mem, task_error=task_error)
    if not has_verdict:
        state.verdict = None

    async def unexpected_close(*args, **kwargs):
        pytest.fail("continue must not write terminal memory or blackboard results")

    monkeypatch.setattr("ctx_weft.core.loop.steps.finalize.apply_task_close", unexpected_close)
    outcome = await FinalizeStep().execute(state, _loop_ctx(mem))
    run_outcome = outcome.state_patch["run_outcome"]
    disp = disposition_for(run_outcome, retry_count=state.task.retry_count,
                           max_retries=state.task.max_retries)
    assert run_outcome.verdict == "continue"
    assert disp.status == "PENDING" and disp.event_type == "TaskRequeued"
    assert disp.payload["retry_count"] == state.task.max_retries
    assert state.task.retry_count == state.task.max_retries
    assert state.task.error == task_error
    assert state.task.error_code is None and state.task.finished_at is None
    assert outcome.events[0].payload["outcome"] == "continue"


@pytest.mark.asyncio
async def test_error_message_prefers_task_error() -> None:
    """observer 判死写了 task_failure_reason（→ task.error）→ error_message 用它，不用过程复述。"""
    mem = InMemoryMemoryProvider()
    state = await _failed_state(
        mem, task_error="第 2 步 API 调用 403：凭据无权限", act_recap="我调了 A、B 两个工具")
    outcome = await FinalizeStep().execute(state, _loop_ctx(mem))
    assert _task_failed_payload(outcome, state.task)["error_message"] == "第 2 步 API 调用 403：凭据无权限"


@pytest.mark.asyncio
async def test_error_message_empty_without_task_error() -> None:
    """无 task.error（如规则 observe 判死）→ error_message 置空，不拿过程复述冒充死因。"""
    mem = InMemoryMemoryProvider()
    state = await _failed_state(mem, task_error=None, act_recap="[test placeholder recap]")
    outcome = await FinalizeStep().execute(state, _loop_ctx(mem))
    assert _task_failed_payload(outcome, state.task)["error_message"] == ""
