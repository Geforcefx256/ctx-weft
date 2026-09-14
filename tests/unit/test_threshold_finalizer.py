"""熔断胶囊闭合（Task 10）：`CtxWeftRuntime._finalize_threshold_memory` memory 闭合。

覆盖 task-10-brief.md 「Runtime 侧（Task 10）」段：
- root finish 对：仅当熔断亲手判死（root_task 非 None 且 started_at 非空）时，在 root scope
  合成 assistant finish_task 调用 + tool `[outcome=fail]` 前缀 + 失败清单文案。
- ack_task 派发对：parent scope 里已存在的 running ack 被 supersede，替换为取消文案。
- root=None：只做 ack 替换，不写 root finish 对。
- best-effort：某条 ack 写失败不阻断其余 ack 的处理。
"""

from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.loop.steps.finalize import _dispatch_running_ack, _ensure_dispatch_frame, _put_dispatch_result
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime

pytestmark = pytest.mark.asyncio


def _runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


def _session(**kw) -> Session:
    return Session(
        id="s1", user_prompt="do it", status="RUNNING", root_agent_id="agt_root",
        tenant_id="default", **kw,
    )


def _root(**kw) -> Task:
    return Task(
        id="root", session_id="s1", status="FAILED", parent_task_id=None,
        creator_agent_id="agt_root", assigned_agent_id="agt_root", title="Root Task",
        **kw,
    )


def _ack_child(tid: str, **kw) -> Task:
    return Task(
        id=tid, session_id="s1", status="CANCELED", parent_task_id="root",
        creator_agent_id="agt_root", assigned_agent_id="agt_child", title=f"Child {tid}",
        started_at=now_utc(), origin_tool_call_id=f"call-{tid}",
        **kw,
    )


async def _seed_running_ack(memory, session, child: Task) -> None:
    """先手工 ingest 一个派发框 + running ack（模拟子任务真正 start 后留下的痕迹）。"""
    parent_scope = MemoryAddress(session_id=session.id, task_id=child.parent_task_id,
                                agent_id=child.creator_agent_id)
    provider_ctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id,
                                   task_id=child.parent_task_id, agent_id=child.creator_agent_id)
    ts, tcid = await _ensure_dispatch_frame(memory, parent_scope, child, provider_ctx)
    await _put_dispatch_result(
        memory, parent_scope, child, _dispatch_running_ack(task_ref(child)), ts, provider_ctx,
        replace=False, tool_call_id=tcid,
    )


async def test_root_finish_pair_written_on_own_scope() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = _session()
    root = _root(started_at=now_utc())
    failures = [("c1", "boom1"), ("c2", "boom2"), ("c3", "boom3")]

    await runtime._finalize_threshold_memory(session, root, [], failures)

    scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    recs = await memory.recall_recent(scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)

    assistant = [r for r in recs if r.role == "assistant"]
    tool = [r for r in recs if r.role == "tool"]
    assert len(assistant) == 1
    assert len(tool) == 1
    tool_calls = assistant[0].metadata.get("tool_calls") or []
    assert any(tc.get("name", "").endswith("finish_task") for tc in tool_calls)
    assert "[outcome=fail]" in tool[0].content
    assert "Failure threshold hit" in tool[0].content
    assert "1) c1: boom1" in tool[0].content
    assert "2) c2: boom2" in tool[0].content
    assert "3) c3: boom3" in tool[0].content


async def test_ack_task_running_ack_replaced_with_cancelled_text() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = _session()
    child = _ack_child("c1")
    await _seed_running_ack(memory, session, child)

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    before = await memory.recall_recent(
        parent_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    tool_before = [r for r in before if r.role == "tool"]
    assert len(tool_before) == 1
    assert "is running now" in tool_before[0].content

    await runtime._finalize_threshold_memory(session, None, [child], [])

    after = await memory.recall_recent(
        parent_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    tool_after = [r for r in after if r.role == "tool"]
    assert len(tool_after) == 1
    assert "was cancelled mid-run" in tool_after[0].content
    assert "session failure threshold hit" in tool_after[0].content
    assert "its partial execution below is incomplete" in tool_after[0].content
    # 旧的 running ack 被 supersede（不再出现在 active recall 里）
    assert tool_after[0].id != tool_before[0].id


async def test_root_none_only_replaces_ack() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = _session()
    child = _ack_child("c1")
    await _seed_running_ack(memory, session, child)

    await runtime._finalize_threshold_memory(session, None, [child], [("x", "y")])

    root_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    # root scope 用的是 root task_id="root" 与 parent scope 相同 task_id（因为 parent_task_id="root"）
    # 但 root finish 对是 assistant + tool 的一对 finish_task 调用；没写 root finish 对时不应出现
    # finish_task tool_call。
    recs = await memory.recall_recent(root_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    assistant = [r for r in recs if r.role == "assistant"
                 and any(tc.get("name", "").endswith("finish_task")
                         for tc in (r.metadata.get("tool_calls") or []))]
    assert assistant == []


async def test_one_bad_ack_task_does_not_block_others() -> None:
    runtime = _runtime()
    memory = runtime.providers.get_memory()
    session = _session()
    good = _ack_child("good")
    bad = _ack_child("bad")
    await _seed_running_ack(memory, session, good)
    await _seed_running_ack(memory, session, bad)

    orig_ingest = memory.ingest

    async def _boom_for_bad(event, ctx):  # noqa: ANN001
        if event.role == "tool" and event.metadata.get("origin_task_id") == "root" \
                and getattr(event, "content", "") and "bad" in event.content:
            raise RuntimeError("simulated ingest failure")
        return await orig_ingest(event, ctx)

    memory.ingest = _boom_for_bad  # type: ignore[assignment]

    # 应不抛出——bad 的写失败被 best-effort 吞掉，good 仍正常替换。
    await runtime._finalize_threshold_memory(session, None, [good, bad], [])

    parent_scope = MemoryAddress(session_id="s1", task_id="root", agent_id="agt_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    recs = await memory.recall_recent(
        parent_scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 2000, ctx)
    tool_recs = {r.metadata.get("tool_call_id"): r for r in recs if r.role == "tool"}
    assert "was cancelled mid-run" in tool_recs["call-good"].content
    # bad 的写失败：旧 running ack 被替换失败前置的 supersede 步骤可能已执行，但至少不应崩溃
    # 整个流程；good 分支已验证不受影响即达成 best-effort 目标。
