"""compact 主键从 session 翻转到 agent（2026-09-04 spec §7.3）。

压缩折的是 agent 层的 dispatch log，本来就是 agent 粒度的操作；
session 只是从 agent 记录反查出来的。

夹具没有走 `start_session` + `wait_for_finish` 再 compact 那条路（controller ruling A）：
一旦 session 跑到 idle，`_release_session` 会回收它的 TaskManager，`ALM.release_session`
连带把 agent 记录也摘掉——`compact_agent` 随即对着一个刚存在过的 agent 抛
`AgentNotFound`。这是既有、已登记在案、明确排除在本 task 改动范围之外的拆卸时序问题
（controller ruling B：不为了让测试过而去改 `_release_session` 的策略）。

改走 `tests/unit/test_compact_session.py` 里既有的手工夹具模式：只灌 `event_store` +
往 memory provider 里 `ingest` `MemoryEvent`，再调
`rt._agent_lifecycle_manager.register_session(...)` /
`rt._agent_lifecycle_manager.materialize(...)` 手工水合出一个 agent record，全程不真的
跑一次 session。
"""

from __future__ import annotations

import dataclasses
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.protocols import (
    LoopConfig,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.protocols.agent import CompactReceipt
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._event_helpers import append_one
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


SID, AID = "ses_ca", "agt_root"


async def _idle_agent():
    """手工搭一个空闲、可 compact 的 session/agent（模式照抄 test_compact_session.py）。"""
    resolver = InlineAgentTemplateProvider()
    tmpl = dataclasses.replace(
        make_echo_template(), loop_config=LoopConfig(compact_keep_last=2),
    )
    resolver.register(tmpl)
    # 富余几条：agent 层大概率触发一次 summarize_for_compact，供了 task_id 的用例
    # 还可能再触发一次 task 层坍缩——两条 mock 响应打底，用不完也不报错。
    llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY")] * 3)
    # 占位 record（手工 materialize 一个从未 instantiate 过的 agent）没有自己的模板——
    # registry 不再借会话那份（2026-09-30）。兜底由 host 声明的母版负责，这里把它配上，
    # 走的是与生产同一条链路：RuntimeConfig.fallback_template_ref → TemplateLookup。
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(fallback_template_ref="agent:tpl_echo"))
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)

    ts = datetime(2026, 6, 16, tzinfo=UTC)
    await append_one(rt.event_store, Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id=SID,
        type=EventType.SESSION_CREATED, timestamp=ts,
        payload={"template_id": f"agent:{tmpl.id}", "user_prompt": "x", "root_agent_id": AID,
                 "llm_model": "mock", "context_limit": 180000},
    ))

    scope = MemoryAddress(session_id=SID, task_id="t_seed", agent_id=AID)
    pctx = ProviderContext(session_id=SID, tenant_id="default", task_id="t_seed", agent_id=AID)
    for i in range(5):
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.USER_PROMPT,
            address=MemoryAddress(session_id=SID, task_id=f"root{i}", agent_id=AID),
            content=f"body {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10)), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"user prompt {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"assistant summary {i}", role="assistant",
            timestamp=ts + timedelta(seconds=i * 10 + 1),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)

    rt._agent_lifecycle_manager.register_session(
        SID, tenant_id="default",
    )
    rt._agent_lifecycle_manager.materialize(AID, session_id=SID, tenant_id="default")
    return rt


@pytest.mark.asyncio
async def test_compact_agent_returns_receipt():
    rt = await _idle_agent()
    await initialize_all(rt)
    r = await rt.compact_agent(AID)
    assert isinstance(r, CompactReceipt)
    assert r.agent_id == AID
    assert r.session_id == SID


@pytest.mark.asyncio
async def test_transient_task_id_is_flagged():
    """不传 task_id 时返回的是内存载体 id，事件库里查不到——这条此前只写在 docstring 里。"""
    rt = await _idle_agent()
    await initialize_all(rt)
    r = await rt.compact_agent(AID)
    assert r.task_id_is_transient is True


@pytest.mark.asyncio
async def test_supplied_task_id_is_not_transient():
    rt = await _idle_agent()
    await initialize_all(rt)
    r = await rt.compact_agent(AID, task_id="root0")
    assert r.task_id == "root0"
    assert r.task_id_is_transient is False


@pytest.mark.asyncio
async def test_unknown_agent_raises():
    from ctx_weft.core.models.errors import AgentNotFound

    rt = await _idle_agent()
    with pytest.raises(AgentNotFound):
        await initialize_all(rt)
        await rt.compact_agent("agt_nope")


@pytest.mark.asyncio
async def test_compact_session_is_gone():
    """不留 shim（spec §1）。"""
    rt = await _idle_agent()
    assert not hasattr(rt, "compact_session")


@pytest.mark.asyncio
async def test_manual_compact_has_formal_step_events_and_monotonic_sequence():
    rt = await _idle_agent()
    await initialize_all(rt)
    events = []
    original_emit = rt._event_bus.emit

    async def emit(event):
        events.append(event)
        await original_emit(event)

    rt._event_bus.emit = emit
    await rt.compact_agent(AID)
    started = next(e for e in events if e.type == EventType.RUN_STARTED
                   and e.payload.get("initial_step") == "compact")
    run = [e for e in events if e.run_id == started.run_id]
    types = [e.type for e in run]
    assert types[0:2] == [EventType.RUN_STARTED, EventType.STEP_STARTED]
    assert run[1].payload["step_name"] == "compact"
    assert types[-2:] == [EventType.STEP_COMPLETED, EventType.RUN_FINISHED]
    assert run[-2].payload["next_step"] is None
    assert run[-1].payload["outcome"] == "completed"
    sequences = [e.sequence for e in run]
    assert sequences == sorted(set(sequences))
    assert not any(e.type in {EventType.TASK_FINISHED, EventType.TASK_FAILED, EventType.TASK_CANCELED}
                   for e in run)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_manual_compact_propagates_failure_and_releases_busy_guard(monkeypatch, error_type):
    from ctx_weft.core.loop.steps.compact import CompactStep
    rt = await _idle_agent()
    await initialize_all(rt)
    events = []
    original_emit = rt._event_bus.emit

    async def emit(event):
        events.append(event)
        await original_emit(event)

    async def fail(self, state, ctx):
        assert ctx.execution.options["after"] is None
        assert ctx.execution.options["force"] is True
        raise error_type("manual compact failed")

    rt._event_bus.emit = emit
    monkeypatch.setattr(CompactStep, "execute", fail)
    with pytest.raises(error_type, match="manual compact failed"):
        await rt.compact_agent(AID)
    assert SID not in rt._busy_sessions
    finished = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert finished.payload["outcome"] == ("canceled" if error_type is asyncio.CancelledError else "interrupted")
    assert not any(e.type == EventType.STEP_COMPLETED for e in events)
