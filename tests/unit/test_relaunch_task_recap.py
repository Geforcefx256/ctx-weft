"""冷恢复调度与共享的持久化 finish 对查找。"""

from __future__ import annotations

import asyncio

import pytest
from tests.step_helpers import patch_observe_start

import ctx_weft.core.runtime as rt_mod
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.model import ResolvedModel
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import qualify
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider



class _Client:
    account = "acct_default"
    model = "mdl_default"
    context_limit = 200_000
    output_reserve = 8192


def _fake_resolved_model() -> ResolvedModel:
    c = _Client()
    return ResolvedModel.from_client(c)


def _make_runtime_and_session():
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider, make_echo_template, make_runtime,
    )

    # 模板要真的能解出来：`_relaunch_observe` 现在按 agent record 的 template_id 现解
    # （从前由调用方连同会话模板一起传进来，那会让子 agent 的 recap 跑 root 的 identity）。
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=provider)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    session = Session(
        id="ses_1", tenant_id="default", user_prompt="x",
        root_agent_id="agt_1", llm_provider="acct1", llm_model="m1",
    )
    task = Task(
        id="tsk_1", session_id="ses_1", status="ACTIVE", tenant_id="default",
        assigned_agent_id="agt_1", creator_agent_id="agt_1", settings=NormalTaskSettings(),
    )
    task_manager = TaskManager(session_id="ses_1", event_bus=runtime.event_bus, max_concurrent=1)
    task_manager.set_session(session)

    # `_relaunch_observe` 不再收模板（它按 agent 自己的出身解），这个占位只为保持
    # fixture 的解包形状不变，本文件的用例都不再把它传下去。
    template = object()
    agent_id = "agt_1"
    return runtime, session, template, task_manager, task, agent_id, memory


async def _seed_finish_pair(memory: InMemoryMemoryProvider, session: Session, task: Task, agent_id: str) -> str:
    from datetime import UTC, datetime

    tcid = "tc_finish_1"
    scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent_id)
    pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id, task_id=task.id, agent_id=agent_id)
    await memory.ingest(
        MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN,
            address=scope,
            content="",
            timestamp=datetime(2026, 6, 12, tzinfo=UTC),
            role="assistant",
            metadata={
                "origin_task_id": task.id,
                "tool_calls": [{"id": tcid, "name": qualify("control:finish_task"), "input": {}}],
            },
        ),
        pctx,
    )
    await memory.ingest(MemoryEvent(
        type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
        content="report", role="tool", timestamp=datetime(2026, 6, 12, tzinfo=UTC), metadata={"origin_task_id": task.id, "tool_call_id": tcid}), pctx)
    return tcid


@pytest.fixture
def minimal_runtime_with_session():
    runtime, session, template, task_manager, task, agent_id, memory = _make_runtime_and_session()
    return runtime, session, template, task_manager, task, agent_id, memory


@pytest.mark.asyncio
async def test_relaunch_does_not_reinfer_finish_policy(minimal_runtime_with_session, monkeypatch):
    """close 边界重跑：从 memory 读到 finish 对 tool_call_id 后 register_close_synth，并以该 boundary 重跑。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session
    await _seed_finish_pair(memory, session, task, agent_id)
    task.status = "FINISHED"

    def _fake_materialize(self, agent_id, *, session_id=None, tenant_id=None):
        return (Agent(id=agent_id, session_id=session.id, template_id="agent:tpl_echo",
                       tenant_id=session.tenant_id), _fake_resolved_model())

    monkeypatch.setattr(AgentLifecycleManager, "materialize", _fake_materialize)

    captured = {}

    def _fake_register(task_id, tool_call_id, scope, outcome, raw_fold_scope=None):
        captured.update(task_id=task_id, tool_call_id=tool_call_id, outcome=outcome,
                        raw_fold_scope=raw_fold_scope, scope=scope)

    monkeypatch.setattr(rt_mod, "register_close_synth", _fake_register, raising=False)

    launched = {}

    def _fake_launch(state, ctx, *, boundary):
        launched["boundary"] = boundary
        return asyncio.create_task(asyncio.sleep(0))

    patch_observe_start(monkeypatch, _fake_launch)

    await runtime._relaunch_observe(
        session=session, task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="finish",
    )
    await asyncio.sleep(0)

    assert captured == {}, "runtime must not recreate a raw-fold decision from task status"
    assert launched["boundary"] == "finish"


@pytest.mark.asyncio
async def test_relaunch_is_best_effort_swallows_errors(minimal_runtime_with_session, monkeypatch):
    """任何一步失败：不得抛出，只记日志跳过（best-effort）。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentLifecycleManager, "materialize", _boom)

    await runtime._relaunch_observe(
        session=session, task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="finish",
    )  # 不应抛出


@pytest.mark.asyncio
async def test_find_finish_pair_tool_call_id_found_and_none():
    _, session, _, _, task, agent_id, memory = _make_runtime_and_session()
    scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent_id)
    pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id, task_id=task.id, agent_id=agent_id)

    from ctx_weft.core.loop.task_memory_fold import find_finish_target
    assert await find_finish_target(memory, pctx, scope, task.id) is None
    tcid = await _seed_finish_pair(memory, session, task, agent_id)
    target = await find_finish_target(memory, pctx, scope, task.id)
    assert target[1] == tcid


@pytest.mark.asyncio
async def test_relaunch_dispatch_boundary_no_close_synth(minimal_runtime_with_session, monkeypatch):
    """非 close 边界（dispatch，spec 2026-07-16）重跑：按原 boundary 重跑、
    不 register_close_synth（那是 close 边界替换占位 finish 对的专属动作）。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session
    task.status = "SUSPENDED"  # 委派挂起中崩溃的形态

    def _fake_materialize(self, agent_id, *, session_id=None, tenant_id=None):
        return (Agent(id=agent_id, session_id=session.id, template_id="agent:tpl_echo",
                       tenant_id=session.tenant_id), _fake_resolved_model())

    monkeypatch.setattr(AgentLifecycleManager, "materialize", _fake_materialize)

    registered = []
    monkeypatch.setattr(rt_mod, "register_close_synth",
                        lambda *a, **k: registered.append(a), raising=False)

    launched = {}

    def _fake_launch(state, ctx, *, boundary):
        launched["boundary"] = boundary
        return asyncio.create_task(asyncio.sleep(0))

    patch_observe_start(monkeypatch, _fake_launch)

    await runtime._relaunch_observe(
        session=session, task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="dispatch",
    )
    await asyncio.sleep(0)

    assert launched["boundary"] == "dispatch"
    assert registered == [], "dispatch 边界不得登记 close_synth"
