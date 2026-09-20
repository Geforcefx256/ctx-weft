"""Task 6: runtime 重跑 recap 辅助方法（_find_finish_pair_tool_call_id / _relaunch_task_recap）。"""

from __future__ import annotations

import asyncio

import pytest

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

pytestmark = pytest.mark.asyncio


class _Client:
    account = "acct_default"
    model = "mdl_default"
    context_limit = 200_000
    output_reserve = 8192


def _fake_resolved_model() -> ResolvedModel:
    c = _Client()
    return ResolvedModel(client=c, account=c.account, model=c.model,
                         context_limit=c.context_limit, reserved_output_tokens=c.output_reserve)


def _make_runtime_and_session():
    from ctx_weft.core import CtxWeftRuntime
    from ctx_weft.providers.llm.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=InlineAgentTemplateProvider())
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    session = Session(
        id="ses_1", tenant_id="default", user_prompt="x", status="RUNNING",
        root_agent_id="agt_1", llm_provider="acct1", llm_model="m1", token_budget=0,
    )
    task = Task(
        id="tsk_1", session_id="ses_1", status="ACTIVE", tenant_id="default",
        assigned_agent_id="agt_1", creator_agent_id="agt_1", settings=NormalTaskSettings(),
    )
    task_manager = TaskManager(session_id="ses_1", event_bus=runtime.event_bus, max_concurrent=1)
    task_manager.set_session(session)

    template = object()  # 具体属性不参与本方法逻辑；仅作为不透明句柄经 state.extra 透传
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
    return tcid


@pytest.fixture
def minimal_runtime_with_session():
    runtime, session, template, task_manager, task, agent_id, memory = _make_runtime_and_session()
    return runtime, session, template, task_manager, task, agent_id, memory


async def test_relaunch_registers_close_synth_for_finish(minimal_runtime_with_session, monkeypatch):
    """close 边界重跑：从 memory 读到 finish 对 tool_call_id 后 register_close_synth，并以该 boundary 重跑。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session
    tcid = await _seed_finish_pair(memory, session, task, agent_id)
    task.status = "FINISHED"

    def _fake_materialize(self, agent_id, *, session_id=None, tenant_id=None):
        return (Agent(id=agent_id, session_id=session.id, template_id="tpl_echo",
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

    monkeypatch.setattr(rt_mod, "launch_background_observe", _fake_launch, raising=False)

    await runtime._relaunch_task_recap(
        session=session, template=template, template_id="tpl_echo", task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="finish",
    )
    await asyncio.sleep(0)

    assert captured["task_id"] == task.id
    assert captured["tool_call_id"] == tcid
    assert captured["outcome"] == "success"
    # 延迟折叠（spec 2026-07-20）：重跑替换成功后按 task scope 补删末段 raw
    assert captured["raw_fold_scope"] == captured["scope"]
    assert launched["boundary"] == "finish"


async def test_relaunch_is_best_effort_swallows_errors(minimal_runtime_with_session, monkeypatch):
    """任何一步失败：不得抛出，只记日志跳过（best-effort）。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session

    def _boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(AgentLifecycleManager, "materialize", _boom)

    await runtime._relaunch_task_recap(
        session=session, template=template, template_id="tpl_echo", task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="finish",
    )  # 不应抛出


async def test_find_finish_pair_tool_call_id_found_and_none():
    _, session, _, _, task, agent_id, memory = _make_runtime_and_session()
    scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent_id)
    pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id, task_id=task.id, agent_id=agent_id)

    from ctx_weft.core import CtxWeftRuntime
    from ctx_weft.providers.llm.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=InlineAgentTemplateProvider())

    # 无占位 finish 对 → None
    got_none = await runtime._find_finish_pair_tool_call_id(memory, scope, task.id, pctx)
    assert got_none is None

    tcid = await _seed_finish_pair(memory, session, task, agent_id)
    got = await runtime._find_finish_pair_tool_call_id(memory, scope, task.id, pctx)
    assert got == tcid


async def test_relaunch_dispatch_boundary_no_close_synth(minimal_runtime_with_session, monkeypatch):
    """非 close 边界（dispatch，spec 2026-07-16）重跑：按原 boundary 重跑、
    不 register_close_synth（那是 close 边界替换占位 finish 对的专属动作）。"""
    runtime, session, template, task_manager, task, agent_id, memory = minimal_runtime_with_session
    task.status = "SUSPENDED"  # 委派挂起中崩溃的形态

    def _fake_materialize(self, agent_id, *, session_id=None, tenant_id=None):
        return (Agent(id=agent_id, session_id=session.id, template_id="tpl_echo",
                       tenant_id=session.tenant_id), _fake_resolved_model())

    monkeypatch.setattr(AgentLifecycleManager, "materialize", _fake_materialize)

    registered = []
    monkeypatch.setattr(rt_mod, "register_close_synth",
                        lambda *a, **k: registered.append(a), raising=False)

    launched = {}

    def _fake_launch(state, ctx, *, boundary):
        launched["boundary"] = boundary
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(rt_mod, "launch_background_observe", _fake_launch, raising=False)

    await runtime._relaunch_task_recap(
        session=session, template=template, template_id="tpl_echo", task_manager=task_manager,
        task=task, agent_id=agent_id, boundary="dispatch",
    )
    await asyncio.sleep(0)

    assert launched["boundary"] == "dispatch"
    assert registered == [], "dispatch 边界不得登记 close_synth"
