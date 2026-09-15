"""子 agent 实例化时必须发 AgentInstantiated，带**自己**的 template_id。

事件流是 agent 出身的唯一记录：`_rebuild_agents` 从 session/task 树推算 AgentView，
推得出 parent/depth，推不出模板。root agent 在 session_registry 里发过这条事件，
子 agent 那条一直没发 —— 于是冷 resume 的 pre_resolved 只能把每个 agent 都填成
session root 的模板，授权（AllowListAuthorizer 按 ctx.agent_template_id 做策略）
会让重建后的子 agent 顶着 root 的模板身份。

对照单测：tests/unit/test_agent_template_id_replay.py 钉的是 reducer 侧的折叠；
本文件钉的是发射侧 —— 两者合起来才让 template_id 真正跨重启存活。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

SUB_TEMPLATE_ID = "tpl_researcher"
# delegate_task 收的是**限定工具名**形式（agent__x）——TemplateLookup.resolve_qualified
# 按 qualify(cap.id) 匹配，裸 id 原样透传给 get_template 后报 TemplateNotFoundError。
SUB_TEMPLATE_REF = "agent__tpl_researcher"


def _researcher_template():
    """与 root 的 tpl_echo 不同的第二个模板——差异正是本测试要观察的东西。"""
    import dataclasses as _dc
    return _dc.replace(make_echo_template(), id=SUB_TEMPLATE_ID, name="researcher_agent")


class _SpawnLLM(MockLLMAdapter):
    """root 的第一次 act 派一个显式指定模板的子 agent；其余一律 finish。"""

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_calls = 0
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}

        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "done",
                                    "task_summary": "done"}),
            ]), request)
        if "control__collect_process_report" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__collect_process_report",
                         arguments={"act_recap": "segment summary"}),
            ]), request)

        self._act_calls += 1
        if self._act_calls == 1:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("del"), name="control__delegate_task",
                         arguments={
                             "title": "research",
                             "task_prompt": "go research",
                             "use_subagent": True,
                             "subagent_template": SUB_TEMPLATE_REF,
                         }),
            ]), request)
        # 交付物 = 收尾回合正文（finish_task 不带未声明参数，spec: capability-gateway）
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=self._id("fin"), name="control__finish_task",
                     arguments={}),
        ]), request)


def _make_runtime(llm) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    resolver.register(_researcher_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def _wait_for_subagent_task(runtime, session_id, timeout=8.0):
    """等到出现一个 use_subagent 且已分配 agent 的 task。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        view = await rebuild_view(runtime.event_store, session_id)
        for t in view.tasks.values():
            if t.settings_raw.get("use_subagent") and t.assigned_agent_id:
                return view, t
        await asyncio.sleep(0.02)
    view = await rebuild_view(runtime.event_store, session_id)
    raise TimeoutError(
        f"no sub-agent task appeared within {timeout}s; "
        f"got {[(t.title, t.status, t.settings_raw) for t in view.tasks.values()]}"
    )


async def test_subagent_instantiation_emits_event_with_own_template():
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    _view, sub_task = await _wait_for_subagent_task(runtime, handle.session_id)
    sub_agent_id = sub_task.assigned_agent_id

    events = await runtime.event_store.read_by_session(handle.session_id)
    instantiated = [
        e for e in events
        if e.type == EventType.AGENT_INSTANTIATED and e.agent_id == sub_agent_id
    ]

    assert instantiated, (
        f"no AgentInstantiated for sub-agent {sub_agent_id}; "
        f"emitted agent ids = "
        f"{[e.agent_id for e in events if e.type == EventType.AGENT_INSTANTIATED]}"
    )
    assert instantiated[-1].payload.get("template_id") == SUB_TEMPLATE_ID


async def test_replayed_view_gives_subagent_its_own_template():
    """端到端闭环：发射 + 折叠合起来，重放出的 AgentView 带子模板而非 root 模板。"""
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    view, sub_task = await _wait_for_subagent_task(runtime, handle.session_id)

    sub_av = view.agents[sub_task.assigned_agent_id]
    root_av = view.agents[view.sessions[handle.session_id].root_agent_id]

    assert sub_av.template_id == SUB_TEMPLATE_ID
    assert root_av.template_id != SUB_TEMPLATE_ID


# ── spawn 动作本身的事件：AgentSpawned / SpawnRejected ────────────────────────
#
# 与 AgentInstantiated 的分工：后者的主语是「这个 agent 自己」（出身配置），
# 前者的主语是「父 agent 的一次 spawn 动作」，且与 SpawnRejected 配对，构成对
# **每一次 spawn 尝试**的完整审计——被拒的那些根本不会有 agent 诞生，
# AgentInstantiated 覆盖不到。
#
# 两者都发在 assemble() 的同一个决定点上：唯一的权限门（深度检查）在
# instantiate 里，派发时才跑。若在 delegate_task 处发 AgentSpawned，
# 会出现「先 Spawned、后 Rejected」的自相矛盾事件对。


async def test_agent_spawned_emitted_with_parent_and_subtask():
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    view, sub_task = await _wait_for_subagent_task(runtime, handle.session_id)
    root_agent_id = view.sessions[handle.session_id].root_agent_id

    events = await runtime.event_store.read_by_session(handle.session_id)
    spawned = [e for e in events if e.type == EventType.AGENT_SPAWNED]

    assert spawned, (
        "no AgentSpawned emitted; agent-domain events = "
        f"{[e.type for e in events if e.type.startswith('Agent') or e.type == 'SpawnRejected']}"
    )
    ev = spawned[-1]
    assert ev.payload.get("parent_agent_id") == root_agent_id
    assert ev.payload.get("subtask_id") == sub_task.id
    # envelope：子 agent 是这次 spawn 的产物，task 是它要跑的子任务
    assert ev.agent_id == sub_task.assigned_agent_id
    assert ev.task_id == sub_task.id


async def test_agent_spawned_precedes_agent_instantiated():
    """因果顺序：先「这次 spawn 被准了」，再「诞生的 agent 长这样」。"""
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    _view, sub_task = await _wait_for_subagent_task(runtime, handle.session_id)
    sub_agent_id = sub_task.assigned_agent_id

    events = await runtime.event_store.read_by_session(handle.session_id)
    order = [
        e.type for e in events
        if e.agent_id == sub_agent_id
        and e.type in (EventType.AGENT_SPAWNED, EventType.AGENT_INSTANTIATED)
    ]

    assert order[:2] == [EventType.AGENT_SPAWNED, EventType.AGENT_INSTANTIATED]
