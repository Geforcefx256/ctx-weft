"""spawn 被权限门拒绝时必须发 SpawnRejected —— AgentInstantiated 覆盖不到的那一半。

`AgentSpawned` / `SpawnRejected` 是一对，构成对**每一次 spawn 尝试**的完整审计：
被拒的尝试根本不会有 agent 诞生，所以 `AgentInstantiated` 里找不到它们的痕迹。
今天唯一的门是 `AgentLifecycleManager.instantiate` 里的深度检查（用**子模板**的
`loop_config.max_spawn_depth`），它抛 SpawnDepthExceeded，此前只会落进 TaskManager
的通用 assembly_failure 分支 —— 事件流里看不出「有人想 spawn 但被挡了」。
"""
from __future__ import annotations

import asyncio
import dataclasses as _dc

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_subagent_instantiated_event import (
    SUB_TEMPLATE_ID,
    _SpawnLLM,
)
from tests._event_helpers import all_events

pytestmark = pytest.mark.asyncio


def _depth_barred_template():
    """子模板把 max_spawn_depth 压到 0 —— 任何子 agent（depth 1）都过不了门。"""
    tmpl = _dc.replace(make_echo_template(), id=SUB_TEMPLATE_ID, name="researcher_agent")
    return _dc.replace(tmpl, loop_config=_dc.replace(tmpl.loop_config, max_spawn_depth=0))


def _make_runtime(llm):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    resolver.register(_depth_barred_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def _wait_for_event(runtime, session_id, event_type, timeout=8.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        events = await all_events(runtime.event_store, session_id)
        hits = [e for e in events if e.type == event_type]
        if hits:
            return hits
        await asyncio.sleep(0.02)
    events = await all_events(runtime.event_store, session_id)
    raise TimeoutError(
        f"{event_type} not emitted within {timeout}s; saw types = "
        f"{sorted({e.type for e in events})}"
    )


async def test_depth_limit_emits_spawn_rejected():
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    rejected = await _wait_for_event(runtime, handle.session_id, EventType.SPAWN_REJECTED)

    ev = rejected[-1]
    assert ev.payload.get("reason") == "depth_limit"
    assert ev.payload.get("fallback_to_inline") is False
    assert ev.payload.get("attempted_subtask_id")


async def test_spawn_rejected_names_the_parent_not_a_phantom_child():
    """子 agent 没诞生，envelope 的 agent_id 必须是**父**——没有子 id 可填。"""
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    rejected = await _wait_for_event(runtime, handle.session_id, EventType.SPAWN_REJECTED)
    view = await rebuild_view(runtime.event_store, handle.session_id)
    root_agent_id = view.sessions[handle.session_id].root_agent_id

    assert rejected[-1].agent_id == root_agent_id


async def test_rejected_spawn_produces_no_agent_instantiated():
    """被拒 = 没有 agent 诞生。两个事件的分工正靠这条守住。"""
    llm = _SpawnLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    rejected = await _wait_for_event(runtime, handle.session_id, EventType.SPAWN_REJECTED)
    attempted = rejected[-1].payload["attempted_subtask_id"]

    events = await all_events(runtime.event_store, handle.session_id)
    born_for_subtask = [
        e for e in events
        if e.type in (EventType.AGENT_INSTANTIATED, EventType.AGENT_SPAWNED)
        and e.task_id == attempted
    ]

    assert born_for_subtask == []
