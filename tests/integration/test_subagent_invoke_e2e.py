"""端到端：subagent 是否真的「调用起来」——不止被实例化，而是完整跑完一圈。

与 test_subagent_instantiated_event.py 的分工：那边只等到「子 agent 被创建并分配」
就停；本文件钉整条生命周期，四件事缺一不可：

1. root actor 调 delegate_task(use_subagent=True, subagent_template=…) → root SUSPENDED
2. 子 agent 以**自己的模板身份**跑 LLM loop：request.system 含 researcher 的
   identity、`## Current Task` 就是委托的 task_prompt（不是 root 的用户输入）
3. 子 agent finish_task(result=…) → 子任务 FINISHED
4. 结果回流：root 恢复后的那次 act request 里能看到子 agent 的 result 文本，
   root 随之 FINISHED —— 两个 task 都到终态

第 4 步是最强的「调用起来」证据：父 agent 的下一轮 LLM 上下文里出现了
子 agent 的产出，说明 dispatch→run→result→resume 整条链路都活着。
"""
from __future__ import annotations

import asyncio
import dataclasses

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
from tests._event_helpers import all_events

pytestmark = pytest.mark.asyncio

SUB_TEMPLATE_ID = "tpl_researcher"
# delegate_task 收限定形式（agent__x），见 test_subagent_instantiated_event.py 的注释
SUB_TEMPLATE_REF = "agent__tpl_researcher"
RESEARCHER_MARK = "You are a meticulous researcher agent"
DELEGATED_PROMPT = "go count the stars"
CHILD_RESULT = "research findings: 42 stars"


def _researcher_template():
    from ctx_weft.protocols import IdentityFacet

    return dataclasses.replace(
        make_echo_template(),
        id=SUB_TEMPLATE_ID,
        name="researcher_agent",
        identity={
            "act": IdentityFacet(text=RESEARCHER_MARK),
            "observe": IdentityFacet(text="You evaluate research quality."),
        },
    )


class _RecordingRouterLLM(MockLLMAdapter):
    """按 system 身份路由的 mock，并把每次 request 快照留档供断言。

    - observer / metadata / background 工具按工具名路由（与既有 e2e 测试同款）
    - act：system 带 researcher 身份 → 子 agent，正文交研究成果（交付物=收尾回合
           正文，finish_task 不带 payload——本架构里 outputs 由 ActStep 从回合
           正文合成，不是工具参数）
           其余是 root act：第 1 次 delegate，之后正文收尾
    """

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self.requests: list[tuple[str, str]] = []  # (system, 全 messages 文本) 快照
        self._root_act_calls = 0
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def _snapshot(self, request) -> tuple[str, str]:
        parts = []
        for m in request.messages:
            c = m.content
            text = c if isinstance(c, str) else " ".join(
                getattr(p, "text", "") for p in (c or []) if hasattr(p, "text")
            )
            parts.append(text)
        return request.system, "\n".join(parts)

    def complete(self, request, stream=True):
        self.last_request = request
        self.requests.append(self._snapshot(request))
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
                         arguments={"act_recap": "delegated and waited",
                                    "task_summary": "segment summary"}),
            ]), request)

        if RESEARCHER_MARK in request.system:
            # 子 agent 的 act：正文=交付物（研究成果），同一回合调 finish_task 收尾
            return self._stream(MockResponse(
                text=CHILD_RESULT,
                tool_calls=[ToolCall(id=self._id("sub"), name="control__finish_task",
                                     arguments={})],
            ), request)

        # root 的 act
        self._root_act_calls += 1
        if self._root_act_calls == 1:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("del"), name="control__delegate_task",
                         arguments={
                             "title": "research",
                             "task_prompt": DELEGATED_PROMPT,
                             "use_subagent": True,
                             "subagent_template": SUB_TEMPLATE_REF,
                         }),
            ]), request)
        return self._stream(MockResponse(
            text="root done",
            tool_calls=[ToolCall(id=self._id("fin"), name="control__finish_task",
                                 arguments={})],
        ), request)


def _make_runtime(llm) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    resolver.register(_researcher_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def _wait_both_finished(runtime, session_id, timeout=10.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        view = await rebuild_view(runtime.event_store, session_id)
        if len(view.tasks) >= 2 and all(
            t.status == "FINISHED" for t in view.tasks.values()
        ):
            return view
        await asyncio.sleep(0.02)
    view = await rebuild_view(runtime.event_store, session_id)
    raise TimeoutError(
        f"not all tasks FINISHED within {timeout}s; "
        f"got {[(t.title, t.status) for t in view.tasks.values()]}"
    )


async def test_subagent_runs_its_own_loop_and_result_flows_back():
    llm = _RecordingRouterLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo",
            user_prompt="delegate the star census",
            context_limit=100_000,
        )
    )
    session_id = handle.session_id
    await handle.wait_for_finish(timeout=10.0)
    view = await _wait_both_finished(runtime, session_id)

    # ── 1. 两个 task 都 FINISHED：root + 委托的 research ──────────────────────
    tasks = list(view.tasks.values())
    assert len(tasks) == 2, f"expected 2 tasks, got {[(t.title, t.status) for t in tasks]}"
    root_task = next(t for t in tasks if t.parent_task_id == "" or t.parent_task_id is None)
    child_task = next(t for t in tasks if t.title == "research")
    assert all(t.status == "FINISHED" for t in tasks)

    # ── 2. 子任务确实派给了独立 subagent（不是 root 自己跑）──────────────────
    root_agent_id = view.sessions[session_id].root_agent_id
    assert child_task.settings_raw.get("use_subagent") is True
    assert child_task.assigned_agent_id, "sub-agent never assigned"
    assert child_task.assigned_agent_id != root_agent_id
    sub_av = view.agents[child_task.assigned_agent_id]
    assert sub_av.parent_agent_id == root_agent_id
    assert sub_av.spawn_depth == 1, f"expected depth 1, got {sub_av.spawn_depth}"

    # ── 3. root 因等子任务而挂起过（真 suspend，不是 detach 直跑）────────────
    events = await all_events(runtime.event_store, session_id)
    suspended = [
        e for e in events
        if e.type == EventType.TASK_SUSPENDED
        and (e.task_id == root_task.id or (e.payload or {}).get("task_id") == root_task.id)
    ]
    assert suspended, (
        f"root task never suspended while waiting for the sub-agent; "
        f"event types = {[e.type for e in events]}"
    )

    # ── 4. 子 agent 以自己的身份跑了 LLM：researcher system + 委托 prompt ────
    sub_requests = [
        (system, body) for (system, body) in llm.requests
        if RESEARCHER_MARK in system and DELEGATED_PROMPT in body
    ]
    assert sub_requests, (
        "sub-agent never ran an LLM turn with its own identity + delegated prompt; "
        f"requests seen: {[(s[:40], b[:60]) for s, b in llm.requests]}"
    )

    # ── 5. 结果回流：root 恢复后的 act 上下文里出现子 agent 的产出 ───────────
    root_resumed_acts = [
        body for (system, body) in llm.requests
        if RESEARCHER_MARK not in system and CHILD_RESULT in body
    ]
    assert root_resumed_acts, (
        "root's post-resume LLM context never saw the sub-agent result "
        f"({CHILD_RESULT!r}); requests: {[(s[:40], b[:80]) for s, b in llm.requests]}"
    )

    # ── 6. 记忆侧：父 scope 派发对已终态化（running ack 被 result 替换）──────
    from ctx_weft.protocols import (
        MemoryAddress, MemoryKind, MemoryScope, ProviderContext,
    )

    memory = runtime.providers.get_memory()
    pctx = ProviderContext(session_id=session_id)
    parent_addr = MemoryAddress(session_id=session_id, agent_id=root_agent_id)
    agent_recs = await memory.load_view(parent_addr, MemoryScope.AGENT, pctx)
    child_results = [
        r for r in agent_recs
        if r.role == "tool" and (r.metadata or {}).get("child_task_id") == child_task.id
    ]
    assert child_results, (
        f"no dispatch tool record for child {child_task.id} in parent agent scope; "
        f"records: {[(r.role, str(r.content)[:60]) for r in agent_recs]}"
    )
    assert any(CHILD_RESULT in str(r.content) for r in child_results), (
        "dispatch tool slot still says 'running' — close never replaced it with the "
        f"terminal result; got: {[str(r.content)[:80] for r in child_results]}"
    )

    # ── 7. blackboard：子任务结果已按 topic 发布（跨 agent 精确召回通道）──────
    pubs = await memory.load_view(
        MemoryAddress(session_id=session_id), MemoryScope.SESSION, pctx,
        kinds=[MemoryKind.PUBLICATION],
    )
    assert any(r.topic == child_task.id and CHILD_RESULT in str(r.content) for r in pubs), (
        f"no blackboard publication for child task; pubs: {[(r.topic, str(r.content)[:60]) for r in pubs]}"
    )


# ── 误用 → 报错回灌 → 改参重试（spec: capability-gateway 的 e2e 锚）─────────────
