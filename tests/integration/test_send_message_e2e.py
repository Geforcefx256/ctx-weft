"""send_message（Task 18）端到端验证：spec §4.2 那条计划自评时标记为「待验证」的链路。

场景：root agent 因 `delegate_task(use_subagent=True)` 处于 `idle`（当前 task
SUSPENDED，等一个独立子 agent 的子任务）—— `send_message` 必须走「注入现有 task」
分支（不新建 task），且子任务收尾后父 task 仍能正常被 `_try_resume_parent` 唤醒、
唤醒后的那轮 act 看得见注入的内容（不能被子任务完成的唤醒逻辑覆盖/丢弃）。

用一个受控 gate 挡住子 agent 的 act 调用，保证 `send_message` 一定发生在子任务
完成**之前**（消除真实场景里本就存在、但会让测试变 flaky 的那个时间窗口）。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

SUB_TEMPLATE_ID = "tpl_researcher"
SUB_TEMPLATE_REF = "agent__tpl_researcher"
MARKER_TEXT = "psst: also double-check the budget cap"


def _researcher_template():
    import dataclasses as _dc
    return _dc.replace(make_echo_template(), id=SUB_TEMPLATE_ID, name="researcher_agent")


class _GatedRouterLLM(MockLLMAdapter):
    """root 的第一次 act 委派一个独立子 agent（不带 finish，root 因此 SUSPEND）；
    其余 act 调用（子 agent 的、以及父 task 被唤醒后的）一律 finish_task，但都要先
    等 `release_gate` 被 set——由测试精确控制"子任务几时才允许完成"，从而保证
    `send_message` 一定发生在子任务完成之前。
    """

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_calls = 0
        self._n = 0
        self.release_gate = asyncio.Event()
        self.act_requests: list = []

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

        # act
        self._act_calls += 1
        self.act_requests.append(request)
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
        return self._gated_finish(request)

    async def _gated_finish(self, request):
        await self.release_gate.wait()
        # 交付物 = 收尾回合正文（finish_task 不带未声明参数，spec: capability-gateway）
        async for chunk in self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=self._id("fin"), name="control__finish_task",
                     arguments={}),
        ]), request):
            yield chunk


def _make_runtime(llm) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    resolver.register(_researcher_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def _wait_until(predicate, timeout=5.0, interval=0.02):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met within timeout")


async def test_send_message_injects_into_suspended_parent_and_survives_child_wakeup():
    llm = _GatedRouterLLM()
    runtime = _make_runtime(llm)

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id=f"agent:{make_echo_template().id}",
            user_prompt="delegate to a researcher",
            context_limit=100_000,
        )
    )
    session_id = handle.session_id
    root_agent_id = handle.agent_id
    root_task_id = handle.task_id

    # 1. 等 root task 真正 SUSPEND（子任务已经派生、还没完成）。
    async def _root_suspended() -> bool:
        view = await rebuild_view(runtime.event_store, session_id)
        t = view.tasks.get(root_task_id)
        return t is not None and t.status == "SUSPENDED"

    await _wait_until(_root_suspended)

    # root agent 此刻必须是 idle（spec 4.2 前提）——current_task 未终态。
    assert runtime._agent_lifecycle_manager.status_of(root_agent_id) == "idle"
    assert runtime._agent_lifecycle_manager._agents[root_agent_id].current_task_id == root_task_id

    # 2. send_message：必须走注入分支（不新建 task），且此刻子任务尚未完成
    #    （被 release_gate 挡住），消除真实场景里的竞态窗口。
    tid = (await runtime.send_message(root_agent_id, MARKER_TEXT)).task_id
    assert tid == root_task_id, "current_task 未终态 -> 必须注入现有 task，不新建"

    # 注入是纯写内存，不该改变 root task 仍 SUSPENDED、子任务仍未完成的事实。
    view = await rebuild_view(runtime.event_store, session_id)
    assert view.tasks[root_task_id].status == "SUSPENDED"

    # 3. 放行子任务完成 -> 触发 _try_resume_parent 唤醒 root task。
    llm.release_gate.set()

    async def _both_finished() -> bool:
        view = await rebuild_view(runtime.event_store, session_id)
        finished = [t for t in view.tasks.values() if t.status == "FINISHED"]
        return len(finished) >= 2

    await _wait_until(_both_finished, timeout=8.0)

    view = await rebuild_view(runtime.event_store, session_id)
    tasks = list(view.tasks.values())
    assert len(tasks) == 2
    assert all(t.status == "FINISHED" for t in tasks), (
        f"parent must resume and finish normally after child completes, got "
        f"{[(t.title, t.status) for t in tasks]}"
    )

    # 4. 核心断言：注入的内容必须真的出现在父任务复跑那轮的装配结果里——
    #    不能被子任务完成的唤醒逻辑覆盖/丢弃。root 的第一次 act（delegate）里没有，
    #    子 agent 的 act 里也不该有（不同 task/agent scope）；只有 root 复跑那次
    #    act 的请求里应该有。
    from ctx_weft.core.utils.content import content_to_text

    def _has_marker(req) -> bool:
        for m in req.messages:
            if MARKER_TEXT in content_to_text(m.content):
                return True
        return False

    matches = [i for i, req in enumerate(llm.act_requests) if _has_marker(req)]
    assert matches, (
        f"injected message never showed up in any act request "
        f"({len(llm.act_requests)} act calls total)"
    )
    assert 0 not in matches, "root's first (delegate) act request predates the injection"
    assert not _has_marker(llm.act_requests[1]), (
        "the injected message leaked into the CHILD agent's act request — wrong scope"
    )
    assert matches == [len(llm.act_requests) - 1], (
        "injected message must show up exactly in the parent's resumed act request "
        f"(last one); matches={matches}, total={len(llm.act_requests)}"
    )

    # 5. memory 层也留痕：写进的是 root task 的 TASK 视图。
    from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext

    memory = runtime.providers.get_memory()
    scope = MemoryAddress(session_id=session_id, task_id=root_task_id, agent_id=root_agent_id)
    ctx = ProviderContext(session_id=session_id, tenant_id="default")
    turns = await memory.load_view(scope, MemoryScope.TASK, ctx)
    assert any(
        r.role == "user" and MARKER_TEXT in content_to_text(r.content)
        for r in turns
    ), "injected message not found in root task's TASK-scope memory"
