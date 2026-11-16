"""E2E（spec 2026-07-16）：父 delegate → 子完成 → 父 resume 时派发前 raw 已折成段摘要。

路由型 mock LLM：act 第 1 次调用（root）只 delegate；第 2 次（子）finish；
第 3 次（root resume 后）finish。bg observe 第 1 次 = dispatch 边界（父挂起时），
产 "DISPATCH段摘要"；此后 = root close 复述。子任务非 root，close 不走 bg observe，
故 bg 调用次序确定。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import MemoryAddress, MemoryEventType, ProviderContext, ToolCall
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



def _is_background_recap(request) -> bool:
    from tests.runtime_helpers import is_recap_request
    return is_recap_request(request)

class _RouterLLM(MockLLMAdapter):
    """按 request.tools 路由；act 次序：root delegate → child finish → root finish。"""

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_calls = 0
        self._bg_calls = 0
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}

        if "control__update_task_metadata" in names:  # recognize_intent
            return self._stream(MockResponse(text=""), request)

        if _is_background_recap(request):  # 只摘要档（dispatch / close 边界）
            self._bg_calls += 1
            report = "DISPATCH段摘要" if self._bg_calls == 1 else "CLOSE复述"
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__report_observation",
                         arguments={"act_recap": report, "task_summary": report}),
            ]), request)

        if "control__report_observation" in names:  # 判定档（这里只有子任务的前台 LLM observe）
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_observation",
                         arguments={"task_status": "success",
                                    "act_recap": "done",
                                    "task_summary": "done"}),
            ]), request)

        # act
        self._act_calls += 1
        if self._act_calls == 1:  # root：只 delegate（派发前已有本轮 raw）
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("del"), name="control__delegate_task",
                         arguments={"title": "child-work",
                                    "task_prompt": "do the delegated work"}),
            ]), request)
        # 子任务 act 与 root resume 后的 act：finish（text=收尾正文，落 task.outputs——
        # 子任务走 LLM observe，report_observation 的 success 护栏要求 task.outputs 非空）
        return self._stream(MockResponse(text="done", tool_calls=[]), request)


async def _wait_all_finished(runtime, session_id, n, timeout=8.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        view = await rebuild_view(runtime.event_store, session_id)
        finished = [t for t in view.tasks.values() if t.status == "FINISHED"]
        if len(finished) >= n:
            return view
        await asyncio.sleep(0.02)
    view = await rebuild_view(runtime.event_store, session_id)
    raise TimeoutError(
        f"expected >={n} FINISHED tasks; got "
        f"{[(t.title, t.status) for t in view.tasks.values()]}"
    )


@pytest.mark.asyncio
async def test_dispatch_boundary_recap_e2e():
    llm = _RouterLLM()
    resolver = InlineAgentTemplateProvider()
    template = make_echo_template()
    # 关短段免折门：本测试的 raw 只有几十 token，默认阈值(400)下必免折、断言不到折叠
    template.loop_config.short_segment_token_threshold = 0
    resolver.register(template)
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    await initialize_all(runtime)
    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="delegate then finish",

        )
    )
    await handle.wait_for_finish(timeout=8.0)
    view = await _wait_all_finished(runtime, handle.session_id, 2)

    root = next(t for t in view.tasks.values() if not t.parent_task_id)
    scope = MemoryAddress(session_id=handle.session_id, task_id=root.id,
                        agent_id=root.assigned_agent_id)
    pctx = ProviderContext(session_id=handle.session_id, tenant_id="default",
                           task_id=root.id, agent_id=root.assigned_agent_id)

    summaries = await memory.recall_recent(
        scope, [MemoryEventType.TASK_COMPACT_SUMMARY], 100, pctx)
    contents = [r.content for r in summaries]
    assert "DISPATCH段摘要" in contents, \
        f"父 task 层必须有 dispatch 段摘要（派发前 raw 的折叠产物），实得 {contents}"

    raws = await memory.recall_recent(
        scope, [MemoryEventType.LLM_RESPONSE], 100, pctx)
    assert raws == [], f"派发前 raw 应已被折掉/胶囊化，实得 {[r.content for r in raws]}"
