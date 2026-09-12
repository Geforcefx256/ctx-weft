"""段 2 · Task 11：端到端与不变式收口。

四条完整链路，均从真实 `CtxWeftRuntime`（`start_session` → 真实 `TaskManager` /
`CapabilityGateway` / `HitlService` / `HitlRegistry` / `HitlWaiter`）起，只在两处打桩
（LLM 与工具 provider——brief 明确允许），不打桩 core 的任何一层：

1. 热审批：改参放行，工具真执行、结果真回灌 LLM。
2. 冷审批：零热窗强制驱逐 → AWAITING_HUMAN → 应答 → reconcile 精确重入，工具**恰好一次**。
3. `ask_user` 冷路径：应答带图，图片以真实 part 落进 TOOL_RESULT、送回模型。
4. 纯文本暂停：PAUSED → 应答注入一条 user 回合、任务续跑；重复应答不产生第二条注入。

装配复用 `tests/integration/test_minimal_loop.py` 的 `InlineAgentTemplateProvider` /
`make_echo_template` / `make_runtime`，与 `tests/integration/test_multimodal_end_to_end.py`
的「按 `request.tools` 路由的 MockLLMAdapter 子类」惯例（root task 恒 interactive，
recognize_intent 与 act 并发跑在同一个共享 LLM 实例上，靠工具名而非调用顺序区分）。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import (
    ImagePart,
    MemoryEventType,
    ProviderContext,
    TextPart,
    ToolCall,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventBlobStore
from ctx_weft.protocols.hitl import HitlReply, ToolResultDelivery, UserTurnDelivery
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio


# ── shared test doubles ─────────────────────────────────────────────────────────


class _StubEventBlobStore(EventBlobStore):
    """携图 HITL 应答要过 event blob 门控（`validate_content`）；最小可外部化桩。"""

    can_externalize = True

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        import hashlib

        from ctx_weft.protocols.memory import BLOB_REF_PREFIX
        ref = f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


class _RecordingBashTool(ToolCapabilityProvider):
    """一个会被 `HumanConfirmationAuthorizer` 门控的真实工具 provider。

    记录每次真正执行时收到的参数——这是「改参真正生效」「恰好执行一次」两条断言的
    唯一证据来源。产出一段可在下一轮 LLM 请求文本里唯一识别的标记，用来证明工具
    结果真的回灌进了模型看到的 prompt。
    """

    name = "fs"

    def __init__(self) -> None:
        self.invocations = 0
        self.last_args: dict | None = None

    async def list(self, ctx: ProviderContext) -> list[ToolCapability]:
        return [self._cap()]

    async def retrieve(self, ctx: ProviderContext) -> list[ToolCapability]:
        return [self._cap()]

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    @staticmethod
    def _cap() -> ToolCapability:
        return ToolCapability(id="fs:bash_exec", name="bash_exec", description="run a shell command")

    def invoke(self, capability_id, args, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run(args)

    async def _run(self, args: dict) -> AsyncIterator[CapabilityEvent]:
        self.invocations += 1
        self.last_args = dict(args)
        yield CapabilityEvent(kind="result", payload={"content": "TOOL_OUTPUT_MARKER: command ran"})

    async def cancel(self, invocation_id, ctx) -> None:
        return None


class _ActRouterLLM(MockLLMAdapter):
    """按 `request.tools` 路由（同 `test_multimodal_end_to_end._RouterLLM`）：
    recognize_intent（工具集含 `control__update_task_metadata`）恒回空文本；
    act 回合按 ``act_responses`` 队列顺序消费——与 recognize_intent 的并发调用互不干扰，
    因为两者用不同分支，act 计数只在 act 分支递增。

    `act_requests` 记录每次 act 回合收到的完整 `LLMRequest`，供断言「工具结果真的
    出现在下一轮送给模型的 prompt 里」（而不仅仅是「memory 里有 TOOL_RESULT」）。
    """

    def __init__(self, act_responses: list[MockResponse], **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_responses = list(act_responses)
        self._act_idx = 0
        self.act_requests: list = []

    def complete(self, request, stream: bool = True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        self.act_requests.append(request)
        response = self._act_responses[self._act_idx]
        self._act_idx += 1
        return self._stream(response, request)


def _all_request_text(request) -> str:
    return request.system + "\n" + "\n".join(content_to_text(m.content) for m in request.messages)


async def _poll(predicate, *, timeout: float = 5.0, interval: float = 0.02):
    """轮询直到 `predicate()` 返回真值，否则超时报错（同 `test_crash_recovery_reconcile` 惯例）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError("timed out waiting for condition")


async def _wait_task_terminal(handle, *, timeout: float = 5.0):
    """等 task 到终态就返回，**不经** `handle.wait_for_finish`。

    `wait_for_finish`（2026-09-04 起）多了一层更强的保证：终态事件到达后，若 close
    边界的后台 observe（段折叠/胶囊化，spec 2026-07-20 延迟折叠）确有在途，还会继续
    等它把折叠落地才返回——这是 host「等 wait_for_finish 返回就去读 memory 渲染」
    这条真实用法需要的durability，不是可选项。

    但下面这条测试要看的恰恰是**折叠落地前**的中间状态（第二轮 act 请求数、工具结果
    回灌），如果借道 `wait_for_finish` 就会连带等到后台 observe 自己也跑一轮 LLM
    调用，把这条测试的桩计数搅乱（它没有像 `test_dispatch_boundary_recap_e2e` 那样
    特判后台 observe 的终止工具名）。这里直接复刻 `wait_for_finish` 曾经的（也是现在
    `TaskFailed`/`TaskCanceled` 仍然沿用的）「见到 task 终态事件就返回」判据，绕开新增
    的那层等待，只钉住「task 到终态」这一刻——断言内容不变，只是不再途经
    `wait_for_finish` 这个更强的契约。"""
    from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT
    from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
    from ctx_weft.protocols.events import EventFilter
    terminal = {et for et, st in TASK_STATUS_BY_EVENT.items() if st in TERMINAL_TASK_STATUSES}
    async with asyncio.timeout(timeout):
        async for ev in handle.event_bus.stream(
            EventFilter(agent_id=handle.agent_id, task_id=handle.task_id)
        ):
            if ev.type in terminal:
                return handle._state
    return handle._state


def _make_runtime_with_bash_tool(llm, *, hitl_timeout_sec: int | None = None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    config = RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec)
    runtime = make_runtime(llm=llm, agent_provider=resolver, config=config)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    tool = _RecordingBashTool()
    runtime.providers.register_capability(
        tool, tool_authorizers={"fs:bash_exec": HumanConfirmationAuthorizer()},
    )
    return runtime, tool


_BASH_CALL = MockResponse(
    text="", tool_calls=[ToolCall(id="tc_bash", name="fs__bash_exec", arguments={"command": "ls"})],
)


def _finish_call(tc_id: str = "tc_finish") -> MockResponse:
    return MockResponse(
        text="done",
        tool_calls=[ToolCall(id=tc_id, name="control__finish_task",
                             arguments={"deliverables_summary": "done"})],
    )


# ── 1. 热审批：改参放行，结果回灌模型 ────────────────────────────────────────────


async def test_hot_approval_rewrites_arguments_and_result_reaches_the_model() -> None:
    """驱动：真实 `start_session` → `ActStep` 经真实 `CapabilityGateway` 触发
    `HumanConfirmationAuthorizer` → 热等待（默认 `hitl_timeout_sec=None`，永不超时）→
    `runtime.reply_to_hitl(accepted, modified_arguments=...)` 就地唤醒。

    会因下列任一项回归而失败：
    - gateway 的热放行短路失效（工具从不执行 / 执行次数不对）；
    - `HumanConfirmationAuthorizer.on_decision` / gateway 丢弃 `modified_arguments`
      （provider 收到原参而非改写后的参）；
    - 工具结果没有真正回灌进下一轮发给 LLM 的 prompt（例如结果只写了 memory 却没被
      装配进下一轮 messages）。
    """
    llm = _ActRouterLLM(act_responses=[_BASH_CALL, _finish_call()])
    runtime, tool = _make_runtime_with_bash_tool(llm)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="please run ls", context_limit=100_000,
    ))
    sid = handle.session_id

    pending = await _poll(lambda: runtime.hitl_registry.list_pending(session_id=sid) or None)
    req = pending[0]
    assert req.form == "approval"
    assert isinstance(req.delivery, ToolResultDelivery)

    view = await runtime.reply_to_hitl(HitlReply(
        hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
        modified_arguments={"command": "ls -l"},
    ))
    assert view is not None and view.outcome == "accepted"

    # 不经 wait_for_finish：这条测试要看 close 边界后台 observe 折叠落地**前**的
    # act 请求数（见 _wait_task_terminal docstring），故只等 task 到终态。
    state = await _wait_task_terminal(handle, timeout=5.0)
    assert state is not None and state.task.status == "FINISHED", (
        f"expected FINISHED, got {state.task.status if state else None}"
    )

    # 工具真的执行了一次，且用的是**改写后**的参数。
    assert tool.invocations == 1
    assert tool.last_args == {"command": "ls -l"}, (
        f"expected rewritten args to reach the provider, got {tool.last_args!r}"
    )

    # 结果真的回灌了模型：第二次 act 回合发给 LLM 的 prompt 里带着工具的输出标记。
    assert len(llm.act_requests) == 2, f"expected 2 act rounds, got {len(llm.act_requests)}"
    assert "TOOL_OUTPUT_MARKER" in _all_request_text(llm.act_requests[1]), (
        "tool result did not reach the model's second-round prompt"
    )


# ── 2. 冷审批：精确重入，恰好一次 ────────────────────────────────────────────────


async def test_cold_approval_reconciles_and_invokes_the_tool_exactly_once() -> None:
    """驱动：`hitl_timeout_sec=0` 强制热窗即刻驱逐 → `ActStep` 抛 `HitlPark` → task
    落 `AWAITING_HUMAN`、工具**尚未**执行 → `reply_to_hitl` 驱动 `recover_agent` →
    `ReconcileStep` 命中 authz 阶段的决定缓存短路，重跑该 dangling tool_call。

    会因下列任一项回归而失败：
    - 冷路径压根没有重新执行工具（人给的答案没有真正驱动续跑）；
    - reconcile 短路失效，工具被执行两次（一次「假想」的热路径 + 一次冷路径）——这正是
      本设计要堵住的洞，故断言的是**计数**而非「发生过」；
    - 改写后的参数在冷路径上丢失（只放行、参数打回原样）。
    """
    llm = _ActRouterLLM(act_responses=[_BASH_CALL, _finish_call()])
    runtime, tool = _make_runtime_with_bash_tool(llm, hitl_timeout_sec=0)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="please run ls", context_limit=100_000,
    ))
    sid = handle.session_id

    state = await handle.wait_for_finish(timeout=5.0)  # RUN_FINISHED 也在冷 park 时发出
    assert state is not None and state.task.status == "AWAITING_HUMAN", (
        f"expected AWAITING_HUMAN (cold park), got {state.task.status if state else None}"
    )
    assert tool.invocations == 0, "tool must not have run before the human answered"

    pending = runtime.hitl_registry.list_pending(session_id=sid)
    assert len(pending) == 1
    req = pending[0]
    assert req.form == "approval"

    view = await runtime.reply_to_hitl(HitlReply(
        hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
        modified_arguments={"command": "ls -l"},
    ))
    assert view is not None and view.outcome == "accepted"

    tm = runtime._task_managers[sid]

    def _final_task():
        t = tm.get_task(req.task_id)
        return t if (t is not None and t.status in ("FINISHED", "FAILED", "CANCELED")) else None

    task = await _poll(_final_task)
    assert task.status == "FINISHED", f"expected FINISHED, got {task.status}"

    assert tool.invocations == 1, (
        f"expected exactly one execution via cold reconcile, got {tool.invocations}"
    )
    assert tool.last_args == {"command": "ls -l"}


# ── 3. ask_user 冷路径带图 ───────────────────────────────────────────────────────


_IMAGE_B64 = "ZmFrZWJhc2U2NGRhdGE="  # 任意合法 base64；内容不重要，重要的是它是不是文本


async def test_ask_user_cold_path_delivers_an_image_into_the_tool_result() -> None:
    """驱动：`ask_user` 让出 `needs_human`（`reply_as_result=True`）→ 零热窗立即驱逐 →
    `reply_to_hitl` 携带多模态 `message`（文本 + 图片）→ `recover_agent` → reconcile
    命中工具阶段的决定缓存短路 → `_human_reply_as_result` 把答复（含图）拼进 TOOL_RESULT。

    会因下列任一项回归而失败：
    - 冷路径下 `ask_user` 的图片被丢弃或降级成文字描述（非文本 part 没有被
      正确拼进最终 content）——断言的是 memory 里 TOOL_RESULT 含**真实 `ImagePart`**、
      逐字节等于原始 base64，而不是「文本里提到了图片」；
    - 冷路径没有真正续跑（任务卡在 AWAITING_HUMAN，断言 FINISHED 会失败）。
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _ActRouterLLM(act_responses=[
        MockResponse(text="", tool_calls=[ToolCall(
            id="tc_ask", name="control__ask_user",
            arguments={"questions": [{"question": "Which DB?"}]},
        )]),
        _finish_call(),
    ])
    config = RuntimeConfig(hitl_timeout_sec=0)
    runtime = make_runtime(llm=llm, agent_provider=resolver, config=config)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)
    runtime.providers.register_event_blob_store(_StubEventBlobStore())

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="set up the db", context_limit=100_000,
    ))
    sid = handle.session_id

    state = await handle.wait_for_finish(timeout=5.0)
    assert state is not None and state.task.status == "AWAITING_HUMAN"

    pending = runtime.hitl_registry.list_pending(session_id=sid)
    assert len(pending) == 1
    req = pending[0]
    assert req.form == "question"
    assert isinstance(req.delivery, ToolResultDelivery)

    reply_message = [
        TextPart(text="It's postgres"),
        ImagePart(data=_IMAGE_B64, media_type="image/png"),
    ]
    view = await runtime.reply_to_hitl(HitlReply(
        hitl_id=req.id, outcome="accepted", agent_id=req.agent_id, message=reply_message,
    ))
    assert view is not None

    tm = runtime._task_managers[sid]

    def _final_task():
        t = tm.get_task(req.task_id)
        return t if (t is not None and t.status in ("FINISHED", "FAILED", "CANCELED")) else None

    task = await _poll(_final_task)
    assert task.status == "FINISHED", f"expected FINISHED, got {task.status}"

    ctxp = ProviderContext(session_id=sid, tenant_id="default")
    scope = state.scope
    results = await memory.recall_recent(scope, [MemoryEventType.TOOL_RESULT], 20, ctxp)
    # tool_call_id 已是摄入点铸造的内部标识（spec: conversation-integrity）——按 tool_name
    # 定位 ask_user 的 TOOL_RESULT；raw id（tc_ask）可在 assistant 回合 metadata.raw_id 追溯。
    matching = [r for r in results if r.metadata.get("tool_name") == "control__ask_user"]
    assert matching, f"no TOOL_RESULT recorded for ask_user; got {[r.metadata for r in results]}"
    content = matching[0].content
    assert isinstance(content, list), (
        f"TOOL_RESULT content should carry parts (text + image), got {type(content).__name__}"
    )
    images = [p for p in content if getattr(p, "type", None) == "image"]
    assert len(images) == 1, f"expected exactly one real ImagePart, got {content!r}"
    assert images[0].data == _IMAGE_B64, "image bytes were not carried through unchanged"
    assert images[0].media_type == "image/png"
    text_parts = [p for p in content if getattr(p, "type", None) == "text"]
    assert any("postgres" in p.text for p in text_parts), (
        "the human's text answer did not survive alongside the image"
    )


# ── 4. 纯文本暂停 + 注入幂等 ──────────────────────────────────────────────────────


async def test_plain_text_pause_injects_reply_once_and_ignores_duplicate() -> None:
    """驱动：act 纯文本回合（无 tool_call）在 interactive 根任务上冷 park（`UserTurnDelivery`）
    → `reply_to_hitl` 注入一条 `user` 回合并把任务重排回 `PENDING` → 续跑完成。第二次对
    **同一个** `hitl_id` 应答须是纯粹 no-op：`resolve()` 对已终局请求幂等返回 `None`，
    `_write_hitl_reply_turn` 的记忆写入还额外带着 `id=f"hitlreply:{hitl_id}"` 的幂等键
    兜底——两层任一失守，重复应答都会在对话里多出一轮。

    会因下列任一项回归而失败：
    - 纯文本没有触发冷 park（task 未落 AWAITING_HUMAN / session 未落 PAUSED）；
    - 应答后用户回复没有以 `user` 回合的形式真正进入 memory；
    - 重复应答被误当作新事实处理，写出第二条注入（无论是因为 `resolve()` 对已终局请求
      不再幂等，还是因为幂等键失效）。
    """
    llm = _ActRouterLLM(act_responses=[
        MockResponse(text="Hi! Anything else?"),  # 纯文本、无 tool_call → 冷 park
        _finish_call(),
    ])
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi", context_limit=100_000,
    ))
    sid = handle.session_id

    state = await handle.wait_for_finish(timeout=5.0)
    assert state is not None
    assert state.task.status == "AWAITING_HUMAN"

    pending = runtime.hitl_registry.list_pending(session_id=sid)
    assert len(pending) == 1
    req = pending[0]
    assert req.form == "wait"
    assert isinstance(req.delivery, UserTurnDelivery)
    assert req.delivery.task_id == req.task_id

    reply = HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
                      message="use postgres too")
    first_view = await runtime.reply_to_hitl(reply)
    assert first_view is not None and first_view.outcome == "accepted"

    tm = runtime._task_managers[sid]

    def _final_task():
        t = tm.get_task(req.task_id)
        return t if (t is not None and t.status in ("FINISHED", "FAILED", "CANCELED")) else None

    task = await _poll(_final_task)
    assert task.status == "FINISHED", f"expected FINISHED, got {task.status}"

    ctxp = ProviderContext(session_id=sid, tenant_id="default")
    scope = state.scope

    recs = await memory.recall_recent(scope, [MemoryEventType.USER_PROMPT], 50, ctxp)
    injected = [r for r in recs if r.metadata.get("source") == "hitl_reply"]
    assert len(injected) == 1, f"expected exactly one injected reply turn, got {len(injected)}"
    assert "use postgres too" in injected[0].content

    # 重复应答：hitl_id 已终局 → resolve() 幂等返回 None，不触发第二次冷续跑。
    second_view = await runtime.reply_to_hitl(reply)
    assert second_view is None, "a duplicate reply to an already-resolved HITL must be a no-op"

    # 给可能被误触发的第二次续跑一点时间窗口，然后确认注入计数纹丝不动。
    await asyncio.sleep(0.05)
    recs_after = await memory.recall_recent(scope, [MemoryEventType.USER_PROMPT], 50, ctxp)
    injected_after = [r for r in recs_after if r.metadata.get("source") == "hitl_reply"]
    assert len(injected_after) == 1, (
        f"duplicate reply produced a second injected turn: {injected_after}"
    )
