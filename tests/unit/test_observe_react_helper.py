"""run_observe_react：跑 ReAct，仅 terminal_tool_name 被调用时终止；不写 task 状态。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import ctx_weft.core.loop.steps.observe as _obs_mod
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.loop.steps.observe import run_observe_react
from ctx_weft.protocols import LLMMessage, LLMUsage, MemoryAddress
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer

pytestmark = pytest.mark.asyncio


# ── minimal fakes ─────────────────────────────────────────────────────────────


class _FakeEventBus:
    async def emit(self, event) -> None:
        pass


class _RecordingEventBus:
    """Records emitted events (and their types) so tests can assert which fired."""

    def __init__(self):
        self.events = []
        self.types = []

    async def emit(self, event) -> None:
        self.events.append(event)
        self.types.append(event.type)


class _FakeGateway:
    """Returns a success result (content=tool_content) for any tool invocation.

    返回形态对齐生产 InvocationResult（含 is_error）——run_observe_react 读该字段
    判定 terminal 失败，缺字段会 AttributeError。
    """

    def __init__(self, tool_content: str):
        self._tool_content = tool_content

    async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
        return SimpleNamespace(content=self._tool_content, is_error=False, metadata={})


def _make_tool_call_chunk(name: str, call_id: str = "tc1"):
    """Chunk that looks like a tool_call kind chunk from stream_llm_resilient."""
    return SimpleNamespace(
        kind="tool_call",
        tool_call=SimpleNamespace(id=call_id, name=name, arguments={}),
        text="",
        usage=None,
    )


def _make_token_chunk(text: str):
    return SimpleNamespace(kind="token", text=text, tool_call=None, usage=None)


def _make_usage_chunk(prompt_tokens: int = 10, completion_tokens: int = 5):
    return SimpleNamespace(
        kind="usage",
        usage=LLMUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        tool_call=None,
        text="",
    )


# ── state / ctx factories ─────────────────────────────────────────────────────


def _make_state():
    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(max_turns_per_observe=3, compact_keep_last=2),
        runtime={"llm_model": "mock"},
        loop_guard=SimpleNamespace(context_limit=100_000, context_tokens=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    task = SimpleNamespace(
        id="t1",
        parent_task_id="p1",
        status="RUNNING",
        observer_outcome=None,
        process_report=None,
    )
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")

    from ctx_weft.core.loop.driver import LoopState
    state = LoopState(
        run_id="run-test",
        session=session,
        task=task,
        agent=agent,
        scope=scope,
        extra={"template": None, "bound_capabilities": []},
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    return state


def _make_ctx(tool_content: str, event_bus=None, gateway=None):
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from ctx_weft.protocols import ProviderContext
    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    class _FakeAssembler:
        async def assemble(self, req):
            return SimpleNamespace(system="SYS", messages=[], tools=[])

    class _FakeLLM:
        tokenizer = HeuristicTokenizer()

        async def complete(self, req, stream=True):
            return
            yield  # unreachable; stream_llm_resilient is monkeypatched in tests

    from ctx_weft.core.loop.driver import LoopContext
    ctx = LoopContext(
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        memory=mem,
        event_bus=event_bus or _FakeEventBus(),
        provider_ctx=pctx,
        capability_gateway=gateway or _FakeGateway(tool_content),
    )
    return ctx


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_helper_returns_tool_content_when_control_tool_called(monkeypatch):
    """Round 1 produces a tool_call; gateway returns ControlResult('REPORT').
    Helper must return ('REPORT', last_text)."""

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("thinking...")
        yield _make_tool_call_chunk("report_task_outcome")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    ctx = _make_ctx(tool_content="REPORT")

    tool_content, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name="report_task_outcome",
    )

    assert tool_content is not None
    assert tool_content.content == "REPORT"


async def test_plain_text_round_nudged_then_terminal(monkeypatch):
    """round 0 纯文本（无 tool call）→ 不应立即放弃：追加催促消息再试一轮；
    round 1 调 terminal 工具 → 返回其 ControlResult。

    催促轮的请求消息里须带上 round 0 的 assistant 文本 + 一条点名 terminal 工具的 user 催促。
    """

    requests = []

    async def _fake_stream(ctx, state, request):
        requests.append(request)
        if len(requests) == 1:
            yield _make_token_chunk("prose recap")
            yield _make_usage_chunk()
        else:
            yield _make_tool_call_chunk("collect_process_report")
            yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    ctx = _make_ctx(tool_content="REPORT")

    result, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name="collect_process_report",
    )

    assert result is not None, "纯文本轮后应催促重试而非直接返回 None"
    assert result.content == "REPORT"
    assert len(requests) == 2
    msgs2 = requests[1].messages
    assert any(m.role == "assistant" and "prose recap" in (m.content or "") for m in msgs2), \
        "催促轮请求须携带上一轮的 assistant 文本"
    assert any(m.role == "user" and "collect_process_report" in str(m.content) for m in msgs2), \
        "催促消息须点名 terminal 工具"


async def test_helper_returns_none_when_no_tool_called(monkeypatch):
    """LLM returns only text (no tool_calls) every round; helper exhausts max_rounds
    then returns (None, last_text)."""

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("just text")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    ctx = _make_ctx(tool_content="IGNORED")

    tool_content, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name="report_task_outcome",
    )

    assert tool_content is None
    assert last_text == "just text"


async def test_helper_last_text_from_final_round(monkeypatch):
    """When the control tool is called, last_text is the text from that round."""

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("final thinking")
        yield _make_tool_call_chunk("report_task_outcome")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    ctx = _make_ctx(tool_content="DONE")

    tool_content, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name="report_task_outcome",
    )

    assert tool_content is not None
    assert tool_content.content == "DONE"
    assert last_text == "final thinking"


async def test_helper_token_accounting(monkeypatch):
    """session.token_used is accumulated from usage chunks."""

    async def _fake_stream(ctx, state, request):
        yield _make_usage_chunk(prompt_tokens=20, completion_tokens=8)

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    state.session.token_used = 0
    ctx = _make_ctx(tool_content="X")

    await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[],
        tools=[],
        max_rounds=1,
        terminal_tool_name="report_task_outcome",
    )

    assert state.session.token_used == 28  # 20 + 8


async def test_ask_user_does_not_terminate_loop(monkeypatch):
    """ask_user（非 terminal 工具）被调用时不应终止循环、不返回其 ControlResult。

    场景：max_rounds=1，round 0 仅调用 ask_user（非 terminal），
    helper 应返回 (None, last_text) 而非 ask_user 的 content。

    这是 Task 5 重构引入的回归：旧代码对任意控制工具终止，修复后只对
    terminal_tool_name 终止。
    """

    ASK_USER_TOOL = "control__ask_user"
    TERMINAL_TOOL = "control__report_task_outcome"

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("thinking about user question")
        yield _make_tool_call_chunk(ASK_USER_TOOL, call_id="tc_ask")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    state = _make_state()
    ctx = _make_ctx(tool_content="ASK_USER_RESULT")

    tool_content, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=1,
        terminal_tool_name=TERMINAL_TOOL,
    )

    # ask_user's ControlResult must NOT be returned as terminal content
    assert tool_content is None, (
        f"Expected None (ask_user is not terminal), got {tool_content!r}. "
        "Bug: run_observe_react terminated on non-terminal ask_user call."
    )
    assert last_text == "thinking about user question"


async def test_response_finished_type_and_origin_follow_state_origin(monkeypatch):
    """Task 5：不再有 event_types 参数区分前台/后台——run_observe_react 自己只发一种类型
    （LLM_RESPONSE_FINISHED，其它 4 个「流式侧」事件由 gateway 发，这里 stream_llm_resilient
    整个被 monkeypatch 掉不会跑到那段）。前台/后台的区别只体现在事件的 origin 字段上，
    origin 随 state.origin 走，run_observe_react 本身不感知也不分支。"""

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("thinking")
        yield _make_tool_call_chunk("collect_process_report")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    for origin in (EventOrigin.LOOP_OBSERVE, EventOrigin.LOOP_BACKGROUND_OBSERVE):
        bus = _RecordingEventBus()
        state = _make_state()
        state.origin = origin
        ctx = _make_ctx(tool_content="REPORT", event_bus=bus)

        await run_observe_react(
            state, ctx,
            system="SYS",
            messages=[],
            tools=[],
            max_rounds=1,
            terminal_tool_name="collect_process_report",
        )

        assert bus.types == [EventType.LLM_RESPONSE_FINISHED], (origin, bus.types)
        assert bus.events[0].origin == origin


async def test_run_observe_react_returns_terminal_controlresult(monkeypatch):
    """Terminal tool call returns the full ControlResult (not just content string)."""

    TERMINAL = "report_task_outcome"

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("thinking")
        yield _make_tool_call_chunk(TERMINAL)
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    class _RichGateway:
        async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
            return SimpleNamespace(
                content="recap", is_error=False, metadata={"task_summary": "sum"})

    state = _make_state()
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from ctx_weft.protocols import ProviderContext
    from ctx_weft.core.loop.driver import LoopContext

    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    class _FakeAssembler2:
        async def assemble(self, req):
            return SimpleNamespace(system="SYS", messages=[], tools=[])

    class _FakeLLM2:
        tokenizer = HeuristicTokenizer()

        async def complete(self, req, stream=True):
            return
            yield

    ctx = LoopContext(
        assembler=_FakeAssembler2(),
        llm=_FakeLLM2(),
        memory=mem,
        event_bus=_FakeEventBus(),
        provider_ctx=pctx,
        capability_gateway=_RichGateway(),
    )

    result, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name=TERMINAL,
    )
    assert result is not None
    assert result.content == "recap"
    assert result.metadata.get("task_summary") == "sum"


class _RecordingBusFull:
    """Records full events (not just types) for payload assertions."""

    def __init__(self):
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


# ── terminal 工具失败不终止循环（spec: capability-gateway）────────────────────


class _FlakyTerminalGateway:
    """第 1 次调用返回 is_error=True（如参数非法被 gateway 拒），之后返回成功。"""

    def __init__(self, error_content: str, ok_content: str):
        self._error = error_content
        self._ok = ok_content
        self.calls = 0

    async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(content=self._error, is_error=True, metadata={})
        return SimpleNamespace(content=self._ok, is_error=False, metadata={})


class _AlwaysErrorGateway:
    """每次调用都返回 is_error=True——钉轮次耗尽的兜底。"""

    def __init__(self, error_content: str):
        self._error = error_content

    async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
        return SimpleNamespace(content=self._error, is_error=True, metadata={})


async def test_terminal_tool_error_does_not_terminate_loop(monkeypatch):
    """terminal 工具首调失败（is_error=True）→ 循环不终止；错误回灌后次调成功 →
    返回成功结果，错误文案不进入结果。"""
    requests = []

    async def _fake_stream(ctx, state, request):
        requests.append(request)
        yield _make_token_chunk("round text")
        yield _make_tool_call_chunk("collect_process_report")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    ERR = "[Error: invalid arguments for 'control__collect_process_report': 'act_recap' is required]"
    OK = "RECAP"
    gw = _FlakyTerminalGateway(ERR, OK)
    state = _make_state()
    ctx = _make_ctx(tool_content="unused", gateway=gw)

    result, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=3,
        terminal_tool_name="collect_process_report",
    )

    assert gw.calls == 2, "首轮错误后应重试而非终止"
    assert result is not None
    assert result.content == OK
    assert ERR not in str(result.content)
    # 错误内容须回灌进第二轮请求（模型据此改参）
    assert any(m.role == "tool" and ERR in str(m.content) for m in requests[1].messages)


async def test_terminal_tool_error_exhausts_rounds_returns_none(monkeypatch):
    """terminal 工具连续失败直至轮次耗尽 → 返回 (None, last_text) 兜底，
    不把错误文案当终止结果。"""

    async def _fake_stream(ctx, state, request):
        yield _make_token_chunk("trying")
        yield _make_tool_call_chunk("report_task_outcome")
        yield _make_usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    ERR = "[Error: invalid arguments for 'control__report_task_outcome': ...]"
    gw = _AlwaysErrorGateway(ERR)
    state = _make_state()
    ctx = _make_ctx(tool_content="unused", gateway=gw)

    result, last_text = await run_observe_react(
        state, ctx,
        system="SYS",
        messages=[LLMMessage(role="user", content="observe this")],
        tools=[],
        max_rounds=2,
        terminal_tool_name="report_task_outcome",
    )

    assert result is None
    assert last_text == "trying"


async def test_response_finished_payload_carries_usage_split(monkeypatch):
    """LLM_RESPONSE_FINISHED 的 payload["usage"] 经 asdict 自动携带七字段拆分。"""

    async def _fake_stream(ctx, state, request):
        yield SimpleNamespace(
            kind="usage",
            usage=LLMUsage(prompt_tokens=127, completion_tokens=5, total_tokens=132,
                           cache_read_tokens=100, cache_write_tokens=20),
            tool_call=None, text="",
        )
        yield _make_tool_call_chunk("report_task_outcome")

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _fake_stream)

    bus = _RecordingBusFull()
    state = _make_state()
    ctx = _make_ctx(tool_content="DONE", event_bus=bus)

    await run_observe_react(
        state, ctx, system="SYS", messages=[], tools=[],
        max_rounds=1,
        terminal_tool_name="report_task_outcome",
    )

    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert finished
    u = finished[0].payload["usage"]
    assert u["prompt_tokens"] == 127
    assert u["cache_read_tokens"] == 100
    assert u["cache_write_tokens"] == 20
    assert u["input_tokens"] == 7
    assert u["reasoning_tokens"] == 0
