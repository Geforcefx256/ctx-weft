"""Task 4：4 种「流式侧」LLM_* 事件收敛到 llm_gateway.stream_llm_resilient。

范围裁定（task-4-brief.md 全量 6 种 → 收窄为 4 种）：
  - 搬进 gateway：LLM_REQUEST_STARTED / LLM_PROMPT_SENT / LLM_TOKEN_STREAMED /
    LLM_REASONING_STREAMED。
  - 留在 act.py：LLM_RESPONSE_FINISHED（payload 依赖软打断决策，见 brief）。
  - 本就在 gateway：LLM_RETRY_TRIGGERED（_emit_retry，Task 3 已加 origin）。
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop import llm_gateway
from ctx_weft.core.loop.steps import act
from ctx_weft.core.loop.steps.act import _run_llm_turn
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols import LLMUsage, MemoryAddress
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer

_MOVED_TYPES = {
    EventType.LLM_REQUEST_STARTED,
    EventType.LLM_PROMPT_SENT,
    EventType.LLM_TOKEN_STREAMED,
    EventType.LLM_REASONING_STREAMED,
}
_GATEWAY_TYPES = _MOVED_TYPES | {EventType.LLM_RETRY_TRIGGERED}


def test_act_no_longer_emits_llm_events():
    """act.py 源码里不应再出现搬走的 4 种 LLM_* 类型；LLM_RESPONSE_FINISHED 仍应保留。"""
    src = inspect.getsource(act)
    leaked = sorted(t.name for t in _MOVED_TYPES if f"EventType.{t.name}" in src)
    assert leaked == [], f"act.py 仍在发射：{leaked}"
    assert "EventType.LLM_RESPONSE_FINISHED" in src, "LLM_RESPONSE_FINISHED 不应被搬走"


def test_gateway_emits_moved_and_retry_events():
    """5 种事件（搬来的 4 个 + 原有的 LLM_RETRY_TRIGGERED）全部由 gateway 发射。"""
    src = inspect.getsource(llm_gateway)
    missing = sorted(t.name for t in _GATEWAY_TYPES if f"EventType.{t.name}" not in src)
    assert missing == [], f"gateway 未接管：{missing}"


# ── request_id 一致性回归（task-4 最容易出错的地方）─────────────────────────────
#
# LLM_REQUEST_STARTED / LLM_PROMPT_SENT 现由 gateway 发射，LLM_RESPONSE_FINISHED 仍由
# act.py 发射——两侧各自用同一个确定性公式 f"req_{agent.id}_{state.sequence_counter}"
# 独立算出 request_id，都在「本次 LLM 调用的任何事件被发射之前」求值，因此理应相等。
# 本测试直接跑一次真实 _run_llm_turn（不 monkeypatch stream_llm_resilient），验证
# 三者的 payload["request_id"] 确实一致——防止两处公式后续被改得不一致而不被察觉。


class _RecordingBus:
    def __init__(self):
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


class _FakeLLM:
    """脚本化 LLMClient：先吐两个 token chunk，再吐一个 usage chunk。记录收到的
    ``request`` 供测试断言**实际发给 LLM client** 的 model/messages——这与事件 payload
    里报的 model 是两回事：payload 由 resolve_llm_identity(state) 独立算出，request.model
    是调用方传给 LLMRequest(...) 构造函数的那个值，两者可能不同源（compact.py 曾经的
    bug 正是这种「payload 对、request 错」的分裂，见
    test_compact_sends_resolved_model_to_llm_client_not_mock_sentinel）。"""
    tokenizer = HeuristicTokenizer()
    context_limit = 100_000

    def __init__(self):
        self.last_request = None

    async def complete(self, req, stream=True):
        self.last_request = req
        yield SimpleNamespace(kind="token", text="he", tool_call=None, usage=None)
        yield SimpleNamespace(kind="token", text="llo", tool_call=None, usage=None)
        yield SimpleNamespace(
            kind="usage", text="", tool_call=None,
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5),
        )


def _make_state():
    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(max_turns_per_observe=3, compact_keep_last=2),
        runtime={},
        loop_guard=SimpleNamespace(context_limit=100_000, context_tokens=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    task = SimpleNamespace(
        id="t1", parent_task_id="p1", status="RUNNING",
        observer_outcome=None, process_report=None,
        # act 的提交点（spec 2026-09-09）会调 persist_user_prompt——这两个字段是它的判据。
        # 本文件只测一次 LLM turn，没有待落库的 user_prompt：标成「已落库」即 no-op。
        user_prompt=None, user_prompt_in_memory=True,
    )
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    from ctx_weft.core.loop.driver import LoopState
    return LoopState(
        run_id="run-test", session=session, task=task, agent=agent, scope=scope,
        extra={"template": None, "bound_capabilities": []},
        resolved_model=SimpleNamespace(model="mock-model", account="mock-acct"),
        origin=EventOrigin.LOOP_ACT,
    )


def _make_ctx(event_bus):
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from ctx_weft.protocols import ProviderContext
    from ctx_weft.core.loop.driver import LoopContext

    class _FakeAssembler:
        async def assemble(self, req):
            return SimpleNamespace(system="SYS", messages=[], tools=[])

    class _FakeGateway:
        async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id=None):
            raise AssertionError("no tool call expected in this test")

    return LoopContext(
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        memory=InMemoryMemoryProvider(),
        event_bus=event_bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="t1", agent_id="a1"),
        capability_gateway=_FakeGateway(),
    )


@pytest.mark.asyncio
async def test_request_started_prompt_sent_response_finished_share_request_id():
    bus = _RecordingBus()
    state = _make_state()
    ctx = _make_ctx(bus)
    prompt = SimpleNamespace(system="SYS", tools=[])

    await _run_llm_turn(state, ctx, prompt, [], 1)

    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    prompt_sent = [e for e in bus.events if e.type == EventType.LLM_PROMPT_SENT]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(prompt_sent) == 1 and len(finished) == 1

    rid = started[0].payload["request_id"]
    assert rid  # 非空
    assert prompt_sent[0].payload["request_id"] == rid
    assert finished[0].payload["request_id"] == rid

    # 顺带确认流式事件也确实搬到了 gateway 侧（TOKEN_STREAMED 携带同一 request_id）。
    tokens = [e for e in bus.events if e.type == EventType.LLM_TOKEN_STREAMED]
    assert len(tokens) == 2
    assert all(e.payload["request_id"] == rid for e in tokens)


# ── 复审修复轮：compact 也要放行 + 补收尾事件；_maybe_predispatch_compact 的 origin ──
#
# 控制方复审发现：
#   1. gateway 的 origin 门禁原来只放行 LOOP_ACT，但 compact（summarize_for_compact）此前
#      一个 LLM_* 事件都不发，放行它是纯增量，不存在 observe/background_observe 那种重复
#      发射/串号风险——理应一并放行。
#   2. 放行 compact 后，summarize_for_compact 必须补发 LLM_RESPONSE_FINISHED，否则
#      REQUEST_STARTED 有了收尾却永远不来，host SSE 看到一条挂死请求。
#   3. act.py::_maybe_predispatch_compact 在 Act 步骤执行期间调用 compact，driver 只在步骤
#      边界切 origin，此刻仍是 LOOP_ACT——必须临时切到 LOOP_COMPACT，调用结束（含异常）后
#      还原，否则这次「实际是 compact 摘要」的调用会被门禁错当成 act 的一次 LLM turn。


class _FakeCompactAssembler:
    async def assemble(self, req):
        return SimpleNamespace(system="SYS-COMPACT", messages=[], tools=[])


def _make_compact_state(*, origin: str = EventOrigin.LOOP_COMPACT):
    agent = SimpleNamespace(
        id="a1", runtime={"llm_model": "mock-model"},
        loop_guard=SimpleNamespace(context_limit=100_000, context_tokens=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    task = SimpleNamespace(id="t1", parent_task_id="p1", status="RUNNING")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    from ctx_weft.core.loop.driver import LoopState
    return LoopState(
        run_id="run-test", session=session, task=task, agent=agent, scope=scope,
        extra={"template": None, "bound_capabilities": []},
        resolved_model=SimpleNamespace(model="mock-model", account="mock-acct"),
        origin=origin,
    )


def _make_compact_ctx(event_bus, *, llm=None):
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from ctx_weft.protocols import ProviderContext
    from ctx_weft.core.loop.driver import LoopContext

    return LoopContext(
        assembler=_FakeCompactAssembler(),
        llm=llm if llm is not None else _FakeLLM(),
        memory=InMemoryMemoryProvider(),
        event_bus=event_bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="t1", agent_id="a1"),
    )


@pytest.mark.asyncio
async def test_gateway_emits_stream_events_for_compact_origin_and_pairs_with_response_finished():
    """origin==LOOP_COMPACT 放行 4 个流式事件；summarize_for_compact 补发的
    LLM_RESPONSE_FINISHED 与它们共享同一个 request_id（同 act.py 那条回归的镜像用例）。"""
    from ctx_weft.core.loop.steps.compact import summarize_for_compact

    bus = _RecordingBus()
    state = _make_compact_state()
    ctx = _make_compact_ctx(bus)

    out = await summarize_for_compact(state, ctx, scope="task")

    # spec: compact-fidelity——返回 ParsedDigest；散文输出整文落库 + degraded。
    assert out.text == "hello" and out.degraded is True
    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    prompt_sent = [e for e in bus.events if e.type == EventType.LLM_PROMPT_SENT]
    tokens = [e for e in bus.events if e.type == EventType.LLM_TOKEN_STREAMED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(prompt_sent) == 1 and len(finished) == 1
    assert len(tokens) == 2

    rid = started[0].payload["request_id"]
    assert rid
    assert prompt_sent[0].payload["request_id"] == rid
    assert finished[0].payload["request_id"] == rid
    assert all(e.payload["request_id"] == rid for e in tokens)
    assert finished[0].payload["content"] == "hello"
    assert finished[0].payload["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_compact_events_carry_resolved_model_not_mock_sentinel():
    """事件 payload 层面的健康检查（非本次改动的验收测试——见下面那条注释）：
    LLM_REQUEST_STARTED/LLM_RESPONSE_FINISHED 的 model/llm_account 都来自
    resolve_llm_identity(state)，不是 "mock" 字面量。

    **复审纠偏**：gateway 的 LLM_REQUEST_STARTED/PROMPT_SENT（llm_gateway.py 内联
    resolve_llm_identity 调用）与 compact 自己的 LLM_RESPONSE_FINISHED，从一开始就各自
    独立算 model/llm_account，从未依赖过 llm_request.model——事件 payload 从来没撒过谎。
    本测试因此测不出"summarize_for_compact 曾经把 model=agent.runtime.get('llm_model',
    'mock') 传给 LLM client"这个真正的 bug（已实测核实：把那行代码改回旧写法，本测试
    仍然全绿）。真正钉住那个 bug 的是下面
    test_compact_sends_resolved_model_to_llm_client_not_mock_sentinel，直接检查发给
    LLM client 的 request.model。本测试保留是因为它验证的东西本身仍然成立、仍然值得
    有回归保护，只是不再声称自己是本次改动的验收测试。"""
    from ctx_weft.core.loop.steps.compact import summarize_for_compact

    bus = _RecordingBus()
    state = _make_compact_state()
    state.agent.runtime = {}  # 正常任务执行下 agent.runtime 就是这样——不含 llm_model
    state.resolved_model = SimpleNamespace(model="claude-real-model", account="acct-real")
    ctx = _make_compact_ctx(bus)

    await summarize_for_compact(state, ctx, scope="task")

    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED][0]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED][0]
    assert started.payload["model"] == "claude-real-model"
    assert started.payload["llm_account"] == "acct-real"
    assert finished.payload["llm_model"] == "claude-real-model"
    assert finished.payload["llm_account"] == "acct-real"
    assert "mock" not in started.payload["model"]
    assert "mock" not in finished.payload["llm_model"]


@pytest.mark.asyncio
async def test_compact_sends_resolved_model_to_llm_client_not_mock_sentinel():
    """真正钉住本次改动的验收测试：断言**实际发给 LLM client 的 request.model**
    （不是事件 payload），验收标准是「把 summarize_for_compact 的
    `model=model` 改回 `model=agent.runtime.get("llm_model", "mock")`，本测试必须变红」
    ——已实测确认这一点（改回旧写法 → 本测试失败 `assert 'mock' == 'claude-real-model'`
    这类；`test_compact_events_carry_resolved_model_not_mock_sentinel` 及全仓其余测试
    在同样的回退下仍然全绿，测不出问题，这正是复审揪出的盲区）。

    走 inline compact 语义的最小化 state/ctx（`_make_compact_state`/`_make_compact_ctx`
    本就是给 inline 路径搭的最小夹具）——**不**走 `compact_agent`，因为那条路径靠
    `runtime.py:1809` 的 `agent.runtime={"llm_model": rm.model}` 桥接，一直是对的，
    测不出 inline 路径这个 bug。
    """
    from ctx_weft.core.loop.steps.compact import summarize_for_compact

    state = _make_compact_state()
    state.agent.runtime = {}  # 正常任务执行下 agent.runtime 就是这样——不含 llm_model
    state.resolved_model = SimpleNamespace(model="claude-real-model", account="acct-real")
    llm = _FakeLLM()
    ctx = _make_compact_ctx(_RecordingBus(), llm=llm)

    await summarize_for_compact(state, ctx, scope="task")

    assert llm.last_request is not None
    assert llm.last_request.model == "claude-real-model"
    assert llm.last_request.model != "mock"


@pytest.mark.asyncio
async def test_compact_stays_silent_on_response_finished_under_wrong_origin():
    """Task 5 更新：gateway 的临时门禁（曾只放行 LOOP_ACT/LOOP_COMPACT）已整个删除——
    gateway 现在对所有 origin 无条件发射 4 个流式事件，下面第一条断言随之翻转。但
    summarize_for_compact 自己的收尾门禁（origin==LOOP_COMPACT）是另一回事、独立保留：
    万一在错误 origin 下被调用，不该补发一条没有配对 REQUEST_STARTED 语境的孤儿
    LLM_RESPONSE_FINISHED（比不发更糟，见该函数文档字符串）。"""
    from ctx_weft.core.loop.steps.compact import summarize_for_compact

    bus = _RecordingBus()
    state = _make_compact_state(origin=EventOrigin.LOOP_OBSERVE)
    ctx = _make_compact_ctx(bus)

    await summarize_for_compact(state, ctx, scope="task")

    assert [e for e in bus.events if e.type in _MOVED_TYPES] != [], (
        "gateway 现在应对所有 origin 无条件发射流式事件（Task 5 删除了临时门禁）"
    )
    assert [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED] == [], (
        "compact 自己的收尾门禁仍应挡住错误 origin 下的孤儿 RESPONSE_FINISHED"
    )


@pytest.mark.asyncio
async def test_predispatch_compact_swaps_origin_to_compact_and_restores_on_success(monkeypatch):
    from ctx_weft.core.loop.steps.act import _maybe_predispatch_compact
    from ctx_weft.core.loop import capability_gateway as _cap_mod
    from ctx_weft.core.loop.steps import compact as _compact_mod

    dispatch_tool_name = next(iter(_cap_mod.DISPATCH_TOOLS))
    tool_calls = [SimpleNamespace(name=dispatch_tool_name, id="tc1", arguments={})]
    seen_origin = {}

    async def _fake_maybe_compact(state, ctx, *, prompt_tokens):
        seen_origin["value"] = state.origin
        return []

    monkeypatch.setattr(_compact_mod, "maybe_compact_before_dispatch", _fake_maybe_compact)
    bus = _RecordingBus()
    state = _make_state()  # origin=LOOP_ACT，同 _run_llm_turn 测试用的那份夹具
    ctx = _make_ctx(bus)

    await _maybe_predispatch_compact(state, ctx, tool_calls, LLMUsage(prompt_tokens=1))

    assert seen_origin["value"] == EventOrigin.LOOP_COMPACT
    assert state.origin == EventOrigin.LOOP_ACT  # 调用结束后已还原


@pytest.mark.asyncio
async def test_predispatch_compact_restores_origin_even_on_exception(monkeypatch):
    from ctx_weft.core.loop.steps.act import _maybe_predispatch_compact
    from ctx_weft.core.loop import capability_gateway as _cap_mod
    from ctx_weft.core.loop.steps import compact as _compact_mod

    dispatch_tool_name = next(iter(_cap_mod.DISPATCH_TOOLS))
    tool_calls = [SimpleNamespace(name=dispatch_tool_name, id="tc1", arguments={})]

    async def _boom(state, ctx, *, prompt_tokens):
        raise RuntimeError("boom")

    monkeypatch.setattr(_compact_mod, "maybe_compact_before_dispatch", _boom)
    bus = _RecordingBus()
    state = _make_state()
    ctx = _make_ctx(bus)

    with pytest.raises(RuntimeError, match="boom"):
        await _maybe_predispatch_compact(state, ctx, tool_calls, LLMUsage(prompt_tokens=1))

    assert state.origin == EventOrigin.LOOP_ACT  # finally 里还原，异常路径也不例外


# ── Task 5：删除 ReactEventTypes 间接层，observe/background_observe 靠 origin 收敛 ──
#
# run_observe_react 不再接收 event_types/request_id_prefix 形参；「观察是前台 observe
# 还是后台 background_observe」不再靠传入不同的一组事件类型区分，改靠调用前已经设好的
# state.origin（观察前台由 driver 按 step 名设好 LOOP_OBSERVE；background_observe 在
# launch 出的独立快照 state 上显式改写 LOOP_BACKGROUND_OBSERVE）。gateway 的临时门禁
# （Task 4 留下的 _STREAM_EVENT_ORIGINS）随之整个删除，对所有 origin 无条件发射。


def test_react_event_types_indirection_is_gone():
    """该间接层唯一的目的是区分两组事件类型，目的消失则层消失。"""
    from ctx_weft.core.loop.steps import observe

    for name in ("ReactEventTypes", "OBSERVE_REACT_EVENTS", "BACKGROUND_OBSERVE_REACT_EVENTS"):
        assert not hasattr(observe, name), f"{name} 应已删除"


def test_run_observe_react_has_no_event_types_param():
    from ctx_weft.core.loop.steps import observe

    sig = inspect.signature(observe.run_observe_react)
    assert "event_types" not in sig.parameters
    # request_id 方案改用与 gateway 相同的确定性公式独立算出，prefix 形参随之失去用途。
    assert "request_id_prefix" not in sig.parameters


def test_background_observe_types_never_emitted_under_core_src():
    """BACKGROUND_OBSERVE_* 四个类型仍留在 EventType 枚举里（退役登记是 Task 7 的事），
    但本任务起 src/ctx_weft/core/ 下不应再有任何发射点——统一走 LLM_* + origin。"""
    import pathlib

    core_dir = pathlib.Path(__file__).resolve().parents[2] / "src" / "ctx_weft" / "core"
    assert core_dir.is_dir(), core_dir
    hits = []
    for py in core_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if "EventType.BACKGROUND_OBSERVE_" in text:
            hits.append(str(py))
    assert hits == [], f"仍在发射 BACKGROUND_OBSERVE_*：{hits}"


@pytest.mark.asyncio
async def test_observe_path_events_share_origin_and_request_id():
    """observe 前台：gateway 的 4 个流式事件 + run_observe_react 自己补发的收尾事件，
    5 个事件共用一个 request_id，且 origin 全部是 loop.observe（state.origin 由调用方
    设好，run_observe_react 不用管、也不该管——它只管跑 ReAct）。"""
    from ctx_weft.core.loop.steps.observe import run_observe_react

    bus = _RecordingBus()
    state = _make_state()
    state.origin = EventOrigin.LOOP_OBSERVE
    ctx = _make_ctx(bus)

    await run_observe_react(
        state, ctx, system="SYS", messages=[], tools=[],
        max_rounds=1, terminal_tool_name="report_task_outcome",
    )

    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    prompt_sent = [e for e in bus.events if e.type == EventType.LLM_PROMPT_SENT]
    tokens = [e for e in bus.events if e.type == EventType.LLM_TOKEN_STREAMED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(prompt_sent) == 1 and len(finished) == 1
    assert len(tokens) == 2

    five = started + prompt_sent + tokens + finished
    assert len(five) == 5
    rid = started[0].payload["request_id"]
    assert rid
    assert all(e.payload["request_id"] == rid for e in five)
    assert all(e.origin == EventOrigin.LOOP_OBSERVE for e in five)


@pytest.mark.asyncio
async def test_background_observe_path_events_share_origin_and_request_id():
    """background_observe：调用前把 state.origin 显式改写为 loop.background_observe
    （同 background_observe.py:_run_background_observe 在调 run_observe_react 前做的事）
    ——同样 5 个事件的 origin 全部随之变成 loop.background_observe，且从不出现已废弃的
    BACKGROUND_OBSERVE_* 类型。"""
    from ctx_weft.core.loop.steps.observe import run_observe_react

    bus = _RecordingBus()
    state = _make_state()
    state.origin = EventOrigin.LOOP_BACKGROUND_OBSERVE
    ctx = _make_ctx(bus)

    await run_observe_react(
        state, ctx, system="SYS", messages=[], tools=[],
        max_rounds=1, terminal_tool_name="collect_process_report",
    )

    five = [e for e in bus.events if e.type in (
        EventType.LLM_REQUEST_STARTED, EventType.LLM_PROMPT_SENT,
        EventType.LLM_TOKEN_STREAMED, EventType.LLM_RESPONSE_FINISHED,
    )]
    assert len(five) == 5
    rid = five[0].payload["request_id"]
    assert rid
    assert all(e.payload["request_id"] == rid for e in five)
    assert all(e.origin == EventOrigin.LOOP_BACKGROUND_OBSERVE for e in five)

    _OBSOLETE = {
        EventType.BACKGROUND_OBSERVE_REQUEST_STARTED, EventType.BACKGROUND_OBSERVE_PROMPT_SENT,
        EventType.BACKGROUND_OBSERVE_TOKEN_STREAMED, EventType.BACKGROUND_OBSERVE_RESPONSE_FINISHED,
    }
    assert [e for e in bus.events if e.type in _OBSOLETE] == []


# ── Task 6：recognize_intent 切 stream_llm_resilient + 补发 LLM_RESPONSE_FINISHED ──
#
# 裸 stream_llm 没有自愈退避——切到 resilient 顺带修掉这个缺陷，并白拿 gateway 那 4 个
# 流式事件。RECOGNIZE_INTENT_LLM_PROMPT（该 step 专属的镜像事件）随之删除，角色由通用
# LLM_PROMPT_SENT 承担；切网关后 LLM_REQUEST_STARTED 有始无终，故本 step 自己补发
# LLM_RESPONSE_FINISHED（payload 结构照抄 act.py::_run_llm_turn），request_id 用同一个
# 确定性公式独立算出，与 gateway 的 4 个天然一致。

from ctx_weft.core.loop.steps import recognize_intent


def test_recognize_intent_uses_resilient_gateway():
    """裸 stream_llm 没有自愈退避——切到 resilient 顺带修掉这个缺陷。"""
    src = inspect.getsource(recognize_intent)
    assert "stream_llm_resilient" in src
    assert "content_to_text" not in src, "脱敏应统一到 redact_content_for_event"


def test_recognize_intent_no_longer_emits_its_mirror_event():
    src = inspect.getsource(recognize_intent)
    assert "RECOGNIZE_INTENT_LLM_PROMPT" not in src


def _make_recognize_intent_state():
    from ctx_weft.protocols.capability import ToolCapability
    agent = SimpleNamespace(
        id="a1", runtime={}, loop_guard=SimpleNamespace(context_limit=100_000, context_tokens=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    # status: launch_recognize_intent._run() 的 RUN_FINISHED payload 读
    # snapshot.task.status（final_status）——即使本测试组大多只驱动
    # RecognizeIntentStep.execute() 本身（用不到这个字段），也一并给上，供驱动
    # launch_recognize_intent 的那条测试复用同一个 fixture。
    task = SimpleNamespace(
        id="t1", parent_task_id=None, title="", status="RUNNING", settings=SimpleNamespace())
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    cap = ToolCapability(id="cap1", name="update_task_metadata", purposes=["recognize_intent"])
    from ctx_weft.core.loop.driver import LoopState
    return LoopState(
        run_id="run-test", session=session, task=task, agent=agent, scope=scope,
        extra={"template": None, "bound_capabilities": [cap]},
        resolved_model=SimpleNamespace(model="mock-model", account="mock-acct"),
        origin=EventOrigin.LOOP_RECOGNIZE_INTENT,
    )


def _make_recognize_intent_ctx(event_bus, *, llm=None, config=None):
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from ctx_weft.protocols import ProviderContext
    from ctx_weft.core.loop.driver import LoopContext

    class _FakeRIAssembler:
        async def assemble(self, req):
            return SimpleNamespace(system="SYS-RI", messages=[], tools=[])

    class _FakeRIGateway:
        async def invoke(self, *, tool_name, arguments, state, ctx):
            pass

    return LoopContext(
        assembler=_FakeRIAssembler(),
        llm=llm if llm is not None else _FakeLLM(),
        memory=InMemoryMemoryProvider(),
        event_bus=event_bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="t1", agent_id="a1"),
        capability_gateway=_FakeRIGateway(),
        config=config,
    )


@pytest.mark.asyncio
async def test_recognize_intent_events_share_origin_and_request_id():
    """gateway 的 4 个流式事件 + recognize_intent 自己补发的收尾事件，5 个共用一个
    request_id，且 origin 全部是 loop.recognize_intent；镜像事件不再出现在事件流里。"""
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    ctx = _make_recognize_intent_ctx(bus)

    outcome = await RecognizeIntentStep().execute(state, ctx)
    assert outcome.next_step is None

    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    prompt_sent = [e for e in bus.events if e.type == EventType.LLM_PROMPT_SENT]
    tokens = [e for e in bus.events if e.type == EventType.LLM_TOKEN_STREAMED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(prompt_sent) == 1 and len(finished) == 1
    assert len(tokens) == 2

    five = started + prompt_sent + tokens + finished
    assert len(five) == 5
    rid = started[0].payload["request_id"]
    assert rid
    assert all(e.payload["request_id"] == rid for e in five)
    assert all(e.origin == EventOrigin.LOOP_RECOGNIZE_INTENT for e in five)

    assert [e for e in bus.events if e.type == "RecognizeIntentLLMPrompt"] == []


@pytest.mark.asyncio
async def test_recognize_intent_self_heals_via_resilient_gateway():
    """切网关的直接收益：裸 stream_llm 没有退避重试，切到 stream_llm_resilient 后，
    首次瞬时故障（outage）应被自愈层原地重试并最终成功，不冒泡成异常。"""
    from ctx_weft.protocols import LLMCallError
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    class _FlakyLLM:
        tokenizer = HeuristicTokenizer()
        context_limit = 100_000

        def __init__(self):
            self.attempts = 0

        async def complete(self, req, stream=True):
            self.attempts += 1
            if self.attempts == 1:
                raise LLMCallError("boom", retriable=True, outage=True)
            yield SimpleNamespace(kind="token", text="ok", tool_call=None, usage=None)
            yield SimpleNamespace(
                kind="usage", text="", tool_call=None,
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1))

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    llm = _FlakyLLM()
    ctx = _make_recognize_intent_ctx(bus, llm=llm, config=SimpleNamespace(
        llm_self_heal_base_delay_sec=0.01, llm_self_heal_max_interval_sec=0.01,
        llm_self_heal_max_attempts=8, llm_self_heal_max_duration_sec=30.0,
    ))

    outcome = await RecognizeIntentStep().execute(state, ctx)

    assert outcome.next_step is None
    assert llm.attempts == 2, "应在瞬时故障后原地重试一次并成功"
    retried = [e for e in bus.events if e.type == EventType.LLM_RETRY_TRIGGERED]
    assert len(retried) == 1
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(finished) == 1


# ── Task 6 复审修复：异常路径不得留孤儿 LLM_REQUEST_STARTED（无配对 RESPONSE_FINISHED）──
#
# gateway 在进重试循环**之前**就无条件发了 LLM_REQUEST_STARTED/LLM_PROMPT_SENT——若本
# step 的 except 分支既不补发 LLM_RESPONSE_FINISHED、异常也不冒泡，host SSE 会看到一条
# 永远等不到收尾的挂死请求。下面两条覆盖「重试耗尽」与「不可重试的永久失败」，均须验证
# REQUEST_STARTED 与 RESPONSE_FINISHED 成对。
#
# outcome 语义的复审结论（详见 task-6-report.md）：`launch_recognize_intent._run()` 把
# 「内部已捕获、降级处理的失败」报成 RUN_FINISHED(outcome=completed) 是 commit 920bb05
# （总账 C5）已经明确裁定的既有设计，本任务未变更——`test_recognize_intent_swallowed_llm_failure_still_reports_run_completed`
# 把这条既有行为钉成回归测试，以便日后若要翻案能在这里看见。


@pytest.mark.asyncio
async def test_recognize_intent_pairs_events_on_permanent_llm_failure():
    """不可重试的永久失败（retriable=False）：LLM_REQUEST_STARTED 已经发出（gateway
    无条件发），except 分支必须补发 LLM_RESPONSE_FINISHED（finish_reason="error"），
    异常本身被就地吞掉、不冒泡（既有设计，见上）。"""
    from ctx_weft.protocols import LLMCallError
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    class _PermFailLLM:
        tokenizer = HeuristicTokenizer()
        context_limit = 100_000

        async def complete(self, req, stream=True):
            raise LLMCallError("bad request", retriable=False, outage=False)
            yield  # pragma: no cover — 使函数成为 async generator，见 stream_llm_resilient 用法

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    ctx = _make_recognize_intent_ctx(bus, llm=_PermFailLLM())

    outcome = await RecognizeIntentStep().execute(state, ctx)

    assert outcome.next_step is None, "异常被就地吞掉，不冒泡"
    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(finished) == 1, "REQUEST_STARTED 必须配对 RESPONSE_FINISHED"
    assert finished[0].payload["request_id"] == started[0].payload["request_id"]
    assert finished[0].payload["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_recognize_intent_pairs_events_on_retry_exhausted():
    """瞬时故障但重试预算耗尽（LLMOutageError）：同上，REQUEST_STARTED 只发一次
    （gateway 在整个重试循环外发一次，不随 attempt 重发），RESPONSE_FINISHED 必须补上。"""
    from ctx_weft.protocols import LLMCallError
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    class _AlwaysOutageLLM:
        tokenizer = HeuristicTokenizer()
        context_limit = 100_000

        def __init__(self):
            self.attempts = 0

        async def complete(self, req, stream=True):
            self.attempts += 1
            raise LLMCallError("outage", retriable=True, outage=True)
            yield  # pragma: no cover

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    llm = _AlwaysOutageLLM()
    ctx = _make_recognize_intent_ctx(bus, llm=llm, config=SimpleNamespace(
        llm_self_heal_base_delay_sec=0.01, llm_self_heal_max_interval_sec=0.01,
        llm_self_heal_max_attempts=2, llm_self_heal_max_duration_sec=30.0,
    ))

    outcome = await RecognizeIntentStep().execute(state, ctx)

    assert outcome.next_step is None
    assert llm.attempts == 2, "应耗尽 max_attempts=2 后放弃"
    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    retried = [e for e in bus.events if e.type == EventType.LLM_RETRY_TRIGGERED]
    assert len(started) == 1, "REQUEST_STARTED 只在重试循环外发一次"
    assert len(finished) == 1
    assert len(retried) == 1, "耗尽前应先原地重试一次"
    assert finished[0].payload["request_id"] == started[0].payload["request_id"]
    assert finished[0].payload["finish_reason"] == "error"


@pytest.mark.asyncio
async def test_recognize_intent_swallowed_llm_failure_still_reports_run_completed():
    """回归钉住既有设计（commit 920bb05 / 总账 C5，本任务未变更）：recognize_intent 内部
    已捕获、降级处理的 LLM 失败，`launch_recognize_intent._run()` 仍把 RUN_FINISHED 报成
    outcome=completed——因为异常从未逃出 RecognizeIntentStep.execute()。这不是「谎报」的
    新缺陷，是该 commit 明确选择的既有口径：只有真正逃出这段代码的未捕获异常才记
    interrupted。若日后要翻案，改这条测试的断言。"""
    from ctx_weft.protocols import LLMCallError
    from ctx_weft.core.loop.driver import LoopState
    from ctx_weft.core.loop.steps.recognize_intent import launch_recognize_intent
    from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind

    class _PermFailLLM:
        tokenizer = HeuristicTokenizer()
        context_limit = 100_000

        async def complete(self, req, stream=True):
            raise LLMCallError("bad request", retriable=False, outage=False)
            yield  # pragma: no cover

    bus = _RecordingBus()
    inner_state = _make_recognize_intent_state()
    ctx = _make_recognize_intent_ctx(bus, llm=_PermFailLLM())
    # launch_recognize_intent 自己会拷贝出一份快照 state，只需要一份带 task/agent/session/
    # scope/extra/resolved_model 的 LoopState 供其读取（origin 由 launch 内部钉死，见源码）。
    outer_state = LoopState(
        run_id="orig-run", session=inner_state.session, task=inner_state.task,
        agent=inner_state.agent, scope=inner_state.scope, extra=inner_state.extra,
        resolved_model=inner_state.resolved_model,
    )

    task = launch_recognize_intent(outer_state, ctx)
    await task

    finished_runs = [e for e in bus.events if e.type == EventType.RUN_FINISHED]
    assert len(finished_runs) == 1
    assert finished_runs[0].payload["outcome"] == RunOutcomeKind.COMPLETED.value
    assert finished_runs[0].payload["error"] is None


# ── Task 6 复审修复第二轮：反向孤儿（FINISHED 无 STARTED 匹配）──
#
# 上一轮修复把「STARTED 已发出」与「补发 FINISHED」两件事绑在了同一个 except 块里，
# 却没区分 except 触发时 STARTED 是否真的已经发出。stream_llm_resilient 是异步生成器，
# LLM_REQUEST_STARTED 是它函数体内的第一段代码——真正进入 async for 迭代之前，生成器体
# 一行都不会跑，STARTED 也就没发出。两条窄窗口会产出无 STARTED 匹配的孤儿 FINISHED：
#   #3 recognize_intent.py 自己的前置计算（request_prompt_estimate）在调用
#      stream_llm_resilient 之前抛异常——此时 gateway 函数体压根没被执行过。
#   #4 gateway 内部的 apply_dynamic_max_tokens 原来排在 STARTED 发射**之前**，抛异常时
#      同样是 STARTED 还没发出。
# 修复：#3 挪进独立的 try（失败不发 FINISHED，直接空手退出）；#4 把 gateway 内的
# apply_dynamic_max_tokens 挪到 STARTED/PROMPT_SENT 发射之后（连带删掉
# recognize_intent.py 里那次已成为多余空操作的重复调用）。下面两条钉住「STARTED 为 0
# 时 FINISHED 也必须为 0」，以及「STARTED 已发出时二者仍正确成对」。


@pytest.mark.asyncio
async def test_recognize_intent_prompt_estimate_failure_emits_no_started_or_finished():
    """path #3：request_prompt_estimate 在 stream_llm_resilient 被调用之前跑，此刻
    LLM_REQUEST_STARTED 不可能已经发出（gateway 函数体还没被执行过一行）。失败必须
    整体空手退出——不能只补一条 FINISHED 而没有配对的 STARTED（那才是真正的反向孤儿：
    无中生有一条收尾事件）。"""
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    class _BrokenTokenizer:
        def count(self, text):
            raise ValueError("boom-tokenizer")

    class _BrokenTokenizerLLM:
        context_limit = 100_000
        tokenizer = _BrokenTokenizer()

        async def complete(self, req, stream=True):
            raise AssertionError("must never reach the actual LLM call")
            yield  # pragma: no cover

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    ctx = _make_recognize_intent_ctx(bus, llm=_BrokenTokenizerLLM())

    outcome = await RecognizeIntentStep().execute(state, ctx)

    assert outcome.next_step is None
    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 0 and len(finished) == 0, "STARTED 为 0 时 FINISHED 也必须为 0"


@pytest.mark.asyncio
async def test_recognize_intent_gateway_max_tokens_failure_still_pairs_events():
    """path #4：gateway 的 apply_dynamic_max_tokens 现在排在 LLM_REQUEST_STARTED/
    LLM_PROMPT_SENT 发射之后（llm_gateway.py 复审修复）——即使它自己抛异常，STARTED
    也已经先发出去了，事件仍然正确成对（不再是反向孤儿）。用一个非法的
    dynamic_max_tokens_margin 配置值诱发 apply_dynamic_max_tokens 内部 int(...) 转换
    抛 ValueError。"""
    from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep

    bus = _RecordingBus()
    state = _make_recognize_intent_state()
    ctx = _make_recognize_intent_ctx(
        bus, config=SimpleNamespace(dynamic_max_tokens_margin="not-a-number"))

    outcome = await RecognizeIntentStep().execute(state, ctx)

    assert outcome.next_step is None
    started = [e for e in bus.events if e.type == EventType.LLM_REQUEST_STARTED]
    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(started) == 1 and len(finished) == 1, "STARTED 已发出，必须配对 FINISHED"
    assert finished[0].payload["request_id"] == started[0].payload["request_id"]
    assert finished[0].payload["finish_reason"] == "error"


# ── Task 5 复审修复 R1：origin 必须在 _run_background_observe 入口钉住，覆盖 ──
# RUN_STARTED/TASK_RECAP_STARTED 以及两条早退路径（re-fold 幂等护栏、短段免折）——
# 这几处此前发射时用的还是调用方快照进来的 origin（LOOP_OBSERVE / LOOP_ACT），只有
# 走到 run_observe_react 那句才被改写成 LOOP_BACKGROUND_OBSERVE，两条早退路径根本
# 走不到那句。


@pytest.mark.asyncio
async def test_background_observe_origin_pinned_on_refold_guard_early_exit(
    fake_state_ctx, monkeypatch,
):
    """re-fold 幂等护栏早退（非 close 边界、视图内无 active raw）：函数体内已经发出的
    RUN_STARTED/TASK_RECAP_STARTED，以及 finally 里的 TASK_RECAP_DONE/RUN_FINISHED，
    origin 必须全部是 loop.background_observe——即使调用方快照带进来的是别的 origin。"""
    from ctx_weft.core.loop.steps import background_observe as bo

    state, ctx = fake_state_ctx
    state.origin = EventOrigin.LOOP_OBSERVE  # 模拟调用方快照带进来的「错误」origin

    async def _empty_view(address, scope, pctx, kinds=None):
        return []

    monkeypatch.setattr(ctx.memory, "load_view", _empty_view)

    await bo._run_background_observe(state, ctx, boundary="interrupt")

    assert ctx.event_bus.emitted
    origins = {e.origin for e in ctx.event_bus.emitted}
    assert origins == {EventOrigin.LOOP_BACKGROUND_OBSERVE}, origins
    types_seen = {e.type for e in ctx.event_bus.emitted}
    assert {EventType.RUN_STARTED, EventType.TASK_RECAP_STARTED,
            EventType.TASK_RECAP_DONE, EventType.RUN_FINISHED} <= types_seen


@pytest.mark.asyncio
async def test_background_observe_origin_pinned_on_short_segment_early_exit(fake_state_ctx):
    """短段免折早退（is_short_segment 命中，同 test_task_recap_refold_guard.py 里
    test_short_segment_kept_raw_no_llm_call 的 fixture 配置：默认 threshold=400、
    种子 raw 远低于此）：同上，origin 必须全部是 loop.background_observe。"""
    from ctx_weft.core.loop.steps import background_observe as bo

    state, ctx = fake_state_ctx
    state.origin = EventOrigin.LOOP_ACT  # 模拟 act.py interrupt 边界快照带进来的「错误」origin

    await bo._run_background_observe(state, ctx, boundary="plain_text")

    assert ctx.event_bus.emitted
    origins = {e.origin for e in ctx.event_bus.emitted}
    assert origins == {EventOrigin.LOOP_BACKGROUND_OBSERVE}, origins
    types_seen = {e.type for e in ctx.event_bus.emitted}
    assert {EventType.RUN_STARTED, EventType.TASK_RECAP_STARTED,
            EventType.TASK_RECAP_DONE, EventType.RUN_FINISHED} <= types_seen


# ── Task 5 复审修复 R3：≥2 轮回归，钉住 request_id/turn 跨轮的行为 ──


@pytest.mark.asyncio
async def test_observe_multi_round_request_id_and_turn_increment():
    """审查者手工验证过多轮机制（round 内 5 个事件共用一个 request_id、跨轮各不相同，
    turn 依次递增）成立，但此前没有回归测试钉住——补上。默认 `_FakeLLM` 每轮都只吐
    纯文本 + usage、从不产 tool_call，配 max_rounds=3 保证跑满 3 轮不提前终止。"""
    from ctx_weft.core.loop.steps.observe import run_observe_react

    bus = _RecordingBus()
    state = _make_state()
    state.origin = EventOrigin.LOOP_OBSERVE
    ctx = _make_ctx(bus)

    await run_observe_react(
        state, ctx, system="SYS", messages=[], tools=[],
        max_rounds=3, terminal_tool_name="report_task_outcome",
    )

    finished = [e for e in bus.events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(finished) == 3, "3 轮都无 tool_call，应跑满 max_rounds"

    per_round_ids = []
    for turn in range(3):
        turn_finished = [e for e in finished if e.payload["turn"] == turn]
        assert len(turn_finished) == 1, (turn, [e.payload for e in finished])
        rid = turn_finished[0].payload["request_id"]
        # 该 turn 对应的 4 个 gateway 流式事件 + 1 个收尾事件应共用同一个 request_id。
        round_events = [e for e in bus.events if e.payload.get("request_id") == rid]
        debug = [(e.type, e.payload.get("request_id")) for e in bus.events]
        assert len(round_events) == 5, (turn, rid, debug)
        started = [e for e in round_events if e.type == EventType.LLM_REQUEST_STARTED]
        assert len(started) == 1
        per_round_ids.append(rid)

    assert len(set(per_round_ids)) == 3, f"跨轮 request_id 应各不相同：{per_round_ids}"
