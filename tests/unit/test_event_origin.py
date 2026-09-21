from __future__ import annotations

from datetime import UTC, datetime

from ctx_weft.protocols.events import Event, EventOrigin


def _ev(**kw) -> Event:
    base = dict(
        id="evt_1", run_id=None, sequence=0, session_id="s1",
        type="TaskStarted", timestamp=datetime.now(UTC),
    )
    base.update(kw)
    return Event(**base)


def test_origin_defaults_to_empty_string():
    """存量事件读出空串——§0 明说不做反推。"""
    assert _ev().origin == ""


def test_origin_round_trips():
    assert _ev(origin=EventOrigin.LOOP_ACT).origin == "loop.act"


def test_event_origin_has_17_values():
    assert len(EventOrigin.all()) == 17


def test_origin_values_are_two_level_dotted_or_bare():
    """§4：两级点号供 host 前缀匹配；分隔符用 . 不用 :（: 留给 capability id）。"""
    for v in EventOrigin.all():
        assert ":" not in v
        assert v.count(".") <= 1
        assert v == v.strip()


def test_loop_prefix_matches_all_loop_origins():
    loop = {v for v in EventOrigin.all() if v.startswith("loop.")}
    assert EventOrigin.LOOP_ACT in loop
    assert EventOrigin.LOOP_BACKGROUND_OBSERVE in loop
    assert EventOrigin.RUNTIME not in loop


# ── Task 2: LoopState.origin 管道与 make_event 默认取值 ──


from ctx_weft.core.loop.driver import make_event
from ctx_weft.protocols.events import EventType


class _FakeState:
    """make_event 只读这几个字段。"""
    def __init__(self, origin: str = ""):
        self.session_id = "s1"
        self.task_id = "t1"
        self.agent_id = "a1"
        self.run_id = "r1"
        self.tenant_id = "default"
        self.sequence_counter = 0
        self.origin = origin
        # 模拟 LoopState 中的 Session/Task/Agent 对象
        self.session = type('Session', (), {'id': 's1', 'tenant_id': 'default'})()
        self.task = type('Task', (), {'id': 't1'})()
        self.agent = type('Agent', (), {'id': 'a1'})()


def test_make_event_takes_origin_from_state():
    """§4 填充方式 1：40+ 个循环内发射点零改动。"""
    ev = make_event(_FakeState(origin=EventOrigin.LOOP_ACT), EventType.ACT_TURN_STARTED, {})
    assert ev.origin == "loop.act"


def test_make_event_explicit_origin_overrides_state():
    """§4 填充方式 2：给 background observe 这类脱离主 driver 序列的场景。"""
    ev = make_event(
        _FakeState(origin=EventOrigin.LOOP_OBSERVE),
        EventType.LLM_PROMPT_SENT,
        {},
        origin=EventOrigin.LOOP_BACKGROUND_OBSERVE,
    )
    assert ev.origin == "loop.background_observe"


# ── Task 3: 循环外发射者补 origin + 端到端不变式 ──

import asyncio
import time

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)
from tests._event_helpers import all_events, append_one


class _RouterLLM(MockLLMAdapter):
    """按 request.tools 路由（同 test_run_id_sequence_integrity.py::_RouterLLM）：

    recognize_intent（工具集含 `control__update_task_metadata`）恒回空文本；
    act 首轮回纯文本、无 tool_call——root task 恒 interactive，纯文本首轮触发
    冷 park，同时并发一次 background observe + recognize_intent。三条并发路径
    叠加主 run，一次跑遍循环外 5 个发射者 + 循环内多个孤儿 run 构造点，是本测试
    想要的覆盖面（不能用单条 MockResponse 队列——并发调用抢同一个响应槽位，
    顺序不确定会导致 RuntimeError: MockLLMAdapter exhausted）。
    """

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)

    def complete(self, request, stream: bool = True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:  # recognize_intent
            return self._stream(MockResponse(text=""), request)
        return self._stream(MockResponse(text="Hi! Anything else?"), request)


async def _poll(predicate, *, timeout: float = 5.0, interval: float = 0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(interval)
    raise AssertionError("timed out waiting for background run(s) to finish")


@pytest.mark.asyncio
async def test_every_emitted_event_has_nonempty_origin():
    """docs/events-v2.md §6 外加条：所有发射出的事件 origin 非空。

    严格版：真正 await 到主 run + background observe + recognize_intent 三条
    并发路径都跑完（不再收窄到 `start_session` 同步返回那一刻），一次性覆盖
    Task 3 触碰过的循环外 5 个发射者、循环内孤儿 run 的 4 个构造点
    （recognize_intent / background_observe / compact / driver 兜底）、以及
    正常经 StepDriver 每步覆盖 origin 的常规循环内事件。

    不设豁免：`SessionWaiting`（`session_registry._emit_session_event`）曾在此临时
    豁免过（当时它还在发射、且计划随会话状态机一并拆掉），现在 `session_state.py`
    已随 Task 15/16 整体退役、`SessionWaiting` 停止发射并转入 L 档只读存量
    （`events.py::L_TIER_EVENT_TYPES`），这条豁免已死，删掉。若还暴露出别的
    origin 为空的事件类型，本测试直接失败（不收窄断言范围）。
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _RouterLLM()
    rt: CtxWeftRuntime = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())

    seen: list[tuple[str, str]] = []

    async def _spy(ev):
        seen.append((ev.type, ev.origin))

    rt._event_bus.subscribe(None, _spy)

    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi", context_limit=100_000,
    ))
    await handle.wait_for_finish(timeout=5.0)
    # background observe / recognize_intent 是 fire-and-forget：主 run 的
    # RunFinished 不代表它们也发完了，等三个 run（主 + recap + recognize_intent）
    # 各自的 RunFinished 落地。
    await _poll(lambda: sum(1 for t, _ in seen if t == EventType.RUN_FINISHED) >= 3)

    assert seen, "没有采集到任何事件"
    blank = sorted({t for t, o in seen if not o})
    assert blank == [], f"这些事件类型的 origin 为空：{blank}"


# ── Task 3 / 补充：三组只被端到端测试路径顺带覆盖、无针对性回归保护的发射点 ──
#
# 上面的 e2e 测试触达的是「act 首轮回纯文本→冷 park」这条路径；HITL_SERVICE
# （只在真正的 approval/input 型 HITL 走到）、runtime.py 的 RUN_* 事件里
# compact-only 支路、capability_gateway 的三条事件（只在 act 真调用一个工具时
# 才发）都不在那条路径上，缺一个误删 `origin=` 也不会变红的回归保护。三个都是
# 直接调用/构造对应代码路径的针对性单测，不驱动完整 loop。


async def test_hitl_service_emits_with_hitl_service_origin():
    """HitlService._emit 发的 HitlOpened/HitlResolved 恒是 EventOrigin.HITL_SERVICE。"""
    from ctx_weft.core.hitl.registry import HitlRegistry
    from ctx_weft.core.hitl.reply_intake import ReplyIntake
    from ctx_weft.core.hitl.service import HitlService
    from ctx_weft.protocols.hitl import HitlAsk, HitlReply, ToolResultDelivery

    class _RecordingBus:
        def __init__(self) -> None:
            self.events: list[Event] = []

        async def emit(self, event: Event) -> None:
            self.events.append(event)

    class _PassthroughNormalizer:
        async def __call__(self, content, session_id, tenant_id):
            return content, content

    bus = _RecordingBus()
    svc = HitlService(
        registry=HitlRegistry(), event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
    )
    req = await svc.open(
        HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id="tc_1")),
        session_id="s1", task_id="t1", stage="tool", unattended=False,
    )
    await svc.resolve(HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id))

    assert len(bus.events) == 2, "HitlOpened + HitlResolved"
    origins = {ev.type: ev.origin for ev in bus.events}
    assert origins == {
        EventType.HITL_OPENED: EventOrigin.HITL_SERVICE,
        EventType.HITL_RESOLVED: EventOrigin.HITL_SERVICE,
    }


async def test_capability_gateway_events_have_loop_capability_gateway_origin():
    """CapabilityGateway.invoke 发的 Invoked/Progress/Finished 恒是 LOOP_CAPABILITY_GATEWAY。"""
    from collections.abc import AsyncIterator
    from types import SimpleNamespace

    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.loop.driver import LoopContext, LoopState
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.protocols import MemoryAddress, ProviderContext
    from ctx_weft.protocols.capability import (
        CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider,
    )
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    class _Echo(ToolCapabilityProvider):
        name = "mcp:a"

        def _cap(self) -> ToolCapability:
            return ToolCapability(id="mcp:a:search", name="search", description="s")

        async def list(self, ctx): return [self._cap()]
        async def retrieve(self, ctx): return [self._cap()]
        async def describe(self, ctx):
            return CapabilityProviderInfo(name=self.name, capability_count=1)

        def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
            async def _run():
                yield CapabilityEvent(kind="progress", payload={"data": "working"})
                yield CapabilityEvent(kind="result", payload={"content": "ok"})
            return _run()

        async def cancel(self, invocation_id, ctx) -> None: return None

    p = _Echo()
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"), agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope, resolved_model=SimpleNamespace(model="mock", account=""),
    )
    bus = InProcessEventBus()
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )
    captured: list[Event] = []

    async def _collect(ev):
        captured.append(ev)

    bus.subscribe(None, _collect)

    cache = CapabilityCache()
    cache.put("agt_1", [p._cap()])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[p], memory=mem, event_bus=bus,
    )
    await gw.invoke("mcp__a__search", {"q": "x"}, state, ctx, tool_call_id="tc_abc")

    gw_types = {EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_PROGRESS,
                EventType.CAPABILITY_FINISHED}
    seen = [ev for ev in captured if ev.type in gw_types]
    assert {ev.type for ev in seen} == gw_types, "三种事件都应发出"
    assert all(ev.origin == EventOrigin.LOOP_CAPABILITY_GATEWAY for ev in seen), (
        [(ev.type, ev.origin) for ev in seen]
    )


async def test_compact_agent_run_events_have_runtime_origin():
    """`CtxWeftRuntime.compact_agent` 的 RunStarted/RunFinished 恒是 EventOrigin.RUNTIME。

    覆盖 runtime.py 里独立于主 `_run_loop` 之外的第二条 RUN_* 发射路径
    （compact-only，走 `origin=EventOrigin.RUNTIME` 显式覆盖，不继承 state.origin）
    ——上面的 e2e 测试走的是「act 首轮冷 park」，不会触达这条 compact-only 支路。
    """
    from datetime import timedelta

    from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY")])
    rt: CtxWeftRuntime = make_runtime(llm=llm, agent_provider=resolver)
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)

    sid, aid = "ses_origin", "agt_root"
    ts = datetime(2026, 9, 3, tzinfo=UTC)
    await append_one(rt.event_store, Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id=sid,
        type=EventType.SESSION_CREATED, timestamp=ts,
        payload={"template_id": "agent:tpl_echo", "user_prompt": "x", "root_agent_id": aid,
                 "llm_model": "mock", "context_limit": 180000},
    ))
    scope = MemoryAddress(session_id=sid, task_id="t_seed", agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id="t_seed", agent_id=aid)
    for i in range(3):
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"turn {i}", role="user", timestamp=ts + timedelta(seconds=i),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)

    rt._agent_lifecycle_manager.register_session(sid, tenant_id="default", fallback_template_id="agent:tpl_echo")
    rt._agent_lifecycle_manager.materialize(aid, session_id=sid, tenant_id="default")

    seen: list[Event] = []

    async def _spy(ev):
        seen.append(ev)

    rt._event_bus.subscribe(None, _spy)

    await rt.compact_agent(aid)

    run_events = [ev for ev in seen if ev.type in (EventType.RUN_STARTED, EventType.RUN_FINISHED)]
    assert len(run_events) == 2, "RunStarted + RunFinished"
    assert all(ev.origin == EventOrigin.RUNTIME for ev in run_events), (
        [(ev.type, ev.origin) for ev in run_events]
    )


# ── Task 3 / R7: SQL EventStore 往返持久化 origin ──


async def test_sql_event_store_round_trips_origin(tmp_path):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        ev = _ev(id="evt_a", type="TaskStarted", origin=EventOrigin.LOOP_ACT)
        await append_one(store, ev)
        loaded = await all_events(store, "s1")
        assert len(loaded) == 1
        assert loaded[0].origin == "loop.act"


async def test_sql_event_store_round_trips_blank_origin(tmp_path):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        ev = _ev(id="evt_b", type="TaskStarted")  # origin 默认 ""
        await append_one(store, ev)
        loaded = await all_events(store, "s1")
        assert len(loaded) == 1
        assert loaded[0].origin == ""
