"""无人值守（`Task.unattended`）：HITL 在唯一登记入口一处被堵死。

后台自治作业没有人可问——`ask_user`、工具的人工授权、纯文本让位，任何一条走到
`HitlService.open()` 都会 park 到死（没人会来应答）。守卫因此落在 `open()` 上，
参数是**必填 keyword-only**：漏传就是 TypeError 当场炸，不会静默失效。

本文件覆盖守卫本身、必填性、设置点的不变式（`unattended ⟹ interaction_mode=="auto"`）、
委派继承，以及两个 catch 点（授权侧拒绝 / `ask_user` 侧自决）。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import (
    ASK_USER_NAME,
    ASK_USER_UNATTENDED_RESULT,
    ControlContext,
    delegate_plan,
    delegate_task,
)
from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL, HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService, UnattendedHitl
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.protocols import (
    AgentCapabilityProvider,
    AgentTemplate,
    CapabilityProviderInfo,
    IdentityFacet,
    LoopConfig,
    MemoryAddress,
    MemoryConfig,
    ProviderContext,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.hitl import HitlAsk, ToolResultDelivery
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

T0 = datetime(2026, 9, 5, tzinfo=UTC)
_AGENT_ID = "agt_1"


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]


def _service(bus, registry: HitlRegistry | None = None) -> HitlService:
    ids = iter(f"hit_{i}" for i in range(1, 100))
    return HitlService(
        registry=registry or HitlRegistry(),
        event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
        id_factory=lambda: next(ids),
        clock=lambda: T0,
    )


def _ask(tool_call_id: str = "call_1") -> HitlAsk:
    return HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id=tool_call_id),
                   prompt="Allow bash?", subject_id="fs:bash_exec")


# ── 1. 守卫本身 ───────────────────────────────────────────────────────────────


async def test_open_raises_when_task_is_unattended() -> None:
    bus = _RecordingBus()
    svc = _service(bus)
    with pytest.raises(UnattendedHitl) as excinfo:
        await svc.open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                       tool_call_id="call_1", stage=HITL_STAGE_AUTHZ, unattended=True)
    # 诊断信息：哪一种 HITL、问的是谁。
    assert "approval" in str(excinfo.value)
    assert "fs:bash_exec" in str(excinfo.value)
    assert excinfo.value.form == "approval"
    assert excinfo.value.subject_id == "fs:bash_exec"
    # 没有半个记录、没有半条事实。
    assert bus.events == []
    assert svc.registry.list_pending() == []


async def test_open_registers_normally_when_attended() -> None:
    bus = _RecordingBus()
    svc = _service(bus)
    req = await svc.open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                         tool_call_id="call_1", stage=HITL_STAGE_AUTHZ, unattended=False)
    assert req.id == "hit_1"
    assert len(bus.events) == 1


# ── 2. 守卫在幂等复用之前 ─────────────────────────────────────────────────────


async def test_guard_precedes_idempotent_reuse() -> None:
    """无人值守的 task 本就不该存在任何「等人回答」的记录——哪怕同键的旧记录还在，
    也不能把它当答案返回。守卫必须排在 `find_for_tool_call` 之前。"""
    bus = _RecordingBus()
    registry = HitlRegistry()
    svc = _service(bus, registry)
    first = await svc.open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                           tool_call_id="call_1", stage=HITL_STAGE_AUTHZ, unattended=False)
    assert registry.find_for_tool_call("s1", "call_1", HITL_STAGE_AUTHZ) is first

    with pytest.raises(UnattendedHitl):
        await svc.open(_ask(), session_id="s1", task_id="t1", agent_id="a1",
                       tool_call_id="call_1", stage=HITL_STAGE_AUTHZ, unattended=True)


# ── 3. 必填性本身 ─────────────────────────────────────────────────────────────


def test_unattended_is_a_required_keyword_argument() -> None:
    """漏传即 TypeError（无默认值）——这是设计的核心，不是风格：有默认值的守卫
    会在下一个调用点被静默绕过。"""
    svc = _service(_RecordingBus())
    with pytest.raises(TypeError):
        svc.open(_ask(), session_id="s1", task_id="t1", stage=HITL_STAGE_AUTHZ)


# ── 6/7. 设置点强制的不变式 ───────────────────────────────────────────────────


class _Resolver(AgentCapabilityProvider):
    name = "agent"

    def __init__(self, t: AgentTemplate) -> None:
        self._t = t

    async def list(self, ctx):
        return []

    async def get_template(self, template_id, version, ctx):
        return self._t

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name)


class _Client:
    def __init__(self) -> None:
        self.account, self.model = "acct", "mdl"
        self.context_limit, self.output_reserve = 200_000, 8192


def _session_registry(bus) -> SessionRegistry:
    template = AgentTemplate(
        id="tpl", name="t", version="1",
        identity={"act": IdentityFacet(text="soul")},
        capability_refs=[], memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )
    reg = ProviderRegistry()
    reg.register_capability(_Resolver(template))
    return SessionRegistry(
        agent_lifecycle_manager=AgentLifecycleManager(
            template_lookup=TemplateLookup(reg), event_bus=bus,
            model_resolver=lambda a, m: _Client(),
        ),
        event_bus=bus,
    )


async def test_unattended_root_task_is_auto() -> None:
    """没有人会发下一条消息，interactive 的纯文本 park 即永久挂起。"""
    sm = _session_registry(InProcessEventBus())
    _s, root, _tm = await sm.create_session(
        template_id="agent:tpl", user_prompt="go", context_limit=1000, unattended=True)
    assert root.unattended is True
    assert root.interaction_mode == "auto"


async def test_attended_root_task_stays_interactive() -> None:
    sm = _session_registry(InProcessEventBus())
    _s, root, _tm = await sm.create_session(
        template_id="agent:tpl", user_prompt="go", context_limit=1000)
    assert root.unattended is False
    assert root.interaction_mode == "interactive"


# ── 8. 委派继承 ───────────────────────────────────────────────────────────────


class _FakeTM:
    def __init__(self) -> None:
        self.staged: list[Task] = []

    def stage_task(self, child: Task, **kwargs) -> None:
        self.staged.append(child)

    def get_task(self, tid: str):
        return None


def _ctl_ctx(tm: _FakeTM, *, unattended: bool) -> ControlContext:
    parent = Task(
        id="p1", session_id="s1", status="ACTIVE", title="P",
        unattended=unattended,
        # 设置点保证的不变式，父任务上原样成立。
        interaction_mode="auto" if unattended else "interactive",
    )
    return ControlContext(session_id="s1", task_id="p1", agent_id="a1", task=parent,
                          task_manager=tm, session=None, tool_call_id="tc")


def test_delegate_task_inherits_unattended() -> None:
    tm = _FakeTM()
    delegate_task(title="c", task_prompt="p", interactive=True,
                  ctx=_ctl_ctx(tm, unattended=True))
    child = tm.staged[0]
    assert child.unattended is True
    # 子任务的 interaction_mode 不用单独处理：`_child_mode` 的「父不 interactive 则
    # 子不 interactive」规则已经接住（父 auto → 子 auto）。
    assert child.interaction_mode == "auto"


def test_delegate_task_does_not_invent_unattended() -> None:
    tm = _FakeTM()
    delegate_task(title="c", task_prompt="p", interactive=True,
                  ctx=_ctl_ctx(tm, unattended=False))
    assert tm.staged[0].unattended is False


def test_delegate_plan_inherits_unattended() -> None:
    tm = _FakeTM()
    delegate_plan(tasks=[{"title": "a", "interactive": True}, {"title": "b"}],
                  ctx=_ctl_ctx(tm, unattended=True))
    assert [c.unattended for c in tm.staged] == [True, True]
    assert [c.interaction_mode for c in tm.staged] == ["auto", "auto"]


def test_delegate_is_not_a_knob_for_the_llm() -> None:
    """不给 LLM 这个旋钮：schema 里不得出现 `unattended`。"""
    import inspect
    for fn in (delegate_task, delegate_plan):
        assert "unattended" not in inspect.signature(fn).parameters


# ── 9/10. 两个 catch 点 ───────────────────────────────────────────────────────


class _GatedTool(ToolCapabilityProvider):
    """一个会被 `HumanConfirmationAuthorizer` 门控的工具；执行次数是「没被放行」的证据。"""

    name = "fs"

    def __init__(self) -> None:
        self.invocations = 0

    async def list(self, ctx):
        return [ToolCapability(id="fs:bash_exec", name="bash_exec", description="run")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run()

    async def _run(self):
        self.invocations += 1
        yield CapabilityEvent(kind="result", payload={"content": "ran"})

    async def cancel(self, iid, ctx) -> None:
        return None


class _AskUserLike(ToolCapabilityProvider):
    """`ask_user` 的形状：声明 needs_human + reply_as_result，自己不 park、不等人。"""

    name = "control"

    async def list(self, ctx):
        return [ToolCapability(id="control:ask_user", name="ask_user", description="ask")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run(ctx)

    async def _run(self, ctx):
        yield CapabilityEvent(kind="needs_human", payload={"ask": HitlAsk(
            form="question",
            delivery=ToolResultDelivery(tool_call_id=ctx.extra["tool_call_id"]),
            prompt="Which region?", subject_id="control:ask_user", reply_as_result=True)})

    async def cancel(self, iid, ctx) -> None:
        return None


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(getattr(p, "text", "") for p in content)


def _gateway_env(provider, cap: ToolCapability, *, authorizers=None):
    bus = InProcessEventBus()
    registry = HitlRegistry()
    service = HitlService(registry=registry, event_bus=bus,
                          reply_intake=ReplyIntake(_PassthroughNormalizer()))
    cache = CapabilityCache()
    cache.put(_AGENT_ID, [cap])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[provider],
        memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_authorizers=authorizers or {},
    )
    return gw, service, HitlWaiter(registry, timeout_sec=None), registry


def _state(*, unattended: bool) -> LoopState:
    from types import SimpleNamespace
    task = Task(id="tsk_1", session_id="s1", status="ACTIVE", unattended=unattended,
                interaction_mode="auto" if unattended else "interactive")
    return LoopState(
        run_id="r1",
        session=SimpleNamespace(id="s1", tenant_id="default"),
        task=task,
        agent=SimpleNamespace(id=_AGENT_ID, template_id="tpl", session_id="s1"),
        scope=MemoryAddress(session_id="s1", task_id="tsk_1", agent_id=_AGENT_ID),
        resolved_model=SimpleNamespace(model="mock", account=""),
    )


def _loop_ctx(service, waiter) -> LoopContext:
    return LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(),
        event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                     task_id="tsk_1", agent_id=_AGENT_ID),
        hitl=service, waiter=waiter,
    )


async def test_human_authorizer_denies_instead_of_raising_when_unattended() -> None:
    tool = _GatedTool()
    gw, svc, waiter, reg = _gateway_env(
        tool, ToolCapability(id="fs:bash_exec", name="bash_exec", description="run"),
        authorizers={"fs:bash_exec": HumanConfirmationAuthorizer()},
    )
    result = await asyncio.wait_for(
        gw.invoke("fs__bash_exec", {"command": "ls"}, _state(unattended=True),
                  _loop_ctx(svc, waiter), tool_call_id="call_1"),
        timeout=5.0,
    )
    text = _text_of(result.content)
    assert "[Blocked by human:" in text, text
    assert "unattended" in text.lower(), text
    assert tool.invocations == 0        # 安全不变式：绝不放行
    assert reg.list_pending() == []     # 也没有留下任何等人的记录


async def test_ask_user_yields_a_self_decide_result_when_unattended() -> None:
    gw, svc, waiter, reg = _gateway_env(
        _AskUserLike(),
        ToolCapability(id="control:ask_user", name="ask_user", description="ask"),
    )
    result = await asyncio.wait_for(
        gw.invoke(ASK_USER_NAME, {"question": "which region?"}, _state(unattended=True),
                  _loop_ctx(svc, waiter), tool_call_id="call_1"),
        timeout=5.0,
    )
    assert _text_of(result.content) == ASK_USER_UNATTENDED_RESULT
    assert "control__finish_task" in ASK_USER_UNATTENDED_RESULT
    assert reg.list_pending() == []


async def test_attended_ask_user_still_parks_for_a_human() -> None:
    """不改既有语义：有人在的时候，还是照常登记、照常等人。"""
    gw, svc, waiter, reg = _gateway_env(
        _AskUserLike(),
        ToolCapability(id="control:ask_user", name="ask_user", description="ask"),
    )
    task = asyncio.create_task(gw.invoke(
        ASK_USER_NAME, {"question": "which region?"}, _state(unattended=False),
        _loop_ctx(svc, waiter), tool_call_id="call_1"))
    try:
        pending = None
        for _ in range(200):
            await asyncio.sleep(0.005)
            if reg.list_pending():
                pending = reg.list_pending()[0]
                break
        assert pending is not None, "attended ask_user must register a pending HITL"
        assert pending.stage == HITL_STAGE_TOOL
    finally:
        task.cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
