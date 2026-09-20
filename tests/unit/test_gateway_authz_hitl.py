"""gateway 的授权侧 HITL 路径（段 2 · Task 4）。

覆盖：热放行、热拒绝、驱逐→park、决定缓存短路（跨重启再入不重问）、
不问人的 authorizer 完全不碰 HITL、契约违例当场报错。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.hitl.snapshot import HitlSnapshot
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.loop.park import HitlPark
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    Authorizer,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.hitl import (
    HITL_FORM_APPROVAL, HitlAsk, HitlDecision, HitlReply, ToolResultDelivery,
)
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

# 用 qualified 名做工具调用入口——与 CapabilityCache 的编码（cap.id 里的 ":" → "__"）一致。
_CAP_ID = "fs:bash_exec"
_TOOL_NAME = "fs__bash_exec"
_AGENT_ID = "agt_1"


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


class _RecordingProvider(ToolCapabilityProvider):
    """记录每次 invoke 的参数；用于验证 Deny 从不到达 provider、改参真正生效。"""

    name = "fs"

    def __init__(self) -> None:
        self.invocations = 0
        self.last_args: dict | None = None

    async def list(self, ctx):
        return [_cap()]

    async def retrieve(self, ctx):
        return [_cap()]

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run(args)

    async def _run(self, args):
        self.invocations += 1
        self.last_args = args
        yield CapabilityEvent(kind="result", payload={"content": "ok"})

    async def cancel(self, iid, ctx):
        return None


class _AllowAll(Authorizer):
    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        return AuthorizationDecision(allowed=True)


class _AsksButNotGated(Authorizer):
    """声明 needs_human 却不实现 HumanGatedAuthorizer（无 on_decision）——契约违例。"""

    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        return AuthorizationDecision(allowed=False, needs_human=HitlAsk(
            form=HITL_FORM_APPROVAL,
            delivery=ToolResultDelivery(tool_call_id=tool_call_id),
            prompt="Allow?",
        ))


def _cap() -> ToolCapability:
    return ToolCapability(id=_CAP_ID, name="bash_exec", description="run shell")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _make_gateway(authorizer, timeout_sec=None):
    """搭一个 gateway + 一套真实的 HitlRegistry/HitlService/HitlWaiter，与
    ``tests/unit/test_hitl_park.py`` 的直接构造 LoopState/LoopContext 惯例一致。

    返回 (gateway, registry, service, provider)，并把 service/waiter 缓存到闭包里，
    供 ``_ctx()`` 复用同一套 HITL 装配（brief 里 ``_ctx()`` 不接参数，只能靠此闭包共享）。
    """
    registry = HitlRegistry()
    bus = InProcessEventBus()
    service = HitlService(
        registry=registry, event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
    )
    waiter = HitlWaiter(registry, timeout_sec=timeout_sec)
    provider = _RecordingProvider()
    cache = CapabilityCache()
    cache.put(_AGENT_ID, [_cap()])
    gw = CapabilityGateway(
        capability_cache=cache,
        capability_providers=[provider],
        memory=InMemoryMemoryProvider(),
        event_bus=bus,
        provider_authorizers={_CAP_ID: authorizer},
    )
    _shared["hitl"] = service
    _shared["waiter"] = waiter
    _shared["provider"] = provider
    return gw, registry, service


# 每个测试通过 _make_gateway 重新填充；_ctx() 读取以搭出带同一套 HITL 装配的 LoopContext。
_shared: dict = {}


def _state() -> LoopState:
    session = SimpleNamespace(id="s1", tenant_id="default")
    task = SimpleNamespace(id="tsk_1", unattended=False)
    agent = SimpleNamespace(id=_AGENT_ID, template_id="tmpl_a", session_id="s1")
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id=_AGENT_ID)
    return LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope, resolved_model=SimpleNamespace(model="mock", account=""))


def _ctx() -> LoopContext:
    return LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(),
        event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id=_AGENT_ID,
        ),
        hitl=_shared.get("hitl"),
        waiter=_shared.get("waiter"),
    )


def _provider_invocations() -> int:
    return _shared["provider"].invocations


def _last_invoked_args() -> dict | None:
    return _shared["provider"].last_args


def _snapshot_with_decision(tool_call_id: str, decision: HitlDecision) -> HitlSnapshot:
    # session_id="s1"：与 _state()/_ctx() 里搭的 session 一致——decision_for 现在按
    # (session_id, tool_call_id, stage) 三维查（Task 4.5），键错一维就查不到。
    return HitlSnapshot(decisions_for={("s1", tool_call_id, HITL_STAGE_AUTHZ): (decision, None)})


# ── 测试 ────────────────────────────────────────────────────────────────────


async def test_hot_approval_lets_the_call_through_with_modified_arguments():
    gw, reg, svc = _make_gateway(HumanConfirmationAuthorizer())
    task = asyncio.create_task(gw.invoke(_TOOL_NAME, {"command": "ls"},
                                         _state(), _ctx(), tool_call_id="call_1"))
    await asyncio.sleep(0)
    pending = reg.list_pending()[0]
    await svc.resolve(HitlReply(hitl_id=pending.id, outcome="accepted", agent_id=_AGENT_ID,
                                modified_arguments={"command": "ls -l"}))
    result = await task
    assert result.is_error is False
    assert _last_invoked_args() == {"command": "ls -l"}      # 改参真正生效


async def test_hot_rejection_never_calls_the_provider():
    """安全不变式：Deny 时 provider.invoke 绝不被调用。"""
    gw, reg, svc = _make_gateway(HumanConfirmationAuthorizer())
    task = asyncio.create_task(gw.invoke(_TOOL_NAME, {"command": "rm -rf /"},
                                         _state(), _ctx(), tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="rejected",
                                agent_id=_AGENT_ID, message="别删"))
    result = await task
    assert result.is_error is True
    assert "别删" in _text_of(result.content)
    assert _provider_invocations() == 0


async def test_eviction_raises_hitl_park_and_does_not_invoke():
    gw, reg, _svc = _make_gateway(HumanConfirmationAuthorizer(), timeout_sec=0)
    with pytest.raises(HitlPark):
        await gw.invoke(_TOOL_NAME, {"command": "ls"}, _state(), _ctx(),
                        tool_call_id="call_1")
    assert _provider_invocations() == 0
    assert reg.list_pending()[0].resolved is False      # 请求仍未决


async def test_a_cached_decision_short_circuits_without_asking_again():
    """冷路径重入：registry 已有该 tool_call 的决定 → 直接 on_decision，不新建请求。"""
    gw, reg, _svc = _make_gateway(HumanConfirmationAuthorizer())
    reg.load_snapshot(_snapshot_with_decision("call_1",
                                              HitlDecision(outcome="accepted")))
    result = await gw.invoke(_TOOL_NAME, {"command": "ls"}, _state(), _ctx(),
                             tool_call_id="call_1")
    assert result.is_error is False
    assert reg.list_pending() == []                    # 没有新建 pending
    assert _provider_invocations() == 1


async def test_a_cached_decision_carries_modified_arguments_into_the_provider():
    """人在批准前**改写的参数**必须跨重启仍然生效——不只是「短路发生了」。

    这是安全相关的一条：人把 `rm -rf /` 改成 `ls -la` 再放行，冷路径若只取 outcome、
    丢掉 `modified_arguments`，provider 收到的就是原参——一次人工把关变成空操作，
    而且只在崩溃之后才出错（最难被发现的时机）。

    旧 `test_hitl_park.py::test_authorize_cold_uses_resolved_decision_no_new_hitl` 断言过
    这一条，随 `HitlManager` 一并删除；在这里补回，且比旧版更进一步——旧版只看
    `authorize()` 的返回值，这里看 **provider 真正收到了什么**。
    """
    gw, reg, _svc = _make_gateway(HumanConfirmationAuthorizer())
    reg.load_snapshot(_snapshot_with_decision("call_1", HitlDecision(
        outcome="accepted", modified_arguments={"command": "ls -la"})))
    result = await gw.invoke(_TOOL_NAME, {"command": "rm -rf /"}, _state(), _ctx(),
                             tool_call_id="call_1")
    assert result.is_error is False
    assert reg.list_pending() == []                    # 没有重新问人
    assert _provider_invocations() == 1
    assert _last_invoked_args() == {"command": "ls -la"}, (
        "人工改写的参数没有跨冷路径生效——provider 拿到的仍是原参"
    )


async def test_a_cached_rejection_still_blocks_the_provider():
    """冷路径的另一半：缓存里是一次**拒绝**时，provider 同样绝不被调用。

    短路是「重用人给过的决定」，不是「放行」——`accepted` 那条用例单独绿着的话，
    一个把 `decision_for` 命中当成放行的实现照样能通过。
    """
    gw, reg, _svc = _make_gateway(HumanConfirmationAuthorizer())
    reg.load_snapshot(_snapshot_with_decision("call_1", HitlDecision(
        outcome="rejected", message="别删")))
    result = await gw.invoke(_TOOL_NAME, {"command": "rm -rf /"}, _state(), _ctx(),
                             tool_call_id="call_1")
    assert result.is_error is True
    assert "别删" in _text_of(result.content)
    assert _provider_invocations() == 0
    assert reg.list_pending() == []


async def test_a_plain_authorizer_never_touches_the_hitl_path():
    """不问人的 authorizer 走的路径与 HITL 无关，一行 HITL 代码都不执行。"""
    gw, reg, _svc = _make_gateway(_AllowAll())
    result = await gw.invoke(_TOOL_NAME, {"command": "ls"}, _state(), _ctx(),
                             tool_call_id="call_1")
    assert result.is_error is False and reg.list_pending() == []


async def test_needs_human_without_the_gated_interface_is_a_contract_violation():
    """声明要问人却没实现 on_decision = 契约违例，当场报错，不静默降级。"""
    gw, reg, svc = _make_gateway(_AsksButNotGated())
    task = asyncio.create_task(gw.invoke(_TOOL_NAME, {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID))
    result = await task
    assert result.is_error is True
    assert "does not implement" in _text_of(result.content)


# ── 决定缓存的第四维：同 tool_call_id 的**另一次**调用（复审 I3）────────────────


async def _approve_hot(gw, reg, svc, args, *, tool_call_id="call_1", modified=None):
    """热路径批准一次调用，返回 InvocationResult。"""
    task = asyncio.create_task(
        gw.invoke(_TOOL_NAME, args, _state(), _ctx(), tool_call_id=tool_call_id))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID, modified_arguments=modified))
    return await task


async def test_re_entering_the_same_call_still_short_circuits():
    """合法重放：park/崩溃后 reconcile 用**同一份**参数重入同一个 tool_call。

    reconcile 拿的是对话里记着的原始 tool_call 参数，所以 invocation_key 逐字节相同 →
    命中缓存 → 不重新问人。这条不能因为 I3 的收紧而被打破。
    """
    gw, reg, svc = _make_gateway(HumanConfirmationAuthorizer())
    await _approve_hot(gw, reg, svc, {"command": "ls"}, modified={"command": "ls -l"})
    assert _provider_invocations() == 1

    result = await gw.invoke(_TOOL_NAME, {"command": "ls"}, _state(), _ctx(),
                             tool_call_id="call_1")
    assert result.is_error is False
    assert reg.list_pending() == []                    # 没有重新问人
    assert _provider_invocations() == 2
    assert _last_invoked_args() == {"command": "ls -l"}  # 改写的参数照旧跟着这次调用


async def test_a_second_distinct_call_reusing_the_tool_call_id_is_not_authorized():
    """模型复用 `call_1`（短 id 是常态）时，第 3 轮的批准不得替第 9 轮的**另一次**调用开门。

    也不得把第 3 轮的 `modified_arguments` 带进去——那是「人批准的是 A、跑的是 B、
    参数还是 A 的」三重错位（复审 I3）。
    """
    gw, reg, svc = _make_gateway(HumanConfirmationAuthorizer(), timeout_sec=0)
    await _approve_hot(gw, reg, svc, {"command": "ls"}, modified={"command": "ls -l"})
    assert _provider_invocations() == 1

    # 同一个 tool_call_id，**另一次**调用（参数不同）→ 必须重新问人。
    # timeout_sec=0 ⟹ 热窗口立刻驱逐 ⟹ park：既没放行、也没拿旧决定。
    with pytest.raises(HitlPark):
        await gw.invoke(_TOOL_NAME, {"command": "rm -rf /"}, _state(), _ctx(),
                        tool_call_id="call_1")
    assert _provider_invocations() == 1                 # provider 没被第二次调用碰过
    fresh = reg.list_pending()
    assert len(fresh) == 1 and fresh[0].tool_call_id == "call_1"  # 新开了一条请求
    assert fresh[0].id != "" and fresh[0].resolved is False
