"""gateway 操作账本串接（spec: tool-operations；wp5-4.3）。

五步执行序、completed 重入短路（O-T08 前半）、裸调旁路、silent 工具入账、
park → waiting_human。
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols.operations import OperationStatus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.operations import InMemoryOperationStore


class _Echo(ToolCapabilityProvider):
    name = "probe"

    def __init__(self) -> None:
        self.calls = 0

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="probe:go", name="go", description="d")

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            self.calls += 1
            yield CapabilityEvent(kind="result", payload={"content": f"done-{self.calls}"})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None: return None


def _fixture(*, op_store=None):
    memory = InMemoryMemoryProvider()
    bus = InProcessEventBus()
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="t1"),
        task=SimpleNamespace(id="task1"), agent=SimpleNamespace(id="a1", template_id="tpl"),
        scope=MemoryAddress(session_id="s1", task_id="task1", agent_id="a1"),
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=memory, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", tenant_id="t1",
                                     task_id="task1", agent_id="a1"),
    )
    tool = _Echo()
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    gateway = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool],
        memory=memory, event_bus=bus,
        operation_store=op_store,
    )
    ctx.capability_gateway = gateway
    return memory, state, ctx, tool, gateway



#: 账本键 = 摄入点铸造的内部 tool_call 标识（gateway 从 invoke 的 tool_call_id 取）。
OP1 = mint_call_id(anchor="rec1", ordinal=0, raw_id="call_1", turn_seq=0)


async def test_full_execution_order_records_completed():
    """五步序：prepare → started → provider（calls=1）→ completed → TOOL_RESULT 事件。"""
    ops = InMemoryOperationStore()
    memory, state, ctx, tool, gateway = _fixture(op_store=ops)

    res = await gateway.invoke("probe__go", {"x": 1}, state, ctx, tool_call_id=OP1)
    assert res.is_error is False and tool.calls == 1
    rec = await ops.get(OP1, ctx.provider_ctx)
    assert rec is not None
    assert rec.status == OperationStatus.COMPLETED
    assert rec.result == "done-1"
    assert rec.attempts and rec.attempts[0].startswith("inv")


async def test_completed_reentry_short_circuits_no_reinvoke():
    """O-T08 前半：同逻辑调用重入——账本 completed → 不再打 provider，回放结果。"""
    ops = InMemoryOperationStore()
    memory, state, ctx, tool, gateway = _fixture(op_store=ops)

    first = await gateway.invoke("probe__go", {}, state, ctx, tool_call_id=OP1)
    assert tool.calls == 1

    # 同逻辑调用重入（如恢复重试）——同一个内部标识
    second = await gateway.invoke("probe__go", {}, state, ctx, tool_call_id=OP1)
    assert tool.calls == 1, "provider must NOT be re-invoked for a completed operation"
    assert second.is_error is False
    assert "done-1" in str(second.content)               # 回放首次结果


async def test_bare_wire_id_bypasses_ledger():
    """裸 wire id（未经铸造）账本全程旁路——既有单测/宿主直构零行为变化。

    判据是「是不是内部标识」，不是「有没有传 id」：模型复用的 call_1 不可信，
    拿它当账本键会让两次不同调用撞同一行。
    """
    ops = InMemoryOperationStore()
    memory, state, ctx, tool, gateway = _fixture(op_store=ops)
    res = await gateway.invoke("probe__go", {}, state, ctx, tool_call_id="call_1")
    assert res.is_error is False and tool.calls == 1
    assert await ops.get("call_1", ctx.provider_ctx) is None

    # 完全不传 id 同样旁路
    res2 = await gateway.invoke("probe__go", {}, state, ctx)
    assert res2.is_error is False and tool.calls == 2


async def test_ledger_failure_raises_persistence_unavailable():
    """账本写失败 → PersistenceUnavailableError（复用 WP3 隔离语义）。"""

    class _Broken(InMemoryOperationStore):
        async def prepare(self, record, ctx):
            raise OSError("ledger down")

    memory, state, ctx, tool, gateway = _fixture(op_store=_Broken())
    from ctx_weft.protocols.events import PersistenceUnavailableError
    try:
        await gateway.invoke("probe__go", {}, state, ctx, tool_call_id=OP1)
        raised = False
    except PersistenceUnavailableError:
        raised = True
    assert raised and tool.calls == 0, "provider must not run when ledger is down"


async def test_identity_is_per_call_not_shared_state():
    """身份是 invoke 的入参，不是共享可变字段——不可能泄漏给下一个无关调用。

    旧设计经 `provider_ctx.operation_id` 转移所有权，忘了清就会让后台 observe 的
    collect_process_report 命中别的操作的 completed 短路（实测回归）。改成入参后
    这个失败模式在结构上不存在。
    """
    ops = InMemoryOperationStore()
    memory, state, ctx, tool, gateway = _fixture(op_store=ops)
    await gateway.invoke("probe__go", {}, state, ctx, tool_call_id=OP1)
    assert tool.calls == 1

    # 紧接着一次不带身份的调用：旁路、正常执行、不命中上一次的 completed
    res = await gateway.invoke("probe__go", {}, state, ctx)
    assert res.is_error is False and tool.calls == 2
    assert not hasattr(ctx.provider_ctx, "operation_id")
