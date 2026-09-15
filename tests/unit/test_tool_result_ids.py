"""TOOL_RESULT 记录 id 的不变式（spec: tool-operations / conversation-integrity）。

一次工具调用的结果可以从五个地方进 task 层对话。「记录 id 由 tool_call_id 确定性派生」
这条不变式此前**没有单一执行点**，五处里只有两处遵守，而唯一的验证者
（`_dangling_tool_calls` 的 memory 通道）只认遵守的那种。本文件钉住收口之后的形态。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ctx_weft.protocols.capability import is_internal_call_id, tool_result_record_id


def test_derivation_is_partial_not_a_colliding_guess():
    """裸 wire id **没有**确定性派生——返回 None，而不是造一个会撞车的值。

    旧实现无条件 `f"res_{x[3:]}"`：`""` → `"res_"`、`"call_1"` → `"res_l_1"`。同一回合
    两条裸 id 的 dangling 作结于是撞同一个 `"res_"`，后一条被 memory 的 ingest 幂等契约
    当 no-op 吞掉：结果从对话里消失，那个 tool_call 从此永久悬挂。
    """
    minted = "tc_1_0_0de0fdd51fdb"
    assert is_internal_call_id(minted)
    assert tool_result_record_id(minted) == "res_1_0_0de0fdd51fdb"
    for bare in ("", "call_1", "call_2", "tcall_01J8", "x"):
        assert not is_internal_call_id(bare)
        assert tool_result_record_id(bare) is None


def test_derivation_is_injective_over_minted_ids():
    """不同的内部标识必须派生出不同的记录 id（撞了就是两次调用共用一条结果）。"""
    from ctx_weft.core.utils.ids import mint_call_id

    ids = [mint_call_id(anchor=a, ordinal=o, raw_id="call_1", turn_seq=t)
           for a in ("asst_1", "asst_2") for o in range(3) for t in (0, 7)]
    derived = [tool_result_record_id(i) for i in ids]
    assert len(set(derived)) == len(ids)
    assert None not in derived


def test_task_scope_tool_results_have_exactly_one_write_site():
    """TASK 层 `role="tool"` 的 MemoryEvent 只许在 `ingest_tool_result` 函数体内构造。

    守卫是**函数级**而不是文件级：它和 `converge_tool_output` 一样住在
    `capability_gateway.py`（那是同一个文件里既有的共享自由函数形态），而那个文件
    1313 行、内部就有三个原本手搓 MemoryEvent 的调用点——按文件放行等于不设防。

    AGENT 层的两处（delegate_plan 配对 ack、子任务 finish 对）不在此列：它们按
    `tcall_...` 配对、由 finalize 的框/对机制管理，dangling 判定从不扫描。
    """
    WRITER = "ingest_tool_result"
    offenders: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self, path: str) -> None:
            self.path, self.stack = path, []

        def _visit_func(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_FunctionDef = visit_AsyncFunctionDef = _visit_func

        def visit_Call(self, node):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name == "MemoryEvent":
                kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
                if kw.get("role") == "'tool'" and "TASK" in kw.get("scope", ""):
                    if WRITER not in self.stack:
                        offenders.append(f"{self.path}:{node.lineno}")
            self.generic_visit(node)

    for p in sorted(pathlib.Path("src").rglob("*.py")):
        src = p.read_text(encoding="utf-8")
        if "MemoryEvent(" in src:
            _Visitor(p.as_posix()).visit(ast.parse(src))

    assert not offenders, (
        f"这些地方直接构造了 TASK 层 TOOL_RESULT，绕过了 {WRITER} —— "
        f"记录 id 的唯一决定点：{offenders}")


@pytest.mark.asyncio
async def test_bare_wire_ids_get_distinct_records_and_are_not_reconcluded():
    """存量裸 wire id：同回合两条各自落库（不撞车），且第二轮 reconcile 不再重复作结。"""
    from types import SimpleNamespace
    from collections.abc import AsyncIterator

    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.loop.driver import LoopContext, LoopState
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
    from ctx_weft.protocols.capability import (
        CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider)
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    class T(ToolCapabilityProvider):
        name = "fx"
        def _cap(self):
            return ToolCapability(id="fx:act", name="act", description="d", side_effects=True)
        async def list(self, ctx): return [self._cap()]
        async def retrieve(self, ctx): return [self._cap()]
        async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)
        def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
            async def _r(): yield CapabilityEvent(kind="result", payload={"content": "x"})
            return _r()
        async def cancel(self, i, ctx): return None

    tool, mem = T(), InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE", parent_task_id=None),
        agent=SimpleNamespace(id="a1", template_id="t"), scope=scope,
        resolved_model=SimpleNamespace(model="m", account=""), sequence_counter=0)
    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
                      provider_ctx=pctx)
    cache = CapabilityCache(); cache.put("a1", [tool._cap()])
    ctx.capability_gateway = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool], memory=mem,
        event_bus=InProcessEventBus(), operation_store=None)

    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=now_utc(), role="assistant",
        metadata={"tool_calls": [{"id": "call_1", "name": "fx__act", "input": {"i": 1}},
                                 {"id": "call_2", "name": "fx__act", "input": {"i": 2}}]}), pctx)

    await ReconcileStep().execute(state, ctx)
    await ReconcileStep().execute(state, ctx)      # 第二轮：wire 配对认得出已完成

    tool_recs = [r for r in await mem.load_view(scope, MemoryScope.TASK, pctx)
                 if r.role == "tool"]
    assert len(tool_recs) == 2, [r.id for r in tool_recs]
    assert len({r.id for r in tool_recs}) == 2, "两条不得撞同一个记录 id"
    assert {r.metadata["tool_call_id"] for r in tool_recs} == {"call_1", "call_2"}
