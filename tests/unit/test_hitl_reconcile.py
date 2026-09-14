"""reconcile step（spec/07 §6）。

原先本文件开头还有三条 `RunStateView.pending_hitl` 的折叠用例。那份投影已删除——
HITL 状态的真相源只剩 `fold_hitl_snapshot` → `HitlRegistry` 一条（口径见
`test_hitl_fold_snapshot.py` / `test_hitl_registry_load.py`）。第二份口径不同的 HITL
折叠留着就是等人再接一次的陈旧副本，正是旧实现「重建了 pending 却没重建已解决」
那类漂移的来源。
"""

from __future__ import annotations

from ctx_weft.core.utils.ids import mint_call_id

#: 摄入点铸造的内部标识——既是消息里的 tool_call id，也是账本键。
TC_A = mint_call_id(anchor="recA", ordinal=0, raw_id="tcA", turn_seq=0)
TC_B = mint_call_id(anchor="recB", ordinal=0, raw_id="tc1", turn_seq=0)

import pytest


pytestmark = pytest.mark.asyncio


class _NullBus:
    async def emit(self, *_a, **_k): ...


async def test_reconcile_invokes_only_dangling_tool_calls() -> None:
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep
    from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    base = datetime(2026, 1, 1, tzinfo=UTC)
    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default")
    sc = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")

    def ev(tp, content, sec, role, **md):
        return MemoryEvent(type=tp, address=sc, content=content,
                           timestamp=base + timedelta(seconds=sec), role=role, metadata=md)

    await mem.ingest(ev(MemoryEventType.LLM_RESPONSE, "", 1, "assistant",
                        tool_calls=[{"id": "tc1", "name": "web", "input": {}},
                                    {"id": "tc2", "name": "ask_user", "input": {"question": "?"}}]), pctx)
    await mem.ingest(ev(MemoryEventType.TOOL_RESULT, "web out", 2, "tool", tool_call_id="tc1"), pctx)

    invoked: list[str] = []
    async def fake_invoke(tool_name, arguments, state, ctx, tool_call_id=""):
        invoked.append(tool_call_id)
        return SimpleNamespace(content="human text", is_error=False)

    gateway = SimpleNamespace(invoke=fake_invoke)
    state = SimpleNamespace(
        scope=sc, agent=SimpleNamespace(id="ag1"),
        task=SimpleNamespace(id="t1", status="ACTIVE"),
        session=SimpleNamespace(id="s1", tenant_id="default"),
        extra={},                          # 无 template → resolve_and_bind 解析为空,不动 cache
    )
    ctx = SimpleNamespace(memory=mem, provider_ctx=pctx, capability_gateway=gateway,
                          cancel_token=None, event_bus=_NullBus(),
                          capability_providers=[], capability_cache=None)

    # wp6（spec: tool-operations）：无账本身份的 dangling **不自动重执行**——tc2 不被
    # invoke。但也不停机：作结「无从查证」写成工具结果，循环继续到 prepare，由 agent
    # 在任务上下文里决定怎么办（结果不确定是一种工具结果，不是一种控制流）。
    state.sequence_counter = 0
    outcome = await ReconcileStep().execute(state, ctx)
    assert invoked == [], "无账本身份的副作用工具不得自动重跑"
    assert outcome.next_step == "prepare", "不停机——把结果交给 agent 继续"


async def test_reconcile_no_dangling_routes_to_prepare() -> None:
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep
    from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    base = datetime(2026, 1, 1, tzinfo=UTC)
    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default")
    sc = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    asst_id = await mem.ingest(MemoryEvent(type=MemoryEventType.LLM_RESPONSE, address=sc, content="",
                                 timestamp=base + timedelta(seconds=1), role="assistant",
                                 metadata={"tool_calls": [{"id": TC_A, "name": "web", "input": {}}]}), pctx)
    # 完成判据 = 确定性 memory id（wire 记录不再判 done——防 call_1 串扰）
    from ctx_weft.protocols.operations import operation_memory_result_id
    op_id = TC_A
    await mem.ingest(MemoryEvent(type=MemoryEventType.TOOL_RESULT, address=sc, content="out",
                                 timestamp=base + timedelta(seconds=2), role="tool",
                                 id=operation_memory_result_id(op_id),
                                 metadata={"tool_call_id": "tcA"}), pctx)
    invoked = []
    async def fake_invoke(**kw): invoked.append(kw); return None
    gateway = SimpleNamespace(invoke=lambda *a, **k: fake_invoke(**k))
    state = SimpleNamespace(scope=sc, agent=SimpleNamespace(id="ag1"),
                            task=SimpleNamespace(id="t1", status="ACTIVE"),
                            session=SimpleNamespace(id="s1", tenant_id="default"), extra={})
    ctx = SimpleNamespace(memory=mem, provider_ctx=pctx, capability_gateway=gateway,
                          cancel_token=None, event_bus=_NullBus())
    outcome = await ReconcileStep().execute(state, ctx)
    assert invoked == []                 # 全部已有 result → 不重跑
    assert outcome.next_step == "prepare"


async def test_resolve_reconcile_detection_helper() -> None:
    from datetime import UTC, datetime, timedelta
    from ctx_weft.core.runtime import _task_has_dangling_tool_call
    from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    base = datetime(2026, 1, 1, tzinfo=UTC)
    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default")
    sc = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")

    def ev(tp, sec, role, **md):
        return MemoryEvent(type=tp, address=sc, content="", timestamp=base + timedelta(seconds=sec),
                           role=role, metadata=md)

    await mem.ingest(ev(MemoryEventType.LLM_RESPONSE, 1, "assistant",
                        tool_calls=[{"id": TC_B, "name": "web", "input": {}}]), pctx)
    # wp6（spec: tool-operations）翻转：完成判据 = 逻辑身份（确定性 memory id），
    # 不再是 wire id 配对——call_1 复用会串扰。旧式 tool 记录（无确定性 id）不再判 done。
    async def _dangling():
        m = __import__("ctx_weft.core.loop.steps.reconcile", fromlist=["_dangling_tool_calls"])
        d, _ = await m._dangling_tool_calls(mem, sc, pctx)
        return d

    assert bool(await _dangling()) is True
    await mem.ingest(ev(MemoryEventType.TOOL_RESULT, 2, "tool", tool_call_id="tc1"), pctx)
    # wire 记录不判 done（防串扰）——仍 dangling
    assert bool(await _dangling()) is True
    # 确定性 id 判 done：按 (record_id, ordinal=0) 派生的 memory id 写 tool 记录
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.protocols.operations import operation_memory_result_id
    op_id = TC_B
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.TOOL_RESULT, address=sc, content="done",
        timestamp=base + timedelta(seconds=3), role="tool",
        id=operation_memory_result_id(op_id)), pctx)
    assert bool(await _dangling()) is False
