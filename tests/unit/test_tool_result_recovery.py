"""tool-result-recovery 回归（spec: tool-result-recovery）。

覆盖：结果存储窗口语义与 LRU 逐出、gateway 收敛（账本全文 + 上下文/ memory 收敛版 +
尾部止血）、尾部证据经 read_tool_output 回取、completed 短路重放收敛与「清空 store 后
从持久账本重放再回读」、store 写失败显式标记。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.capability_gateway import CapabilityGateway, converge_tool_output
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.results import READ_TOOL_QUALIFIED_NAME
from ctx_weft.providers.capability_results import ResultsCapabilityProvider
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.operations import InMemoryOperationStore
from ctx_weft.providers.results import InMemoryToolResultStore

# ── 结果存储 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_pagination_and_tail():
    store = InMemoryToolResultStore()
    text = "".join(chr(ord("a") + i % 26) for i in range(5_000))
    await store.put("inv1", text)
    # 分页拼接 == 原文（spec R1 场景）。
    pages, off = [], 0
    while True:
        chunk = await store.get("inv1", offset=off, limit=700)
        assert chunk is not None
        pages.append(chunk)
        if len(chunk) < 700:
            break
        off += 700
    assert "".join(pages) == text
    # tail 直读末尾。
    assert await store.get("inv1", tail=100) == text[-100:]


@pytest.mark.asyncio
async def test_store_lru_eviction_yields_explicit_miss():
    store = InMemoryToolResultStore(max_entries=1)
    await store.put("old", "o" * 10)
    await store.put("new", "n" * 10)
    assert await store.get("old") is None     # 逐出 → None（显式未命中由调用方转译）
    assert await store.get("new") == "n" * 10


# ── gateway 收敛 ─────────────────────────────────────────────────────────────


class _Big(ToolCapabilityProvider):
    name = "mcp:b"

    def __init__(self, text: str, spillable: bool = True) -> None:
        self._text = text
        self._spillable = spillable

    def _cap(self):
        return ToolCapability(
            id="mcp:b:dump", name="dump", description="d", spillable=self._spillable)

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            yield CapabilityEvent(kind="result", payload={"content": self._text})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None: return None


def _harness(provider, *, store=None, ledger=None, threshold=1000):
    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    cache = CapabilityCache()
    cache.put("agt_1", [provider._cap()])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[provider],
        memory=mem, event_bus=bus, spill_threshold=threshold,
        spill_preview_chars=100, spill_tail_chars=120,
        result_store=store, operation_store=ledger,
    )
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"), agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )
    return gw, mem, state, ctx, scope


@pytest.mark.asyncio
async def test_converge_ledger_full_context_converged_with_tail():
    tail_marker = "TAIL_MARKER_9876543210"
    big = "x" * 5_000 + tail_marker
    store = InMemoryToolResultStore()
    ledger = InMemoryOperationStore()
    provider = _Big(big)
    gw, mem, state, ctx, scope = _harness(provider, store=store, ledger=ledger)

    # 铸 operation 身份（act 路径口径），使账本 completed 生效。
    ctx.provider_ctx.operation_id = "op_test_1"
    res = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id="tc_1_0_a")

    # 上下文 = 收敛版：引用 + 全长 + 头预览 + 尾预览（尾部止血）。
    assert tail_marker in res.content                       # 尾部证据直接可见
    assert str(len(big)) in res.content
    assert READ_TOOL_QUALIFIED_NAME in res.content
    assert "x" * 5_000 not in res.content                   # 全文未直灌
    # memory TOOL_RESULT 同为收敛版。
    recs = await mem.load_view(scope, MemoryScope.TASK, ctx.provider_ctx)
    tool_rec = [r for r in recs if r.role == "tool"][0]
    assert tool_rec.content == res.content
    # 账本 completed = 收敛前全文（修 tool-operations 偏离）。
    op_rec = await ledger.get("op_test_1", ctx.provider_ctx)
    assert op_rec.result == big
    assert len(op_rec.result) == len(big)
    # store 持全文，可窗口回读。
    assert await store.get(op_rec.attempts[-1]) == big


@pytest.mark.asyncio
async def test_under_threshold_untouched():
    store = InMemoryToolResultStore()
    provider = _Big("short output")
    gw, mem, state, ctx, _ = _harness(provider, store=store)
    res = await gw.invoke("mcp__b__dump", {}, state, ctx)
    assert res.content == "short output"


@pytest.mark.asyncio
async def test_tail_evidence_reachable_via_read_tool_output():
    tail_marker = "CRITICAL_ERROR_AT_END_1234567890"
    big = "y" * 30_000 + tail_marker
    store = InMemoryToolResultStore()
    provider = _Big(big)
    gw, mem, state, ctx, _ = _harness(provider, store=store)

    res = await gw.invoke("mcp__b__dump", {}, state, ctx)
    assert tail_marker in res.content                       # 尾部预览直接呈现
    inv_id = res.invocation_id
    reader = ResultsCapabilityProvider(lambda: store)

    async def _read(args):
        out = ""
        async for ev in reader.invoke("results:read_tool_output", args, ctx.provider_ctx):
            out = ev.payload.get("content", out)
        return out

    # tail 模式读回含标记的全文片段；分页模式拼回全文。
    assert tail_marker in await _read({"invocation_id": inv_id, "tail": 200})
    chunks, off = [], 0
    while True:
        part = await _read({"invocation_id": inv_id, "offset": off, "limit": 8000})
        if part.startswith("[no stored output"):
            break
        chunks.append(part.split("\n[window")[0])
        off += 8000
        if off >= len(big):
            break
    assert "".join(chunks) == big


# ── 重放：统一收敛 + 持久账本重放入库 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_shortcircuit_replay_converges_and_reputs():
    big = "z" * 4_000
    store = InMemoryToolResultStore()
    ledger = InMemoryOperationStore()
    provider = _Big(big)
    gw, mem, state, ctx, _ = _harness(provider, store=store, ledger=ledger)

    ctx.provider_ctx.operation_id = "op_replay_1"
    first = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id="tc_1_0_a")
    first_inv = first.invocation_id
    assert READ_TOOL_QUALIFIED_NAME in first.content

    # 清空内存 store（模拟逐出/重启）→ 重放须以账本全文重新入库且形态收敛。
    store2 = InMemoryToolResultStore()
    gw2, *_ = _harness(_Big("unused"), store=store2, ledger=ledger)
    # gateway 默认 store 是独立实例——把重放 gateway 指向被清空的 store2：
    gw2._result_store = store2
    ctx.provider_ctx.operation_id = "op_replay_1"
    replay = await gw2.invoke("mcp__b__dump", {}, state, ctx, tool_call_id="tc_9_0_b")

    assert READ_TOOL_QUALIFIED_NAME in replay.content      # 收敛版，非全文直灌
    assert replay.content.count("z" * 100) < 10
    # 引用身份 = 账本原执行 invocation_id（非重放新铸值）。
    assert first_inv in replay.content
    # store2 已由账本全文重新入库 → 可回读（spec R4 场景「清空存储后重放仍可回读」）。
    assert await store2.get(first_inv) == big
    assert await store2.get(replay.invocation_id) is None  # 重放新 id 不入库


# ── store 写失败显式标记 ─────────────────────────────────────────────────────


class _FailingStore(InMemoryToolResultStore):
    async def put(self, invocation_id, text, ctx=None):
        raise RuntimeError("disk on fire")


@pytest.mark.asyncio
async def test_store_write_failure_marks_unavailability():
    provider = _Big("w" * 3_000)
    gw, mem, state, ctx, _ = _harness(provider, store=_FailingStore())
    res = await gw.invoke("mcp__b__dump", {}, state, ctx)
    assert "full output unavailable" in res.content
    assert "result store write failed" in res.content
    assert "w" * 100 in res.content          # 头部预览仍在（不静默、不空转）


# ── reconcile 补写入口（账本完成而 memory 缺失 → 收敛补写） ────────────────────


@pytest.mark.asyncio
async def test_reconcile_backfill_converges_ledger_result():
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep

    big = "r" * 4_000
    store = InMemoryToolResultStore()
    ledger = InMemoryOperationStore()
    provider = _Big(big)
    gw, mem, state, ctx, scope = _harness(provider, store=store, ledger=ledger)
    ctx.provider_ctx.operation_id = "op_bf_1"
    res = await gw.invoke("mcp__b__dump", {}, state, ctx)   # 原执行（attempts 落账本）

    rec = await ledger.get("op_bf_1", ctx.provider_ctx)
    step = ReconcileStep()
    tc = {"id": "tc_1_0_r", "name": "mcp__b__dump", "input": {}}
    await step._backfill_memory(
        state, ctx, "op_bf_1", rec.result, tc, gateway=gw, rec=rec, via="ledger-completed")

    recs = await mem.load_view(scope, MemoryScope.TASK, ctx.provider_ctx)
    bf = [r for r in recs if r.metadata.get("recovered_via") == "ledger-completed"][0]
    assert READ_TOOL_QUALIFIED_NAME in bf.content            # 收敛版，非全文直灌
    assert res.invocation_id in bf.content                   # 引用 = 账本原执行 id


@pytest.mark.asyncio
async def test_pure_converge_no_sink_no_store_still_bounded():
    """纯函数直调：无 store（None）→ 显式不可用；尾部预览仍在。"""
    out = await converge_tool_output(
        "q" * 2_500 + "END_MARK", "inv_x", None, None, None,
        threshold=1000, preview_chars=50, tail_chars=60)
    assert "full output unavailable" in out
    assert "END_MARK" in out
