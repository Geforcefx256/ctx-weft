"""tool-result-recovery 回归（spec: tool-result-recovery）。

覆盖：结果存储窗口语义与 LRU 逐出、gateway 收敛（账本全文 + 上下文/ memory 收敛版 +
尾部止血）、尾部证据经 read_tool_output 回取、completed 短路重放收敛与「清空 store 后
从持久账本重放再回读」、store 写失败显式标记。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.utils.ids import mint_call_id

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.capability_gateway import CapabilityGateway, converge_tool_output
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.core.loop.capability_gateway import tool_result_record_id
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.capability_results import (
    READ_TOOL_QUALIFIED_NAME,
    ResultsCapabilityProvider,
)
from ctx_weft.providers.capability_results import ResultsCapabilityProvider
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
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


#: 同一个结果存储共享同一个事件库——`_harness` 被调两次模拟「同一会话的第二次
#: 进入」（重放 / 补写），两次必须看见同一条事件流。
_SHARED_EVENT_STORE: dict = {}


def _harness(provider, *, store=None, threshold=1000, sink=True):
    """`sink=True` 注册一个可回读的 SpillSink（ResultsCapabilityProvider）。

    core 只认 SpillSink——「能不能回读」由宿主注册哪种实现决定，不是 core 的协议分支。
    """
    bus = InProcessEventBus()
    # 事件库 + 提交门：重入判据来自 capability 事件流，所以夹具要跟生产同构。
    from ctx_weft.core.events.commit_gate import CommitGate
    from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
    event_store = _SHARED_EVENT_STORE.setdefault(id(store), InMemoryEventStore())
    bus.attach_commit_gate(CommitGate(event_store))
    mem = InMemoryMemoryProvider()
    cache = CapabilityCache()
    cache.put("agt_1", [provider._cap()])
    providers = [provider]
    if sink:
        providers.append(ResultsCapabilityProvider(store))
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=providers,
        memory=mem, event_bus=bus, spill_threshold=threshold,
        spill_preview_chars=100, spill_tail_chars=120,
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
        event_store=event_store,
    )
    return gw, mem, state, ctx, scope


@pytest.mark.asyncio
async def test_converged_context_and_full_text_in_the_sink():
    """收敛分层：上下文拿收敛版，全文归 sink——**核心不留第二份副本**。"""
    tail_marker = "TAIL_MARKER_9876543210"
    big = "x" * 5_000 + tail_marker
    store = InMemoryToolResultStore()
    provider = _Big(big)
    gw, mem, state, ctx, scope = _harness(provider, store=store)

    res = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id=OP_A)

    # 上下文 = 收敛版：引用 + 全长 + 头预览 + 尾预览（尾部止血）。
    assert tail_marker in res.content
    assert str(len(big)) in res.content
    assert READ_TOOL_QUALIFIED_NAME in res.content
    assert "x" * 5_000 not in res.content                   # 全文未直灌
    # memory TOOL_RESULT 同为收敛版。
    recs = await mem.load_view(scope, MemoryScope.TASK, ctx.provider_ctx)
    assert [r for r in recs if r.role == "tool"][0].content == res.content
    # 全文只在 sink 里，键 = 这次执行的 invocation_id。
    assert await store.get(res.invocation_id) == big


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
    reader = ResultsCapabilityProvider(store)

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


#: 合法的内部标识（账本键 = tool_call_id；裸 wire id 会走旁路）。
OP_A = mint_call_id(anchor="recA", ordinal=0, raw_id="call_1", turn_seq=1)
OP_B = mint_call_id(anchor="recB", ordinal=0, raw_id="call_1", turn_seq=9)


# ── 重放：结果来自事件流 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_reentry_replays_the_recorded_result():
    """同逻辑调用重入 → 复用**事件里记着的那份**，不再打 provider、不再收敛一遍。"""
    store = InMemoryToolResultStore()
    provider = _Big("z" * 4_000)
    gw, mem, state, ctx, _ = _harness(provider, store=store)

    first = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id=OP_A)
    assert READ_TOOL_QUALIFIED_NAME in first.content
    replay = await gw.invoke("mcp__b__dump", {}, state, ctx,
                             tool_call_id=OP_A, reentry=True)
    assert replay.content == first.content, "照抄事件里那份，逐字节一致"


@pytest.mark.asyncio
async def test_full_text_survives_only_as_long_as_the_sink_does():
    """sink 被清空 → 全文取不回来了。这是**有意的**。

    账本时代核心另存一份收敛前全文，逐出后拿它重新入库。撤销之后，「全文能存活多久」
    完全由宿主注册的 sink 决定——不持久就是不持久，核心不再存副本假装兜得住，而收敛版
    里本来就有「NOT recoverable」的如实说法。
    """
    store = InMemoryToolResultStore()
    provider = _Big("z" * 4_000)
    gw, mem, state, ctx, _ = _harness(provider, store=store)
    res = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id=OP_A)

    assert await store.get(res.invocation_id) == "z" * 4_000
    assert await InMemoryToolResultStore().get(res.invocation_id) is None   # 逐出/重启



# ── store 写失败显式标记 ─────────────────────────────────────────────────────


class _FailingStore(InMemoryToolResultStore):
    async def put(self, invocation_id, text, ctx=None):
        raise RuntimeError("disk on fire")


@pytest.mark.asyncio
async def test_spill_failure_marks_unrecoverable():
    """sink 存不下 → 显式「不可取回」，不留一个取不回来的引用骗模型。"""
    provider = _Big("w" * 3_000)
    gw, mem, state, ctx, _ = _harness(provider, store=_FailingStore())
    res = await gw.invoke("mcp__b__dump", {}, state, ctx)
    assert "NOT recoverable" in res.content
    assert "w" * 100 in res.content          # 头部预览仍在（不静默、不空转）


# ── reconcile 补写入口（FINISHED 已发而 memory 缺失 → 照抄事件里那份） ──────────


@pytest.mark.asyncio
async def test_reconcile_backfills_from_the_event_without_reconverging():
    """「已完成而 memory 缺失」的补写：用事件里那份，**不再收敛一遍**。

    那份就是当初进对话的收敛版；再过一次收敛出口会把收敛说明自己当正文又切一刀。
    """
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep

    big = "r" * 4_000
    store = InMemoryToolResultStore()
    provider = _Big(big)
    gw, mem, state, ctx, scope = _harness(provider, store=store)
    res = await gw.invoke("mcp__b__dump", {}, state, ctx, tool_call_id=OP_B)

    # 「FINISHED 已发而 TOOL_RESULT 从未写成」那个崩溃窗口——换一个空 memory，而不是
    # 把已写的那条抹掉：记录 id 是确定性的，抹掉走 fold 会给它立墓碑，同 id 再也回不来。
    gw2, mem2, state2, ctx2, scope2 = _harness(provider, store=store)
    tc = {"id": OP_B, "name": "mcp__b__dump", "input": {}}
    for _ in range(2):          # 补两次，验幂等
        await ReconcileStep()._backfill_memory(
            state2, ctx2, OP_B, tc, res.content, gateway=gw2,
            attempts=(res.invocation_id,), converge=False, via="event-finished")

    recs = await mem2.load_view(scope2, MemoryScope.TASK, ctx2.provider_ctx)
    tool_recs = [r for r in recs if r.role == "tool"]
    assert len(tool_recs) == 1, "同一确定性 id，补两次只落一条"
    assert tool_recs[0].id == tool_result_record_id(OP_B)
    assert tool_recs[0].content == res.content, "照抄，不重新收敛"


@pytest.mark.asyncio
async def test_pure_converge_without_sink_still_bounded():
    """纯函数直调：无 sink → 显式不可取回；头尾预览仍在，输出仍有界。"""
    out = await converge_tool_output(
        "q" * 2_500 + "END_MARK", "inv_x", None, None,
        threshold=1000, preview_chars=50, tail_chars=60)
    assert "NOT recoverable" in out
    assert "END_MARK" in out
