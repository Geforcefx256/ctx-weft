"""压缩的粒度：L3 按回合切、L2 从最老的胶囊开始逐颗降级。

两个问题同源——**按条目数/整批操作，而不是按语义单元操作**：

- **L3**（`collapse_task_layer`）原先按**记录条数**数 `keep_last`，且把不渲染的
  `TOOL_AUDIT` 也算进名额。切点可能落在 assistant 与它的工具结果之间：结果留在保留区、
  assistant 被折走 → 结果成孤儿 → 发送前被 `drop_orphan_tool_results` **静默**丢掉。一轮并行
  调两个工具时，「保留最近 3 条」留下的是 `[result, audit, result]`，渲染出来 **0 条**。
  恢复路径上这几乎必然发生：act 段总在工具执行完后才退出，时间线末尾一定是工具结果。

- **L2**（`demote_kept_capsules`）原先对 L1 保下的全部胶囊**一次性**降级。哪怕降一颗就
  够达标，6 颗也一起被掏空（USER_PROMPT 原文、段摘要、act_recap 全删，只剩综合总结）。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps import compact as cm
from ctx_weft.protocols import (
    MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

_BASE = datetime(2026, 9, 16, tzinfo=UTC)
_ROOT = MemoryAddress(session_id="s1", task_id="root", agent_id="a1")


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="tn")


class _Seeder:
    """按调用顺序递增时间戳灌记录——时间线就是写入顺序，断言里好推。"""

    def __init__(self, mem, address=_ROOT) -> None:
        self.mem, self.address, self.i = mem, address, 0

    async def _put(self, *, kind, scope, role, content, metadata, address=None) -> str:
        self.i += 1
        return await self.mem.ingest(MemoryEvent(
            kind=kind, scope=scope, address=address or self.address, content=content,
            timestamp=_BASE + timedelta(seconds=self.i), role=role,
            metadata=metadata), _pctx())

    async def user(self, text):
        return await self._put(kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                               role="user", content=text, metadata={})

    async def assistant(self, text, *call_ids):
        return await self._put(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, role="assistant",
            content=text,
            metadata={"tool_calls": [{"id": c, "name": "probe__noop"} for c in call_ids]})

    async def audit(self, call_id):
        """与 capability_gateway 写的形态一致：role=assistant、无 tool_calls、**不渲染**。"""
        return await self._put(
            kind=MemoryKind.TOOL_AUDIT, scope=MemoryScope.TASK, role="assistant",
            content=f"probe__noop({{}})",
            metadata={"tool_call_id": call_id, "tool_name": "probe__noop"})

    async def result(self, call_id, text):
        return await self._put(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, role="tool",
            content=text, metadata={"tool_call_id": call_id})

    async def round(self, text, *call_ids):
        """一轮：assistant 发起 → 每个工具「审计 → 结果」。与 act 的真实落库顺序一致。"""
        await self.assistant(text, *call_ids)
        for c in call_ids:
            await self.audit(c)
            await self.result(c, f"result of {c}")


def _ctx(mem):
    tokenizer = SimpleNamespace(count=lambda text: max(1, len(text) // 4))
    return SimpleNamespace(memory=mem, provider_ctx=_pctx(),
                           llm=SimpleNamespace(tokenizer=tokenizer), blob_store=None)


def _l3_state(address=_ROOT):
    return SimpleNamespace(scope=address, task=SimpleNamespace(id=address.task_id),
                           agent=SimpleNamespace(), session=SimpleNamespace())


async def _task_view(mem, address=_ROOT):
    return await mem.load_view(address, MemoryScope.TASK, _pctx(),
                               kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY,
                                      MemoryKind.TOOL_AUDIT])


def _rendered(view):
    """装配会渲染的那部分（`AgentRecallSource` 只取对话回合 + 摘要，TOOL_AUDIT 不进 prompt）。"""
    return [r for r in view if r.kind is not MemoryKind.TOOL_AUDIT]


def _orphans(view):
    """保留下来、却找不到发起它的 assistant 的工具结果——发送前会被静默丢掉。"""
    owned = {tc.get("id") for r in view if r.role == "assistant"
             for tc in (r.metadata.get("tool_calls") or [])}
    return [r for r in view if r.role == "tool" and r.metadata.get("tool_call_id") not in owned]


# ── L3：按回合切 ───────────────────────────────────────────────────────────────


async def test_l3_never_orphans_parallel_tool_results():
    """最后一轮并行调两个工具：保留区必须带上那一轮的 assistant，两条结果都不成孤儿。"""
    mem = InMemoryMemoryProvider()
    s = _Seeder(mem)
    await s.user("refactor X")
    await s.round("look first", "a")
    await s.round("now two at once", "b", "c")

    n = await cm.collapse_task_layer(_l3_state(), _ctx(mem), 3, "digest")

    assert n > 0, "前一轮应被折掉"
    view = await _task_view(mem)
    assert not _orphans(view), f"保留区不得有孤儿工具结果：{[r.content for r in _orphans(view)]}"
    kept = [r.content for r in _rendered(view)]
    assert "now two at once" in kept and "result of b" in kept and "result of c" in kept, kept
    # 对照：更早那一轮确实被折走了（否则上面的断言可能只是「什么都没折」）
    assert "look first" not in kept and "result of a" not in kept, kept


async def test_l3_audit_records_do_not_consume_keep_slots():
    """`keep_last` 数的是**会被渲染**的记录：审计不进 prompt，不该占保留名额。"""
    mem = InMemoryMemoryProvider()
    s = _Seeder(mem)
    await s.user("refactor X")
    await s.round("r1", "a")
    await s.round("r2", "b")
    await s.round("r3", "c")

    await cm.collapse_task_layer(_l3_state(), _ctx(mem), 3, "digest")

    view = await _task_view(mem)
    raw_kept = [r for r in _rendered(view) if not r.metadata.get("collapsed")]
    assert len(raw_kept) >= 3, f"应至少保留 3 条可见记录，实际 {[r.content for r in raw_kept]}"
    assert not _orphans(view)


async def test_l3_single_round_is_left_alone_without_spending_a_summary():
    """整个 task 只有一轮（一次并行调三个工具）：按回合切就没有可折的东西。

    此时必须 no-op，**且不调摘要 LLM**——否则就是白烧一次调用，把 USER_PROMPT 换成
    「USER_PROMPT + 摘要」，token 反而变多。摘要以零参函数传入，只在确定要折时才求值。
    """
    mem = InMemoryMemoryProvider()
    s = _Seeder(mem)
    await s.user("refactor X")
    await s.round("all at once", "a", "b", "c")
    calls: list[str] = []

    async def _summary():
        calls.append("summarize")
        return "digest"

    n = await cm.collapse_task_layer(_l3_state(), _ctx(mem), 3, _summary)

    assert n == 0
    assert calls == [], "没有东西可折时不得调摘要"
    # 对照：同一个函数在有东西可折时确实会被调用
    await s.round("second round", "d")
    await s.round("third round", "e")
    assert await cm.collapse_task_layer(_l3_state(), _ctx(mem), 3, _summary) > 0
    assert calls == ["summarize"]


async def test_escalating_passes_l3_summary_lazily(monkeypatch):
    """`escalating_compact` 交给 L3 的是**取摘要的函数**，不是先算好的文本（同 L1 的做法）。

    先算好有两个代价：按回合切后 L3 可能 no-op，摘要白算；以及折区图片降级发生在
    `collapse_task_layer` 内部，先算摘要会让摘要输入里仍是真图（L1 为此专门改过顺序）。
    """
    summarized: list[str] = []
    captured: list = []
    monkeypatch.setattr(cm, "summarize_for_compact",
                        lambda s, c, *, scope="task": (summarized.append(scope) or _const("d")))
    monkeypatch.setattr(cm, "_count_root_residues", lambda s, c: _const(0))
    monkeypatch.setattr(cm, "_kept_origin_ids", lambda s, c, k: _const([]))
    monkeypatch.setattr(cm, "_active_memory_tokens", lambda s, c: _const(900))

    async def _fake_collapse(state, ctx, keep, summary):
        captured.append(summary)
        return 0
    monkeypatch.setattr(cm, "collapse_task_layer", _fake_collapse)

    mem = InMemoryMemoryProvider()
    s = _Seeder(mem)
    await s.user("x")
    for i in range(4):
        await s.round(f"r{i}", f"c{i}")

    await cm.escalating_compact(_escalating_state(limit=1000, target=0.5), _ctx(mem),
                                token_estimate=900)

    assert len(captured) == 1 and callable(captured[0]), captured
    assert summarized == [], "L3 放弃折叠时摘要不该已经被算过"


# ── L2：从最老的胶囊开始逐颗降级 ───────────────────────────────────────────────


async def _capsule(mem, oid: str, t: int, body_chars: int):
    """一颗本 agent 亲做的 rich 胶囊：task 层 body（请求 + 段摘要）+ agent 层 finish 对。"""
    addr = MemoryAddress(session_id="s1", task_id=oid, agent_id="a1")
    ts = _BASE + timedelta(minutes=t)
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=addr,
        content=f"{oid} request " + "x" * body_chars, timestamp=ts, role="user",
        metadata={}), _pctx())
    await mem.ingest(MemoryEvent(
        kind=MemoryKind.SUMMARY, scope=MemoryScope.TASK, address=addr,
        content=f"{oid} segment summary", timestamp=ts + timedelta(seconds=1),
        role="assistant", metadata={}), _pctx())
    half = MemoryAddress(session_id="s1", agent_id="a1")
    for role, content, md in (
        ("assistant", f"{oid} act_recap",
         {"tool_calls": [{"id": f"f_{oid}", "name": "control__finish_task"}]}),
        ("tool", f"{oid} overall summary", {"tool_call_id": f"f_{oid}"}),
    ):
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=half,
            content=content, timestamp=ts + timedelta(seconds=2), role=role,
            metadata={"origin_task_id": oid, "parent_task_id": None, **md}), _pctx())


async def _is_rich(mem, oid: str) -> bool:
    addr = MemoryAddress(session_id="s1", task_id=oid, agent_id="a1")
    return bool(await mem.load_view(addr, MemoryScope.TASK, _pctx()))


def _escalating_state(*, limit, target):
    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(
            compact_keep_last=6, collapse_keep_last=3, compact_token_ratio=0.8,
            compact_target_ratio=target, compact_keep_recent_images=2),
        loop_guard=SimpleNamespace(context_limit=limit))
    return SimpleNamespace(scope=_ROOT, task=SimpleNamespace(id="root"), agent=agent,
                           session=SimpleNamespace(id="s1", tenant_id="tn"),
                           extra={}, run_id="r1", sequence_counter=0)


async def test_kept_capsules_are_ordered_oldest_first():
    """逐颗降级的前提：`_kept_origin_ids` 必须给出**从老到新**的顺序（曾是无序 set）。"""
    mem = InMemoryMemoryProvider()
    # 刻意乱序写入：顺序必须来自胶囊的时间，而不是写入先后
    await _capsule(mem, "c_old", t=1, body_chars=10)
    await _capsule(mem, "c_new", t=3, body_chars=10)
    await _capsule(mem, "c_mid", t=2, body_chars=10)

    kept = await cm._kept_origin_ids(_escalating_state(limit=1000, target=0.5), _ctx(mem), 6)

    assert list(kept) == ["c_old", "c_mid", "c_new"], kept


async def test_l2_demotes_oldest_capsule_first_and_stops_at_target():
    """降一颗就达标 → 只降最老那颗，新的保持 rich。

    三颗胶囊 body 各 ~1000 token（HeuristicTokenizer 口径 len//4）。target=0.6×5000=3000，
    估算从 3400 起：降掉最老一颗（-1000）即 < 3000，必须就此停下。原先的整批降级会把三颗
    一起掏空。
    """
    mem = InMemoryMemoryProvider()
    await _capsule(mem, "c1", t=1, body_chars=4000)
    await _capsule(mem, "c2", t=2, body_chars=4000)
    await _capsule(mem, "c3", t=3, body_chars=4000)

    events = await cm.escalating_compact(_escalating_state(limit=5000, target=0.6), _ctx(mem),
                                         token_estimate=3400)

    assert not await _is_rich(mem, "c1"), "最老一颗应被降成 lean"
    assert await _is_rich(mem, "c2"), "达标后不得继续降"
    assert await _is_rich(mem, "c3"), "最新一颗不得被降"
    l2 = [e.payload for e in events if e.payload.get("source") == "demote_lean"]
    assert len(l2) == 1 and l2[0]["demoted_capsules"] == 1, l2


async def test_l2_keeps_going_until_target_when_one_is_not_enough():
    """对照：一颗不够就接着降第二颗，仍然按从老到新的顺序。"""
    mem = InMemoryMemoryProvider()
    await _capsule(mem, "c1", t=1, body_chars=4000)
    await _capsule(mem, "c2", t=2, body_chars=4000)
    await _capsule(mem, "c3", t=3, body_chars=4000)

    events = await cm.escalating_compact(_escalating_state(limit=5000, target=0.4), _ctx(mem),
                                         token_estimate=3400)

    assert not await _is_rich(mem, "c1") and not await _is_rich(mem, "c2")
    assert await _is_rich(mem, "c3"), "两颗就够时第三颗（最新）必须保住"
    l2 = [e.payload for e in events if e.payload.get("source") == "demote_lean"]
    assert len(l2) == 1 and l2[0]["demoted_capsules"] == 2, l2


async def _const(v):
    return v
