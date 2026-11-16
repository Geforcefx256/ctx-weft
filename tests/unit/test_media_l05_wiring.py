"""Phase 4 Task 5：L0.5 接入 `escalating_compact` + §6.1 L1/L3 折叠前置降级。

前四个任务造的零件（占位编解码 / 降级 / 带 part 的工具结果 / get_image）在此之前
**谁都没被真正调用**。本文件钉住「接上电」的四件事：

1. L0.5 跑在 L1 **之前**（子设计 §6：无 LLM、单位收益最高、且可逆——L1/L2/L3 折的是
   记录本身，一旦执行位置就没了，所以先花可逆的额度）；
2. L0.5 产出 `MemoryCompacted(source="demote_images")` 且 `freed_tokens` **真的 > 0**
   （Phase 0 之前 `_active_memory_tokens` 不计图片，这里恒为 0，L0.5 等于白跑）；
3. §6.1：`collapse_task_layer` 折叠前无条件降级折区内的残留真图，且**降级之后重新
   `load_view`**——降级换掉了 record id，拿旧 id 去 fold 会让同一段对话出现两次，
   而摘要输入里仍是真图、照旧被 `content_to_text` 静默拍扁；
4. 未接 `MemoryBlobStore` 时行为与改造前一致（一次 `fold()` 都不发、没有 L0.5 事件）。

⚠️ 断言口径：本文件里「某件事没有发生」型断言（不升级到 L1 / 不降级 / 不重复）
一律配一个「确实发生了」的对照，写在同一个用例内；Step 6 另做变异验证。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ctx_weft.providers.memory import organize as memory_ops
import pytest

from ctx_weft.core.utils.content import collect_blob_refs
from tests import compaction_testkit as cm
from ctx_weft.core.media import demote_for_budget, get_image
from ctx_weft.core.media.refs import (
    decode_image_placeholder,
    encode_image_placeholder,
    find_image_placeholders,
)
from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols import (
    ImagePart,
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    TextPart,
    load_records,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

_BASE = datetime(2026, 8, 27, tzinfo=UTC)
_SCOPE = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")

# ── 脚手架 ────────────────────────────────────────────────────────────────────


def _ref(n: int) -> str:
    return f"blob:{n:064x}"


def _img(n: int) -> ImagePart:
    """ref 形态的图：byte_size=4096 → image_tokens 落在下界 1600（Phase 0/3c 口径）。"""
    return ImagePart(data=_ref(n), media_type="image/png", source_type="ref",
                     byte_size=4096)


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="tn")


class _CountingMemory(InMemoryMemoryProvider):
    """数 fold 次数——「未接 MemoryBlobStore 一次都不写」需要能观测到写。"""

    def __init__(self) -> None:
        super().__init__()
        self.fold_calls = 0

    async def fold(self, supersede_ids, replacements, ctx):
        self.fold_calls += 1
        return await super().fold(supersede_ids, replacements, ctx)

    async def _commit_organize(self, ids, events, ctx):
        self.fold_calls += 1
        return await super()._commit_organize(ids, events, ctx)


class _Blobs:
    """能外部化的 blob store 桩（`can_externalize=True` 是 L0.5 的总闸）。"""

    can_externalize = True

    def __init__(self, data: dict[str, bytes] | None = None) -> None:
        self._data = data or {}

    async def put(self, data, media_type, ctx):  # pragma: no cover - 本文件不写入
        raise NotImplementedError

    async def get(self, ref, ctx):
        raw = self._data.get(ref)
        return (raw, "image/png") if raw is not None else None


def _state(*, target_ratio=0.5, limit=10000, collapse_keep=99, keep_recent_images=2):
    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(
            compact_keep_last=6, collapse_keep_last=collapse_keep,
            compact_token_ratio=0.8, compact_target_ratio=target_ratio,
            compact_keep_recent_images=keep_recent_images),
        loop_guard=SimpleNamespace(context_limit=limit))
    return SimpleNamespace(resolved_model=model_snapshot(context_limit=limit), scope=_SCOPE, task=SimpleNamespace(id="t1"), agent=agent,
                           session=SimpleNamespace(id="s1", tenant_id="tn"),
                           extra={}, run_id="r1", sequence_counter=0)


def _ctx(memory, *, blobs: _Blobs | None = None):
    tokenizer = SimpleNamespace(count=lambda text: max(1, len(text) // 4))
    return SimpleNamespace(memory=memory, provider_ctx=_pctx(),
                           llm=SimpleNamespace(tokenizer=tokenizer),
                           blob_store=blobs)


async def _seed(mem, contents, *, role="user", address=_SCOPE):
    ids = []
    for i, content in enumerate(contents):
        ids.append(await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=address,
            content=content, timestamp=_BASE + timedelta(seconds=i), role=role,
            metadata={"i": i}), _pctx()))
    return ids


async def _view(mem, address=_SCOPE):
    return await load_records(mem, address, MemoryScope.TASK, _pctx(),
                               kinds=cm._TASK_VIEW_KINDS)


def _all_text(records) -> str:
    out = []
    for r in records:
        c = r.content
        out.append(c if isinstance(c, str) else "".join(
            getattr(p, "text", "") for p in c))
    return "\n".join(out)


async def _const(v):
    return v


# ── 1. L0.5 跑在 L1 之前 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l05_runs_before_l1(monkeypatch):
    """顺序钉死：把 L0.5 挪到 L1 之后，本条必红。

    L1 的 guard 被强行放行（`_count_root_residues` → 99），且预算门保持敞开
    （est 远高于 target），故两级都**确实跑到**——记录到的是真顺序，不是「L1 没跑」。
    """
    mem = _CountingMemory()
    await _seed(mem, [[TextPart(text="look"), _img(1)], [_img(2)], [_img(3)],
                      [TextPart(text="ok")]])

    from tests.unit.test_compact_history_priority import capsule
    state, ctx = _state(target_ratio=0.01), _ctx(mem, blobs=_Blobs())
    await capsule(state, ctx, "old", 0)
    order: list[str] = []
    real = memory_ops.organize
    async def spy_organize(*args, **kwargs):
        result = await real(*args, **kwargs)
        if kwargs["operation"] == "demote_images" and kwargs.get("record_ids") is None:
            order.append("L0.5")
        elif kwargs["operation"] == "reduce_history" and result.changed:
            order.append("L1")
        return result
    monkeypatch.setattr(memory_ops, "organize", spy_organize)
    monkeypatch.setattr(cm, "_count_root_residues", lambda s, c: _const(99))
    monkeypatch.setattr(cm, "_kept_origin_ids", lambda s, c, k: _const(["old"]))
    cm.patch_summary(monkeypatch,
                        lambda s, c, *, scope="task": _const("S"))

    events = await cm.escalating_compact(
        state, ctx, token_estimate=9000, trigger="compact")

    assert order == ["L0.5", "L1"], order
    # 对照：两级都真的产出了事件（否则「顺序对」可能只是因为某级没跑）
    assert [e.payload.get("source") for e in events
            if e.type == "MemoryCompacted"][:2] == ["demote_images", "demote_lean"]


# ── 2./3. 事件形状 + freed_tokens 真的 > 0 ────────────────────────────────────


@pytest.mark.asyncio
async def test_l05_emits_memory_compacted_with_real_freed_tokens():
    """L0.5 事件 `source="demote_images"`，且 `freed_tokens` **不是 0**。

    Phase 0 之前 `_active_memory_tokens` 把图片一律算 0，这里恒为 0 → est 不减 →
    编排误判本级白跑而继续升级。故这条断的是「> 0」而不是「>= 0」，并进一步要求
    它落在「两张图（各 1600）减去两条占位文本」的量级上。
    """
    mem = _CountingMemory()
    await _seed(mem, [[TextPart(text="a"), _img(1)], [_img(2)],
                      [TextPart(text="b"), _img(3)], [_img(4)]])

    events = await cm.escalating_compact(
        _state(), _ctx(mem, blobs=_Blobs()), token_estimate=9000, trigger="compact")

    l05 = [e for e in events if e.type == "MemoryCompacted"
           and e.payload.get("source") == "demote_images"]
    assert len(l05) == 1
    p = l05[0].payload
    assert p["demoted_images"] == 2 and p["superseded_count"] == 2   # keep_recent=2
    assert p["layer"] == "task" and p["trigger"] == "compact"
    assert p["freed_tokens"] > 3000, p            # 2 × 1600 − 两条占位的文本 token
    # 视图侧对照：确实是「最早两张」变成了可解码的占位，最近两张仍是真图
    view = await _view(mem)
    found = find_image_placeholders(_all_text(view))
    assert [r for r, _ in found] == [_ref(1), _ref(2)]
    assert sum(1 for r in view for p_ in (r.content or [])
               if not hasattr(p_, "text")) == 2


# ── 4. L0.5 顶用时不再升级到 L1 ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l05_alone_can_stop_escalation(monkeypatch):
    """est 被 L0.5 压到 target 以下 → 不进 L1。

    L1 的 guard 被强行放行，所以「没进 L1」的唯一可能原因就是预算已达标；
    对照断言：同一批种子在 `keep_recent` 大到一张都不降时，L1 **确实会跑**。
    """
    calls: list[str] = []

    real = memory_ops.organize
    async def spy_organize(*args, **kwargs):
        result = await real(*args, **kwargs)
        if kwargs["operation"] == "reduce_history" and result.changed:
            calls.append("L1")
        return result
    monkeypatch.setattr(memory_ops, "organize", spy_organize)
    monkeypatch.setattr(cm, "_count_root_residues", lambda s, c: _const(99))
    monkeypatch.setattr(cm, "_kept_origin_ids", lambda s, c, k: _const(["old"]))
    cm.patch_summary(monkeypatch,
                        lambda s, c, *, scope="task": _const("S"))

    seed = [[TextPart(text="a"), _img(1)], [_img(2)], [_img(3)], [_img(4)]]

    mem = _CountingMemory()
    await _seed(mem, seed)
    from tests.unit.test_compact_history_priority import capsule
    state = _state()
    state.scope = MemoryAddress(session_id="s1", agent_id="a1", task_id="cur")
    await capsule(state, _ctx(mem, blobs=_Blobs()), "old", 0)
    # eff=10000, target_ratio=0.5 → target=5000；est 6000 − freed(≈3100) < 5000
    events = await cm.escalating_compact(
        state, _ctx(mem, blobs=_Blobs()), token_estimate=6000, trigger="compact")
    assert calls == []
    assert events[-1].payload["levels"] == ["demote_images"]
    assert events[-1].payload["est_after"] < 5000

    # 对照：keep_recent 大到一张都不降 → est 不动 → 照旧升级到 L1
    mem2 = _CountingMemory()
    await _seed(mem2, seed)
    await capsule(state, _ctx(mem2, blobs=_Blobs()), "old", 0)
    await cm.escalating_compact(
        _state(keep_recent_images=99), _ctx(mem2, blobs=_Blobs()),
        token_estimate=6000, trigger="compact")
    assert calls == ["L1"]


@pytest.mark.asyncio
async def test_keep_recent_images_is_read_from_loop_config():
    """`keep_recent` 从 `loop_config` 读，不是硬编码的 2。"""
    for keep, expect_demoted in ((0, 3), (1, 2), (3, 0)):
        mem = _CountingMemory()
        await _seed(mem, [[_img(1)], [_img(2)], [_img(3)]])
        events = await cm.escalating_compact(
            _state(keep_recent_images=keep), _ctx(mem, blobs=_Blobs()),
            token_estimate=9000, trigger="compact")
        got = [e.payload["demoted_images"] for e in events
               if e.type == "MemoryCompacted" and e.payload.get("source") == "demote_images"]
        assert got == ([expect_demoted] if expect_demoted else []), keep


# ── 5./6. §6.1：L3 折叠前置降级 + 降级后重新 load_view ────────────────────────


async def _seed_for_collapse(mem):
    """折区（最早 3 条）里放一张真图——它正是 L0.5 的 `keep_recent` 保下来的那种。"""
    await _seed(mem, [[TextPart(text="ORIGINAL-MSG"), _img(7)]])
    await _seed(mem, [[TextPart(text=f"turn-{i}")] for i in range(1, 5)],
                role="assistant")


@pytest.mark.asyncio
async def test_collapse_demotes_images_in_fold_range_before_folding():
    """§6.1（第 5 条）：折区里的真图先降级，ref 随「原始消息」节活下来。

    不降级的话取 `original` 节走 `content_to_text`，这张图会被**静默拍扁**——而它恰
    是最新的那几张之一，结果是老图留下可取回的占位、最新的图彻底消失。
    """
    mem = _CountingMemory()
    await _seed_for_collapse(mem)

    n = await cm.collapse_task_layer(
        _state(), _ctx(mem, blobs=_Blobs()), 2, "SUMMARY-TEXT")
    assert n > 0

    view = await _view(mem)
    collapsed = [r for r in view if r.metadata.get("collapsed")]
    assert len(collapsed) == 1
    original = cm._original_section(collapsed[0].content)
    assert "ORIGINAL-MSG" in original
    # ref 确实活在「原始消息」节里，且解得回来（不是被拍扁成空）
    assert decode_image_placeholder(original) == (_ref(7), "image/png")


async def _seed_two_images_for_collapse(mem):
    """折区（最早 3 条）里两条 user 回合各一张真图。

    坍缩物的「原始消息」节只取**折区最早那条 user 回合**，故 `_ref(7)` 的占位会随它
    活下来、`_ref(8)` 的不会——这正是「声明边界必须与可取回边界重合」的检验样本。
    """
    await _seed(mem, [[TextPart(text="ORIGINAL-MSG"), _img(7)]])
    await _seed(mem, [[TextPart(text="SECOND-MSG"), _img(8)]])
    await _seed(mem, [[TextPart(text=f"turn-{i}")] for i in range(1, 4)],
                role="assistant")


async def _live_refs(mem) -> set[str]:
    """活引用集的等价物：`SqlMemoryProvider.live_blob_refs` 就是对未 supersede 的记录
    逐条取 `collect_blob_refs` 的并集（provider 建引用边只走这个真源）。"""
    refs: set[str] = set()
    for r in await _view(mem):
        refs |= set(collect_blob_refs(r))
    return refs


@pytest.mark.asyncio
async def test_collapse_declares_the_placeholder_ref_it_carried_over():
    """🔴 坍缩物必须**声明**它带过来的那个 ref，否则字节会被 GC 收走而占位还在。

    上一条钉的是「ref 随 `original` 节活下来」——那只保住了**文本**。字节的死活由另一
    条判据决定：`collect_blob_refs` 只看结构化字段，刻意绝不解析占位文案。坍缩把原记录
    全部 supersede，占位却是文本，不显式声明 → 该 ref 掉出活引用集 → 宽限期一过字节被
    回收，而占位仍在视图里、`get_image` 仍定位得到它，换回来的是 `[image unavailable]`。
    这是缺陷 2026-08-27（`fold._rebuild` 的 `blob_refs` 累积）在折叠侧的同一个失败面。

    对照（防永真）：`_ref(8)` 的占位**没有**进坍缩物，故它不该被声明——声明边界必须与
    「还能被 get_image 取回的边界」严格重合，宽一格就是字节永远回收不掉。
    """
    mem = _CountingMemory()
    await _seed_two_images_for_collapse(mem)
    assert await _live_refs(mem) == {_ref(7), _ref(8)}      # 折叠前两张都活着

    await cm.collapse_task_layer(
        _state(), _ctx(mem, blobs=_Blobs()), 2, "SUMMARY-TEXT")

    view = await _view(mem)
    collapsed = [r for r in view if r.metadata.get("collapsed")]
    assert len(collapsed) == 1
    # 占位确实幸存在坍缩物里（前提，非永真）
    assert find_image_placeholders(collapsed[0].content) == [(_ref(7), "image/png")]
    # 且它被显式声明了 —— 修复前这里是 set()
    assert await _live_refs(mem) == {_ref(7)}


@pytest.mark.asyncio
async def test_get_image_still_works_after_collapse_and_a_real_sweep():
    """端到端：坍缩 → 按活引用集真扫一遍字节 → 幸存占位仍取得回图。

    上一条断的是 mark 输入，这一条断它**换来的结果**：sweep 之后 `get_image` 对
    `_ref(7)` 仍返回 `ImagePart`（而不是 `[image unavailable]`），对没能幸存的
    `_ref(8)` 则诚实地说「本 task 里没有这个 ref」——两种失败模式分得开。
    """
    mem = _CountingMemory()
    await _seed_two_images_for_collapse(mem)
    blobs = _Blobs({_ref(7): b"SEVEN", _ref(8): b"EIGHT"})

    await cm.collapse_task_layer(_state(), _ctx(mem, blobs=blobs), 2, "SUMMARY-TEXT")

    # mark-sweep：活引用集之外的字节被回收（`FsBlobStore.collect` 的判据一，宽限期已过）
    live = await _live_refs(mem)
    for ref in list(blobs._data):
        if ref not in live:
            del blobs._data[ref]

    parts = await get_image(mem, _SCOPE, _pctx(), _ref(7), blob_store=blobs,
                            kinds=cm._TASK_VIEW_KINDS)
    assert any(not hasattr(p, "text") for p in parts), parts      # 图真的回来了

    parts8 = await get_image(mem, _SCOPE, _pctx(), _ref(8), blob_store=blobs,
                             kinds=cm._TASK_VIEW_KINDS)
    assert all(hasattr(p, "text") for p in parts8), parts8
    assert "present in this task" in "".join(p.text for p in parts8)


@pytest.mark.asyncio
async def test_l1_summary_declares_a_ref_only_when_it_actually_echoed_one(monkeypatch):
    """L1 的摘要是 LLM 写的：抄了 ref 就得声明，没抄就不该凭空声明。

    两半都断，因为这一处的 `placeholder_refs` 在健康数据上恒返回空表——只断「没抄不
    声明」会是一条永真断言。
    """
    mem = _CountingMemory()
    await _seed_two_top_units(mem)
    ctx = _ctx(mem, blobs=_Blobs())
    state = _l1_state()

    # ① 摘要里逐字抄了折区那个占位 → 必须声明
    ph = encode_image_placeholder(_ref(20), "image/png")
    await cm.fold_root_experience(state, ctx, 1, lambda: _const(f"SUMMARY {ph}"))
    agent_view = await load_records(mem, _AGENT_HALF, MemoryScope.AGENT, _pctx())
    summaries = [r for r in agent_view if r.kind is MemoryKind.SUMMARY]
    assert len(summaries) == 1
    assert set(collect_blob_refs(summaries[0])) == {_ref(20)}

    # ② 对照：纯文本摘要不声明任何 ref
    mem2 = _CountingMemory()
    await _seed_two_top_units(mem2)
    await cm.fold_root_experience(
        _l1_state(), _ctx(mem2, blobs=_Blobs()), 1, lambda: _const("PLAIN SUMMARY"))
    agent_view2 = await load_records(mem2, _AGENT_HALF, MemoryScope.AGENT, _pctx())
    s2 = [r for r in agent_view2 if r.kind is MemoryKind.SUMMARY]
    assert len(s2) == 1 and collect_blob_refs(s2[0]) == []


@pytest.mark.asyncio
async def test_collapse_reloads_view_after_demote_so_history_is_not_duplicated():
    """🔴 第 6 条（防静默失效）：`demote_all` 换掉了 record id，之后**必须重新
    `load_view`**。

    不重新加载的话，下面那次 `fold()` 拿的是降级**前**的 id：被降过的那条一条都
    supersede 不掉，于是同一段对话在视图里出现两次（旧记录的降级版还活着，坍缩物里
    又抄了一份原文）；且手里那份旧 records 里仍是真图，取 `original` 节照旧被拍扁。

    两条断言分别钉这两个后果，且都**非永真**：
    - `ORIGINAL-MSG` 恰好出现一次——「什么都不做」（不坍缩）时它也只出现一次，故补
      `collapsed` 记录必须存在、且折区记录必须真的消失了；
    - `original` 节里解得出 ref——旧 records 走 `content_to_text` 解不出。
    """
    mem = _CountingMemory()
    await _seed_for_collapse(mem)
    before = await _view(mem)
    assert len(before) == 5

    await cm.collapse_task_layer(_state(), _ctx(mem, blobs=_Blobs()), 2, "SUMMARY-TEXT")

    view = await _view(mem)
    # 坍缩确实发生了（对照，防「什么都没做」）
    collapsed = [r for r in view if r.metadata.get("collapsed")]
    assert len(collapsed) == 1
    # 同一段对话没有出现两次（不重新 load_view 时这里恰好是 2：被降级的那条 supersede
    # 不掉、活了下来，坍缩物里又抄了一份原文）
    assert _all_text(view).count("ORIGINAL-MSG") == 1, _all_text(view)
    # 折区 3 条 + 保留 2 条 → 坍缩物 1 条 + 保留 2 条
    assert len(view) == 3, [str(r.content)[:60] for r in view]
    # 且用的是**降级后**的内容：ref 在，真图不在
    assert decode_image_placeholder(cm._original_section(collapsed[0].content)) \
        == (_ref(7), "image/png")
    assert not any(not hasattr(p, "text")
                   for r in view if not isinstance(r.content, str)
                   for p in (r.content or []))


@pytest.mark.asyncio
async def test_fold_root_experience_demotes_its_range_before_folding(monkeypatch):
    """§6.1 的另一半：`fold_root_experience` 也在动手前降一次自己的折区。"""
    mem = _CountingMemory()
    ranges: list[list[str]] = []
    real = memory_ops._demote_images

    async def spy(memory, pctx, live, ids, keep):
        ranges.append(list(ids))
        return await real(memory, pctx, live, ids, keep)

    monkeypatch.setattr(memory_ops, "_demote_images", spy)

    # 两个已完成顶层单元（各一条 finish 对 assistant 回合）+ 各自的 task 层胶囊
    for i, tid in enumerate(("old", "new")):
        addr = MemoryAddress(session_id="s1", task_id=tid, agent_id="a1")
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=addr,
            content=[TextPart(text=f"body-{tid}"), _img(20 + i)],
            timestamp=_BASE + timedelta(seconds=i), role="user",
            metadata={"task_id": tid}), _pctx())
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
            address=MemoryAddress(session_id="s1", agent_id="a1"),
            content=f"done-{tid}", timestamp=_BASE + timedelta(seconds=i),
            role="assistant",
            metadata={"origin_task_id": tid, "parent_task_id": None,
                      "tool_calls": [{"name": "finish_task"}]}), _pctx())

    state = _state()
    state.scope = MemoryAddress(session_id="s1", task_id="cur", agent_id="a1")
    n = await cm.fold_root_experience(state, _ctx(mem, blobs=_Blobs()), 1, "EXP")

    assert n > 0
    assert ranges and ranges[0], "折区为空 → 前置降级没被真的调用"
    # 折走的是最老那个单元（old）；它的图在被折之前先变成了占位
    half = MemoryAddress(session_id="s1", agent_id="a1")
    remaining = await load_records(mem, half, MemoryScope.TASK, _pctx(),
                                    kinds=cm._TASK_VIEW_KINDS)
    assert [r.address.task_id for r in remaining] == ["new"]


# ── 7. 未注册 MemoryBlobStore：行为与改造前一致 ────────────────────────────────────


@pytest.mark.asyncio
async def test_without_blob_store_l05_is_a_noop(monkeypatch):
    """未接 `MemoryBlobStore` → 一次 `fold()` 都不发、没有 L0.5 事件、内容逐字节不变。

    对照写在同一用例内：同样的种子接上 blob store 后**确实**降级并写了 memory——
    否则「什么都没做」的实现也能让上半段通过。
    """
    monkeypatch.setattr(cm, "_count_root_residues", lambda s, c: _const(0))
    monkeypatch.setattr(cm, "_kept_origin_ids", lambda s, c, k: _const(set()))
    seed = [[TextPart(text="a"), _img(1)], [_img(2)], [_img(3)]]

    mem = _CountingMemory()
    await _seed(mem, seed)
    events = await cm.escalating_compact(
        _state(), _ctx(mem, blobs=None), token_estimate=9000, trigger="compact")

    assert mem.fold_calls == 0
    assert not [e for e in events if e.type == "MemoryCompacted"
                and e.payload.get("source") == "demote_images"]
    view = await _view(mem)
    assert [p.data for r in view for p in r.content if not hasattr(p, "text")] \
        == [_ref(1), _ref(2), _ref(3)]

    # 对照：接上 blob store 的同一批种子确实被降级、确实写了 memory
    mem2 = _CountingMemory()
    await _seed(mem2, seed)
    events2 = await cm.escalating_compact(
        _state(), _ctx(mem2, blobs=_Blobs()), token_estimate=9000, trigger="compact")
    assert mem2.fold_calls > 0
    assert [e.payload["source"] for e in events2 if e.type == "MemoryCompacted"] \
        == ["demote_images"]


@pytest.mark.asyncio
async def test_null_blob_store_is_treated_as_unregistered():
    """`NullMemoryBlobStore`（`can_externalize=False`）与未注册同路——生产里 registry 给的
    正是它，不是 `None`。"""
    mem = _CountingMemory()
    await _seed(mem, [[_img(1)], [_img(2)], [_img(3)]])
    null = _Blobs()
    null.can_externalize = False

    events = await cm.escalating_compact(
        _state(), _ctx(mem, blobs=null), token_estimate=9000, trigger="compact")

    assert mem.fold_calls == 0
    assert not [e for e in events if e.type == "MemoryCompacted"
                and e.payload.get("source") == "demote_images"]


# ── 8. 裁定 R2：衰减之后 get_image 给出诚实的「不在这里」 ─────────────────────


@pytest.mark.asyncio
async def test_get_image_is_honest_after_the_placeholder_is_folded_away():
    """裁定 R2（子设计 §6 优先于 §5）：图被 L1/L3 折进摘要后不再提供取回。

    这条验的是**那条路径是诚实的**——模型得到清楚的「本 task 里没有这个 ref」，
    而不是崩溃、也不是错图。对照：占位还在时同一次调用**确实**返回 ImagePart，
    且 blob store 里的字节自始至终都在（判据是「本视图占位里有没有它」，
    不是「blob 存不存在」）。
    """
    mem = _CountingMemory()
    ids = await _seed(mem, [[TextPart(text="x"), _img(9)]])
    blobs = _Blobs({_ref(9): b"PNGBYTES"})

    # 先降级，占位落库 → 取得回来
    assert await demote_for_budget(mem, _SCOPE, _pctx(), keep_recent=0,
                                   kinds=cm._TASK_VIEW_KINDS) == 1
    parts = await get_image(mem, _SCOPE, _pctx(), _ref(9), blob_store=blobs,
                            kinds=cm._TASK_VIEW_KINDS)
    assert any(not hasattr(p, "text") for p in parts), parts

    # 衰减：承载占位的记录被折进摘要（纯遗忘，模拟 L1）
    view = await _view(mem)
    await mem.fold([r.id for r in view], [], _pctx())
    assert ids  # 原 id 早已被降级换掉，这里只是记明「折的是新 id」

    parts = await get_image(mem, _SCOPE, _pctx(), _ref(9), blob_store=blobs,
                            kinds=cm._TASK_VIEW_KINDS)
    assert all(hasattr(p, "text") for p in parts), parts        # 没有错图
    text = "".join(p.text for p in parts)
    assert _ref(9) in text and "present in this task" in text   # 说得清楚
    assert await blobs.get(_ref(9), _pctx()) is not None        # 字节其实还在


# ── 9. Task 5b：§6.1 对 L1 真的生效——降级必须早于摘要（修 P4-L10） ───────────
#
# Task 5 按简报原文把 `demote_all` 放进了 `fold_root_experience` 内部，而调用方是
# **先** `summarize_for_compact` 算好摘要、**再**把它传进来。于是被 `keep_recent`
# 保住的那几张最新的图先被 `content_to_text` 无痕拍扁进摘要输入（连占位都没有、
# ref 彻底没了），紧接着记录被 supersede——正是 §6.1 开头要防的「优先级完全颠倒」。
#
# 下面三条断的分别是：摘要输入里逐字有 ref（不是「摘要非空」这种永真）／降级确实排在
# 摘要之前／降级省下的 token 归属正确（不重复计入、也不丢）。


_AGENT_HALF = MemoryAddress(session_id="s1", agent_id="a1")


async def _seed_two_top_units(mem):
    """两个已完成顶层单元 old/new：各一条**含真图**的 task 层胶囊 + 一条 agent 层 finish 对。

    两张图合计 2 张 = `keep_recent_images=2` 的名额，故 L0.5 一张都不降——正是简报要求的
    「被 keep_recent 保住的最新的图落进 L1 折叠范围」。
    """
    for i, tid in enumerate(("old", "new")):
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=MemoryAddress(session_id="s1", task_id=tid, agent_id="a1"),
            content=[TextPart(text=f"body-{tid}"), _img(20 + i)],
            timestamp=_BASE + timedelta(seconds=i), role="user",
            metadata={"task_id": tid}), _pctx())
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
            address=_AGENT_HALF, content=f"done-{tid}",
            timestamp=_BASE + timedelta(seconds=i), role="assistant",
            metadata={"origin_task_id": tid, "parent_task_id": None,
                      "tool_calls": [{"name": "finish_task"}]}), _pctx())


def _l1_state(**kw):
    """当前 task 是 `cur`（不可折），keep_last=1 → 只折最老那个单元 `old`。"""
    st = _state(**kw)
    st.scope = MemoryAddress(session_id="s1", task_id="cur", agent_id="a1")
    st.agent.loop_config.compact_keep_last = 1
    return st


async def _summary_input_text(mem) -> str:
    """摘要输入的等价物：装配器读的就是这两个视图，每条 content 过 `content_to_text`
    （真图在这一步被静默拍扁，占位则是文本、原样留下）。"""
    recs = await load_records(mem, _AGENT_HALF, MemoryScope.AGENT, _pctx())
    recs += await load_records(mem, _AGENT_HALF, MemoryScope.TASK, _pctx(),
                                kinds=cm._TASK_VIEW_KINDS)
    return "\n".join(r.content if isinstance(r.content, str) else content_to_text(r.content)
                     for r in recs)


@pytest.mark.asyncio
async def test_l1_summary_input_holds_the_placeholder_not_a_vanished_image(monkeypatch):
    """🔴 本任务存在的理由：L1 的**摘要输入**里，折区那张最新的图是**含 ref 的占位**。

    逐字断言 ref 本身出现在摘要输入里——「摘要输入非空」之类是永真的。
    修复前（摘要先算、降级后跑）这里逐字得到的是 `body-old` 而 ref 无影无踪。
    """
    mem = _CountingMemory()
    await _seed_two_top_units(mem)

    captured: dict[str, str] = {}

    async def _fake_summ(state, ctx, *, scope="task"):
        captured[scope] = await _summary_input_text(mem)
        return f"SUM-{scope}"

    cm.patch_summary(monkeypatch, _fake_summ)

    events = await cm.escalating_compact(
        _l1_state(target_ratio=0.8), _ctx(mem, blobs=_Blobs()),
        token_estimate=9000, trigger="compact")

    sources = [e.payload["source"] for e in events if e.type == "MemoryCompacted"]
    # 前提：L0.5 一张都没降（两张图正好被 keep_recent=2 保住）→ 图是「最新的」那种
    assert "demote_images" not in sources, sources
    # 对照：L1 **确实**跑了并折了东西（否则下面的断言可能只是因为什么都没发生）
    assert sources == ["root_experience"], sources

    text = captured["agent"]
    assert "body-old" in text                                  # 折区那条确实进了摘要输入
    assert _ref(20) in text, text                              # ← 逐字：ref 还在
    assert find_image_placeholders(text) == [(_ref(20), "image/png")], text
    # 折区那条随后就被 supersede 了，摘要文本是它唯一的痕迹——`ref` 在里面就是全部证据
    body = await load_records(mem, _AGENT_HALF, MemoryScope.TASK, _pctx(),
                               kinds=cm._TASK_VIEW_KINDS)
    assert [r.address.task_id for r in body] == ["new"]
    # 折区外那张（`new`，记录并不消失）仍是真图 —— §6.1 只降自己的折区，
    # 也说明上面 `text` 里那个 ref 不可能来自它（真图过 content_to_text 一点不剩）
    assert any(not hasattr(p, "text") for p in body[0].content)
    assert _ref(21) not in text


@pytest.mark.asyncio
async def test_l1_demotion_runs_before_the_summary_is_generated(monkeypatch):
    """顺序钉死在 `fold_root_experience` 内部：两次 `demote_all` 都在摘要求值之前。

    调用形态因此改成「传取摘要的函数」而不是摘要文本——折区 ids 只有函数内部算得出，
    所以是把摘要推迟进去，而不是把降级提到外面。
    """
    mem = _CountingMemory()
    await _seed_two_top_units(mem)

    order: list[str] = []
    real = memory_ops._demote_images

    async def spy_demote(*args, **kwargs):
        order.append("demote")
        return await real(*args, **kwargs)

    monkeypatch.setattr(memory_ops, "_demote_images", spy_demote)

    seen: list[str] = []

    async def _summary() -> str:
        order.append("summarize")
        seen.append(await _summary_input_text(mem))
        return "EXP"

    n = await cm.fold_root_experience(
        _l1_state(), _ctx(mem, blobs=_Blobs()), 1, _summary)

    assert n > 0                                   # 对照：确实折了
    assert order == ["demote", "summarize"], order  # one selected mixed-scope transformation
    assert seen and _ref(20) in seen[0], seen      # 摘要看到的是占位不是空白
    # 摘要真的落进了新的 AGENT_COMPACT_SUMMARY（证明 `_summary()` 的返回值被用上）
    agent_recs = await load_records(mem, _AGENT_HALF, MemoryScope.AGENT, _pctx())
    assert [r.content for r in agent_recs if r.kind is MemoryKind.SUMMARY] == ["EXP"]


@pytest.mark.asyncio
async def test_l1_freed_tokens_account_for_the_demotion_exactly_once(monkeypatch):
    """freed 归属：降级发生在 L1 的 `_apply` **内部**，故它省下的 token 恰好计一次。

    整级（降级 + 摘要 + fold）是交给 `_apply` 的一步，`before` 仍是 L0.5 的 `after`
    ——把降级挪到 `_apply` 之外再显式重新测量，这一段就会从账上消失（est_after 偏高）；
    不重新测量而在别处又减一次，则会被重复计入。这条断的是**逐字的守恒**。
    """
    mem = _CountingMemory()
    await _seed_two_top_units(mem)
    cm.patch_summary(monkeypatch,
                        lambda s, c, *, scope="task": _const(f"SUM-{scope}"))
    state = _l1_state(target_ratio=0.8)
    ctx = _ctx(mem, blobs=_Blobs())
    before = await cm._active_memory_tokens(state, ctx)
    events = await cm.escalating_compact(state, ctx, token_estimate=9000,
                                         trigger="compact")
    after = await cm._active_memory_tokens(state, ctx)

    folded = [e for e in events if e.type == "MemoryCompacted"]
    assert [e.payload["source"] for e in folded] == ["root_experience"]
    fin = events[-1].payload
    assert before - after > 1500                       # 图片额度扣除 lean 收尾记录的文本开销
    assert fin["freed_tokens"] == before - after       # 不重复、不遗漏
    assert fin["est_after"] == 9000 - (before - after)
