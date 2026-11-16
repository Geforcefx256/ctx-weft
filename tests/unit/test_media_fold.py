"""Phase 4 Task 2：L0.5 降级（`core/media/policy.py` + `fold.py`）。

L0.5 把 memory 记录里的真图换成**含 ref 的文本占位并落库**，模型之后靠
`media:get_image` 取回。与 per-purpose 降级（`content.py::downgrade_images_to_text`，
不落库、占位不可回读）并存、互不替代。

本文件钉住三处判断（理由见 `policy.py` 模块 docstring）：

1. `keep_recent` 按**图片张数**数，一条记录可以只有一部分图被降；
2. 只降 `source_type == "ref"` 的图——inline base64 换成占位就永久丢失，
   这条同时兜住「`MemoryBlobStore` 未注册 → 返回 0、不写 memory」；
3. 同刻 tie 组**整段同批重写**，否则被降的那条会掉到组尾（`seq_no` 由 provider
   在 ingest 时分配，带不过去；真正保住位置的是 timestamp）。

⚠️ 「保住最近 N 张」「未注册不降级」这类断言对**什么都不做**的实现同样成立，故每处
都配了「确实降了东西」的对照断言（见各用例内注释），Step 6 另做变异验证。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.media import demote_all, demote_for_budget
from ctx_weft.core.media.fold import rebuild_demoted_record as _rebuild
from ctx_weft.core.media.policy import demotable_ref, plan_demotions
from ctx_weft.core.media.refs import decode_image_placeholder
from ctx_weft.core.utils.content import image_part_count, image_tokens
from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols import (
    ImagePart,
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    ProviderContext,
    TextPart,
    load_records,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

_BASE = datetime(2026, 8, 27, tzinfo=UTC)
_ADDR = MemoryAddress(session_id="s", task_id="t1", agent_id="a1")


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s", tenant_id="tn")


def _ref(n: int) -> str:
    return f"blob:{n:064x}"


def _img(n: int) -> ImagePart:
    return ImagePart(data=_ref(n), media_type="image/png", source_type="ref",
                     byte_size=4096)


def _b64img() -> ImagePart:
    """未接 MemoryBlobStore 时的图：inline base64，没有任何 ref 可写进占位。"""
    return ImagePart(data="ZGF0YQ==", media_type="image/png")


class _Rec:
    """policy 是纯函数——只要 id / content / timestamp 三个字段，不必造真 MemoryRecord。"""

    def __init__(self, rid, content, ts=_BASE):
        self.id, self.content, self.timestamp = rid, content, ts


class _CountingMemory(InMemoryMemoryProvider):
    """数写入次数；可让第 N 次 fold 抛异常。"""

    def __init__(self, fail_on: set[int] | None = None):
        super().__init__()
        self.fold_calls = 0
        self.ingest_calls = 0
        self._fail_on = fail_on or set()

    async def ingest(self, event, ctx):
        self.ingest_calls += 1
        return await super().ingest(event, ctx)

    async def fold(self, supersede_ids, replacements, ctx):
        self.fold_calls += 1
        if self.fold_calls in self._fail_on:
            raise RuntimeError("boom")
        return await super().fold(supersede_ids, replacements, ctx)


async def _seed(mem, contents, *, same_ts=False, role="user"):
    """按顺序写入若干条 CONVERSATION_TURN；返回 id 列表。"""
    ids = []
    for i, content in enumerate(contents):
        ts = _BASE if same_ts else _BASE + timedelta(seconds=i)
        ids.append(await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=_ADDR,
            content=content, timestamp=ts, role=role, metadata={"i": i}), _pctx()))
    return ids


async def _view(mem):
    return await load_records(mem, _ADDR, MemoryScope.TASK, _pctx())


def _make_record(*, content, blob_refs=None):
    """`_rebuild` 的直接测试用——不经 provider 造一条 `MemoryRecord`（Ruling 3）。"""
    return MemoryRecord(
        id="rec_1", type="conversation_turn", content=content,
        timestamp=_BASE, kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=_ADDR, blob_refs=list(blob_refs or []),
    )


# ── 1. policy 纯函数：选中集合 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "keep_recent, expect",
    [
        (0, {"r0": (1,), "r1": (0, 2), "r2": (0,)}),   # 全降：四张图一张不留
        (1, {"r0": (1,), "r1": (0, 2)}),               # 最后一张（r2）保住
        (2, {"r0": (1,), "r1": (0,)}),                 # r1 的第二张也保住 → 记录内部分降级
        (3, {"r0": (1,)}),
        (4, {}),                                        # 恰好等于总张数 → 一张都不降
        (99, {}),                                       # 超过总张数：仍是全保（不得反向变少）
    ],
)
def test_policy_keep_recent_counts_images_not_records(keep_recent, expect):
    """`keep_recent` 数的是**图片张数**：保护名额可以落在一条记录的中间。"""
    records = [
        _Rec("r0", [TextPart(text="a"), _img(0)]),
        _Rec("r1", [_img(1), TextPart(text="b"), _img(2)]),
        _Rec("r2", [_img(3)]),
    ]
    plan = plan_demotions(records, keep_recent=keep_recent)
    assert dict(plan.demote_indices) == expect
    assert plan.image_count == sum(len(v) for v in expect.values())


def test_policy_negative_keep_recent_is_zero():
    plan = plan_demotions([_Rec("r0", [_img(0)])], keep_recent=-5)
    assert dict(plan.demote_indices) == {"r0": (0,)}


def test_policy_protection_slot_is_spent_by_undemotable_image():
    """不可降的 base64 图照样占一个保护名额——`keep_recent` 保的是「模型还看得见的最近 N 张」。"""
    mixed = [_Rec("r0", [_img(0)]), _Rec("r1", [_b64img()])]
    # 唯一的保护名额被 r1 的 base64 图占掉 → r0 的 ref 图照降。
    # 若名额只按「可降的图」数，这里会是 {}——两种口径在此处分开。
    assert dict(plan_demotions(mixed, keep_recent=1).demote_indices) == {"r0": (0,)}
    # 对照：同位置换成可降的 ref 图时，同一个名额确实能把 r0 保住
    both_ref = [_Rec("r0", [_img(0)]), _Rec("r1", [_img(1)])]
    assert dict(plan_demotions(both_ref, keep_recent=1).demote_indices) == {"r0": (0,)}
    assert dict(plan_demotions(both_ref, keep_recent=2).demote_indices) == {}


def test_policy_only_ids_limits_scope():
    records = [_Rec("r0", [_img(0)]), _Rec("r1", [_img(1)])]
    assert dict(plan_demotions(records, keep_recent=0, only_ids={"r0"}).demote_indices) \
        == {"r0": (0,)}


def test_policy_ignores_str_content_and_text_only():
    records = [_Rec("r0", "plain"), _Rec("r1", [TextPart(text="x")])]
    assert plan_demotions(records, keep_recent=0) == plan_demotions([], keep_recent=0)


# ── 2. 只降 ref 形态的图（判断题 2） ─────────────────────────────────────────


@pytest.mark.parametrize("part, ok", [
    (_img(7), True),
    (_b64img(), False),                                              # inline base64
    (ImagePart(data="https://x/y.png", media_type="image/png", source_type="url"), False),
    (ImagePart(data="blob: 00ff", media_type="image/png", source_type="ref"), False),  # 含空白
    (ImagePart(data="blob:aa]bb", media_type="image/png", source_type="ref"), False),  # 含 ]
    (TextPart(text="hi"), False),
])
def test_demotable_ref_criterion(part, ok):
    """ref 畸形时 `fold.py` 自己先判、直接跳过——不指望 `refs.encode` 抛出来兜底。"""
    assert (demotable_ref(part) is not None) is ok


# ── 3. 降级落库：占位可解回 ref、image_tokens 归零、位置不变 ──────────────────


async def test_demote_replaces_image_with_decodable_placeholder():
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[TextPart(text="look"), _img(0)]])
    n = await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0)
    assert n == 1
    rec = (await _view(mem))[0]
    assert image_part_count(rec.content) == 0        # 真图没了
    assert image_tokens(rec.content) == 0            # 预算压力归零（freed_tokens 靠这条）
    assert decode_image_placeholder(content_to_text(rec.content)) == (_ref(0), "image/png")
    assert "look" in content_to_text(rec.content)    # 同条里的文本原样保留
    assert rec.role == "user" and rec.metadata["i"] == 0  # 语义字段照抄


async def test_demote_keeps_position_when_timestamps_differ():
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[TextPart(text="m0"), _img(0)],
                      [TextPart(text="m1")],
                      [TextPart(text="m2"), _img(1)]])
    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0) == 2
    view = await _view(mem)
    assert [content_to_text(r.content)[:2] for r in view] == ["m0", "m1", "m2"]
    # 对照：确实降了——两条都不再含真图
    assert sum(image_part_count(r.content) for r in view) == 0


async def test_demote_rewrites_whole_tie_group_to_keep_order():
    """同刻 tie 组：只降组内第一条会让它掉到组尾，故自第一条被降者起整段同批重写。

    生产里同刻不是测试造作——`finalize.py` 的派发框与其 result 共用同一个锚点 ts，
    并明写「框与 result 同锚、严格相邻」是不变量。
    """
    mem = InMemoryMemoryProvider()
    ids = await _seed(mem, [[TextPart(text="m0"), _img(0)],
                            [TextPart(text="m1")],
                            [TextPart(text="m2")]], same_ts=True)
    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0) == 1
    view = await _view(mem)
    assert [content_to_text(r.content)[:2] for r in view] == ["m0", "m1", "m2"]
    # 钉住选定行为：整组都被重写（id 全新），而不是只重写 m0
    assert all(r.id not in ids for r in view)
    assert sum(image_part_count(r.content) for r in view) == 0   # 对照：确实降了


async def test_tie_group_records_before_first_demoted_are_left_alone():
    """组内**位于第一条被降者之前**的记录 seq_no 更小、天然仍在前面，不必重写。"""
    mem = InMemoryMemoryProvider()
    ids = await _seed(mem, [[TextPart(text="m0")],
                            [TextPart(text="m1"), _img(0)]], same_ts=True)
    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0) == 1
    view = await _view(mem)
    assert [content_to_text(r.content)[:2] for r in view] == ["m0", "m1"]
    assert view[0].id == ids[0] and view[1].id != ids[1]


# ── 4. keep_recent 真的保住最近 N 张（配「确实降了」的对照） ──────────────────


async def test_keep_recent_protects_last_two_images_and_demotes_the_rest():
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[_img(0)], [_img(1)], [_img(2)], [_img(3)]])
    n = await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=2)

    view = await _view(mem)
    kept = [p.data for r in view for p in r.content if not hasattr(p, "text")]
    demoted = [decode_image_placeholder(content_to_text(r.content)) for r in view]

    assert n == 2                                   # 返回值 = 实际降级的图片张数
    assert kept == [_ref(2), _ref(3)]               # 最近两张仍是真图
    # ⚠️ 对照（缺了这条，「什么都不降」也能通过上面全部断言）：更早的两张确实变成了占位
    assert demoted[:2] == [(_ref(0), "image/png"), (_ref(1), "image/png")]
    assert [image_part_count(r.content) for r in view] == [0, 0, 1, 1]


async def test_keep_recent_partially_demotes_a_single_record():
    """保护名额落在一条记录中间时：该条内被保护的图仍是真图，其余就地变占位。"""
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[_img(0), TextPart(text="mid"), _img(1), _img(2)]])
    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=1) == 2
    content = (await _view(mem))[0].content
    assert [type(p).__name__ for p in content] == \
        ["TextPart", "TextPart", "TextPart", "ImagePart"]
    assert content[3].data == _ref(2)                       # 最近一张原样保留
    assert decode_image_placeholder(content[0].text) == (_ref(0), "image/png")
    assert decode_image_placeholder(content[2].text) == (_ref(1), "image/png")
    assert content[1].text == "mid"                          # 原文本没被挪位


# ── 5. 未注册 MemoryBlobStore：返回 0 且 memory 一个字节都没被写 ────────────────────


async def test_no_blob_store_means_no_demotion_and_no_write():
    """不接 MemoryBlobStore 时图是 inline base64，没有 ref 可写进占位 → 不降级、不写 memory。"""
    mem = _CountingMemory()
    await _seed(mem, [[TextPart(text="m0"), _b64img()], [_b64img()]])
    writes_before = mem.ingest_calls

    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0) == 0
    assert await demote_all(mem, [r.id for r in await _view(mem)], _pctx(),
                            address=_ADDR) == 0
    assert (mem.fold_calls, mem.ingest_calls) == (0, writes_before)
    view = await _view(mem)
    assert [image_part_count(r.content) for r in view] == [1, 1]   # 图原样还在

    # ⚠️ 对照：同一段代码路径在**有** ref 图时确实会写 memory——否则上面全是永真断言
    await _seed(mem, [[_img(9)]])
    assert await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0) == 1
    assert mem.fold_calls == 1


# ── 6. fold() 失败：跳过继续，返回值反映实际降级数 ───────────────────────────


async def test_fold_failure_skips_that_record_and_continues():
    mem = _CountingMemory(fail_on={1})
    await _seed(mem, [[_img(0)], [_img(1)]])
    n = await demote_for_budget(mem, _ADDR, _pctx(), keep_recent=0)
    assert n == 1                                   # 只算成功的那一条
    assert mem.fold_calls == 2                      # 失败之后没有中断
    view = await _view(mem)
    assert [image_part_count(r.content) for r in view] == [1, 0]


# ── 7. demote_all 不受 keep_recent 保护 ──────────────────────────────────────


async def test_demote_all_ignores_keep_recent_protection():
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[_img(0)], [_img(1)], [_img(2)]])
    ids = [r.id for r in await _view(mem)]
    assert await demote_all(mem, ids, _pctx(), address=_ADDR) == 3
    view = await _view(mem)
    assert sum(image_part_count(r.content) for r in view) == 0
    assert [decode_image_placeholder(content_to_text(r.content))[0] for r in view] == \
        [_ref(0), _ref(1), _ref(2)]


async def test_demote_all_limits_to_given_ids():
    mem = InMemoryMemoryProvider()
    await _seed(mem, [[_img(0)], [_img(1)]])
    ids = [r.id for r in await _view(mem)]
    assert await demote_all(mem, ids[:1], _pctx(), address=_ADDR) == 1
    assert [image_part_count(r.content) for r in await _view(mem)] == [0, 1]


async def test_demote_all_empty_id_list_is_a_no_op():
    mem = _CountingMemory()
    await _seed(mem, [[_img(0)]])
    assert await demote_all(mem, [], _pctx(), address=_ADDR) == 0
    assert mem.fold_calls == 0


# ── 8. `_rebuild` 显式声明降级掉的 ref（缺陷 2026-08-27） ──────────────────────


def test_rebuild_declares_demoted_refs() -> None:
    """降级掉的 ref 必须进 blob_refs——否则 GC 采不到，图过宽限期被删。"""
    rec = _make_record(content=[
        TextPart(text="看图"),
        ImagePart(data="blob:aaaa", media_type="image/png", source_type="ref"),
    ])
    ev = _rebuild(rec, (1,), MemoryScope.TASK)
    assert ev is not None
    assert ev.blob_refs == ["blob:aaaa"]
    assert not hasattr(ev.content[1], "data"), "降级后该 part 应是 TextPart 占位"


def test_rebuild_accumulates_previously_declared_refs() -> None:
    """两次降级：第一次降的 ref 不能在第二次重建时丢掉。"""
    rec = _make_record(
        content=[
            TextPart(text="[image blob:aaaa media_type=image/png]"),
            ImagePart(data="blob:bbbb", media_type="image/png", source_type="ref"),
        ],
        blob_refs=["blob:aaaa"],
    )
    ev = _rebuild(rec, (1,), MemoryScope.TASK)
    assert ev is not None
    assert ev.blob_refs == ["blob:aaaa", "blob:bbbb"]


def test_rebuild_without_demotion_keeps_existing_refs() -> None:
    """同刻组里被原样重写的记录：不降级，但已有的声明要照抄。"""
    rec = _make_record(content=[TextPart(text="纯文本")], blob_refs=["blob:aaaa"])
    ev = _rebuild(rec, (), MemoryScope.TASK)
    assert ev is not None
    assert ev.blob_refs == ["blob:aaaa"]
