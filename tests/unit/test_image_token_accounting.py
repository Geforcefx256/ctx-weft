"""图片 token 估算改成**体积相关**（Phase 3c Task D，用户裁定 D2）。

背景：``image_tokens`` 原本是 ``1600 * 图片张数``——对唯一真正变化的维度（字节体积）
毫无反应。5 MiB 截图与 50 KiB 缩略图同价，于是唯一能阻止请求体无限膨胀的机制
（token 预算）对真实失败模式完全失明：5 张满额图账面才 8000 token（远不触发 compact），
实际请求体已 30 MB+ 被 provider 拒，且每次 act 都重发全部历史图 → 会话永久卡死。

本文件钉住修好之后的口径与**整条自愈链**：

    图变大 → 估算上升 → 越过 compact 触发比 → fold 掉旧记录
           → load_view 不再返回 → act 装配携带的图片数下降 → 请求缩小

**估算建模的是字节压力，不是计费 token**（provider 侧会降采样，单图计费封顶约 1600）。
拿它去算成本是误用，见 ``core/utils.image_tokens`` 的注释。
"""

from __future__ import annotations


from tests.model_testkit import model_snapshot

import base64
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.builder import PromptBuilder
from ctx_weft.core.assembler.sources._history import record_to_history_block
from ctx_weft.protocols.content import content_from_jsonable, content_to_jsonable
from ctx_weft.core.models.errors import ContextOverflowError
from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.core.utils.content import _IMAGE_BYTES_PER_TOKEN, _IMAGE_PART_TOKENS, image_byte_size, image_part_count, image_tokens
from ctx_weft.core.utils.estimate import effective_limit, estimate_content_tokens, estimate_tokens
from ctx_weft.protocols import (
    ImagePart,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    ProviderContext,
    TextPart,
    LLMMessage,
    load_records,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compile_prompt, prompt_request

# 仓内典型预算（见 core/state/models.py:132、core/control/types.py:32-33，
# 以及 protocols/template.py:79 的 compact_token_ratio）。
_CONTEXT_LIMIT = 180_000
_RESERVE = 8_192
_EFF = effective_limit(_CONTEXT_LIMIT, _RESERVE)      # 171_808
_COMPACT_RATIO = 0.8

_MAX_IMAGE_BYTES = 5 * 1024 * 1024                    # core/content.py:251 的单图上限
_SMALL_IMAGE_BYTES = 50 * 1024                        # 缩略图量级


def _ref_img(byte_size: int | None = None) -> ImagePart:
    """外部化之后的形态：data 是 ``blob:<sha>``，与真实体积毫无关系。"""
    return ImagePart(
        data="blob:" + "de" * 32,
        media_type="image/png",
        source_type="ref",
        byte_size=byte_size,
    )


def _inline_img(raw_bytes: int) -> ImagePart:
    """未外部化的形态：真的带 ``raw_bytes`` 字节的 base64 载荷。"""
    return ImagePart(
        data=base64.b64encode(b"\x00" * raw_bytes).decode("ascii"),
        media_type="image/png",
    )


# ── 1. 不再是平的 ─────────────────────────────────────────────────────────────


def test_large_image_costs_far_more_than_small_image():
    """钉住「体积相关」：满额图与缩略图不再同价，差距与字节比同量级。"""
    big = image_tokens([_ref_img(_MAX_IMAGE_BYTES)])
    small = image_tokens([_ref_img(_SMALL_IMAGE_BYTES)])
    assert big > small * 10, f"满额图 {big} 应远高于缩略图 {small}"
    assert big == _MAX_IMAGE_BYTES // _IMAGE_BYTES_PER_TOKEN


def test_several_full_size_images_exceed_a_typical_budget():
    """标定目标（简报「若干张满额图即应超出典型预算」）：

    典型 effective_limit = 180_000 − 8_192 = 171_808；compact 触发比 0.8。
    4 张满额图必须**已经**越过触发比，5 张必须直接超出整个预算。
    """
    four = image_tokens([_ref_img(_MAX_IMAGE_BYTES) for _ in range(4)])
    five = image_tokens([_ref_img(_MAX_IMAGE_BYTES) for _ in range(5)])
    assert four / _EFF >= _COMPACT_RATIO, f"4 张满额图 {four} 未触发 compact"
    assert five > _EFF, f"5 张满额图 {five} 未超出 effective_limit {_EFF}"


def test_old_flat_constant_would_not_have_triggered_anything():
    """对照：旧口径（1600/张）下 5 张满额图才 8000 token——占典型预算不到 5%。
    这条是本任务存在的理由，钉住它别被将来「简化」回去。"""
    assert 5 * _IMAGE_PART_TOKENS / _EFF < 0.05


# ── 2. 纯文本路径恒为 0（既有不变量，不可破）────────────────────────────────────


@pytest.mark.parametrize("content", [None, "", "hello world", [], [TextPart(text="a")],
                                     [TextPart(text="a"), TextPart(text="b")]])
def test_plain_text_paths_stay_zero(content):
    assert image_tokens(content) == 0
    assert image_part_count(content) == 0


# ── 3. ref（带 byte_size）与同尺寸 inline base64 同价 ──────────────────────────


@pytest.mark.parametrize("raw_bytes", [1, 3, 1024, 1024 * 1024])
def test_ref_with_byte_size_matches_inline_base64_of_same_size(raw_bytes):
    """外部化不得改变预算口径——否则同一张图在 put 前后算出两个数。"""
    inline = _inline_img(raw_bytes)
    ref = _ref_img(raw_bytes)
    assert image_tokens([inline]) == image_tokens([ref])


@pytest.mark.parametrize("raw_bytes", [1, 2, 3, 4, 5, 1024, 1024 * 1024])
def test_image_byte_size_is_exact_for_inline_base64(raw_bytes):
    """反解必须**精确**——含 base64 padding 修正。经 image_tokens 看不见这 1~2 字节
    误差（地板 + 整除会把它吞掉），但 byte_size 是要被写进事件载荷带走的口径，
    差 2 字节会让「同一张图在外部化前后同价」这条不变量在边界尺寸上失守。"""
    assert image_byte_size(_inline_img(raw_bytes)) == raw_bytes


def test_image_byte_size_unknown_for_ref_without_field():
    """ref 形态的 data 是 blob:<sha>，长度与体积无关——必须报 None（交回落），
    绝不能拿 len(data)*3//4 当体积（那会算出约 51 字节的假值）。"""
    assert image_byte_size(_ref_img(None)) is None


def test_inline_base64_size_is_derived_without_decoding():
    """inline 形态即使没有 byte_size 也必须按载荷体积算（base64 膨胀 4/3 反解，
    含 padding 修正），不能退回平常数。"""
    got = image_tokens([_inline_img(1024 * 1024)])
    assert got == (1024 * 1024) // _IMAGE_BYTES_PER_TOKEN


# ── 4. 存量数据回落 ───────────────────────────────────────────────────────────


def test_ref_without_byte_size_falls_back_to_legacy_constant():
    """存量记录 / 不接 blob 的宿主：ref 且无 byte_size → 回落旧常数。
    既不抛，也不算成 0（算成 0 = 预算对图片再次失明）。"""
    assert image_tokens([_ref_img(None)]) == _IMAGE_PART_TOKENS
    assert image_tokens([TextPart(text="a"), _ref_img(None), _ref_img(None)]) == \
        2 * _IMAGE_PART_TOKENS


def test_small_images_still_cost_the_legacy_floor():
    """地板：小于 ``_IMAGE_PART_TOKENS * _IMAGE_BYTES_PER_TOKEN`` 的图仍按旧常数计。
    模型侧对小图的固定开销本就在 1600 量级，往下折算会低估。"""
    assert image_tokens([_ref_img(1)]) == _IMAGE_PART_TOKENS
    assert image_tokens([_ref_img(_SMALL_IMAGE_BYTES)]) == _IMAGE_PART_TOKENS


def test_image_part_count_unaffected_by_byte_size():
    """图片**张数**口径不受本任务影响（budget.py 的报错文案据此报数）。"""
    content = [TextPart(text="a"), _ref_img(None), _ref_img(_MAX_IMAGE_BYTES)]
    assert image_part_count(content) == 2


# ── 5. jsonable 往返 ──────────────────────────────────────────────────────────


def test_jsonable_roundtrip_preserves_byte_size():
    content = [TextPart(text="hi"), _ref_img(_MAX_IMAGE_BYTES)]
    back = content_from_jsonable(content_to_jsonable(content))
    assert isinstance(back, list)
    assert back[1].byte_size == _MAX_IMAGE_BYTES
    assert image_tokens(back) == image_tokens(content)


def test_jsonable_roundtrip_of_legacy_row_yields_none():
    """存量行没有 byte_size 键 → None → 回落，不抛。"""
    legacy = [{"type": "image", "data": "blob:ab", "media_type": "image/png",
               "source_type": "ref"}]
    back = content_from_jsonable(legacy)
    assert isinstance(back, list)
    assert back[0].byte_size is None
    assert image_tokens(back) == _IMAGE_PART_TOKENS


def test_jsonable_omits_byte_size_when_absent():
    """没有 byte_size 的 part 序列化后**不多写键**——存量事件载荷逐字节不变。"""
    out = content_to_jsonable([_ref_img(None)])
    assert isinstance(out, list)
    assert "byte_size" not in out[0]


# ── 5b. 边界处填 byte_size 的两条路 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_externalize_records_byte_size_so_ref_keeps_its_size():
    """normalize_content 是最后一个还握着 raw bytes 的地方——必须在那里记下体积，
    否则外部化之后 data 变成 ``blob:<sha>``，体积信息永久丢失（image_tokens 是同步的，
    不能回 MemoryBlobStore 取回来）。"""
    from ctx_weft.protocols.content import normalize_content

    raw_len = 1024 * 1024

    class _Store:
        can_externalize = True

        async def put(self, data, media_type, ctx):
            return "blob:" + "ab" * 32

    inline = [TextPart(text="hi"), _inline_img(raw_len)]
    out = await normalize_content(inline, blob_store=_Store(), ctx=None)
    assert isinstance(out, list)
    assert out[1].source_type == "ref"
    assert out[1].byte_size == raw_len
    assert image_tokens(out) == image_tokens(inline), "外部化前后预算口径必须一致"


def test_validate_content_does_not_mutate_input():
    """入口校验是校验器，不改内容——``task.user_prompt == 原 content`` 这条既有不变量
    （tests/unit/test_multimodal_entry.py）依赖它。inline 形态的体积由 image_tokens
    自己从载荷长度反解，validate 无需（也不该）就地写 byte_size。"""
    from ctx_weft.protocols.content import validate_content

    content = [TextPart(text="hi"), _inline_img(16)]
    validate_content(content)
    assert content[1].byte_size is None
    assert content == [TextPart(text="hi"), _inline_img(16)]


# ── 6. 自愈链端到端 ───────────────────────────────────────────────────────────


def _record(rid: str, content, ts: datetime) -> MemoryEvent:
    return MemoryEvent(
        id=rid,
        type=MemoryEventType.USER_PROMPT,
        address=MemoryAddress(session_id="s", task_id="t1", agent_id="a"),
        content=content,
        timestamp=ts,
        role="user",
        metadata={"task_id": "t1"},
    )


def _assembler_request():
    task = SimpleNamespace(user_prompt_in_memory=True, process_report=None,
                           title="T", description="D", user_prompt="look", id="t1",
                           outputs=None, unattended=False)
    template = SimpleNamespace(identity={"act": SimpleNamespace(text="ACT-SOUL", style=None)})
    return prompt_request(task_id="t1", task=task, template=template,
                          resolved_model=model_snapshot(_CONTEXT_LIMIT, _RESERVE),
                          session=SimpleNamespace(
                              context_limit=_CONTEXT_LIMIT,
                              reserved_output_tokens=_RESERVE,
                          ),
                          token_counter=estimate_tokens)


def _blocks_from(records, request):
    return [
        record_to_history_block(r, "task_conversation", i)
        for i, r in enumerate(records)
    ]


def _images_in(prompt) -> int:
    n = 0
    for m in prompt.messages:
        if isinstance(m.content, list):
            n += sum(1 for p in m.content if not hasattr(p, "text"))
    return n


async def _should_compact(total_tokens: int) -> bool:
    """走真实的 PrepareStep._should_compact，而不是复述它的算式。"""
    state = SimpleNamespace(resolved_model=model_snapshot(context_limit=_CONTEXT_LIMIT, output_reserve=_RESERVE), agent=SimpleNamespace(
        loop_config=SimpleNamespace(compact_token_ratio=_COMPACT_RATIO),
        loop_guard=SimpleNamespace(context_limit=_CONTEXT_LIMIT,
                                   reserved_output_tokens=_RESERVE),
    ))
    return await PrepareStep()._should_compact(state, None, total_tokens)


@pytest.mark.asyncio
async def test_self_healing_chain_big_images_trigger_compact_then_fold_drops_them():
    """端到端走「memory → load_view → 历史块估算 → compact 触发判定 → fold →
    再装配」这条链，证明修好 image_tokens 之后**既有机制自己就闭合了**。"""
    memory = InMemoryMemoryProvider()
    ctxp = ProviderContext(session_id="s", task_id="t1", agent_id="a")
    ids = [f"mem_{i}" for i in range(4)]
    for i, rid in enumerate(ids):
        await memory.ingest(
            _record(rid, [TextPart(text="look"), _ref_img(_MAX_IMAGE_BYTES)],
                    datetime(2026, 8, 1, 0, i, tzinfo=UTC)),
            ctxp,
        )

    addr = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    request = _assembler_request()

    before = await load_records(memory, addr, MemoryScope.TASK, ctxp)
    assert len(before) == 4
    blocks_before = _blocks_from(before, request)
    total_before = sum(
        estimate_content_tokens(block.value.content, count=request.token_counter)
        for block in blocks_before)

    # ① 估算真的把字节压力算进去了，且越过了 compact 触发比。
    assert await _should_compact(total_before), \
        f"4 张满额图共 {total_before} token 应触发 compact（eff={_EFF}）"
    # ② 旧口径（平 1600）下同一份内容**不会**触发——这正是缺陷所在。
    legacy_total = sum(
        request.token_counter("look") + _IMAGE_PART_TOKENS for _ in before
    )
    assert not await _should_compact(legacy_total), \
        f"旧口径 {legacy_total} token 不该触发 compact（这就是缺陷）"

    prompt_before = compile_prompt(blocks_before, request)
    assert _images_in(prompt_before) == 4

    # ③ fold 掉最老的三条（compact 的既有动作），不写补偿摘要以隔离变量。
    await memory.fold(ids[:3], [], ctxp)

    after = await load_records(memory, addr, MemoryScope.TASK, ctxp)
    assert len(after) == 1, "fold 后 load_view 只应返回幸存的一条"
    blocks_after = _blocks_from(after, request)
    total_after = sum(
        estimate_content_tokens(block.value.content, count=request.token_counter)
        for block in blocks_after)

    # ④ 链条闭合：装配携带的图片数下降，估算回落到触发比以下。
    prompt_after = compile_prompt(blocks_after, request)
    assert _images_in(prompt_after) == 1
    assert total_after < total_before
    assert not await _should_compact(total_after)


# ── 7. 地板超限 → ContextOverflowError 且 image_count 正确 ─────────────────────


@pytest.mark.asyncio
async def test_floor_of_big_images_overflows_with_correct_image_count():
    """pin 住的当前消息里全是满额图 → 地板本身就超预算 → 抛 ContextOverflowError，
    ``image_count`` 精确等于图片张数（用户该做的是删图而非删字，spec §6.5）。

    与既有 test_budget_strategy 那条的区别：这里的 ``token_estimate`` 不是手工填的，
    而是由 ``record_to_history_block`` 按真实字节体积算出来的——即本任务修好之后，
    大图**自己**就能把地板顶穿。
    """
    n = 5
    record = MemoryRecord(
        id="mem_pin",
        type=MemoryEventType.USER_PROMPT,
        kind=MemoryKind.CONVERSATION_TURN,
        scope=MemoryScope.TASK,
        address=MemoryAddress(session_id="s", task_id="t1", agent_id="a"),
        content=[TextPart(text="hi"), *[_ref_img(_MAX_IMAGE_BYTES) for _ in range(n)]],
        timestamp=datetime(2026, 8, 1, tzinfo=UTC),
        role="user",
        metadata={"task_id": "t1", "type": "user_prompt"},
    )
    assert estimate_content_tokens(record.content, count=estimate_tokens) > _EFF
    builder = PromptBuilder(estimate_tokens)
    builder.message(
        LLMMessage(role="user", content=record.content),
        ref="current", priority=0, required=True,
    )
    with pytest.raises(ContextOverflowError) as ei:
        builder.build(
            _EFF,
            context_limit=_CONTEXT_LIMIT,
            reserved_output_tokens=_RESERVE,
        )
    assert ei.value.image_count == n
    assert ei.value.effective_limit == _EFF
    assert ei.value.context_limit == _CONTEXT_LIMIT
