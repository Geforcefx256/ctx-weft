"""MemoryRecord 在边界把 dict 形态 content 归一成 dataclass（Phase 3c Task E）。

裁定 D1：判据 ``not hasattr(p, "text")`` **不解冻**。dict 形态是**协议违规**——
``MemoryRecord.content`` 的类型声明就是 ``str | list[ContentPart]``，契约第 2 条
要求「召回时原样返回」。修法是在类型自己的边界归一，而不是给每个读取方加 dict
兜底（后者把协议约束往下腐蚀，且随新调用点腐化）。

落在 ``MemoryRecord.__post_init__`` 而不是包装 provider：core 读 memory 有 11 个
调用点，而**任何 provider 都必须构造 MemoryRecord**——类型自己把住 = 结构性覆盖。
"""

from __future__ import annotations

from datetime import datetime

from ctx_weft.core.utils.content import image_part_count
from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols import ImagePart, TextPart
from ctx_weft.protocols.memory import MemoryEventType, MemoryRecord

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _rec(content) -> MemoryRecord:
    return MemoryRecord(
        id="r1",
        type=MemoryEventType.USER_PROMPT,
        content=content,
        timestamp=datetime(2026, 8, 26, 12, 0, 0),
    )


# ── 1. dict 文本 part → dataclass TextPart ────────────────────────────────────


def test_dict_text_part_normalized_to_dataclass():
    rec = _rec([{"type": "text", "text": "hi"}])
    assert isinstance(rec.content, list), "归一后仍应是 list（不是被拍成 str）"
    assert rec.content == [TextPart(text="hi")]
    part = rec.content[0]
    assert not isinstance(part, dict), "dict 形态必须在构造时就消失"
    # 陷阱：hasattr 断言在 str 上恒为 True，故上面先钉住 isinstance(list)。
    assert hasattr(part, "text") and part.text == "hi"


# ── 2. dict 图片 part → dataclass ImagePart，字段不丢 ─────────────────────────


def test_dict_image_part_normalized_with_all_fields():
    rec = _rec([{
        "type": "image", "data": _PNG_B64, "media_type": "image/png",
        "source_type": "ref",
    }])
    assert isinstance(rec.content, list)
    part = rec.content[0]
    assert isinstance(part, ImagePart)
    assert part.data == _PNG_B64
    assert part.media_type == "image/png"
    assert part.source_type == "ref", "source_type 不得在归一中丢失/被默认值覆盖"


# ── 3. 归一后 Phase 3b 实测的两个坏值消失 ────────────────────────────────────


def test_dict_text_record_no_longer_counted_as_image():
    """实测坏值 1：``image_part_count([{'type':'text',...}])`` 曾为 1（多算 1600 token）。"""
    rec = _rec([{"type": "text", "text": "hello world"}])
    assert image_part_count(rec.content) == 0


def test_dict_text_record_is_visible_to_summarizer():
    """实测坏值 2：``content_to_text([{'type':'text',...}])`` 曾为 ''（摘要器看不见）。"""
    rec = _rec([{"type": "text", "text": "hello world"}])
    assert content_to_text(rec.content) == "hello world"


def test_dict_image_record_still_counted_as_one_image():
    """归一不得反过来把真图片计漏。"""
    rec = _rec([{"type": "image", "data": _PNG_B64, "media_type": "image/png"}])
    assert image_part_count(rec.content) == 1
    assert content_to_text(rec.content) == ""


# ── 4/5. 快路径：不该发生重建的两种形态是**同一对象** ────────────────────────


def test_str_content_is_same_object():
    """纯文本快路径：``isinstance(str)`` 立即返回，连新字符串都不该产生。"""
    s = "plain text"
    rec = _rec(s)
    assert rec.content is s, "str content 必须是同一对象（零开销快路径）"


def test_dataclass_list_is_same_object():
    """已合规的 dataclass 列表不做无谓重建（省一次 list 分配 + 保住调用方的别名语义）。"""
    parts = [TextPart(text="a"), ImagePart(data=_PNG_B64, media_type="image/png")]
    rec = _rec(parts)
    assert rec.content is parts, "已是 dataclass 形态时必须原样返回同一个 list 对象"


def test_empty_list_is_same_object():
    empty: list = []
    rec = _rec(empty)
    assert rec.content is empty


# ── 6. 混合列表：部分 dict 部分 dataclass → 全部归一 ─────────────────────────


def test_mixed_list_fully_normalized():
    parts = [
        TextPart(text="a"),
        {"type": "text", "text": "b"},
        {"type": "image", "data": _PNG_B64, "media_type": "image/jpeg"},
        ImagePart(data=_PNG_B64, media_type="image/png"),
    ]
    rec = _rec(parts)
    assert isinstance(rec.content, list)
    assert len(rec.content) == 4, "归一不得增删 part"
    assert not any(isinstance(p, dict) for p in rec.content), "混合列表里的 dict 必须全部消失"
    assert rec.content == [
        TextPart(text="a"),
        TextPart(text="b"),
        ImagePart(data=_PNG_B64, media_type="image/jpeg"),
        ImagePart(data=_PNG_B64, media_type="image/png"),
    ]
    assert rec.content is not parts, "含 dict 时必然重建（不得原地改调用方的 list）"
    assert parts[1] == {"type": "text", "text": "b"}, "不得就地改写调用方传入的 list"


def test_unknown_dict_part_type_is_dropped_not_raised():
    """未知 type 跳过而不抛——沿用 ``content_from_jsonable`` 的既有降级语义
    （事件流只增，旧版本会读到比自己新的数据）。"""
    rec = _rec([{"type": "video", "data": "x"}, {"type": "text", "text": "ok"}])
    assert rec.content == [TextPart(text="ok")]
