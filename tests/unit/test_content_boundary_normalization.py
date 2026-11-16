"""三处边界共用同一份 dict→dataclass 归一（Phase 3c Task E2）。

Task E 只在 ``MemoryRecord.__post_init__`` 归一，断言「dict 在 core 内部结构性消失」——
该断言**只对经 memory 的路径成立**：``LLMMessage`` 是 dataclass 且无 ``__post_init__``，
宿主/自定义流程可直接构造；``rehydrate_content`` 又刻意是「dict 进 dict 出」。于是
Task E 删掉 adapter 的 dict 分支后，dict 图片在出网路径上被**静默丢弃**（controller 实测
``_parts_to_blocks([{...image...}]) → []``，无 raise 无 log）——比以前的错误兜底更糟。

修法仍是归一（不是在 adapter 里 raise——adapter 在同步出网主路径上，抛异常会掀掉整个
LLM 请求，同 Phase 3b 对 ``MemoryBlobStore.get`` 恒不抛的取向）。三处边界
（``MemoryRecord`` / ``MemoryEvent`` / ``LLMMessage``）共用 ``core.utils.content`` 里的
**同一个** ``normalize_content_parts``——spec §3① 明令形态转换收在归一层，不得散成三份。
"""

from __future__ import annotations

from datetime import datetime

from ctx_weft.protocols.content import normalize_content_parts
from ctx_weft.protocols import ImagePart, LLMMessage, TextPart
from ctx_weft.protocols.memory import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryRecord,
)
from ctx_weft.providers.llm.adapters.anthropic import _serialize_messages as anth
from ctx_weft.providers.llm.adapters.openai_compatible import _serialize_messages as oai

_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


def _dict_image(**over) -> dict:
    d = {"type": "image", "data": _PNG_B64, "media_type": "image/png",
         "source_type": "base64"}
    d.update(over)
    return d


def _event(content) -> MemoryEvent:
    return MemoryEvent(
        type=MemoryEventType.USER_PROMPT,
        address=MemoryAddress(session_id="s1"),
        content=content,
        timestamp=datetime(2026, 8, 27, 12, 0, 0),
    )


def _record(content) -> MemoryRecord:
    return MemoryRecord(
        id="r1",
        type=MemoryEventType.USER_PROMPT,
        content=content,
        timestamp=datetime(2026, 8, 27, 12, 0, 0),
    )


# ── 1. LLMMessage 直构路径：dict 图片归一成 ImagePart，字段不丢 ────────────────


def test_llm_message_dict_image_normalized_with_all_fields():
    msg = LLMMessage(role="user", content=[_dict_image(source_type="ref")])
    assert isinstance(msg.content, list), "归一后仍是 list（陷阱：str 上 hasattr 恒真）"
    part = msg.content[0]
    assert isinstance(part, ImagePart), "dict 形态必须在构造时就消失"
    assert part.data == _PNG_B64
    assert part.media_type == "image/png"
    assert part.source_type == "ref", "source_type 不得被默认值覆盖"


def test_llm_message_dict_text_normalized():
    msg = LLMMessage(role="user", content=[{"type": "text", "text": "hi"}])
    assert isinstance(msg.content, list)
    assert msg.content == [TextPart(text="hi")]


# ── 2. 该图能真的出网（直接钉住 controller 实测的 `[]`）──────────────────────


def test_anthropic_dict_image_actually_reaches_wire():
    """Task E 后实测：``_parts_to_blocks([dict image]) → []``（静默丢图）。"""
    out = anth([LLMMessage(role="user", content=[_dict_image()])])
    blocks = out[0]["content"]
    assert isinstance(blocks, list)
    assert blocks == [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _PNG_B64},
    }], "dict 形态图片必须出网，不得被静默丢弃"


def test_openai_dict_image_actually_reaches_wire():
    out = oai("", [LLMMessage(role="user", content=[_dict_image()])])
    blocks = out[-1]["content"]
    assert isinstance(blocks, list)
    assert blocks == [{
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_PNG_B64}"},
    }], "dict 形态图片必须出网，不得被静默丢弃"


def test_dict_text_part_still_reaches_wire():
    """归一不得让 dict 文本反而丢掉（Task E 后它同样落在被跳过的最后一支）。"""
    out = anth([LLMMessage(role="user", content=[{"type": "text", "text": "看图"}])])
    assert out[0]["content"] == [{"type": "text", "text": "看图"}]


# ── 3. 纯文本快路径：同一对象（"没有发生某件事"型，须变异验证）────────────────


def test_llm_message_str_content_is_same_object():
    s = "plain text"
    msg = LLMMessage(role="user", content=s)
    assert msg.content is s, "str content 必须是同一对象（零开销快路径）"


def test_memory_event_str_content_is_same_object():
    s = "plain text"
    assert _event(s).content is s


def test_memory_record_str_content_is_same_object():
    s = "plain text"
    assert _record(s).content is s


def test_plain_text_wire_is_byte_identical():
    """三处归一后纯文本 wire 形态逐字节不变（Global Constraint 第 1 条）。"""
    msgs = [LLMMessage(role="user", content="hello"),
            LLMMessage(role="assistant", content="hi"),
            LLMMessage(role="tool", content="result", tool_call_id="tc1")]
    assert anth(msgs) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tc1", "content": "result"}]},
    ]


# ── 4. 已合规的 dataclass 列表：同一对象，不重建 ──────────────────────────────


def test_llm_message_dataclass_list_is_same_object():
    parts = [TextPart(text="a"), ImagePart(data=_PNG_B64, media_type="image/png")]
    msg = LLMMessage(role="user", content=parts)
    assert msg.content is parts, "已是 dataclass 形态时必须原样返回同一个 list"


def test_memory_event_dataclass_list_is_same_object():
    parts = [TextPart(text="a")]
    assert _event(parts).content is parts


def test_llm_message_empty_list_is_same_object():
    empty: list = []
    assert LLMMessage(role="user", content=empty).content is empty


# ── 5. MemoryEvent 写侧归一（与读侧 MemoryRecord 对称）───────────────────────


def test_memory_event_dict_content_normalized():
    ev = _event([{"type": "text", "text": "b"}, _dict_image(media_type="image/jpeg")])
    assert isinstance(ev.content, list)
    assert ev.content == [
        TextPart(text="b"),
        ImagePart(data=_PNG_B64, media_type="image/jpeg", source_type="base64"),
    ]


def test_memory_event_does_not_mutate_caller_list():
    parts = [{"type": "text", "text": "b"}]
    ev = _event(parts)
    assert ev.content is not parts, "含 dict 必然重建"
    assert parts == [{"type": "text", "text": "b"}], "不得就地改写调用方传入的 list"


def test_memory_event_validation_still_runs():
    """归一不得抢在既有校验之前把 __post_init__ 的报错路径改掉。"""
    import pytest
    with pytest.raises(ValueError):
        MemoryEvent(
            type=MemoryEventType.USER_PROMPT,
            address=MemoryAddress(session_id="s1"),
            content=None,
            timestamp=datetime(2026, 8, 27, 12, 0, 0),
        )


# ── 6. 三处共用同一实现（可观测断言，不是"看代码"）──────────────────────────


def test_three_boundaries_call_the_one_shared_implementation(monkeypatch):
    """三处都必须走 ``protocols.content.normalize_content_parts``——三份复制粘贴的实现
    在这里必然漏网（spec §3① 要防的正是这种散点）。

    三处通过协议模块调用，替换模块属性可以检查是否共享同一实现。
    """
    import ctx_weft.protocols.content as content_mod

    calls: list = []
    real = content_mod.normalize_content_parts

    def spy(content):
        calls.append(content)
        return real(content)

    monkeypatch.setattr(content_mod, "normalize_content_parts", spy)

    a = [{"type": "text", "text": "a"}]
    b = [{"type": "text", "text": "b"}]
    c = [{"type": "text", "text": "c"}]
    LLMMessage(role="user", content=a)
    _event(b)
    _record(c)

    assert calls == [a, b, c], (
        "三处边界必须调用同一个共用实现；缺哪个说明那处是自己抄了一份"
    )


# ── 7. 共用实现本身的直测 ────────────────────────────────────────────────────


def test_shared_normalizer_passes_through_none_and_str():
    assert normalize_content_parts(None) is None
    s = "x"
    assert normalize_content_parts(s) is s


def test_shared_normalizer_drops_unknown_type_without_raising():
    """未知 type 跳过而不抛——沿用 ``content_from_jsonable`` 的既有降级语义。"""
    out = normalize_content_parts([{"type": "video", "data": "x"},
                                   {"type": "text", "text": "ok"}])
    assert out == [TextPart(text="ok")]
