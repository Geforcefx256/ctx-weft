import json

from ctx_weft.core.utils.content import redact_content_for_event
from ctx_weft.protocols.content import content_from_jsonable, content_to_jsonable
from ctx_weft.protocols import ImagePart, TextPart


def _img():
    return ImagePart(data="ZGF0YWRhdGFkYXRh", media_type="image/png")


# ── str / None 恒等 ───────────────────────────────────────────────────────

def test_str_passes_through_unchanged():
    assert content_to_jsonable("hello") == "hello"
    assert content_from_jsonable("hello") == "hello"


def test_none_passes_through():
    assert content_to_jsonable(None) is None
    assert content_from_jsonable(None) is None


# ── 往返 ─────────────────────────────────────────────────────────────────

def test_roundtrip_preserves_parts():
    src = [TextPart(text="look"), _img()]
    assert content_from_jsonable(content_to_jsonable(src)) == src


def test_jsonable_output_is_actually_json_serializable():
    """这是本函数存在的理由：ContentPart 是普通 dataclass，json.dumps 会直接炸。"""
    payload = content_to_jsonable([TextPart(text="look"), _img()])
    json.dumps(payload)  # 不抛即通过


def test_image_fields_survive_roundtrip():
    out = content_from_jsonable(content_to_jsonable([_img()]))
    assert out[0].data == "ZGF0YWRhdGFkYXRh"
    assert out[0].media_type == "image/png"
    assert out[0].source_type == "base64"


# ── 前向兼容 ─────────────────────────────────────────────────────────────

def test_unknown_part_type_is_skipped_not_raised():
    raw = [{"type": "text", "text": "keep"}, {"type": "video", "data": "x"}]
    out = content_from_jsonable(raw)
    assert len(out) == 1 and out[0].text == "keep"


# ── 脱敏 ─────────────────────────────────────────────────────────────────

def test_redact_leaves_plain_text_untouched():
    assert redact_content_for_event("hello") == "hello"


def test_redact_replaces_image_with_marker():
    out = redact_content_for_event([TextPart(text="look"), _img()])
    assert "look" in out
    assert "ZGF0YWRhdGFkYXRh" not in out, "脱敏后不得残留完整 base64"
    assert "image/png" in out


def test_redact_none_is_empty_string():
    assert redact_content_for_event(None) == ""
