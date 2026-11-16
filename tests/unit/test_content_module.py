from ctx_weft.protocols.content import content_to_text, content_with_prefix, content_with_suffix
from ctx_weft.protocols import ImagePart, TextPart


def _img():
    return ImagePart(data="ZGF0YQ==", media_type="image/png")


# ── str 路径：与朴素字符串拼接逐字节相同 ──────────────────────────────────

def test_prefix_on_str_is_plain_concat():
    assert content_with_prefix("body", "head") == "headbody"


def test_suffix_on_str_is_plain_concat():
    assert content_with_suffix("body", "tail") == "bodytail"


def test_empty_text_is_noop_on_str():
    assert content_with_prefix("body", "") == "body"
    assert content_with_suffix("body", "") == "body"


def test_none_content_yields_text():
    assert content_with_prefix(None, "head") == "head"
    assert content_with_suffix(None, "tail") == "tail"


# ── list 路径：并入首/末个 TextPart，图片不动 ──────────────────────────────

def test_prefix_merges_into_leading_text_part():
    out = content_with_prefix([TextPart(text="body"), _img()], "head")
    assert [type(p).__name__ for p in out] == ["TextPart", "ImagePart"]
    assert out[0].text == "headbody"


def test_suffix_merges_into_trailing_text_part():
    out = content_with_suffix([_img(), TextPart(text="body")], "tail")
    assert [type(p).__name__ for p in out] == ["ImagePart", "TextPart"]
    assert out[1].text == "bodytail"


def test_prefix_inserts_new_part_when_leading_is_image():
    out = content_with_prefix([_img(), TextPart(text="body")], "head")
    assert [type(p).__name__ for p in out] == ["TextPart", "ImagePart", "TextPart"]
    assert out[0].text == "head"


def test_suffix_appends_new_part_when_trailing_is_image():
    out = content_with_suffix([TextPart(text="body"), _img()], "tail")
    assert [type(p).__name__ for p in out] == ["TextPart", "ImagePart", "TextPart"]
    assert out[2].text == "tail"


def test_empty_text_is_noop_on_list():
    src = [TextPart(text="body"), _img()]
    assert content_with_suffix(src, "") is src


def test_input_list_is_not_mutated():
    src = [TextPart(text="body"), _img()]
    content_with_prefix(src, "head")
    assert src[0].text == "body", "拼接必须返回新列表，不得就地改写调用方的 part"


def test_empty_list_yields_single_text_part():
    out = content_with_suffix([], "tail")
    assert len(out) == 1 and out[0].text == "tail"


# ── re-export ────────────────────────────────────────────────────────────

def test_content_to_text_reexported():
    from ctx_weft.protocols.content import content_to_text as util_impl
    assert content_to_text is util_impl
