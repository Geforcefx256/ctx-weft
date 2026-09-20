from ctx_weft.protocols import ImagePart, TextPart


def _content():
    return [TextPart(text="这是我的答复"), ImagePart(data="ZGF0YQ==", media_type="image/png")]


# 「HITL 应答内容可以是 parts 列表」这条契约现在钉在 `HitlDecision.message` /
# `HitlReply.message` 的类型上，端到端由 test_hitl_multimodal_validation.py 覆盖；
# 下面两条钉的是拼接助手本身——它们才是图片真正会丢的地方。


def test_rejected_reply_keeps_image_via_prefix():
    """拒绝路径把「Human declined:」拼到回复前——必须保 parts。"""
    from ctx_weft.core.utils.content import content_with_prefix
    out = content_with_prefix(_content(), "Human declined: ")
    assert any(not hasattr(p, "text") for p in out), "图片不得在拼接中丢失"
    assert out[0].text.startswith("Human declined: ")


def test_interrupt_edit_prefix_keeps_image_via_content_with_prefix():
    """① 打断续接说明拼到多模态回复前——必须保 parts，走 content_with_prefix 而非 f-string。"""
    from ctx_weft.core.utils.content import content_with_prefix
    from ctx_weft.core.loop.steps.act import _interrupt_edit_prefix

    prefix = _interrupt_edit_prefix("do X")
    assert prefix
    out = content_with_prefix(_content(), prefix)
    assert any(not hasattr(p, "text") for p in out), "图片不得在拼接中丢失"
    assert out[0].text.startswith(prefix)
