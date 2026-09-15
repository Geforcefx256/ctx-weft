"""「当前消息框」保 parts：图片在装配链上最后一次消失的地方。

与 test_current_message_framing.py 覆盖同一函数 `_frame_current_message`，但聚焦
多模态：加框（## Opening Message / ## Current Message）不得把 list[ContentPart]
拍扁成 str，图片必须原样存活在装饰后的 content 里。

三条覆盖：
1. 纯文本框架文案与改造前逐字节相同（期望值取自改造前的真实运行输出，见下方
   docstring 注明的抄录来源，而非凭代码推导）。
2. 多模态时 `## Current Message` 前缀存在且图片仍在。
3. 首条 != 末条（interactive 多轮）时，两个框各自贴上、各自的图片都在。
"""
from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.protocols import ImagePart, LLMMessage, TextPart
from ctx_weft.core.assembler.composer import DefaultComposer


def _comp() -> DefaultComposer:
    return DefaultComposer.__new__(DefaultComposer)


def _u(c): return LLMMessage(role="user", content=c)
def _a(c): return LLMMessage(role="assistant", content=c)


def _task(in_mem=True, title="PPTX转PDF", desc="转 PDF", prompt="把这个 ppt 转 pdf"):
    return SimpleNamespace(user_prompt_in_memory=in_mem, title=title, description=desc,
                           user_prompt=prompt, process_report=None, id="t1")


def _img(tag: str) -> ImagePart:
    return ImagePart(data=f"data-{tag}", media_type="image/png", source_type="base64")


# ── 1) 纯文本：逐字节不变 ────────────────────────────────────────────────────
#
# 期望值抄自改造前（工作树 HEAD，composer.py 尚未接入 content_with_prefix/suffix 时）
# 实际运行 `_frame_current_message` 的输出，用如下脚本得到：
#
#   comp = _comp()
#   history_pairs = [
#       (_u("检查工作目录"), "agent_recall", "user_prompt", "t1"),
#       (_a("好的"), "agent_recall", "llm_response", "t1"),
#       (_u("把这个 ppt 转 pdf"), "agent_recall", "user_prompt", "t1"),
#   ]
#   messages = [m for m, *_ in history_pairs]
#   comp._frame_current_message(messages, history_pairs, _task())
#   print(repr(messages[0].content)); print(repr(messages[2].content))
#
# 真实输出（逐字节抄录，未经推导）：
#   messages[0].content ==
#     '## Current Task\n'PPTX转PDF' (t1)\n转 PDF\n\n## Opening Message\n检查工作目录'
#   messages[2].content ==
#     '## Current Message\n把这个 ppt 转 pdf\n\n'
#     '（Reply in the same language as the Current Message above.）'
#
# 单条（首条即最新，A 形态合并框）：
#   messages[0].content ==
#     '## Current Task\n'PPTX转PDF' (t1)\n转 PDF\n\n## Current Message\n把这个 ppt 转 pdf\n\n'
#     '（Reply in the same language as the Current Message above.）'

_EXPECT_OPENING = "## Current Task\n'PPTX转PDF' (t1)\n转 PDF\n\n## Opening Message\n检查工作目录"
_EXPECT_CURRENT = (
    "## Current Message\n把这个 ppt 转 pdf\n\n"
    "（Reply in the same language as the Current Message above.）"
)
_EXPECT_COMBINED = (
    "## Current Task\n'PPTX转PDF' (t1)\n转 PDF\n\n## Current Message\n把这个 ppt 转 pdf\n\n"
    "（Reply in the same language as the Current Message above.）"
)


def test_plain_text_framing_byte_identical_to_pre_refactor():
    comp = _comp()
    history_pairs = [
        (_u("检查工作目录"), "agent_recall", "user_prompt", "t1"),
        (_a("好的"), "agent_recall", "llm_response", "t1"),
        (_u("把这个 ppt 转 pdf"), "agent_recall", "user_prompt", "t1"),
    ]
    messages = [m for m, *_ in history_pairs]
    comp._frame_current_message(messages, history_pairs, _task())
    assert messages[0].content == _EXPECT_OPENING
    assert messages[2].content == _EXPECT_CURRENT


def test_plain_text_single_message_combined_frame_byte_identical():
    comp = _comp()
    history_pairs = [(_u("把这个 ppt 转 pdf"), "agent_recall", "user_prompt", "t1")]
    messages = [m for m, *_ in history_pairs]
    comp._frame_current_message(messages, history_pairs, _task())
    assert messages[0].content == _EXPECT_COMBINED


# ── 2) 多模态：图片在加框后仍存活 ────────────────────────────────────────────

def test_multimodal_current_message_keeps_image_and_gets_prefix():
    comp = _comp()
    content = [TextPart(text="把这个 ppt 转 pdf"), _img("a")]
    history_pairs = [(_u(content), "agent_recall", "user_prompt", "t1")]
    messages = [m for m, *_ in history_pairs]
    comp._frame_current_message(messages, history_pairs, _task())
    result = messages[0].content
    assert isinstance(result, list)
    assert any(isinstance(p, ImagePart) and p.data == "data-a" for p in result)
    text_parts = [p.text for p in result if isinstance(p, TextPart)]
    joined = "".join(text_parts)
    assert "## Current Task" in joined
    assert "## Current Message" in joined
    assert "把这个 ppt 转 pdf" in joined
    assert "Reply in the same language" in joined


# ── 3) 首条 != 末条：两个框各自贴上、各自的图片都在 ─────────────────────────

def test_multimodal_multi_turn_both_frames_keep_their_own_images():
    comp = _comp()
    opening_content = [TextPart(text="检查工作目录"), _img("open")]
    latest_content = [TextPart(text="把这个 ppt 转 pdf"), _img("latest")]
    history_pairs = [
        (_u(opening_content), "agent_recall", "user_prompt", "t1"),
        (_a("好的"), "agent_recall", "llm_response", "t1"),
        (_u(latest_content), "agent_recall", "user_prompt", "t1"),
    ]
    messages = [m for m, *_ in history_pairs]
    comp._frame_current_message(messages, history_pairs, _task())

    opening_result = messages[0].content
    assert isinstance(opening_result, list)
    assert any(isinstance(p, ImagePart) and p.data == "data-open" for p in opening_result)
    opening_text = "".join(p.text for p in opening_result if isinstance(p, TextPart))
    assert "## Opening Message" in opening_text
    assert "检查工作目录" in opening_text
    assert "## Current Message" not in opening_text

    latest_result = messages[2].content
    assert isinstance(latest_result, list)
    assert any(isinstance(p, ImagePart) and p.data == "data-latest" for p in latest_result)
    latest_text = "".join(p.text for p in latest_result if isinstance(p, TextPart))
    assert "## Current Message" in latest_text
    assert "把这个 ppt 转 pdf" in latest_text
    assert "Reply in the same language" in latest_text
    assert "## Current Task" not in latest_text
