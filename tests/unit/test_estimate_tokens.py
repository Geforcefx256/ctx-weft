"""estimate_tokens：CJK/其他分开、刻意往大了估（避免 len//4 对中文系统性低估）。

口径：CJK 每字 ceil(1.5*n) token；其余每 ceil(len/3) token。两段相加，非空至少 1。
"""
from __future__ import annotations

from math import ceil

from ctx_weft.core.utils.estimate import estimate_tokens


def test_empty_is_zero():
    assert estimate_tokens("") == 0


def test_english_over_estimated_ceil_len_over_3():
    # 11 ASCII → ceil(11/3) = 4（旧 //4 只有 2）
    assert estimate_tokens("hello world") == 4


def test_cjk_one_and_half_per_char():
    # 4 CJK → ceil(1.5*4) = 6（旧 len//4 = 1）
    assert estimate_tokens("你好世界") == 6


def test_cjk_odd_count_rounds_up():
    # 1 CJK → ceil(1.5) = 2；3 CJK → ceil(4.5) = 5
    assert estimate_tokens("你") == 2
    assert estimate_tokens("你好啊") == 5


def test_mixed_cjk_and_ascii_added():
    # 2 CJK + 3 ASCII → ceil(1.5*2)=3 + ceil(3/3)=1 = 4
    assert estimate_tokens("你好abc") == 4


def test_japanese_kana_and_hangul_are_cjk():
    assert estimate_tokens("あ") == 2   # 平假名
    assert estimate_tokens("ア") == 2   # 片假名
    assert estimate_tokens("한") == 2   # 谚文


def test_cjk_punctuation_and_fullwidth_are_cjk():
    # 全角标点/形式区按 CJK 计（每字 1.5）
    assert estimate_tokens("。") == 2
    assert estimate_tokens("！") == 2


def test_supplementary_cjk_ext_b():
    # 扩展 B（U+20000+，如 𠀀）也算 CJK
    assert estimate_tokens("\U00020000") == 2


def test_always_over_estimates_vs_old_len_div_4():
    # 对任意含 CJK/普通文本，新估算 >= 旧 len//4（刻意偏大）
    for txt in ["hello world", "你好世界", "这是一段中文说明", "mixed 混合 text 文本"]:
        assert estimate_tokens(txt) >= max(1, len(txt) // 4)


def test_matches_documented_formula():
    # 白盒：与 CJK ceil(1.5n) + 其他 ceil(len/3) 一致
    txt = "混合abc123，测试！"
    cjk = sum(1 for c in txt if c in "混合测试，！")
    other = len(txt) - cjk
    assert estimate_tokens(txt) == ceil(1.5 * cjk) + ceil(other / 3)


# ── 高熵长串（URL 段/UUID/哈希/base64/hex）按 0.6 token/字符计费 ────────────────
# 实测（tiktoken cl100k/o200k）：这类"随机 ASCII"真实约 0.51~0.54 token/字符。费率取 0.6
# 保证冷启动单边高估（校准系数常态 <1、只向下回收）：len/2 仍会低估 8-11%，收敛前
# 方向是危险的低估；len/3 更是低估 35-40%——工具结果里 id/哈希/base64 击穿 margin 的洞。


def test_dense_ascii_run_billed_at_point_six():
    # 40 连续 [A-Za-z0-9+/=_-] → ceil(0.6*40)=24（len/2=20 冷启动仍偏低）
    assert estimate_tokens("Ab3dEf6hIj9lMnOpQr2tUv5xYz8Ab3dEf6hIj9lM") == 24


def test_sha256_hex_billed_at_point_six():
    h = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # 64 hex
    assert estimate_tokens(h) == 39  # ceil(38.4)


def test_base64_billed_at_point_six():
    b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"  # 44 chars
    assert estimate_tokens(b64) == 27  # ceil(26.4)


def test_run_below_threshold_keeps_len_over_3():
    # 19 连续字符未达 20 阈值 → 仍按 len/3
    assert estimate_tokens("a" * 19) == ceil(19 / 3)


def test_run_at_threshold_switches_to_point_six():
    assert estimate_tokens("a" * 20) == 12


def test_prose_with_spaces_unaffected():
    # 空格切断长串：散文/正常词不受高熵费率影响（回归）
    assert estimate_tokens("hello world") == 4


def test_mixed_cjk_dense_other_formula():
    # CJK ceil(1.5n) + 高熵段 ceil(0.6n) + 其余 ceil(len/3) 三段相加
    txt = "哈希是e3b0c44298fc1c149afbf4c8996fb924，请核对。"
    # 8 CJK → 12；32 hex（≥20 连续）→ ceil(19.2)=20；other 0
    assert estimate_tokens(txt) == 12 + 20


# ── 非 ASCII 非 CJK（emoji/西里尔/组合符等）按 1 token/字符 ─────────────────────
# 实测 emoji 混排在 cl100k 下 est/real=0.96（低估）：这些字符真实 1~3 token/字，
# 旧 len/3 严重低估。1/字符保证冷启动高估方向（西里尔等会 ~2x 高估，方向安全）。


def test_emoji_billed_per_char():
    assert estimate_tokens("✅") == 1
    assert estimate_tokens("✅🎉👀") == 3


def test_variation_selector_counted():
    # ⚠️ = U+26A0 + U+FE0F 两个码点 → 2
    assert estimate_tokens("⚠️") == 2


def test_cyrillic_billed_per_char():
    assert estimate_tokens("привет") == 6  # 旧 ceil(6/3)=2 严重低估


def test_cjk_not_double_counted_as_non_ascii():
    # CJK 仍按 1.5/字，不落入非 ASCII 1/字桶
    assert estimate_tokens("你好世界") == 6


# ── estimate_content_tokens / estimate_tool_calls_tokens（gateway 与 prepare/composer 共用）──
from ctx_weft.core.utils.estimate import estimate_content_tokens, estimate_tool_calls_tokens
from ctx_weft.protocols.content import ImagePart, TextPart


def test_content_tokens_adds_framing_even_when_empty():
    assert estimate_content_tokens("") == 4  # 仅 framing


def test_content_tokens_none_is_framing_only():
    # 回归：content 为 None（旧/导入 memory 记录、None 工具结果等）时，estimate_content_tokens
    # 曾在 `for p in content` 迭代 None 崩溃（prepare 的 token 估算路径）。应与 "" 同为仅 framing。
    assert estimate_content_tokens(None) == 4


def test_content_tokens_text_plus_framing():
    # "hello world"(ceil(11/3)=4) + framing(4)
    assert estimate_content_tokens("hello world") == 4 + 4


def test_content_tokens_image_by_fixed_constant():
    c = [TextPart(text="x"), ImagePart(data="A" * 99_999, media_type="image/png")]
    # framing(4) + estimate("x")=1 + 图片常数(1600)；不含 base64 长度
    assert 1600 <= estimate_content_tokens(c) <= 1700


def test_tool_calls_tokens_counts_name_and_args():
    assert estimate_tool_calls_tokens([{"name": "write_file", "input": {"content": "x" * 6000}}]) >= 2000


def test_tool_calls_tokens_empty_is_zero():
    assert estimate_tool_calls_tokens(None) == 0
    assert estimate_tool_calls_tokens([]) == 0


def test_tool_calls_tokens_accepts_both_arguments_and_input_keys():
    a = estimate_tool_calls_tokens([{"name": "w", "arguments": {"c": "x" * 3000}}])
    b = estimate_tool_calls_tokens([{"name": "w", "input": {"c": "x" * 3000}}])
    assert a == b >= 900


# ── Step 1: count 回调测试 ──────────────────────────────────────────────


def test_content_tokens_routes_text_through_count_callback():
    # 文本费率经回调；framing 常数不过回调
    assert estimate_content_tokens("hello world", count=lambda t: 100) == 4 + 100


def test_content_tokens_callback_default_is_heuristic():
    assert estimate_content_tokens("hello world") == estimate_content_tokens(
        "hello world", count=estimate_tokens
    )


def test_tool_calls_tokens_routes_through_count_callback():
    tc = [{"name": "w", "input": {"c": "x"}}]
    assert estimate_tool_calls_tokens(tc, count=lambda t: 10) == 20  # name + args 各 10


def test_image_constant_not_routed_through_callback():
    c = [TextPart(text="x"), ImagePart(data="A" * 100, media_type="image/png")]
    got = estimate_content_tokens(c, count=lambda t: 0)
    assert got == 4 + 0 + 1600
