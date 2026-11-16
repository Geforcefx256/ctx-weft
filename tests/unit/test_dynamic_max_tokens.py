"""dynamic_max_tokens + 请求携带 prompt 估算值：按窗口实时算输出上限。

架构：caller 侧 request_prompt_estimate 算 used（真实基线 + 本轮增量）→ 挂到
request.prompt_token_estimate → 网关 apply_dynamic_max_tokens 纯消费。网关不再自估。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
from ctx_weft.protocols import ModelCallConfig

from ctx_weft.core.orchestrator.model import ResolvedModel

from types import SimpleNamespace

from ctx_weft.core.loop.llm_gateway import (
    PROMPT_EST_BASE_KEY,
    PROMPT_EST_SEG_KEY,
    _estimate_message_tokens,
    _estimate_request_tokens,
    apply_dynamic_max_tokens,
    request_prompt_estimate,
)
from ctx_weft.core.utils.estimate import dynamic_max_tokens, estimate_tokens
from ctx_weft.protocols import LLMMessage, LLMRequest
from ctx_weft.protocols import OutputPolicy
from ctx_weft.protocols.content import ImagePart, TextPart
from ctx_weft.providers.llm import ModelCapabilities, ModelLimits
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer


def _tok(factor_from: "tuple[int, int] | None" = None) -> HeuristicTokenizer:
    t = HeuristicTokenizer()
    if factor_from:
        t.observe(*factor_from)
    return t


# ── 纯算术：dynamic_max_tokens(context_limit, used, ceiling) ────────────────────


def test_empty_window_releases_full_remaining():
    # used 小 + 默认 ceiling=context_limit → ≈ context_limit − margin
    got = dynamic_max_tokens(200_000, 100, ceiling=200_000, margin=4096, floor=1024)
    assert got == 200_000 - 100 - 4096


def test_used_directly_subtracted():
    got = dynamic_max_tokens(128_000, 3000, ceiling=128_000, margin=4096, floor=1024)
    assert got == 128_000 - 3000 - 4096


def test_ceiling_clamps_when_configured_smaller():
    # 配了 output_ceiling(< 剩余) → 被夹到 ceiling（Anthropic 硬输出上限场景）
    got = dynamic_max_tokens(200_000, 100, ceiling=8192, margin=4096, floor=1024)
    assert got == 8192


def test_floor_when_window_nearly_full():
    # 剩余 < floor → 被 floor 兜住
    got = dynamic_max_tokens(200_000, 199_500, ceiling=200_000, margin=4096, floor=1024)
    assert got == 500  # The floor must not exceed the physical remaining window.


# ── helper: request_prompt_estimate（caller 侧 used 估算）────────────────────────


def _req(**kw):
    base = dict(model="m", system="sys", messages=[LLMMessage(role="user", content="hi")], tools=[])
    if "max_tokens" in kw:
        kw["config"] = ModelCallConfig(max_tokens=kw.pop("max_tokens"))
    base.update(kw)
    return LLMRequest(**base)


def _guard(context_limit=200_000, context_tokens=0):
    return SimpleNamespace(context_limit=context_limit, context_tokens=context_tokens)


def test_estimate_first_call_uses_full_when_larger():
    # baseline=None → max(整份估算, context_tokens)；整份更大时用整份
    req = _req(messages=[LLMMessage(role="user", content="X" * 40_000)])
    est = request_prompt_estimate(_tok(), req, _guard(context_tokens=0), None)
    assert est == _estimate_request_tokens(req, estimate_tokens)
    assert est >= 10_000


def test_estimate_first_call_floors_at_context_tokens():
    # baseline=None、上步真实 context_tokens 更大 → 不低于它（更保守，永不 400）
    req = _req(messages=[LLMMessage(role="user", content="hi")])
    est = request_prompt_estimate(_tok(), req, _guard(context_tokens=90_000), None)
    assert est == 90_000


def test_estimate_incremental_baseline_plus_delta():
    # 循环 2+：真实基线 + 仅新增尾段（本轮 tool result）；基线前的历史不重估
    req = _req(
        messages=[
            LLMMessage(role="user", content="old" * 10_000),  # 基线内，不计
            LLMMessage(role="assistant", content="prev"),  # 基线内，不计
            LLMMessage(role="tool", content="R" * 4000, tool_call_id="t1"),  # 新增，计
        ]
    )
    est = request_prompt_estimate(_tok(), req, _guard(context_tokens=50_000), 2)
    # 50_000（真实基线）+ 每条消息估算：framing(4) + ceil(0.6*4000)=2400（连续 R 串按高熵费率）；
    # 历史大 user 不被重估
    assert est == 50_000 + 4 + 2400


def test_estimate_incremental_falls_back_without_real_baseline():
    # context_tokens=0（无真实基线）即使给了 baseline_msg_count 也退回首次分支
    req = _req(messages=[LLMMessage(role="tool", content="R" * 4000, tool_call_id="t1")])
    est = request_prompt_estimate(_tok(), req, _guard(context_tokens=0), 0)
    assert est == max(_estimate_request_tokens(req, estimate_tokens), 0)


def test_estimate_includes_tool_result_messages():
    # role="tool" 消息必须计入（整份估算分支）
    req = _req(
        messages=[
            LLMMessage(role="user", content="hi"),
            LLMMessage(role="tool", content="X" * 4000, tool_call_id="t1"),
        ]
    )
    est = _estimate_request_tokens(req, estimate_tokens)
    assert est >= 1000


def test_estimate_incremental_delta_calibrated():
    # tokenizer 学到 2x → 增量段 ×2、真实基线不乘
    req = _req(
        messages=[
            LLMMessage(role="user", content="old"),
            LLMMessage(role="assistant", content="prev"),
            LLMMessage(role="tool", content="R" * 4000, tool_call_id="t1"),
        ]
    )
    est = request_prompt_estimate(_tok((1000, 2000)), req, _guard(context_tokens=50_000), 2)
    # delta = framing(4，常数不乘) + count("R"*4000)=4800（2400×2） = 4804
    assert est == 50_000 + 4 + 4800
    assert req.metadata[PROMPT_EST_BASE_KEY] == 50_000


def test_estimate_full_path_calibrated_and_base_zero():
    req = _req(messages=[LLMMessage(role="user", content="X" * 40_000)])
    tok = _tok((1000, 2000))
    est = request_prompt_estimate(tok, req, _guard(context_tokens=0), None)
    assert est == _estimate_request_tokens(req, tok.count)
    assert req.metadata[PROMPT_EST_BASE_KEY] == 0


def test_estimate_full_path_floored_seg_key_stays_uncalibrated_estimate():
    # 整份路径被 floor（真实 context_tokens 远大于整份估算）时，返回值是 floor 后的
    # ctx_tokens；但 metadata[PROMPT_EST_SEG_KEY] 必须仍是 floor 前的估算段（tokenizer.count
    # 直接产出），供 act 回喂使用——否则回喂会用 floor 后的假值污染伺服校准（I1）。
    req = _req(messages=[LLMMessage(role="user", content="hi")])
    tok = _tok()
    est = request_prompt_estimate(tok, req, _guard(context_tokens=90_000), None)
    assert est == 90_000  # floor 生效
    assert req.metadata[PROMPT_EST_SEG_KEY] == _estimate_request_tokens(req, tok.count)
    assert req.metadata[PROMPT_EST_SEG_KEY] != est


# ── 此前数不到的几类：tool_calls 参数 / reasoning / 图片 / framing ─────────────────


def test_message_counts_tool_call_arguments():
    # 纯工具调用回合：content 空，体量全在 tool_calls 的 arguments 里——必须计入（旧口径 ≈0 → 400 洞）
    big = "x" * 6000
    m = LLMMessage(
        role="assistant",
        content="",
        tool_calls=[{"id": "c1", "name": "write_file", "input": {"content": big}}],
    )
    assert _estimate_message_tokens(m, estimate_tokens) >= 2000  # ceil(~6000/3) 量级，远超旧的 ~0


def test_message_counts_reasoning_content():
    with_r = _estimate_message_tokens(
        LLMMessage(role="assistant", content="hi", reasoning_content="R" * 3000), estimate_tokens
    )
    without = _estimate_message_tokens(LLMMessage(role="assistant", content="hi"), estimate_tokens)
    assert with_r - without >= 900  # ceil(3000/3)=1000


def test_message_counts_image_parts_by_fixed_constant():
    # 图片按固定保守常数（~1600），不是 base64 的 10 万字符（否则反向严重高估）
    m = LLMMessage(
        role="user",
        content=[
            TextPart(text="见图"),
            ImagePart(data="A" * 100_000, media_type="image/png"),
        ],
    )
    assert 1500 <= _estimate_message_tokens(m, estimate_tokens) <= 2200


def test_message_framing_overhead_added():
    # 空 content 消息也有固定 framing 开销（provider 每条消息都要计）
    assert _estimate_message_tokens(LLMMessage(role="user", content=""), estimate_tokens) >= 4


def test_tool_call_args_counted_in_request_estimate():
    # 整份估算里也要含 tool_calls 参数
    req = _req(
        messages=[
            LLMMessage(
                role="assistant",
                content="",
                tool_calls=[{"id": "c1", "name": "w", "input": {"c": "x" * 9000}}],
            ),
        ]
    )
    assert _estimate_request_tokens(req, estimate_tokens) >= 3000  # ceil(~9000/3)


# ── 网关消费者：apply_dynamic_max_tokens 读 request.prompt_token_estimate ─────────


def _ctx(llm, config=None):
    return SimpleNamespace(llm=llm, config=config)


def _llm(context_limit=200_000, output_ceiling=None, output_policy=OutputPolicy()):
    return SimpleNamespace(context_limit=context_limit, output_ceiling=output_ceiling,
                           output_reserve=0, output_policy=output_policy)


def test_apply_soft_cap_binds_when_window_has_room():
    # 窗口有余量：ceiling = 软顶 0.2*200k = 40k（不再整窗放输出）
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm()), req, ResolvedModel.from_client(_llm()))
    assert req.config.max_tokens == 40_000


def test_apply_tail_regime_when_window_tight():
    # used 大到 L-used-margin 跌破软顶 → 回落紧缩段 context-used-margin；
    # margin 比例制：max(8192, 5%*170_000=8500) = 8500
    req = _req(prompt_token_estimate=170_000)
    apply_dynamic_max_tokens(_ctx(_llm()), req, ResolvedModel.from_client(_llm()))
    assert req.config.max_tokens == 200_000 - 170_000 - 8500  # 21500 < 40k 软顶


def test_apply_fixed_margin_floors_when_ratio_smaller():
    # 5%*155_000=7750 < 8192 → 固定 margin 兜底
    req = _req(prompt_token_estimate=155_000)
    apply_dynamic_max_tokens(_ctx(_llm()), req, ResolvedModel.from_client(_llm()))
    assert req.config.max_tokens == 200_000 - 155_000 - 8192  # 36808 < 40k 软顶


def test_apply_margin_ratio_configurable():
    # config 提高比例 → margin 随之放大
    req = _req(prompt_token_estimate=170_000)
    apply_dynamic_max_tokens(_ctx(_llm(output_policy=OutputPolicy(margin_ratio=0.1))), req, ResolvedModel.from_client(_llm(output_policy=OutputPolicy(margin_ratio=0.1))))
    assert req.config.max_tokens == 200_000 - 170_000 - 17_000


def test_apply_output_min_floors_soft_cap_on_small_window():
    # 小窗口：0.2*16k=3200 < 4096 → 软顶兜到 4096
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm(context_limit=16_000)), req, ResolvedModel.from_client(_llm(context_limit=16000)))
    assert req.config.max_tokens == 4096


def test_apply_hard_ceiling_above_soft_clamped_to_soft():
    # 显式 output_ceiling 高于软顶 → 被软顶夹下（min 语义）
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm(output_ceiling=64_000)), req, ResolvedModel.from_client(_llm(output_ceiling=64000)))
    assert req.config.max_tokens == 40_000  # min(64k, 0.2*200k)


def test_apply_noop_when_estimate_missing():
    # caller 未挂估算 → 网关不设 max_tokens（feature off，无旧回退自估）
    req = _req()  # prompt_token_estimate=None
    apply_dynamic_max_tokens(_ctx(_llm()), req, ResolvedModel.from_client(_llm()))
    assert req.config.max_tokens is None


def test_apply_respects_explicit_max_tokens():
    req = _req(max_tokens=16, prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm()), req, ResolvedModel.from_client(_llm()))
    assert req.config.max_tokens == 16  # 不覆盖显式值


def test_apply_noop_when_loop_guard_none():
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm()), req, None)
    assert req.config.max_tokens is None


def test_apply_soft_cap_default_when_ceiling_unset():
    # output_ceiling 缺省(None) → ceiling = 软顶 0.2*50k = 10k（不再是整窗）
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(
        _ctx(_llm(context_limit=50_000, output_ceiling=None)), req, ResolvedModel.from_client(_llm(context_limit=50000, output_ceiling=None))
    )
    assert req.config.max_tokens == 10_000


def test_apply_hard_ceiling_below_soft_wins():
    # 硬上限 8192 低于软顶 40k → 取硬上限（Anthropic 场景）
    req = _req(prompt_token_estimate=100)
    apply_dynamic_max_tokens(_ctx(_llm(output_ceiling=8192)), req, ResolvedModel.from_client(_llm(output_ceiling=8192)))
    assert req.config.max_tokens == 8192


# ── 瞬态字段不进线上 payload ─────────────────────────────────────────────────────
from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter


def test_prompt_token_estimate_not_in_payload():
    adapter = OpenAICompatibleAdapter(api_key="k", model="m")
    req = _req(prompt_token_estimate=12345, max_tokens=100)
    payload = adapter._build_payload(req)
    assert "prompt_token_estimate" not in payload
    assert payload["max_tokens"] == 100


# ── output_reserve / output_ceiling 配置透传 ─────────────────────────────────────
from ctx_weft.core.utils.estimate import default_output_reserve
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.llm.provider import LLMProvider, ModelConfig, _FixedModelClient


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")


def test_fixed_model_client_exposes_reserve_and_ceiling():
    adapter = MockLLMAdapter(responses=[])
    client = _FixedModelClient(
        adapter, "m", context_limit=200_000, output_reserve=8192, output_ceiling=64_000
    )
    assert client.output_reserve == 8192
    assert client.output_ceiling == 64_000


def test_fixed_model_client_output_ceiling_defaults_none():
    adapter = MockLLMAdapter(responses=[])
    client = _FixedModelClient(adapter, "m", context_limit=200_000, output_reserve=8192)
    assert client.output_ceiling is None


def test_model_config_fields_default_none():
    cfg = ModelConfig(
        name="m",
        output_reserve=16000,
        limit=ModelLimits(200000, output=32000),
        protocol="openai",
        capabilities=ModelCapabilities(),
    )
    assert cfg.output_reserve == 16_000
    assert cfg.limit.output == 32_000
    m2 = ModelConfig(
        name="m2",
        limit=ModelLimits(100000, output=None),
        protocol="openai",
        capabilities=ModelCapabilities(),
    )
    assert m2.output_reserve is None and m2.limit.output is None


def test_default_output_reserve_scales_with_window():
    # max(L//16, 4096)：小窗兜到 4096、大窗按比例放大
    assert default_output_reserve(8_000) == 4000  # 8000//16=500 → 4096 兜底
    assert default_output_reserve(128_000) == 8_000  # 128000//16
    assert default_output_reserve(1_000_000) == 62_500


class _StoreStub:
    def save(self, a): ...
    def delete(self, n, *, namespace="global"):
        return True

    def list_all(self):
        return []


def test_get_client_resolves_none_reserve_by_window():
    from ctx_weft.providers.llm.provider import LLMAccount

    p = LLMProvider(_StoreStub())
    p.register_account(
        LLMAccount(
            name="a",
            api_key="k",
            base_url="https://x/v1",
            models=[
                ModelConfig(
                    name="m",
                    limit=ModelLimits(200000, output=None),
                    protocol="openai",
                    capabilities=ModelCapabilities(),
                )
            ],
            default_model="m",
        ),
        persist=False,
    )
    client = p.get_client("a", "m", ctx=_PROVIDER_CONTEXT)
    assert client.output_reserve == default_output_reserve(200_000) == 12_500


def test_get_client_honors_explicit_zero_reserve():
    from ctx_weft.providers.llm.provider import LLMAccount

    p = LLMProvider(_StoreStub())
    p.register_account(
        LLMAccount(
            name="a",
            api_key="k",
            base_url="https://x/v1",
            models=[
                ModelConfig(
                    name="m",
                    output_reserve=0,
                    limit=ModelLimits(200000, output=None),
                    protocol="openai",
                    capabilities=ModelCapabilities(),
                )
            ],
            default_model="m",
        ),
        persist=False,
    )
    # 显式 0 不被"按尺寸默认"覆盖（is not None 判定）
    assert p.get_client("a", "m", ctx=_PROVIDER_CONTEXT).output_reserve == 0
