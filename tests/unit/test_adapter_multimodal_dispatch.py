"""内置 adapter 的模态分流：类型即声明（spec 2026-08-28-multimodal-adapter-dispatch）。

纯文本 adapter 把图降级成占位 + warning；多模态子类原样透传。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
import base64
import logging

import pytest

from ctx_weft.protocols import ImagePart, LLMMessage, LLMRequest, TextPart
from ctx_weft.providers.llm import ModelCapabilities, ModelConfig, ModelLimits


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode()


def _img(media_type: str = "image/png") -> ImagePart:
    return ImagePart(data=_PNG_B64, media_type=media_type)


def _req(*messages: LLMMessage) -> LLMRequest:
    return LLMRequest(model="m-1", system="", messages=list(messages))


# ── Task 1：降级辅助 ──────────────────────────────────────────────────────


def test_downgrade_replaces_image_with_deterministic_placeholder():
    from ctx_weft.providers.llm.adapters._modality import downgrade_for_text_only

    req = _req(LLMMessage(role="user", content=[TextPart(text="看图"), _img()]))
    out = downgrade_for_text_only(req, adapter_hint="AnthropicMultimodalAdapter")

    assert [p.text for p in out[0].content] == ["看图", "[image image/png]"]


def test_downgrade_is_identity_for_text_only_messages():
    """纯文本路径逐字节不变：返回同一个 list 对象，不重建。"""
    from ctx_weft.providers.llm.adapters._modality import downgrade_for_text_only

    req = _req(LLMMessage(role="user", content="纯文本"))
    assert downgrade_for_text_only(req, adapter_hint="X") is req.messages


def test_downgrade_warns_once_with_model_count_and_fix_hint(caplog):
    from ctx_weft.providers.llm.adapters._modality import downgrade_for_text_only

    req = _req(
        LLMMessage(role="user", content=[_img(), _img()]),
        LLMMessage(role="tool", content=[_img()], tool_call_id="tc1"),
    )
    with caplog.at_level(logging.WARNING):
        downgrade_for_text_only(req, adapter_hint="OpenAICompatibleMultimodalAdapter")

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1, "每次请求最多一条 warning"
    msg = records[0].getMessage()
    assert "m-1" in msg and "3" in msg and "OpenAICompatibleMultimodalAdapter" in msg


def test_downgrade_emits_no_log_for_text_only(caplog):
    from ctx_weft.providers.llm.adapters._modality import downgrade_for_text_only

    req = _req(LLMMessage(role="user", content="纯文本"))
    with caplog.at_level(logging.WARNING):
        downgrade_for_text_only(req, adapter_hint="X")
    assert caplog.records == []


def test_downgrade_does_not_mutate_input_messages():
    from ctx_weft.providers.llm.adapters._modality import downgrade_for_text_only

    original = LLMMessage(role="user", content=[_img()])
    req = _req(original)
    downgrade_for_text_only(req, adapter_hint="X")
    assert isinstance(original.content[0], ImagePart), "入参消息不得被就地改写"


# ── Task 2：Anthropic 分流 ────────────────────────────────────────────────


def _anth(multimodal: bool):
    from ctx_weft.providers.llm.adapters.anthropic import (
        AnthropicAdapter,
        AnthropicMultimodalAdapter,
    )

    cls = AnthropicMultimodalAdapter if multimodal else AnthropicAdapter
    return cls(api_key="k")


def test_anthropic_text_only_downgrades_user_image():
    payload = _anth(multimodal=False)._build_payload(
        _req(LLMMessage(role="user", content=[TextPart(text="看图"), _img()]))
    )
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text"]
    assert blocks[1]["text"] == "[image image/png]"


def test_anthropic_multimodal_keeps_user_image():
    payload = _anth(multimodal=True)._build_payload(
        _req(LLMMessage(role="user", content=[TextPart(text="看图"), _img()]))
    )
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image"]
    assert blocks[1]["source"]["data"] == _PNG_B64


def test_anthropic_text_only_downgrades_assistant_image():
    payload = _anth(multimodal=False)._build_payload(
        _req(LLMMessage(role="assistant", content=[_img()]))
    )
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text"]


def test_anthropic_text_only_downgrades_tool_result_image():
    payload = _anth(multimodal=False)._build_payload(
        _req(
            LLMMessage(role="user", content="q"),
            LLMMessage(role="tool", content=[_img()], tool_call_id="tc1"),
        )
    )
    tool_msg = payload["messages"][-1]["content"][0]
    assert [b["type"] for b in tool_msg["content"]] == ["text"]


def test_anthropic_multimodal_keeps_tool_result_image():
    payload = _anth(multimodal=True)._build_payload(
        _req(
            LLMMessage(role="user", content="q"),
            LLMMessage(role="tool", content=[_img()], tool_call_id="tc1"),
        )
    )
    tool_msg = payload["messages"][-1]["content"][0]
    assert [b["type"] for b in tool_msg["content"]] == ["image"]


def test_anthropic_plain_text_payload_unchanged_between_classes():
    """纯文本会话在两个类上产出完全相同的 wire。"""
    req = _req(LLMMessage(role="user", content="纯文本"))
    assert _anth(multimodal=False)._build_payload(req) == _anth(multimodal=True)._build_payload(req)


# ── Task 3：OpenAI 分流 ───────────────────────────────────────────────────


def _oai(multimodal: bool):
    from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter, OpenAICompatibleMultimodalAdapter

    cls = OpenAICompatibleMultimodalAdapter if multimodal else OpenAICompatibleAdapter
    return cls(api_key="k")


def _roles(payload) -> list[str]:
    return [m["role"] for m in payload["messages"]]


def test_openai_text_only_downgrades_user_image():
    payload = _oai(multimodal=False)._build_payload(
        _req(LLMMessage(role="user", content=[TextPart(text="看图"), _img()]))
    )
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "text"]
    assert blocks[1]["text"] == "[image image/png]"


def test_openai_multimodal_keeps_user_image_as_data_url():
    payload = _oai(multimodal=True)._build_payload(
        _req(LLMMessage(role="user", content=[TextPart(text="看图"), _img()]))
    )
    blocks = payload["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["text", "image_url"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_text_only_emits_no_relocated_user_message():
    """纯文本 adapter：tool 图已降级 → 不产生重定位的 user 消息。"""
    payload = _oai(multimodal=False)._build_payload(
        _req(
            LLMMessage(role="user", content="q"),
            LLMMessage(role="tool", content=[_img()], tool_call_id="tc1"),
        )
    )
    assert _roles(payload) == ["user", "tool"]
    assert "[image see the following message]" not in payload["messages"][-1]["content"]


def test_openai_multimodal_relocates_tool_image_to_following_user_message():
    payload = _oai(multimodal=True)._build_payload(
        _req(
            LLMMessage(role="user", content="q"),
            LLMMessage(role="tool", content=[_img()], tool_call_id="tc1"),
        )
    )
    assert _roles(payload) == ["user", "tool", "user"]
    assert [b["type"] for b in payload["messages"][-1]["content"]] == ["image_url"]


def test_openai_text_only_downgrades_assistant_image():
    payload = _oai(multimodal=False)._build_payload(
        _req(LLMMessage(role="assistant", content=[_img()]))
    )
    assert [b["type"] for b in payload["messages"][0]["content"]] == ["text"]


def test_openai_plain_text_payload_unchanged_between_classes():
    req = _req(LLMMessage(role="user", content="纯文本"))
    assert _oai(multimodal=False)._build_payload(req) == _oai(multimodal=True)._build_payload(req)


# ── Task 4：LLMProvider 接线 ─────────────────────────────────────────────


class _StoreStub:
    def save(self, a): ...
    def delete(self, n, *, namespace="global"):
        return True

    def list_all(self):
        return []


def _provider_with(style: str):
    from ctx_weft.providers.llm import LLMAccount, LLMProvider, ModelConfig

    p = LLMProvider(_StoreStub())
    p.register_account(
        LLMAccount(
            name="a",
            api_key="k",
            base_url="https://x",
            models=[
                ModelConfig(
                    name="m",
                    limit=ModelLimits(200000, output=None),
                    protocol=style.removesuffix("-multimodal"),
                    capabilities=ModelCapabilities(
                        ["text", "image"] if style.endswith("-multimodal") else ["text"]
                    ),
                )
            ],
            default_model="m",
        ),
        persist=False,
    )
    return p


@pytest.mark.parametrize(
    ("style", "cls_name"),
    [
        ("anthropic", "AnthropicAdapter"),
        ("anthropic-multimodal", "AnthropicMultimodalAdapter"),
        ("openai", "OpenAICompatibleAdapter"),
        ("openai-multimodal", "OpenAICompatibleMultimodalAdapter"),
    ],
)
def test_build_adapter_dispatches_on_style(style, cls_name):
    p = _provider_with(style)
    assert type(p.get_client("a", ctx=_PROVIDER_CONTEXT)._adapter).__name__ == cls_name


def test_unknown_style_still_rejected():
    from ctx_weft.providers.llm import LLMAccount, LLMProvider

    p = LLMProvider(_StoreStub())
    with pytest.raises(ValueError):
        p.register_account(
            LLMAccount(
                name="a",
                api_key="k",
                base_url="https://api.anthropic.com"
                if "gemini" == "anthropic"
                else "https://api.openai.com",
                models=[
                    ModelConfig(
                        name="m",
                        protocol="gemini",
                        limit=ModelLimits(128000),
                        capabilities=ModelCapabilities(),
                    )
                ],
            ),
            persist=False,
        )


def test_multimodal_classes_exported_from_package():
    import ctx_weft.providers.llm as pkg

    assert pkg.AnthropicMultimodalAdapter.__name__ == "AnthropicMultimodalAdapter"
    assert pkg.OpenAICompatibleMultimodalAdapter.__name__ == "OpenAICompatibleMultimodalAdapter"


@pytest.mark.asyncio
async def test_near_miss_style_rejected_on_ungated_path():
    """fetch_models / verify_model 不经 SUPPORTED_STYLES 门控，靠 _build_adapter 自己拒——
    故分派必须精确匹配：前缀相同但不在清单里的 style 也要抛。"""
    p = _provider_with("anthropic")  # Create a provider instance
    # Now test ungated path with a near-miss style
    with pytest.raises(ValueError, match="Unsupported protocol"):
        await p.fetch_models("anthropicX", "k", "https://x")
