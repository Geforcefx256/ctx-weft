"""LLMProvider.fetch_models / fetch_models_for_account / ping / verify_model —— 复用 _build_adapter，不改 protocol。"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
import base64
import struct
import zlib

import httpx
import pytest

from ctx_weft.protocols import ImagePart, LLMCallError, LLMChunk, TextPart
from ctx_weft.providers.llm import ModelCapabilities, ModelLimits
from ctx_weft.providers.llm.provider import LLMAccount, LLMProvider, ModelConfig


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")


class _FakeStore:
    def save(self, account) -> None: ...
    def delete(self, name) -> bool:
        return True

    def list_all(self) -> list:
        return []


class _FakeAdapter:
    """最小 LLMClient：记录收到的 request，complete 产出 chunk 或抛 LLMCallError。"""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.calls: list = []

    def complete(self, request, stream: bool = True):
        self.calls.append(request)
        return self._gen()

    async def _gen(self):
        if self._fail:
            raise LLMCallError("API error 404: model not found", status_code=404, retriable=False)
        yield LLMChunk(kind="token", text="hi")
        yield LLMChunk(kind="done", finish_reason="stop")


def _provider() -> LLMProvider:
    return LLMProvider(_FakeStore())


def _mock(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_models_for_account():
    p = _provider()
    p.register_account(
        LLMAccount(
            name="acc",
            api_key="sk",
            base_url="https://api.openai.com",
            models=[
                ModelConfig(
                    name="gpt-4o",
                    limit=ModelLimits(128000, output=None),
                    protocol="openai",
                    capabilities=ModelCapabilities(),
                )
            ],
            default_model="gpt-4o",
        ),
        persist=False,
    )
    p.get_client("acc", ctx=_PROVIDER_CONTEXT)._adapter._client = _mock(
        lambda req: httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "o3"}]})
    )
    assert await p.fetch_models_for_account("acc", protocol="openai") == ["gpt-4o", "o3"]


async def test_fetch_models_dynamic(monkeypatch):
    p = _provider()
    captured: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        return httpx.Response(200, json={"data": [{"id": "claude-opus-4-8"}]})

    real_build = p._build_adapter

    def fake_build(acc, model):
        a = real_build(acc, model)
        a._client = _mock(handler)
        return a

    monkeypatch.setattr(p, "_build_adapter", fake_build)
    models = await p.fetch_models(
        protocol="anthropic", api_key="k", base_url="https://api.anthropic.com"
    )
    assert models == ["claude-opus-4-8"]
    assert captured["url"] == "https://api.anthropic.com/v1/models"


async def test_ping_returns_latency(monkeypatch):
    p = _provider()

    async def fake_fetch(style, api_key, base_url=""):
        return ["m"]

    monkeypatch.setattr(p, "fetch_models", fake_fetch)
    ms = await p.ping(protocol="openai", api_key="k", base_url="https://api.openai.com")
    assert isinstance(ms, float) and ms >= 0


async def test_fetch_models_unknown_account_raises():
    p = _provider()
    with pytest.raises(KeyError):
        await p.fetch_models_for_account("nope", protocol="openai")


async def test_verify_model_returns_latency_and_uses_configured_model(monkeypatch):
    p = _provider()
    fake = _FakeAdapter()
    monkeypatch.setattr(p, "_build_adapter", lambda acc, model: fake)
    ms = await p.verify_model(
        protocol="anthropic",
        api_key="k",
        base_url="https://api.anthropic.com",
        model="claude-opus-4-8",
    )
    assert isinstance(ms, float) and ms >= 0
    assert len(fake.calls) == 1
    req = fake.calls[0]
    assert req.model == "claude-opus-4-8"
    assert req.config.max_tokens > 1  # 显式不用 1：避开个别推理模型对极小 max_tokens 的拒绝


async def test_verify_model_bad_model_raises(monkeypatch):
    p = _provider()
    monkeypatch.setattr(p, "_build_adapter", lambda acc, model: _FakeAdapter(fail=True))
    with pytest.raises(LLMCallError):
        await p.verify_model(
            protocol="anthropic", api_key="k", base_url="https://api.anthropic.com", model="nope"
        )


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("inputs", [["text"], ["text", "image"], ["image"]])
async def test_verify_model_exercises_declared_inputs(monkeypatch, protocol, inputs):
    p = _provider()
    fake = _FakeAdapter()
    monkeypatch.setattr(p, "_build_adapter", lambda acc, model: fake)
    await p.verify_model(
        protocol, "key", "https://gateway.test", model="candidate",
        capabilities=ModelCapabilities(input=inputs),
    )
    assert len(fake.calls) == 1
    content = fake.calls[0].messages[0].content
    if "image" not in inputs:
        assert content == "ping"
        return
    assert isinstance(content, list)
    assert any(isinstance(part, TextPart) for part in content) == ("text" in inputs)
    image = next(part for part in content if isinstance(part, ImagePart))
    assert image.media_type == "image/png" and image.source_type == "base64"
    data = base64.b64decode(image.data, validate=True)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (1, 1)
    assert len(data) == 69
    compressed = b""
    offset = 8
    while offset < len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + size]
        crc = struct.unpack(">I", data[offset + 8 + size : offset + 12 + size])[0]
        assert zlib.crc32(kind + payload) == crc
        if kind == b"IDAT":
            compressed += payload
        offset += size + 12
    assert len(zlib.decompress(compressed)) == height * (1 + width * 3)


async def test_routine_ping_uses_text_without_changing_vision_config(monkeypatch):
    p = _provider()
    config = ModelConfig(
        "vision", "anthropic", ModelLimits(128000),
        ModelCapabilities(input=["text", "image"]),
    )
    p.register_account(LLMAccount("acc", "key", "https://gateway.test", [config]), persist=False)
    fake = _FakeAdapter()
    monkeypatch.setattr(p, "_build_adapter", lambda acc, model: fake)
    await p.verify_model_for_account("acc", "vision")
    latency = await p.ping_model_for_account("acc", "vision")
    assert latency >= 0
    assert any(isinstance(part, ImagePart) for part in fake.calls[0].messages[0].content)
    assert fake.calls[1].messages[0].content == "ping"
    assert config.capabilities.input == ["text", "image"]
    assert len(p._adapters[p.get_account("acc").key]) == 1


async def test_verify_model_for_account_reuses_live_adapter():
    p = _provider()
    p.register_account(
        LLMAccount(
            name="acc",
            api_key="sk",
            base_url="https://api.openai.com",
            models=[
                ModelConfig(
                    name="gpt-4o",
                    limit=ModelLimits(128000, output=None),
                    protocol="openai",
                    capabilities=ModelCapabilities(),
                )
            ],
            default_model="gpt-4o",
        ),
        persist=False,
    )
    fake = _FakeAdapter()
    p.get_client("acc", ctx=_PROVIDER_CONTEXT)
    p._adapters[p.get_account("acc").key][("openai", False)] = fake
    ms = await p.verify_model_for_account("acc", "gpt-4o")
    assert isinstance(ms, float) and ms >= 0
    assert fake.calls[0].model == "gpt-4o"


async def test_verify_model_for_account_unknown_raises():
    p = _provider()
    with pytest.raises(KeyError):
        await p.verify_model_for_account("nope", "m")


def _acc_with_no_models(p: LLMProvider) -> None:
    p.register_account(
        LLMAccount(
            name="acc", api_key="sk", base_url="https://api.openai.com", models=[], default_model=""
        ),
        persist=False,
    )


def test_add_model_threads_output_ceiling():
    p = _provider()
    _acc_with_no_models(p)
    p.add_model(
        "acc",
        ModelConfig("claude-x", "anthropic", ModelLimits(200_000, 64_000), output_reserve=8192),
    )
    m = next(m for m in p.get_account("acc").models if m.name == "claude-x")
    assert m.limit.output == 64_000
    assert m.output_reserve == 8192


def test_add_model_output_ceiling_defaults_none():
    p = _provider()
    _acc_with_no_models(p)
    p.add_model("acc", ModelConfig("gpt-4o", "openai", ModelLimits(128_000)))
    m = next(m for m in p.get_account("acc").models if m.name == "gpt-4o")
    assert m.limit.output is None
