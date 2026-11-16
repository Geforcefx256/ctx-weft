"""Direct provider calls (including settings probes) perform one HTTP attempt."""

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
import httpx
import pytest

from ctx_weft.protocols import LLMCallError, LLMRequest
from ctx_weft.providers.llm import ModelCapabilities, ModelLimits
from ctx_weft.providers.llm.provider import LLMAccount, LLMProvider, ModelConfig


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")


class _MemStore:
    def save(self, account): ...
    def delete(self, name, *, namespace="global"): ...
    def list_all(self):
        return []


def _acct(style):
    return LLMAccount(
        name="a",
        api_key="k",
        models=[
            ModelConfig(
                name="m",
                output_reserve=100,
                limit=ModelLimits(1000, output=None),
                protocol=style.removesuffix("-multimodal"),
                capabilities=ModelCapabilities(
                    ["text", "image"] if style.endswith("-multimodal") else ["text"]
                ),
            )
        ],
        default_model="m",
        base_url="https://api.anthropic.com"
        if style.removesuffix("-multimodal") == "anthropic"
        else "https://api.openai.com",
    )


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
@pytest.mark.parametrize("entry", ["complete", "ping", "verify"])
async def test_provider_direct_request_has_no_hidden_retry(protocol, entry):
    p = LLMProvider(_MemStore())
    p.register_account(_acct(protocol), persist=False)
    client = p.get_client("a", ctx=_PROVIDER_CONTEXT)
    adapter = client._adapter
    await adapter._client.aclose()
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(503, text="unavailable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        adapter._client = http
        with pytest.raises(LLMCallError) as error:
            if entry == "complete":
                async for _ in client.complete(LLMRequest(model="m", system="", messages=[])):
                    pass
            elif entry == "ping":
                await p.ping_model_for_account("a", "m")
            else:
                await p.verify_model_for_account("a", "m")
    assert error.value.outage is True
    assert len(calls) == 1
