"""Model metadata and actual clients use the same contextual account resolution."""

from ctx_weft.protocols.context import ProviderContext
from ctx_weft.providers.llm import LLMAccount, LLMProvider, ModelConfig, ModelLimits


class _Store:
    def save(self, account):
        pass

    def delete(self, name, *, namespace="global"):
        return True

    def list_all(self):
        return []


class _ContextProvider(LLMProvider):
    def __init__(self):
        super().__init__(_Store())
        self.seen = []

    def _resolve_account(self, account, model, *, ctx):
        self.seen.append(ctx)
        return super()._resolve_account(ctx.session_id, model, ctx=ctx)


def test_metadata_and_client_share_the_current_context_and_budget():
    provider = _ContextProvider()
    for name, context in [("first", 16000), ("second", 64000)]:
        provider.register_account(LLMAccount(
            name, name + "-key", "https://example.test",
            [ModelConfig("m", "openai-compatible", ModelLimits(context))],
        ), persist=False)
    for name, expected in [("first", 16000), ("second", 64000), ("first", 16000)]:
        ctx = ProviderContext(session_id=name, tenant_id="opaque-context")
        config = provider.get_model_config("logical-name", "m", ctx=ctx)
        client = provider.get_client("logical-name", "m", ctx=ctx)
        assert config.limit.context == client.context_limit == expected
        assert client.account == provider.get_account(name).resource_id
        assert all(value is ctx for value in provider.seen[-2:])
