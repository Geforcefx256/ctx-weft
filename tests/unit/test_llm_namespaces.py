"""Identical account names resolve to independent configs and clients by context."""
from ctx_weft.protocols import ProviderContext
from ctx_weft.providers.llm import LLMAccount, LLMProvider, ModelConfig, ModelLimits


class Store:
    def __init__(self):
        self.accounts = {}

    def save(self, account):
        self.accounts[account.key] = account

    def delete(self, resource_id, *, namespace="global"):
        return self.accounts.pop((namespace, resource_id), None) is not None

    def list_all(self, *, namespace=None):
        return [v for v in self.accounts.values() if namespace is None or v.namespace == namespace]


class Scoped(LLMProvider):
    def _account_visible(self, account, ctx):
        return ctx is not None and account.namespace == ctx.tenant_id


def test_same_names_have_independent_storage_clients_and_mutations():
    store = Store()
    provider = Scoped(store)
    records = []
    for scope, limit in [("a", 16000), ("b", 64000)]:
        account = LLMAccount("same", scope + "-secret", "https://example.test",
                             [ModelConfig("m", "openai-compatible", ModelLimits(limit))],
                             namespace=scope)
        provider.register_account(account)
        records.append(account)
    a, b = [ProviderContext(session_id=s, tenant_id=s) for s in ("a", "b")]
    first = provider.get_client("same", ctx=a)
    second = provider.get_client("same", ctx=b)
    assert first.context_limit == 16000 and second.context_limit == 64000
    assert first._adapter is not second._adapter
    assert len(store.accounts) == 2
    provider.delete_account(records[0].resource_id, ctx=a)
    assert provider.get_client("same", ctx=b).context_limit == 64000
    assert len(store.accounts) == 1


def test_reasoning_codec_is_preserved_and_invalid_levels_are_rejected():
    from dataclasses import asdict
    import pytest
    from ctx_weft.providers.llm import ModelCapabilities, ReasoningCapabilities, ReasoningLevel

    descriptor = ReasoningCapabilities(True, [ReasoningLevel("high", {
        "type": "base", "version": 1, "values": {"reasoning_effort": "high"},
    })])
    config = ModelCapabilities(reasoning=descriptor)
    restored = ModelCapabilities(**asdict(config))
    assert restored == config
    with pytest.raises(ValueError, match="Duplicate"):
        ReasoningCapabilities(True, descriptor.effort * 2)
