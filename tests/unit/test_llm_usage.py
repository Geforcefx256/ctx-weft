"""LLMUsage distinguishes unreported counters from explicitly reported zero."""
from __future__ import annotations

import dataclasses

import pytest

from ctx_weft.protocols import LLMUsage


def test_missing_cache_counters_do_not_claim_uncached_input():
    u = LLMUsage(prompt_tokens=100, completion_tokens=5, total_tokens=105)
    assert u.input_tokens is None
    assert u.cache_read_tokens is None and u.cache_write_tokens is None
    assert u.reasoning_tokens is None


def test_unknown_and_explicit_zero_are_distinct():
    assert all(value is None for value in dataclasses.asdict(LLMUsage()).values())
    zero = LLMUsage(prompt_tokens=0, completion_tokens=0, cache_read_tokens=0, cache_write_tokens=0)
    assert zero.total_tokens == zero.input_tokens == 0
    assert zero.reasoning_tokens is None


@pytest.mark.parametrize("field", [field.name for field in dataclasses.fields(LLMUsage)])
@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_reported_counter_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        LLMUsage(**{field: value})


def test_total_is_derived_only_from_known_operands():
    assert LLMUsage(prompt_tokens=3).total_tokens is None
    assert LLMUsage(completion_tokens=4).total_tokens is None
    assert LLMUsage(prompt_tokens=3, completion_tokens=4).total_tokens == 7
    assert LLMUsage(prompt_tokens=3, completion_tokens=4, total_tokens=8).total_tokens == 8


def test_derivation_with_cache_split():
    u = LLMUsage(prompt_tokens=127, cache_read_tokens=100, cache_write_tokens=20)
    assert u.input_tokens == 7


def test_explicit_input_tokens_preserved():
    u = LLMUsage(prompt_tokens=127, cache_read_tokens=100, cache_write_tokens=20,
                 input_tokens=7)
    assert u.input_tokens == 7


def test_abnormal_cache_split_remains_unknown_instead_of_claiming_zero():
    u = LLMUsage(prompt_tokens=5, cache_read_tokens=10, cache_write_tokens=0)
    assert u.input_tokens is None
    assert u.prompt_tokens == 5 and u.cache_read_tokens == 10


def test_asdict_carries_seven_keys():
    d = dataclasses.asdict(LLMUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13))
    assert set(d) == {
        "prompt_tokens", "completion_tokens", "total_tokens",
        "cache_read_tokens", "cache_write_tokens", "input_tokens", "reasoning_tokens",
    }
    assert d["input_tokens"] is None


async def test_mock_adapter_carries_cache_split():
    from ctx_weft.protocols import LLMMessage, LLMRequest
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse

    adapter = MockLLMAdapter([MockResponse(text="hi", cache_read_tokens=3, reasoning_tokens=1)])
    req = LLMRequest(model="mock", system="sys prompt",
                     messages=[LLMMessage(role="user", content="question")])
    chunks = [c async for c in adapter.complete(req)]
    u = [c for c in chunks if c.kind == "usage"][0].usage
    assert u.cache_read_tokens == 3
    assert u.reasoning_tokens == 1
    assert u.input_tokens == u.prompt_tokens - 3  # 自动派生
