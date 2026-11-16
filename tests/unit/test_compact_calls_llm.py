
from tests.model_testkit import model_snapshot
"""summarize_for_compact must produce its summary from the LLM (no truncation fallback).

A transient outage that exhausts self-heal must propagate LLMOutageError, not return "".
"""

from types import SimpleNamespace

import pytest

from tests import compaction_testkit as compact_mod
from ctx_weft.protocols import LLMChunk, LLMOutageError
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs



class _Assembler:
    async def compile(self, inputs, **kwargs):
        return await compiled_test_inputs(inputs, **kwargs)

    async def assemble(self, request):
        return scripted_inputs(SimpleNamespace(system="sys", messages=[]), purpose=request.purpose)


def _state_ctx():
    agent = SimpleNamespace(runtime={"llm_model": "mock"})
    # resolved_model 是 resolve_llm_identity 的真值来源（task-4 复审第二轮：
    # summarize_for_compact 不再读 agent.runtime.get("llm_model")）。
    state = SimpleNamespace(agent=agent, scope=None, task=None, session=None,
                            extra={}, transcript=[],
                            resolved_model=model_snapshot(model="mock", account=""))
    ctx = SimpleNamespace(assembler=_Assembler(), llm=SimpleNamespace(tokenizer=HeuristicTokenizer()),
                          cancel_token=None, event_bus=None, config=None)
    return state, ctx


@pytest.mark.asyncio
async def test_summarize_for_compact_returns_llm_text(monkeypatch):
    async def _ok(ctx, state, request, **kwargs):
        for t in ["sum", "mary"]:
            yield LLMChunk(kind="token", text=t)
    monkeypatch.setattr(compact_mod.compact_step, "stream_llm_resilient", _ok)
    state, ctx = _state_ctx()
    out = await compact_mod.summarize_for_compact(state, ctx)
    # spec: compact-fidelity——返回解析后的 digest：散文输出（无五小节）整文落库 + degraded。
    assert out.text == "summary"
    assert out.degraded is True


@pytest.mark.asyncio
async def test_summarize_for_compact_propagates_outage(monkeypatch):
    async def _raise(ctx, state, request, **kwargs):
        raise LLMOutageError("self-heal exhausted")
        yield  # pragma: no cover  (make this an async generator)
    monkeypatch.setattr(compact_mod.compact_step, "stream_llm_resilient", _raise)
    state, ctx = _state_ctx()
    with pytest.raises(LLMOutageError):
        await compact_mod.summarize_for_compact(state, ctx)
