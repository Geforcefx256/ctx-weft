from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
import pytest

from tests.compaction_testkit import _active_memory_tokens
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import MemoryEvent, MemoryEventType as T, MemoryAddress, ProviderContext

_BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _pctx():
    return ProviderContext(session_id="s", tenant_id="tn")


@pytest.mark.asyncio
async def test_active_tokens_sums_and_drops_after_supersede():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    ids = []
    for i in range(3):
        await mem.ingest(MemoryEvent(type=T.LLM_RESPONSE, address=scope, content="x" * 400,
                                     timestamp=_BASE + timedelta(seconds=i), role="assistant",
                                     metadata={"task_id": "t1"}), _pctx())
    state = SimpleNamespace(scope=scope, agent=SimpleNamespace())
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(),
                          llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))
    before = await _active_memory_tokens(state, ctx)
    assert before > 0
    # supersede 掉最老一条后总量下降
    recs = await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _pctx())
    await mem.supersede([recs[-1].id], _pctx())
    after = await _active_memory_tokens(state, ctx)
    assert after < before


@pytest.mark.asyncio
async def test_active_tokens_counts_image_parts():
    from ctx_weft.protocols import ImagePart, TextPart
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=scope,
                                 content=[TextPart(text="look"),
                                          ImagePart(data="ZGF0YQ==", media_type="image/png")],
                                 timestamp=_BASE, role="user",
                                 metadata={"task_id": "t1"}), _pctx())
    state = SimpleNamespace(scope=scope, agent=SimpleNamespace())
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(),
                          llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))
    total = await _active_memory_tokens(state, ctx)
    text_only = HeuristicTokenizer().count("look")
    assert total == text_only + 1600 + 4  # Message framing matches prepare/gateway.


@pytest.mark.asyncio
async def test_active_tokens_plain_text_includes_message_framing():
    """Count each emitted message's framing as well as its text."""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    for i in range(3):
        await mem.ingest(MemoryEvent(type=T.LLM_RESPONSE, address=scope, content="x" * 400,
                                     timestamp=_BASE + timedelta(seconds=i), role="assistant",
                                     metadata={"task_id": "t1"}), _pctx())
    state = SimpleNamespace(scope=scope, agent=SimpleNamespace())
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(),
                          llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))
    total = await _active_memory_tokens(state, ctx)
    assert total == 3 * (HeuristicTokenizer().count("x" * 400) + 4)
