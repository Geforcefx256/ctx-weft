from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import _is_short_leaf
from ctx_weft.protocols import (
    ImagePart, MemoryAddress, MemoryEvent, MemoryEventType as MT, ProviderContext, TextPart,
)
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


_PCTX = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")
_SCOPE = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
_BASE = datetime(2026, 8, 1, tzinfo=UTC)


def _ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_PCTX,
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _loop_config():
    return SimpleNamespace(short_task_turn_cap=10, short_task_token_threshold=400)


async def _seed(mem, contents):
    for i, c in enumerate(contents):
        await mem.ingest(MemoryEvent(
            type=MT.LLM_RESPONSE, address=_SCOPE, content=c,
            timestamp=_BASE + timedelta(seconds=i), role="assistant"), _PCTX)


@pytest.mark.asyncio
async def test_short_leaf_true_for_small_text():
    """纯文本短任务仍判 short——改造前行为不得漂移。"""
    mem = InMemoryMemoryProvider()
    await _seed(mem, ["ok", "done"])
    got = await _is_short_leaf(
        mem, _SCOPE, SimpleNamespace(id="t1"), _loop_config(), _ctx(mem), False)
    assert got is True


@pytest.mark.asyncio
async def test_short_leaf_false_when_images_exceed_threshold():
    """一张图 1600 token 已超阈值 400，不得因图算 0 而误判 short。"""
    mem = InMemoryMemoryProvider()
    await _seed(mem, ["ok", [TextPart(text="done"),
                             ImagePart(data="ZGF0YQ==", media_type="image/png")]])
    got = await _is_short_leaf(
        mem, _SCOPE, SimpleNamespace(id="t1"), _loop_config(), _ctx(mem), False)
    assert got is False
