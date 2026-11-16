"""Interrupt at an act checkpoint: park (wait_for_user) instead of cancel."""

import asyncio

import pytest

from ctx_weft.core.control.tokens import CancelToken, PauseToken
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.loop.steps.act import INTERRUPTED_MARK, ActStep, interrupt_edit_note
from ctx_weft.protocols import LLMChunk, MemoryEventType
from ctx_weft.protocols.hitl import (
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    UserTurnDelivery,
)
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from tests.integration.test_interactive_task import _act_state_ctx



class _PauseMidStream:
    """LLM stub that streams one token, fires the pause, then more tokens."""

    def __init__(self, token: PauseToken, text: str = "partial reply") -> None:
        self._token = token
        self._text = text
        self.tokenizer = HeuristicTokenizer()

    @property
    def context_limit(self) -> int:
        return 100_000

    @property
    def output_reserve(self) -> int:
        return 4096

    @property
    def supports_tool_calling(self) -> bool:
        return True

    async def complete(self, request, stream=True):
        yield LLMChunk(kind="token", text=self._text)
        self._token.pause()                          # soft interrupt fires mid-stream
        yield LLMChunk(kind="token", text=" MORE")   # should not be consumed


@pytest.mark.asyncio
async def test_interrupt_parks_instead_of_cancel():
    llm = MockLLMAdapter(responses=[MockResponse(text="hello")])
    state, ctx, task, hitl, _mem = _act_state_ctx(False, llm)
    pause = PauseToken()
    pause.pause()
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    # park 不写状态：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落（Task 4）
    assert task.status == "ACTIVE"
    pend = hitl.list_pending("s1")
    # 旧断言看的是 capability_id sentinel；新契约用 delivery 表达同一件事（spec §5）。
    assert pend and isinstance(pend[0].delivery, UserTurnDelivery)


@pytest.mark.asyncio
async def test_hard_cancel_raises_cancellederror():
    llm = MockLLMAdapter(responses=[MockResponse(text="hello")])
    state, ctx, task, hitl, _mem = _act_state_ctx(False, llm)
    tok = CancelToken()
    tok.cancel()
    ctx.cancel_token = tok

    with pytest.raises(asyncio.CancelledError):
        await ActStep().execute(state, ctx)
    assert hitl.list_pending("s1") == []


@pytest.mark.asyncio
async def test_interrupt_midstream_commits_partial_marked():
    pause = PauseToken()
    llm = _PauseMidStream(pause, "partial reply")
    state, ctx, task, hitl, mem = _act_state_ctx(False, llm)
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    # park 不写状态：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落（Task 4）
    assert task.status == "ACTIVE"
    recs = await mem.recall_recent(state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx)
    hit = [r for r in recs if "partial reply" in (r.content or "")]
    assert hit, "partial assistant text should be persisted"
    rec = hit[-1]
    assert INTERRUPTED_MARK in rec.content
    assert "MORE" not in rec.content          # token after interrupt not consumed
    assert rec.metadata.get("interrupted") is True


def test_interrupt_edit_note_wraps_prev_and_new():
    note = interrupt_edit_note("do X", "do Y instead")
    assert "do X" in note
    assert "do Y instead" in note
    assert "cancelled" in note


def test_interrupt_edit_note_empty_prev_returns_new():
    assert interrupt_edit_note("", "only this") == "only this"


@pytest.mark.asyncio
async def test_interrupt_before_token_marks_edit_phase():
    # ① 未吐 token → preface = interrupt_edit（续接需补说明；旧字段名 context）。
    pause = PauseToken()
    pause.pause()
    llm = MockLLMAdapter(responses=[MockResponse(text="x")])
    state, ctx, task, hitl, _mem = _act_state_ctx(False, llm)
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)
    assert hitl.list_pending("s1")[0].delivery.preface == PREFACE_AFTER_INTERRUPT_EDIT


@pytest.mark.asyncio
async def test_interrupt_after_token_is_not_edit_phase():
    # ② 已吐 token → 非 edit（preface = interrupt）。
    pause = PauseToken()
    llm = _PauseMidStream(pause, "partial")
    state, ctx, task, hitl, _mem = _act_state_ctx(False, llm)
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)
    assert hitl.list_pending("s1")[0].delivery.preface == PREFACE_AFTER_INTERRUPT


@pytest.mark.asyncio
async def test_interrupt_before_any_token_does_not_persist_response():
    # No token streamed yet (phase ①) → nothing to commit, just park.
    pause = PauseToken()
    pause.pause()                               # paused before streaming starts
    llm = MockLLMAdapter(responses=[MockResponse(text="unused")])
    state, ctx, task, hitl, mem = _act_state_ctx(False, llm)
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    recs = await mem.recall_recent(state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx)
    assert all(not r.metadata.get("interrupted") for r in recs)


def test_cancel_token_is_pure_signal():
    tok = CancelToken()
    assert not hasattr(tok, "mode")
    assert not hasattr(tok, "produced")
    assert not hasattr(tok, "in_tool_loop")
