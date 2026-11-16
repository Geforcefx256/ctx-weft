"""Trailing-segment fold anchor + resume barrier (multi-turn "continue task" bug).

Root cause of the intermittent "user input ignored → continue-task cue → blank reply":
a detached plain_text background observe folds the just-finished segment into a
TASK_COMPACT_SUMMARY. When the folded block has no surviving event after it (the
single-segment plain_text case: [USER_PROMPT, LLM_RESPONSE]), apply_compact used to
anchor the summary to wall-clock now(). If that detached fold completes AFTER the next
turn's USER_PROMPT is already ingested, the now()-stamped summary sorts AFTER the new
user message, so the assembler sees history ending in a (assistant-role) summary,
appends the "continue task" resume cue, and buries the real new request → blank reply.

Fix 2 (this file, provider level): a trailing-segment fold summary must be anchored to
the folded segment's own position (the last archived event's timestamp), never to now(),
so a later-arriving USER_PROMPT always sorts after it.

Fix 1 (this file, runtime level): _inject_user_reply must await any in-flight background
observe for the resumed task before ingesting the new USER_PROMPT, so the fold's write
always precedes the new turn's prompt.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryScope,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.core.hitl.registry import PendingHitl
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_NORMAL,
    HitlDecision,
    UserTurnDelivery,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")


def _scope() -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")


def _ts(offset_us: int) -> datetime:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(microseconds=offset_us)


async def _fold_trailing_segment(mem: InMemoryMemoryProvider) -> None:
    """Ingest a single plain_text segment [UP1, LLM] and fold it (plain_text boundary)."""
    pctx = _pctx()
    scope = _scope()
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=scope, content="原始诉求",
                                 timestamp=_ts(1), role="user"), pctx)
    await mem.ingest(MemoryEvent(type=T.LLM_RESPONSE, address=scope, content="上一轮回复",
                                 timestamp=_ts(2), role="assistant"), pctx)
    from tests.segment_fold_testkit import segment_fold
    await segment_fold(mem, scope, MemoryScope.TASK, "段摘要", pctx)


# ── Fix 2: provider anchor ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trailing_fold_summary_anchored_to_segment_not_now():
    """A trailing-segment fold summary is anchored to the last archived event's timestamp,
    NOT wall-clock now()."""
    mem = InMemoryMemoryProvider()
    await _fold_trailing_segment(mem)

    summaries = await mem.recall_recent(_scope(), [T.TASK_COMPACT_SUMMARY], 10, _pctx())
    assert len(summaries) == 1
    # The folded LLM_RESPONSE sat at _ts(2); the summary must inherit that position.
    assert summaries[0].timestamp == _ts(2), (
        f"summary anchored to {summaries[0].timestamp} (now()?) instead of the folded "
        f"segment position {_ts(2)}"
    )


@pytest.mark.asyncio
async def test_user_prompt_after_trailing_fold_sorts_newest():
    """After folding a trailing segment, a freshly ingested USER_PROMPT must be the newest
    record — the fold summary must never leapfrog the new user message."""
    mem = InMemoryMemoryProvider()
    await _fold_trailing_segment(mem)

    # Next turn: the user's new message is injected (later than the folded segment).
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=_scope(), content="新问题",
                                 timestamp=_ts(1000), role="user"), _pctx())

    recent = await mem.recall_recent(
        _scope(),
        [T.USER_PROMPT, T.LLM_RESPONSE, T.TASK_COMPACT_SUMMARY],
        10, _pctx(),
    )
    # recall_recent returns newest-first.
    assert recent[0].content == "新问题", (
        f"newest record is {recent[0].type}:{recent[0].content!r}, not the new USER_PROMPT — "
        "the fold summary leapfrogged the new user message"
    )


# ── Fix 1: resume barrier ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_inject_user_reply_does_not_await_pending_observe(monkeypatch):
    """_inject_user_reply 不再等在途后台折叠——2026-09-22 拆除那道屏障。

    原先要等，理由是「折叠摘要的时间戳必须早于新消息，否则迟到的摘要会越到新消息之后、
    令下一轮装配误判续跑并埋掉新输入」。问题的根其实不在时间戳（`segment_fold` 的锚点
    早就取 `following[0].timestamp - 1μs`），在段界是动态查找的——新 USER_PROMPT 一落库
    就成了「最后一条 user 回合」，折叠池随之变空。改由 `launch_recap` 钉住
    段界水位线（见 tests/unit/test_segment_fold.py），迟到的折叠自己落回原位，这里不必
    再等。人的回复因此不为任何后台 LLM 往返买单。
    """
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
    from ctx_weft.core import CtxWeftRuntime

    order: list[str] = []

    class _RecordingMem(InMemoryMemoryProvider):
        async def ingest(self, event, ctx):
            from ctx_weft.protocols import MemoryKind
            # v2 词汇：user 回合 = kind CONVERSATION_TURN + role user（旧 type 兜底兼容）
            if (event.type is T.USER_PROMPT
                    or (event.kind is MemoryKind.CONVERSATION_TURN and event.role == "user")):
                order.append("ingest")
            return await super().ingest(event, ctx)

    runtime = make_runtime(agent_provider=InlineAgentTemplateProvider())
    mem = _RecordingMem()
    runtime.providers.register_memory(mem)

    from ctx_weft.core.loop.driver import ExecutionRegistry
    registry = runtime._step_executions.setdefault("s1", ExecutionRegistry())

    async def _slow_fold():
        await asyncio.sleep(0.05)
        order.append("fold")

    pending = asyncio.create_task(_slow_fold())
    registry.register("recap-run", "t1", "observe", pending, agent_id="a1")

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = Task(id="t1", session_id="s1", status="SUSPENDED",
                assigned_agent_id="a1", creator_agent_id="a1")
    async def _mark_human_resolved(tid, *, hitl_id):
        if tid == "t1" and task.status not in ("FINISHED", "FAILED", "CANCELED"):
            task.status = "PENDING"

    task_manager = SimpleNamespace(get_task=lambda tid: task if tid == "t1" else None,
                                   children_of=lambda tid: set(),
                                   mark_human_resolved=_mark_human_resolved)
    req = PendingHitl(
        id="h1", form=HITL_FORM_WAIT, session_id="s1", task_id="t1", agent_id="a1",
        delivery=UserTurnDelivery(task_id="t1", preface=PREFACE_NORMAL),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    req.decision = HitlDecision(outcome="accepted", message="新问题")

    await runtime._inject_user_reply(req, session, task_manager)

    assert order == ["ingest"], (
        f"回复的注入不该等折叠落地（段界水位线已接管正确性），got {order}"
    )
    await pending
    assert order == ["ingest", "fold"]
