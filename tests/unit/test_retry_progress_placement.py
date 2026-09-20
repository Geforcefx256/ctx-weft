"""Observe finish_task output labeling (send-only).

The old process_report-based "Current Progress" chronological placement + dedup
mechanism tested here previously was retired in Task 3 (2026-07-01): retry
feedback is now carried entirely by the TASK_COMPACT_SUMMARY segment summary
(folded by observe, rendered by _history.py with PROGRESS_SO_FAR_HEADING) —
finalize no longer writes task.process_report/process_report_at, and composer
no longer has a separate process_report-driven Progress So Far render path.
Coverage for that mechanism now lives in test_compaction.py /
test_finalize_retry_no_process_report.py. Only the still-live finish_task
output labeling behavior (unrelated to process_report) remains here.
"""
from __future__ import annotations
from datetime import UTC, datetime
from types import SimpleNamespace
import pytest

from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.assembler.composer import DefaultComposer

pytestmark = pytest.mark.asyncio


def _ts(sec: int) -> str:
    return datetime(2026, 1, 1, 0, 0, sec, tzinfo=UTC).isoformat()


def _hist(role: str, content: str, sec: int, **md) -> ContextBlock:
    return ContextBlock(id=f"b{sec}", source="task_conversation", kind="history",
                        target="messages", content=content, priority=3, token_estimate=1,
                        metadata={"role": role, "timestamp": _ts(sec), **md})


def _task(**over) -> SimpleNamespace:
    base = dict(id="R", user_prompt_in_memory=True,
                process_report=None, process_report_at=None,
                outputs="THE FINAL ANSWER", title="", description="", user_prompt="do X",
                parent_task_id=None)
    base.update(over)
    return SimpleNamespace(**base)


async def test_observe_includes_finish_output_labeled() -> None:
    """Observer sees the finish_task output, labeled as the act-phase finish_task result."""
    blocks = [
        _hist("user", "## Current Message\ndo X", 1),
        _hist("assistant", "ATTEMPT2 did Y", 10),
    ]
    req = SimpleNamespace(purpose="observe", task=_task(), session=SimpleNamespace(user_prompt="do X"),
                          template=None)
    msgs = DefaultComposer()._build_observer_messages(blocks, req)
    joined = "\n".join(m.content for m in msgs if isinstance(m.content, str))
    assert "THE FINAL ANSWER" in joined
    assert "finish_task" in joined  # explicit label that this was the act-phase finish_task result
