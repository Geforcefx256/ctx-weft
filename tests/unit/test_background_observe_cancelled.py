"""F2：后台 observe 被真正取消（`asyncio.CancelledError`）时，`RunFinished` 不得谎报 completed。

`CancelledError` 不是 `Exception` 的子类（3.8+ 改継承 `BaseException`），
`_run_background_observe` 原来只 `except Exception as exc: run_error = exc`——
取消时这条不命中，`run_error` 仍是 `None`，`finally` 里的
`RunOutcomeKind.COMPLETED.value if run_error is None else INTERRUPTED` 就把一次
真取消报成了「跑完了」。对照 `runtime.py::_run_loop` 已有的
`except asyncio.CancelledError:` 处理（同一口径，R1 定下）：取消时 `outcome` 走
`RunOutcomeKind.CANCELED.value`。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import ctx_weft.core.loop.steps.background_observe as bo
import ctx_weft.core.loop.steps.observe as _obs_mod
from ctx_weft.core.orchestrator.task_disposition import RunOutcomeKind
from ctx_weft.protocols.events import EventType


async def test_run_finished_reports_canceled_not_completed_on_cancellation(
        monkeypatch, fake_state_ctx):
    state, ctx = fake_state_ctx
    ctx.capability_gateway = None
    state.agent.loop_config = SimpleNamespace(
        compact_keep_last=2, max_turns_per_observe=3, short_segment_token_threshold=0)
    state.session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)

    async def _cancelled(c, s, req):
        raise asyncio.CancelledError()
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _cancelled)

    t = bo.launch_background_observe(state, ctx, boundary="interrupt")
    await t  # 不得重新抛出（与 _run_loop 对 CancelledError 的处理同一口径：吞、不重抛）

    assert t.exception() is None
    assert not t.cancelled()

    finished = [e for e in ctx.event_bus.emitted if e.type == EventType.RUN_FINISHED]
    assert len(finished) == 1
    payload = finished[0].payload
    assert payload["outcome"] == RunOutcomeKind.CANCELED.value
    assert payload["outcome"] != "completed"
