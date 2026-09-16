"""F2：后台 observe 被真正取消（`asyncio.CancelledError`）时，两件事都要钉住：

1. `RunFinished` 不得谎报 `completed`——`outcome` 必须是 `canceled`，且必须在异常继续
   传播之前发出（`finally` 天然保证这点）。
2. `CancelledError` 必须重新抛出，不能被吞。

`CancelledError` 不是 `Exception` 的子类（3.8+ 改继承 `BaseException`），
`_run_background_observe` 原来只 `except Exception as exc: run_error = exc`——
取消时这条不命中，`run_error` 仍是 `None`，`finally` 里的
`RunOutcomeKind.COMPLETED.value if run_error is None else INTERRUPTED` 就把一次
真取消报成了「跑完了」，且 `CancelledError` 未经任何处理直接从 `await t` 冒出来。

**为什么这里要重新抛出，而 `runtime.py::_run_loop` 的同款 `except asyncio.CancelledError`
不重新抛出**（协调方裁定，订正了 F2 最初的 brief）：`_run_loop` 吞是因为它把取消结果
转成了 `RunOutcome{kind=CANCELED}` 这个**返回值契约**塞回调用方——取消信息没丢，换了
载体。`_run_background_observe` 是 fire-and-forget 的 `asyncio.Task`，没有这种返回值
契约；吞掉 `CancelledError` 会让 `task.cancel()` 之后 `await task` 拿到一个看似正常的
返回值，取消信息凭空消失——必须重新抛出，让 `task.cancelled()` 如实反映发生过什么。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import ctx_weft.core.loop.steps.background_observe as bo
import ctx_weft.core.loop.steps.observe as _obs_mod
from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
from ctx_weft.protocols.events import EventType


async def test_cancellation_reraises_and_run_finished_reports_canceled_first(
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
    with pytest.raises(asyncio.CancelledError):
        await t  # 必须重新抛出——不是「吞、不重抛」（那是 _run_loop 的口径，这里不适用）

    assert t.cancelled(), "取消信息不得被悄悄吸收——task 必须如实反映自己被取消过"

    # RunFinished 是在 CancelledError 继续传播之前、finally 里发出的——即便异常最终
    # 传播出去，事件也已经落了，不谎报 completed。
    finished = [e for e in ctx.event_bus.emitted if e.type == EventType.RUN_FINISHED]
    assert len(finished) == 1
    payload = finished[0].payload
    assert payload["outcome"] == RunOutcomeKind.CANCELED.value
    assert payload["outcome"] != "completed"
