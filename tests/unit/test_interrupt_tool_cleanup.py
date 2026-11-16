"""Interrupting an in-flight tool: cancellation reaches the tool's cleanup,
and the gateway calls the provider's cancel() hook as a safety net."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import AsyncIterator

import psutil
import pytest

from ctx_weft.providers._script_runner import run_with_liveness

from ctx_weft.protocols.capability import (
    CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider,
)
from tests.unit.test_gateway_qualified import _gw, _state_ctx



class _BlockingTool(ToolCapabilityProvider):
    name = "mcp:a"

    def __init__(self) -> None:
        self.cancelled_with: str | None = None
        self.invoke_saw_invocation_id: str | None = None
        self.invoke_finally_ran = False
        self.started = asyncio.Event()

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="mcp:a:search", name="search", description="s")

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        self.invoke_saw_invocation_id = ctx.invocation_id   # provider correlates cancel by this
        async def _run():
            self.started.set()
            try:
                await asyncio.Event().wait()        # block until cancelled
                yield CapabilityEvent(kind="result", payload={"content": "ok"})
            finally:
                self.invoke_finally_ran = True       # tool's own cleanup (e.g. bash terminate_tree)
        return _run()

    async def cancel(self, invocation_id, ctx) -> None:
        self.cancelled_with = invocation_id


@pytest.mark.asyncio
async def test_interrupt_reaches_tool_cleanup_and_calls_provider_cancel():
    p = _BlockingTool()
    mem, state, ctx = _state_ctx()
    gw = _gw(p, mem)

    task = asyncio.ensure_future(gw.invoke("mcp__a__search", {}, state, ctx))
    await p.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # (a) CancelledError propagated into the tool → its finally ran (bash terminate_tree path).
    assert p.invoke_finally_ran is True
    # (b) gateway invoked the provider's cancel() hook as a safety net (MCP etc.).
    assert p.cancelled_with is not None
    # (c) invoke received the invocation_id via ctx → cancel() can correlate (same id).
    assert p.invoke_saw_invocation_id is not None
    assert p.invoke_saw_invocation_id == p.cancelled_with


def _marked_pids(marker: str) -> list[int]:
    out: list[int] = []
    for pr in psutil.process_iter(["cmdline"]):
        try:
            cl = pr.info.get("cmdline") or []
            if any(marker in (a or "") for a in cl):
                out.append(pr.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


async def _wait_until(pred, timeout=8.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if pred():
            return True
        await asyncio.sleep(0.05)
    return pred()


@pytest.mark.asyncio
async def test_real_subprocess_killed_on_interrupt():
    # End-to-end: a shell tool's child process tree must die when the tool is cancelled.
    marker = f"LOOMEX_INT_MARK_{os.getpid()}"
    cmd = f'"{sys.executable}" -c "import time; time.sleep(60)  # {marker}"'
    task = asyncio.ensure_future(run_with_liveness(
        cmd, cwd=None, env=None, idle_timeout_sec=999, hard_cap_sec=999, output_limit_bytes=4096,
    ))
    try:
        assert await _wait_until(lambda: bool(_marked_pids(marker))), "subprocess should have started"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert await _wait_until(lambda: not _marked_pids(marker)), "subprocess must be killed on interrupt"
    finally:
        # safety: ensure no leak even if the assertions fail
        for pid in _marked_pids(marker):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                psutil.Process(pid).kill()
