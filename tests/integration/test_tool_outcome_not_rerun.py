"""H3 契约锚：工具副作用已发生、结果写入失败 → 恢复**不重跑**（spec: tool-operations）。

本文件原是 WP0 的**缺陷基线**夹具（钉住 2026-09-11 可靠性方案 H3 的旧行为：
ReconcileStep 对无配对 TOOL_RESULT 的 dangling tool_call 一律经 gateway 重新执行，
外部副作用被执行两次），在 WP5/WP6 落地后**有意翻转**为现契约。

外部副作用用**独立子进程 + SQLite 计数器**承载（design D5）：副作用证据落在外部
文件里，不随测试进程内存消失——「Runtime 崩溃后副作用仍在」由新开连接可读来模拟。
真子进程强退的形态见 `test_operation_crash_matrix.py`（O-T05/O-T06）。

⚠️ 本例走的是**无账本记录**那条分支：`tool_call_id` 是裸 wire id（`effect_test`，非
摄入点铸造的 `tc_...`），`is_internal_call_id` 判假 → 账本全程旁路 → reconcile 读不到
记录 → 作结「无账本记录，无从查证」。它钉的是「一律不盲重跑」这条底线，**不**覆盖
`started + reviewed` 的裁决链——那条由 `tests/unit/test_operation_recovery_policy.py` 钉。
"""
from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.steps.reconcile import ReconcileStep
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

# 子进程副作用计数器：一次 INSERT = 一次外部副作用。只用 stdlib，跨平台无端口。
_COUNTER_SCRIPT = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("CREATE TABLE IF NOT EXISTS effects (n INTEGER)")
conn.execute("INSERT INTO effects VALUES (1)")
conn.commit()
conn.close()
"""


def _read_effects(db_path) -> int:
    """新开连接读取副作用计数——模拟 Runtime 进程退出后证据仍可核验。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM effects").fetchone()
        return int(rows[0])
    except sqlite3.OperationalError:  # 表还没建
        return 0
    finally:
        conn.close()


class _ExternalEffectTool(ToolCapabilityProvider):
    """每次 invoke 起一个子进程做一次「外部副作用」（SQLite 计数 +1）。"""

    name = "probe"

    def __init__(self, db_path) -> None:
        self.db_path = str(db_path)
        self.invocations: list[str] = []

    def capability(self) -> ToolCapability:
        return ToolCapability(
            id="probe:record", name="record", description="Simulated external operation",
            side_effects=True,
            # 显式写出默认值：reviewed = core 绝不自行重跑（本例实际走无记录分支）
            recovery_policy="reviewed",
        )

    async def list(self, ctx):
        return [self.capability()]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx):
        async def _run():
            self.invocations.append(ctx.invocation_id)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _COUNTER_SCRIPT, self.db_path,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            await proc.wait()
            yield CapabilityEvent(kind="result", payload={"content": "operation completed"})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None:
        return None


def _fixture(db_path):
    memory = InMemoryMemoryProvider()
    bus = InProcessEventBus()
    state = LoopState(
        run_id="r1",
        session=type("S", (), {"id": "s1", "tenant_id": "default"})(),
        task=type("T", (), {"id": "task1"})(),
        agent=type("A", (), {"id": "agent1", "template_id": "template"})(),
        scope=MemoryAddress(session_id="s1", task_id="task1", agent_id="agent1"),
        resolved_model=type("M", (), {"model": "mock", "account": ""})(),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=memory, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", task_id="task1", agent_id="agent1"),
    )
    tool = _ExternalEffectTool(db_path)
    cache = CapabilityCache()
    cache.put("agent1", [tool.capability()])
    gateway = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool],
        memory=memory, event_bus=bus,
    )
    ctx.capability_gateway = gateway
    return memory, state, ctx, tool


async def test_result_write_failure_then_reconcile_does_not_rerun(tmp_path):
    """契约锚：副作用完成后结果写失败 → 恢复作结不重跑 → 外部计数 = 1。"""
    db = tmp_path / "effects.sqlite"
    memory, state, ctx, tool = _fixture(db)

    # 崩溃前持久化了 assistant 回合（带 dangling tool_call），但 TOOL_RESULT 没写成功
    await memory.ingest(MemoryEvent(
        type=MemoryEventType.LLM_RESPONSE, address=state.scope, content="",
        role="assistant", timestamp=datetime.now(UTC),
        metadata={"tool_calls": [{
            "id": "effect_test", "name": "probe__record",
            "input": {"operation": "increment"},
        }]},
    ), ctx.provider_ctx)

    # 第一次执行：子进程副作用成功；结果写 memory 时模拟崩溃（role=tool 的 ingest 抛错）
    ingest = memory.ingest

    async def _fail_tool_writes(item, provider_ctx):
        if item.role == "tool":
            raise OSError("simulated tool-result persistence failure")
        return await ingest(item, provider_ctx)

    write_failed = False
    with patch.object(memory, "ingest", side_effect=_fail_tool_writes):
        try:
            await ctx.capability_gateway.invoke(
                "probe__record", {"operation": "increment"}, state, ctx,
                tool_call_id="effect_test",
            )
        except OSError:
            write_failed = True
    assert write_failed, "result write should have failed (simulated crash point)"
    assert _read_effects(db) == 1, "first execution: external side effect happened exactly once"

    # 恢复（真实 ReconcileStep，只绕过能力发现——能力已绑定；与探针同一口径）
    with patch("ctx_weft.core.loop.steps.reconcile.resolve_and_bind", new=AsyncMock()):
        await ReconcileStep().execute(state, ctx)

    # 翻转后契约（spec: tool-operations）：无账本记录 → 作结「无从查证」，副作用保持
    # 1 次。作结写成工具结果后循环继续，不停机（见 test_operation_recovery_policy）
    external_effects = _read_effects(db)
    assert external_effects == 1, (
        f"recovery must NOT re-run the completed side effect (got {external_effects})")
    assert len(set(tool.invocations)) == 1
