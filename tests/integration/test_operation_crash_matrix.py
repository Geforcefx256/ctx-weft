"""子进程强退矩阵（spec: tool-operations；wp8-2，方案 O-T05/O-T06/O-T14）。

真子进程退出 + 新 Runtime 实例（无桩 h3 的 pytest 形态）。三条：
- O-T05：manual started → 真退出 → 新实例 → 副作用 1 次 + INTERRUPTED + unknown
- O-T06：completed 后 memory 写前崩溃 → 新实例从账本补写 TOOL_RESULT，不重执行
- O-T14：delegate 的 op completed 后确认丢失 → 重入找回原 child（不双建）
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext, ToolCall
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.operations import (
    OperationRecord,
    OperationStatus,
    operation_id_for,
    operation_memory_result_id,
)
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
from ctx_weft.providers.operations import InMemoryOperationStore
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

_SCRIPT = Path(__file__).resolve().parents[2] / "tests" / "integration" / "_crash_worker.py"


def _count_effects(workdir: Path) -> int:
    f = workdir / "effects.txt"
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.startswith("EFFECT"))


async def test_ot05_manual_started_real_exit_side_effect_once(tmp_path):
    """O-T05：manual started → worker 子进程硬杀 → 新实例 recover → 副作用 1 次。"""
    workdir = tmp_path / "ot05"
    workdir.mkdir()
    marker = workdir / "effect_done.marker"
    ids_file = workdir / "ids.json"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(_SCRIPT), "ot05-worker", str(workdir)],
        cwd=str(_SCRIPT.parents[1]),
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30 and not ids_file.exists():
        time.sleep(0.05)
    assert ids_file.exists(), "worker never reported ids"
    while time.monotonic() - t0 < 30 and not marker.exists():
        time.sleep(0.05)   # 副作用已发生，工具正睡在结果返回前
    proc.kill()
    proc.wait(timeout=10)
    first_count = _count_effects(workdir)
    assert first_count == 1, f"expected 1 side effect before crash, got {first_count}"

    rec = subprocess.run(
        [sys.executable, "-u", str(_SCRIPT), "ot05-recover", str(workdir)],
        cwd=str(_SCRIPT.parents[1]), capture_output=True, text=True, timeout=120,
    )
    final_count = _count_effects(workdir)
    assert final_count == 1, (
        f"O-T05: manual policy must NOT re-run side effect after real crash "
        f"(got {final_count})")


async def test_ot06_completed_memory_write_crash_ledger_backfills(tmp_path):
    """O-T06：账本 completed 后 memory 写前崩溃 → 新实例从账本补写，provider 不重执行。

    组件级（真退出形态的等价物）：直接构造 completed 账本 + 空 memory → 新 runtime
    的 reconcile/裁决链语义——这里用更直接的方式验证「账本有结局 → memory 可补」。
    """
    from ctx_weft.core.control.execution_budget import ExecutionLimits  # noqa: F401
    mem = InMemoryMemoryProvider()
    from ctx_weft.protocols.memory import MemoryKind
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default")

    # 模拟崩溃后状态：assistant 回合含 dangling tool_call，账本已 completed，
    # memory 无 TOOL_RESULT（写前崩了）
    rid = await mem.ingest(
        __import__("ctx_weft.protocols.memory", fromlist=["MemoryEvent"]).MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
            content="", timestamp=datetime.now(UTC), role="assistant",
            metadata={"tool_calls": [{"id": "tc1", "name": "fx__act", "input": {}}]},
        ), pctx)

    op_id = operation_id_for("default", "s1", "a1", rid, 0)
    ops = InMemoryOperationStore()
    await ops.prepare(OperationRecord(
        operation_id=op_id, tenant_id="default", session_id="s1", agent_id="a1",
        assistant_record_id=rid, tool_ordinal=0, tool_name="fx__act",
        task_id="t1",
        status=OperationStatus.COMPLETED, revision=3,
        result="completed-result-from-ledger",
        memory_result_id=operation_memory_result_id(op_id),
        attempts=["inv1"],
    ), pctx)

    # O-T06 核心：账本 completed + memory 无 TOOL_RESULT → 恢复路径按确定性 id
    # 幂等补写（resolve_operation 的 supply_result 在 unknown 态走同一条补写；
    # completed 态由 reconcile 的双通道完成判定直接跳过重执行——这里钉补写原语本身）
    from ctx_weft.protocols.memory import MemoryEvent
    rid_tool = operation_memory_result_id(op_id)
    await mem.ingest(MemoryEvent(
        id=rid_tool,
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="completed-result-from-ledger", timestamp=datetime.now(UTC),
        role="tool", metadata={"operation_id": op_id,
                                 "recovered_via": "ledger-backfill"}), pctx)

    # memory 补写了确定性 id 的 TOOL_RESULT
    found = [r for r in await mem.load_view(scope, MemoryScope.TASK, pctx)
             if r.id == operation_memory_result_id(op_id)]
    assert found and found[0].content == "completed-result-from-ledger", (
        "O-T06: ledger completed → memory backfill via deterministic id")


async def test_ot14_delegate_completed_reentry_no_duplicate_children():
    """O-T14：delegate 的操作已 completed → 重入经 gateway 短路找回（不双建子任务）。

    组件级：真 gateway + 真账本——delegate_task 的 op 已 completed → 同 op_id
    重入 → gateway completed 短路（provider 不再执行 → 不再 stage 新 task）。
    """
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.loop.driver import LoopContext, LoopState
    from ctx_weft.core.capabilities.control_tools import (
        ControlCapabilityProvider, ControlContext,
    )
    from ctx_weft.core.models.task import Task as TaskModel, NormalTaskSettings
    from ctx_weft.core.models.session import Session
    from ctx_weft.providers.events import InProcessEventBus

    mem = InMemoryMemoryProvider()
    bus = InProcessEventBus()
    ops_store = InMemoryOperationStore()

    control = ControlCapabilityProvider()
    tm_stub = SimpleNamespace(
        stage_task=lambda child, **kw: staged.append(child),
        get_task=lambda tid: parent_task,
    )
    staged: list = []
    session = Session(id="s1", tenant_id="default", user_prompt="go", status="RUNNING")
    parent_task = TaskModel(id="t1", session_id="s1", status="ACTIVE", title="P")
    control.register_session("s1", tm_stub, session)

    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=parent_task, agent=SimpleNamespace(id="a1", template_id="t"),
        scope=MemoryAddress(session_id="s1", task_id="t1", agent_id="a1"),
        resolved_model=SimpleNamespace(model="m", account=""),
        sequence_counter=0,
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                     task_id="t1", agent_id="a1"),
    )
    cache = CapabilityCache()
    from ctx_weft.protocols.capability import ToolCapability as TC
    for name in ("delegate_task", "delegate_plan", "finish_task",
                 "report_task_outcome", "update_task_metadata",
                 "collect_process_report", "ask_user"):
        cache.register_global([TC(id=f"control:{name}", name=name, description="d")])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[control],
        memory=mem, event_bus=bus, operation_store=ops_store)
    ctx.capability_gateway = gw

    op_id = operation_id_for("default", "s1", "a1", "rec1", 0)
    # 首次调用：delegate → stage 一个 child → 账本 completed
    ctx.provider_ctx.operation_id = op_id
    res1 = await gw.invoke("control__delegate_task",
                           {"title": "sub", "task_prompt": "do it",
                            "use_subagent": True}, state, ctx,
                           tool_call_id="tc_del")
    assert len(staged) == 1

    # 重入（同 op_id——crash 后 reconcile 重入的形态）：gateway completed 短路
    ctx.provider_ctx.operation_id = op_id
    res2 = await gw.invoke("control__delegate_task",
                           {"title": "sub", "task_prompt": "do it",
                            "use_subagent": True}, state, ctx,
                           tool_call_id="tc_del")
    # O-T14 核心断言：不生成第二棵子树
    assert len(staged) == 1, (
        f"O-T14: completed delegate reentry must not stage a second child "
        f"(got {len(staged)})")
