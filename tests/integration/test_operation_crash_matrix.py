"""子进程强退矩阵（spec: tool-operations；wp8-2，方案 O-T05/O-T06/O-T14）。

真子进程退出 + 新 Runtime 实例（无桩 h3 的 pytest 形态）。三条：
- O-T05：reviewed started → 真退出 → 新实例 → 副作用 1 次（不重跑，作结后续跑）
- O-T06：FINISHED 已发而 memory 写前崩溃 → 从事件里那份补写 TOOL_RESULT，不重执行
- O-T14：delegate 已完成后确认丢失 → 重入折出结局、找回原 child（不双建）
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
from ctx_weft.core.utils.ids import mint_call_id

#: 摄入点铸造的内部标识——既是 tool_call id 也是事件 payload 的配对键。
TC1 = mint_call_id(anchor="rec1", ordinal=0, raw_id="tc1", turn_seq=0)
TC_DEL = mint_call_id(anchor="rec1", ordinal=1, raw_id="tc_del", turn_seq=0)

from ctx_weft.core.loop.capability_gateway import tool_result_record_id
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
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


async def test_ot05_reviewed_started_real_exit_side_effect_once(tmp_path):
    """O-T05：reviewed started → worker 子进程硬杀 → 新实例 recover → 副作用 1 次。"""
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
        f"O-T05: reviewed policy must NOT re-run side effect after real crash "
        f"(got {final_count})")


async def test_ot06_finished_but_memory_write_crashed_backfills(tmp_path):
    """O-T06：`CapabilityFinished` 已发而 TOOL_RESULT 写前崩溃 → 从事件里那份补写。

    组件级（真退出形态的等价物）：播出 INVOKED + FINISHED、memory 无 TOOL_RESULT，
    跑真 reconcile，验它按确定性 id 幂等补写、且不重执行 provider。
    """
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.events.commit_gate import CommitGate
    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.loop.driver import LoopContext, LoopState
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep
    from ctx_weft.protocols.capability import (
        CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider)
    from ctx_weft.protocols.events import Event, EventType
    from ctx_weft.protocols.memory import MemoryEvent, MemoryKind
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore

    calls: list[int] = []

    class _Tool(ToolCapabilityProvider):
        name = "fx"
        def _cap(self):
            return ToolCapability(id="fx:act", name="act", description="d",
                                  side_effects=True)
        async def list(self, ctx): return [self._cap()]
        async def retrieve(self, ctx): return [self._cap()]
        async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)
        async def cancel(self, i, ctx): return None
        def invoke(self, cid, args, ctx):
            async def _r():
                calls.append(1)
                yield CapabilityEvent(kind="result", payload={"content": "re-run!"})
            return _r()

    mem, tool = InMemoryMemoryProvider(), _Tool()
    event_store, bus = InMemoryEventStore(), InProcessEventBus()
    bus.attach_commit_gate(CommitGate(event_store))
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=datetime.now(UTC), role="assistant",
        metadata={"tool_calls": [{"id": TC1, "name": "fx__act", "input": {}}]}), pctx)

    # 崩溃前的事件：调用过、也完成了——只是 TOOL_RESULT 没写成。
    for i, (etype, payload) in enumerate((
        (EventType.CAPABILITY_INVOKED, {"tool_call_id": TC1, "invocation_id": "inv1"}),
        (EventType.CAPABILITY_FINISHED, {"tool_call_id": TC1, "invocation_id": "inv1",
                                         "outcome": "success",
                                         "result": "finished-result-from-event",
                                         "result_length": len("finished-result-from-event")}),
    ), start=1):
        await event_store.append(Event(
            id=f"evt_seed_{i}", type=etype, session_id="s1", run_id="r0", sequence=i,
            timestamp=datetime.now(UTC), payload=payload))


    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE"),
        agent=SimpleNamespace(id="a1", template_id="t"), scope=scope,
        resolved_model=SimpleNamespace(model="m", account=""), sequence_counter=0)
    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx, event_store=event_store)
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    ctx.capability_gateway = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool], memory=mem, event_bus=bus)

    await ReconcileStep().execute(state, ctx)

    assert not calls, "O-T06: 已完成的调用不得重执行"
    found = [r for r in await mem.load_view(scope, MemoryScope.TASK, pctx)
             if r.id == tool_result_record_id(TC1)]
    assert found and found[0].content == "finished-result-from-event", (
        "O-T06: FINISHED 已发 → 按确定性 id 从事件里那份补写 memory")


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

    from ctx_weft.core.events.commit_gate import CommitGate
    from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore

    mem = InMemoryMemoryProvider()
    bus = InProcessEventBus()
    event_store = InMemoryEventStore()
    bus.attach_commit_gate(CommitGate(event_store))

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
        event_store=event_store,
    )
    cache = CapabilityCache()
    from ctx_weft.protocols.capability import ToolCapability as TC
    for name in ("delegate_task", "delegate_plan", "finish_task",
                 "report_task_outcome", "update_task_metadata",
                 "collect_process_report", "ask_user"):
        cache.register_global([TC(id=f"control:{name}", name=name, description="d")])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[control],
        memory=mem, event_bus=bus)
    ctx.capability_gateway = gw

    # 首次调用：delegate → stage 一个 child → 账本 completed
    res1 = await gw.invoke("control__delegate_task",
                           {"title": "sub", "task_prompt": "do it",
                            "use_subagent": True}, state, ctx,
                           tool_call_id=TC_DEL)
    assert len(staged) == 1

    # 重入（同一个内部标识——crash 后 reconcile 重入的形态）：gateway 折出「已完成」
    # 后短路，provider 不再执行 → 不再 stage 第二棵子树。
    res2 = await gw.invoke("control__delegate_task",
                           {"title": "sub", "task_prompt": "do it",
                            "use_subagent": True}, state, ctx,
                           tool_call_id=TC_DEL, reentry=True)
    # O-T14 核心断言：不生成第二棵子树
    assert len(staged) == 1, (
        f"O-T14: completed delegate reentry must not stage a second child "
        f"(got {len(staged)})")
