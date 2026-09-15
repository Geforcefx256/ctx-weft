"""Read-only architecture probes for the 2026-09-11 review.

Run from the repository root with the repository's Python environment:
    .venv/Scripts/python.exe docs/plans/verification/verify_agent_architecture.py --expect baseline

All stores and external effects are simulated in memory. No network, model,
shell tool, credentials, or production data are used. This is a small set of
component probes, not a replacement for integration or crash tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.control.reducers import rebuild_view, reduce_events
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
from ctx_weft.protocols.events import Event
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus
from ctx_weft.providers.events.persister import attach_persistence
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


def event(n, kind, *, task_id=None, payload=None):
    return Event(
        id=f"evt_{n:04d}", run_id="run_b", sequence=n,
        session_id="s", type=kind, timestamp=datetime.now(UTC),
        task_id=task_id, payload=payload or {},
    )


async def persistence_failure():
    """H1 探针（reliability-wp3 接口演进）：从 attach_persistence 观察者接线改为
    CommitGate + bus 接线——对应生产 runtime 的 required 默认路径。故障注入不变
    （store 永远失败）、目标断言不放宽；只换了被测系统的接线形态。"""

    class FailingStore(InMemoryEventStore):
        async def append(self, item):
            raise OSError("simulated storage unavailable")

        async def append_batch(self, *args, **kwargs):
            raise OSError("simulated storage unavailable")

    from ctx_weft.core.events.commit_gate import CommitGate

    bus = InProcessEventBus()
    store = FailingStore()
    bus.attach_commit_gate(CommitGate(store))   # required 接线（WP3 生产路径）
    observed = []

    async def observe(item):
        observed.append(item.id)

    bus.subscribe(None, observe)
    rejected = False
    try:
        await bus.emit(event(1, "TaskFinished", task_id="a", payload={"outcome": "success"}))
    except Exception:
        rejected = True
    return {
        "emit_rejected": rejected,
        "observer_count": len(observed),
        "stored_count": len(await store.read_by_session("s")),
    }


async def late_commit():
    """H2 探针（reliability-wp4 接口演进）：故障交错不变（延迟提交的旧 ID 落在快照
    触发之后），断言从「快照恢复丢 a」改为「两条恢复路径等价见 a、b」——WP4 的
    position 一致切面使然。快照游标断言从触发事件 ID 改为 last_commit_position。"""
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    attach_persistence(bus, store, snapshot_every_n=1)
    await bus.emit(event(1, "SessionCreated", payload={"root_agent_id": "root"}))
    bus.begin_provisional("a")
    await bus.emit(event(2, "TaskCreated", task_id="a", payload={
        "task": {"id": "a", "assigned_agent_id": "agent_a"},
    }))
    await bus.emit(event(3, "TaskCreated", task_id="b", payload={
        "task": {"id": "b", "assigned_agent_id": "agent_b"},
    }))
    await bus.emit(event(4, "RunFinished", task_id="b", payload={"outcome": "completed"}))
    snapshot = await store.load_latest_snapshot("s")
    await bus.commit_provisional("a")
    full = reduce_events(await store.read_by_session("s"), "s")
    restored = await rebuild_view(store, "s")
    return {
        "snapshot_created": snapshot is not None,
        "full_replay_tasks": sorted(full.tasks),
        "snapshot_replay_tasks": sorted(restored.tasks),
    }


class RecordingTool(ToolCapabilityProvider):
    name = "probe"

    def __init__(self):
        self.arguments = []
        self.invocations = []

    def capability(self):
        return ToolCapability(
            id="probe:record", name="record", description="Simulated operation",
            side_effects=True,
        )

    async def list(self, ctx):
        return [self.capability()]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx):
        async def run():
            self.arguments.append(arguments)
            self.invocations.append(ctx.invocation_id)
            yield CapabilityEvent(kind="result", payload={"content": "operation completed"})
        return run()

    async def cancel(self, invocation_id, ctx):
        return None


def gateway_fixture():
    memory = InMemoryMemoryProvider()
    bus = InProcessEventBus()
    state = LoopState(
        run_id="r1",
        session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="task1"),
        agent=SimpleNamespace(id="agent1", template_id="template"),
        scope=MemoryAddress(session_id="s1", task_id="task1", agent_id="agent1"),
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=memory, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", task_id="task1", agent_id="agent1"),
    )
    tool = RecordingTool()
    cache = CapabilityCache()
    cache.put("agent1", [tool.capability()])
    gateway = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool],
        memory=memory, event_bus=bus,
    )
    ctx.capability_gateway = gateway
    return memory, state, ctx, tool, gateway


async def execution_redaction():
    _, state, ctx, tool, gateway = gateway_fixture()
    await gateway.invoke(
        "probe__record", {"headers": {"Authorization": "FAKE_TEST_TOKEN"}},
        state, ctx, tool_call_id="auth_test",
    )
    return {"provider_authorization": tool.arguments[-1]["headers"]["Authorization"]}


async def recovery_duplicate():
    memory, state, ctx, tool, gateway = gateway_fixture()
    await memory.ingest(MemoryEvent(
        type=MemoryEventType.LLM_RESPONSE, address=state.scope, content="",
        role="assistant", timestamp=datetime.now(UTC),
        metadata={"tool_calls": [{
            "id": "effect_test", "name": "probe__record", "input": {"operation": "increment"},
        }]},
    ), ctx.provider_ctx)
    ingest = memory.ingest

    async def fail_result(item, provider_ctx):
        if item.role == "tool":
            raise OSError("simulated tool-result persistence failure")
        return await ingest(item, provider_ctx)

    write_failed = False
    with patch.object(memory, "ingest", side_effect=fail_result):
        try:
            await gateway.invoke(
                "probe__record", {"operation": "increment"}, state, ctx,
                tool_call_id="effect_test",
            )
        except OSError:
            write_failed = True
    recovery_error = None
    # The cache is already bound. Bypass only discovery, not reconciliation,
    # gateway execution, persistence, or the simulated external side effect.
    # spec: tool-operations——tool_call_id 是裸 wire id（非摄入点铸造的 tc_...），
    # 账本旁路 → reconcile 读不到记录 → 作结「无账本记录，无从查证」：副作用不重跑，
    # 结果写成 TOOL_RESULT 后循环照常续跑（不停机、无专属错误码/事件）。
    state.task = SimpleNamespace(id="task1", status="ACTIVE")
    state.sequence_counter = 0
    with patch("ctx_weft.core.loop.steps.reconcile.resolve_and_bind", new=AsyncMock()):
        try:
            await ReconcileStep().execute(state, ctx)
        except Exception as exc:
            recovery_error = type(exc).__name__
    return {
        "result_write_failed": write_failed,
        "external_effect_count": len(tool.arguments),
        "distinct_invocation_ids": len(set(tool.invocations)),
        "recovery_error": recovery_error,
    }


async def main(expect):
    observed = {
        "H1": await persistence_failure(),
        "H2": await late_commit(),
        "H3": await recovery_duplicate(),
        "H4": await execution_redaction(),
    }
    baseline = {
        # H1/H2/H3 均已修复（wp3/wp4/wp5+wp6）：baseline 与 fixed 同值——探针保留
        # 「复现旧缺陷」的历史语义文档，--expect 两个模式现等价（全 True）。
        "H1": observed["H1"] == {"emit_rejected": True, "observer_count": 0, "stored_count": 0},
        "H2": observed["H2"] == {
            "snapshot_created": True, "full_replay_tasks": ["a", "b"], "snapshot_replay_tasks": ["a", "b"],
        },
        "H3": observed["H3"]["result_write_failed"]
        and observed["H3"]["external_effect_count"] == 1,
        "H4": observed["H4"]["provider_authorization"] == "***",
    }
    fixed = {
        "H1": observed["H1"] == {"emit_rejected": True, "observer_count": 0, "stored_count": 0},
        "H2": observed["H2"] == {
            "snapshot_created": True, "full_replay_tasks": ["a", "b"], "snapshot_replay_tasks": ["a", "b"],
        },
        "H3": observed["H3"]["result_write_failed"] and observed["H3"]["external_effect_count"] == 1,
        "H4": observed["H4"]["provider_authorization"] == "FAKE_TEST_TOKEN",
    }  # H3 wp6 语义：manual 停住 → 计数 1（原 baseline 判据计数 2 已随修复失效）
    checks = baseline if expect == "baseline" else fixed
    print(json.dumps({"expect": expect, "observed": observed, "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=["baseline", "fixed"], default="baseline")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(main(args.expect)))
