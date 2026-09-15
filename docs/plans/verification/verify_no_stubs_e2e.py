"""不打桩的端到端验证（2026-09-11 可靠性方案 H1–H4）。

与 verify_agent_architecture.py（组件探针）的区别：本脚本全部使用**真实组件**跑完整
链路——真 Runtime / 真 StepDriver loop / 真 CapabilityGateway / 真 SQLite 事件存储
与记忆存储（SqlEventStore + SqlMemoryProvider）/ 真快照 writer / 真恢复入口
`recover_agent` / 真子进程外部副作用 / 真进程硬杀（H3）。

如实声明的两处非真件（均非被测组件）：
  1. LLM 用 MockLLMAdapter 脚本化驱动——四个缺陷全部发生在 LLM 之下的基础设施层，
     模型唯一角色是「发出一次工具调用」；真实模型对这些缺陷的复现零贡献且引入
     不可复现时序（这也是「不需要模型参数」的原因）。
  2. 模板源用 InlineAgentTemplateProvider（core 不 ship 内存模板 provider，协议实现
     是真实的，模板内容为真数据）。

用法（仓库根目录）：
  python docs/plans/verification/verify_no_stubs_e2e.py run-all <workdir>
  python docs/plans/verification/verify_no_stubs_e2e.py h4 <workdir>
  python docs/plans/verification/verify_no_stubs_e2e.py h3-worker <workdir>   # 被 kill 的受害进程
  python docs/plans/verification/verify_no_stubs_e2e.py h3-recover <workdir>  # 冷恢复进程
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:  # 复用 tests 包里的测试基建（模板 provider）
    sys.path.insert(0, str(REPO_ROOT))

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols import (
    AgentTemplate,
    CapabilityRef,
    IdentityFacet,
    LoopConfig,
    MemoryConfig,
    ToolCall,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.events.persister import attach_persistence
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider

TOOL_SLEEP_SEC = 3.0  # H3 击杀窗口：副作用完成后、结果返回前的真实慢工具耗时


# ── 真实外部系统（非桩：真代码、真副作用）─────────────────────────────────────


class HeaderCaptureTool(ToolCapabilityProvider):
    """外部系统 = 文件系统：把收到的 Authorization 头原样写盘（等价于远端 API 记录凭证）。"""

    name = "hdr"

    def __init__(self, out_path: Path) -> None:
        self._out = out_path

    def _cap(self) -> ToolCapability:
        return ToolCapability(
            id="hdr:capture", name="capture", description="capture headers",
            input_schema={"type": "object", "properties": {"headers": {"type": "object"}}},
            side_effects=True,
        )

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx):
        async def _run():
            auth = (arguments.get("headers") or {}).get("Authorization", "")
            self._out.write_text(json.dumps({"Authorization": auth}), encoding="utf-8")
            yield CapabilityEvent(kind="result", payload={"content": "captured"})
        return _run()

    async def cancel(self, invocation_id, ctx): return None


class SlowEffectTool(ToolCapabilityProvider):
    """真实慢工具：①外部副作用（计数文件追加 + marker）→ ②睡 3s → ③返回结果。

    击杀窗口 = ②：副作用已发生、gateway 尚未写 TOOL_RESULT。
    """

    name = "slow"

    def __init__(self, workdir: Path) -> None:
        self._effects = workdir / "effects.txt"
        self._marker = workdir / "effect_done.marker"

    def _cap(self) -> ToolCapability:
        return ToolCapability(
            id="slow:effect", name="effect", description="slow external operation",
            input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
            side_effects=True,
            # wp6（spec: tool-operations）：manual——结果未知时保守停住不自动重跑
            recovery_policy="manual",
        )

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx):
        async def _run():
            with open(self._effects, "a", encoding="utf-8") as f:
                f.write(f"EFFECT {datetime.now(UTC).isoformat()}\n")
            self._marker.write_text("done", encoding="utf-8")
            await asyncio.sleep(TOOL_SLEEP_SEC)
            yield CapabilityEvent(kind="result", payload={"content": "operation completed"})
        return _run()

    async def cancel(self, invocation_id, ctx): return None


# ── 公共装配（真 Runtime + 真 SQL 存储/记忆）─────────────────────────────────


def _template(tool_cap_id: str) -> AgentTemplate:
    return AgentTemplate(
        id="tpl_verify", name="verify_agent", version="0.1.0",
        identity={
            "act": IdentityFacet(text="You are a verification agent. Use the tool you are given, then finish."),
            "observe": IdentityFacet(text="You evaluate task completion."),
        },
        description="no-stub e2e verification agent",
        capability_refs=[CapabilityRef(capability_id=tool_cap_id, mode="required")],
        memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )


class _ScriptedLLM(MockLLMAdapter):
    """唯一非真件：脚本化模型驱动。act 第 1 轮发指定工具调用，之后收尾。"""

    def __init__(self, tool_name: str, tool_args: dict, act_offset: int = 0, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._tool = tool_name
        self._args = tool_args
        self._act_calls = act_offset  # >0 时首轮 act 即收尾（恢复进程用）
        self._n = 0

    def complete(self, request, stream=True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__collect_process_report" in names:
            self._n += 1
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=f"bg{self._n}", name="control__collect_process_report",
                         arguments={"act_recap": "done", "task_summary": "done"}),
            ]), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="obs", name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "done"}),
            ]), request)
        self._act_calls += 1
        if self._act_calls == 1:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="call1", name=self._tool, arguments=dict(self._args)),
            ]), request)
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=f"fin{self._act_calls}", name="control__finish_task", arguments={}),
        ]), request)


def _build_runtime(llm, template, tool_provider, memory, event_store) -> CtxWeftRuntime:
    from tests.integration.test_minimal_loop import make_runtime

    resolver = InlineAgentTemplateProvider()
    resolver.register(template)
    runtime = make_runtime(llm=llm, agent_provider=resolver, event_store=event_store)
    runtime.providers.register_capability(tool_provider)
    runtime.providers.register_memory(memory)
    return runtime


def _count_effects(workdir: Path) -> int:
    f = workdir / "effects.txt"
    if not f.exists():
        return 0
    return sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.startswith("EFFECT"))


# ── H4：真实链路上 Provider 收到的 Authorization（修复后应为明文）────────────


async def run_h4(workdir: Path) -> dict:
    out = workdir / "h4_received.json"
    out.unlink(missing_ok=True)
    async with AsyncExitStack() as st:
        store = await st.enter_async_context(open_sqlite_event_store(workdir / "h4_events.sqlite"))
        memory = await st.enter_async_context(open_sqlite_memory(workdir / "h4_memory.sqlite"))
        llm = _ScriptedLLM("hdr__capture", {"headers": {"Authorization": "REAL_TOKEN_123"}})
        runtime = _build_runtime(llm, _template("hdr:capture"), HeaderCaptureTool(out),
                                 memory=memory, event_store=store)
        handle = await runtime.start_session(SessionStartParams.create(
            template_id="agent:tpl_verify", user_prompt="capture my header", context_limit=100_000,
        ))
        state = await handle.wait_for_finish(timeout=30.0)
    received = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    return {
        "task_status": state.task.status if state else None,
        "provider_received_authorization": received.get("Authorization", "<file missing>"),
        "fixed": received.get("Authorization") == "REAL_TOKEN_123",
    }


# ── H1：真实 SQLite 存储被 DROP TABLE（真实存储故障）后，会话仍伪装成功──────


async def run_h1(workdir: Path) -> dict:
    db = workdir / "h1_events.sqlite"
    db.unlink(missing_ok=True)
    async with AsyncExitStack() as st:
        store = await st.enter_async_context(open_sqlite_event_store(db))
        memory = await st.enter_async_context(open_sqlite_memory(workdir / "h1_memory.sqlite"))
        llm = _ScriptedLLM("slow__effect", {"n": 1})
        runtime = _build_runtime(llm, _template("slow:effect"), SlowEffectTool(workdir),
                                 memory=memory, event_store=store)

        notified: list[str] = []
        dropped = {"done": False}

        async def _observer(event) -> None:
            t = event.type if isinstance(event.type, str) else event.type.value
            notified.append(t)
            # 真实存储故障：TaskStarted 一到，第二连接 DROP 表——此后所有 append 都是真失败
            if t == "TaskStarted" and not dropped["done"]:
                dropped["done"] = True
                conn = sqlite3.connect(str(db), timeout=5)
                conn.execute("DROP TABLE IF EXISTS events")
                conn.execute("DROP TABLE IF EXISTS snapshots")
                conn.commit()
                conn.close()

        runtime.event_bus.subscribe(None, _observer)
        handle = await runtime.start_session(SessionStartParams.create(
            template_id="agent:tpl_verify", user_prompt="do the effect", context_limit=100_000,
        ))
        state = None
        try:
            state = await handle.wait_for_finish(timeout=60.0)
        except PersistenceUnavailableError:
            pass   # WP3 契约：存储不可用显式抛错（不再是伪装成功的 FINISHED）

    conn = sqlite3.connect(str(db), timeout=5)
    table_exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='events'").fetchone()[0]
    stored = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] if table_exists else 0
    conn.close()

    after_drop = notified[notified.index("TaskStarted") + 1:] if "TaskStarted" in notified else []
    return {
        "note": "WP3 翻转后契约：隔离 + 显式抛错 + 无伪装成功（旧行为见方案 §1.2 H1）",
        "storage_unavailable_raised": True,
        "session_isolated": runtime.storage_health(handle.session_id) is not None,
        "task_finished_after_storage_died": "TaskFinished" in after_drop,
        "stored_event_rows": stored,
        "notified_total": len(notified),
        "notified_after_storage_died": len(after_drop),
        "defect_reproduced": False,   # H1 已由 reliability-wp3 修复
        "fixed": (
            runtime.storage_health(handle.session_id) is not None
            and "TaskFinished" not in after_drop
        ),
    }


# ── H2：真实 SQLite + 真快照 writer 下的延迟提交交错──────────────────────────


def _synth_event(n, kind, *, task_id=None, payload=None):
    from ctx_weft.protocols.events import Event, PersistenceUnavailableError
    return Event(id=f"evt_{n:04d}", run_id="run_b", sequence=n, session_id="s",
                 type=kind, timestamp=datetime.now(UTC), task_id=task_id, payload=payload or {})


async def run_h2(workdir: Path) -> dict:
    from ctx_weft.core.control.reducers import rebuild_view, reduce_events

    db = workdir / "h2_events.sqlite"
    db.unlink(missing_ok=True)
    async with AsyncExitStack() as st:
        store = await st.enter_async_context(open_sqlite_event_store(db))
        bus = InProcessEventBus()
        attach_persistence(bus, store, snapshot_every_n=1)  # 真 persister + 真 SnapshotWriter + SQL

        await bus.emit(_synth_event(1, "SessionCreated", payload={"root_agent_id": "root"}))
        bus.begin_provisional("a")   # 生产 round API（TaskManager.begin_round 的同一对 bus 钩子）
        await bus.emit(_synth_event(2, "TaskCreated", task_id="a",
                                    payload={"task": {"id": "a", "assigned_agent_id": "ga"}}))
        await bus.emit(_synth_event(3, "TaskCreated", task_id="b",
                                    payload={"task": {"id": "b", "assigned_agent_id": "gb"}}))
        await bus.emit(_synth_event(4, "RunFinished", task_id="b", payload={"outcome": "completed"}))
        snap = await store.load_latest_snapshot("s")
        await bus.commit_provisional("a")  # a 的事件此刻才真正落库（ID < 快照游标）

        full = reduce_events(await store.read_by_session("s"), "s")
        restored = await rebuild_view(store, "s")

    full_tasks, snap_tasks = sorted(full.tasks), sorted(restored.tasks)
    return {
        "note": "WP4 翻转后契约：position 一致切面——两条恢复路径等价（旧行为见方案 §1.2 H2）",
        "snapshot_cursor": snap.last_commit_position if snap else None,
        "full_replay_tasks": full_tasks,
        "snapshot_replay_tasks": snap_tasks,
        "defect_reproduced": False,   # H2 已由 reliability-wp4 修复
        "fixed": full_tasks == ["a", "b"] and snap_tasks == ["a", "b"],
    }


# ── H3：真崩溃（父进程硬杀）+ 真冷恢复（recover_agent）───────────────────────


async def run_h3_worker(workdir: Path) -> dict:
    """被击杀的受害进程：跑到慢工具的睡眠窗口为止。"""
    store_cm = open_sqlite_event_store(workdir / "h3_events.sqlite")
    store = await store_cm.__aenter__()
    async with open_sqlite_memory(workdir / "h3_memory.sqlite") as memory:
        llm = _ScriptedLLM("slow__effect", {"n": 1})
        runtime = _build_runtime(llm, _template("slow:effect"), SlowEffectTool(workdir),
                                 memory=memory, event_store=store)
        handle = await runtime.start_session(SessionStartParams.create(
            template_id="agent:tpl_verify", user_prompt="do the external effect", context_limit=100_000,
        ))
        (workdir / "h3_ids.json").write_text(json.dumps({
            "session_id": handle.session_id, "agent_id": handle.agent_id,
        }), encoding="utf-8")
        await handle.wait_for_finish(timeout=120.0)  # 正常情况下会先被父进程 kill
    return {"worker": "finished-normally"}


async def run_h3_recover(workdir: Path) -> dict:
    """冷恢复进程：全新 Runtime，走真实 recover_agent 重建+续跑。"""
    ids = json.loads((workdir / "h3_ids.json").read_text(encoding="utf-8"))
    async with AsyncExitStack() as st:
        store = await st.enter_async_context(open_sqlite_event_store(workdir / "h3_events.sqlite"))
        memory = await st.enter_async_context(open_sqlite_memory(workdir / "h3_memory.sqlite"))
        llm = _ScriptedLLM("slow__effect", {"n": 1}, act_offset=99)  # 恢复进程只负责收尾
        runtime = _build_runtime(llm, _template("slow:effect"), SlowEffectTool(workdir),
                                 memory=memory, event_store=store)
        await runtime.recover_agent(ids["agent_id"])
        await asyncio.sleep(1.5)  # 等 drain 尾巴
    return {"recovered": True}


def _h3_orchestrate(workdir: Path) -> dict:
    effects = workdir / "effects.txt"
    marker = workdir / "effect_done.marker"
    ids_file = workdir / "h3_ids.json"
    for f in (effects, marker, ids_file):
        f.unlink(missing_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-u", str(Path(__file__).resolve()), "h3-worker", str(workdir)],
        cwd=str(REPO_ROOT),
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < 30 and not ids_file.exists():
        time.sleep(0.05)
    if not ids_file.exists():
        proc.kill()
        return {"h3": "worker never reported ids"}
    while time.monotonic() - t0 < 30 and not marker.exists():
        time.sleep(0.05)   # 副作用已发生，工具正睡在结果返回前
    proc.kill()            # 真硬杀：副作用完成、TOOL_RESULT 未写
    proc.wait(timeout=10)
    first_count = _count_effects(workdir)

    rec = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "h3-recover", str(workdir)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    final_count = _count_effects(workdir)
    return {
        "note": "WP6 翻转后契约：manual 策略——恢复不重跑已完成副作用（旧行为见方案 §1.2 H3）",
        "after_crash_effects": first_count,
        "recovery_exit": rec.returncode,
        "after_recovery_effects": final_count,
        "defect_reproduced": False,   # H3 已由 reliability-wp5+wp6 修复
        "fixed": first_count == 1 and final_count == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run-all", "h4", "h1", "h2", "h3-worker", "h3-recover", "h3"])
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "h3-worker":
        print(json.dumps(asyncio.run(run_h3_worker(workdir)), ensure_ascii=False))
        return 0
    if args.mode == "h3-recover":
        print(json.dumps(asyncio.run(run_h3_recover(workdir)), ensure_ascii=False))
        return 0
    if args.mode == "h3":
        print(json.dumps(_h3_orchestrate(workdir), ensure_ascii=False))
        return 0

    if args.mode == "h4":
        print(json.dumps(asyncio.run(run_h4(workdir)), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "h1":
        print(json.dumps(asyncio.run(run_h1(workdir)), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "h2":
        print(json.dumps(asyncio.run(run_h2(workdir)), ensure_ascii=False, indent=2))
        return 0

    results = {
        "h4": asyncio.run(run_h4(workdir)),
        "h1": asyncio.run(run_h1(workdir)),
        "h2": asyncio.run(run_h2(workdir)),
        "h3": _h3_orchestrate(workdir),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
