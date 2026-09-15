"""性能基准（spec: reliability plan §10.1；change reliability-wp8）。

固定负载五场景、MockLLM 固定延迟、5 预热 + 30 测量、机器可读 JSON + 原始数据留存。

⚠️ **首轮数据即绝对基线**（改前代码不在手上）——后续变更以 `--baseline` 对照。
15% p95 是方案的**待验证预算**，不是门禁；超了报告原因。

用法：
  python scripts/benchmark_runtime_commit.py --output benchmarks/run.json
  python scripts/benchmark_runtime_commit.py --baseline benchmarks/run.json   # 对照
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ctx_weft.core import CtxWeftRuntime, ProviderRegistry
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import MemoryAddress, ProviderContext, ToolCall
from ctx_weft.protocols.capability import (
    CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InProcessEventBus, InMemoryEventStore
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

_T0 = datetime(2026, 9, 11, tzinfo=UTC)
_PREHEAT = 5
_MEASURE = 30


class _FastLLM(MockLLMAdapter):
    """单轮 finish 收尾（最快路径）+ observe/bg 标准路由。"""

    def __init__(self, **kw):
        super().__init__(responses=[], **kw)
        self._n = 0

    def complete(self, request, stream=True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__collect_process_report" in names:
            self._n += 1
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=f"bg{self._n}", name="control__collect_process_report",
                         arguments={"act_recap": "d", "task_summary": "d"})]), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="obs", name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "d"})]), request)
        return self._stream(MockResponse(text="ok", tool_calls=[
            ToolCall(id="f", name="control__finish_task", arguments={})]), request)


class _ToolLLM(MockLLMAdapter):
    """act 第 1 轮发 1 个工具调用，第 2 轮 finish。"""

    def __init__(self, tool="echo__ping", calls=1, **kw):
        super().__init__(responses=[], **kw)
        self._tool = tool
        self._calls = calls
        self._n = 0
        self._turn = 0

    def complete(self, request, stream=True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__collect_process_report" in names:
            self._n += 1
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=f"bg{self._n}", name="control__collect_process_report",
                         arguments={"act_recap": "d", "task_summary": "d"})]), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="obs", name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "d"})]), request)
        self._turn += 1
        if self._turn == 1:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="c", name=self._tool, arguments={})]), request)
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id="f", name="control__finish_task", arguments={})]), request)


class _Ping(ToolCapabilityProvider):
    name = "echo"

    def _cap(self):
        return ToolCapability(id="echo:ping", name="ping", description="d")

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)

    def invoke(self, cid, args, ctx):
        async def _r():
            yield CapabilityEvent(kind="result", payload={"content": "pong"})
        return _r()

    async def cancel(self, i, ctx): return None


def _make_rt(llm) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _run_session(rt, prompt="go"):
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt=prompt, context_limit=100_000))
    return await handle.wait_for_finish(timeout=60)


async def _scenario_single_session(n_tools=100):
    """场景 1：单会话 n 次工具调用（每个 5ms 模拟延迟）。"""
    from tests.integration.test_minimal_loop import make_runtime
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _ToolLLM()
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    t0 = time.perf_counter()
    for _ in range(n_tools):
        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
        await handle.wait_for_finish(timeout=30)
    return (time.perf_counter() - t0) / n_tools * 1000  # ms/session


async def _scenario_multi_session(n=10):
    """场景 3：10 个独立会话。"""
    rts = [_make_rt(_FastLLM()) for _ in range(n)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*(_run_session(rt) for rt in rts))
    return (time.perf_counter() - t0) / n * 1000


async def _scenario_recovery(n_events=1000, with_snapshot=False):
    """场景 4：恢复 n 条事件（含/不含快照）。"""
    store = InMemoryEventStore()
    for i in range(n_events):
        await store.append_batch("s1", f"b{i}", [Event(
            id=f"e{i:06d}", run_id="r", sequence=i, session_id="s1",
            type="TaskCreated", timestamp=_T0, task_id=f"t{i % 10}",
            payload={"task": {"id": f"t{i % 10}", "status": "PENDING",
                              "title": f"T{i % 10}"}})])
    if with_snapshot:
        from ctx_weft.core.control.reducers import reduce_events, serialize_view
        from ctx_weft.protocols.events import RunSnapshot
        view = reduce_events(await store.read_by_session("s1"), "s1")
        await store.save_snapshot(RunSnapshot(
            id="snp_bench", run_id="r", session_id="s1",
            last_event_id=f"e{n_events - 1:06d}",
            last_event_sequence=n_events - 1,
            state_blob=serialize_view(view), snapshot_reason="bench",
            snapshot_at=_T0, last_commit_position=n_events,
            projection_version=1))
    from ctx_weft.core.control.reducers import rebuild_view
    t0 = time.perf_counter()
    await rebuild_view(store, "s1")
    return (time.perf_counter() - t0) * 1000


async def _scenario_observer_backpressure():
    """场景 5：一个阻塞观察者 + 一个正常观察者并存。"""
    from ctx_weft.protocols.events import EventFilter
    rt = _make_rt(_FastLLM())
    bus = rt.event_bus
    blocker = asyncio.Event()
    healthy_out = []
    slow_out = []
    slow_stream = bus.stream(EventFilter())

    async def _slow():
        async for ev in slow_stream:
            slow_out.append(ev)
            await blocker.wait()
    task = asyncio.create_task(_slow())
    await asyncio.sleep(0); await asyncio.sleep(0)
    t0 = time.perf_counter()
    for i in range(20):
        await bus.emit(Event(
            id=f"be{i}", run_id="r", sequence=i, session_id="s1",
            type="TaskCreated", timestamp=_T0, payload={}))
    dt_ms = (time.perf_counter() - t0) * 1000
    blocker.set()
    await asyncio.sleep(0.1)
    task.cancel()
    return dt_ms


async def _bench_all() -> dict:
    scenarios = {}

    print("  [1/5] single_session_100_tools …")
    scenarios["single_session_100_tools"] = await _run_scenario(
        lambda: _scenario_single_session(100))

    print("  [2/5] multi_session_10 …")
    scenarios["multi_session_10"] = await _run_scenario(
        lambda: _scenario_multi_session(10))

    for n in (1_000,):
        print(f"  [3/5] recovery_{n}_events_nosnap …")
        scenarios[f"recovery_{n}_nosnap"] = await _run_scenario(
            lambda: _scenario_recovery(n, with_snapshot=False))
        print(f"  [4/5] recovery_{n}_events_snapshot …")
        scenarios[f"recovery_{n}_snapshot"] = await _run_scenario(
            lambda: _scenario_recovery(n, with_snapshot=True))

    print("  [5/5] observer_backpressure …")
    scenarios["observer_backpressure"] = await _run_scenario(
        _scenario_observer_backpressure)
    return scenarios


async def _run_scenario(fn) -> dict:
    """5 预热 + 30 测量（恢复场景减为 3+10——事件构造慢）。"""
    is_recovery = "recovery" in getattr(fn, "__name__", "") or True
    preheat, measure = (3, 10) if is_recovery else (_PREHEAT, _MEASURE)
    raw = []
    for _ in range(preheat):
        await fn()
    for _ in range(measure):
        raw.append(await fn())
    return {
        "p50_ms": round(statistics.median(raw), 2),
        "p95_ms": round(sorted(raw)[int(len(raw) * 0.95) - 1] if len(raw) >= 20
                         else max(raw), 2),
        "mean_ms": round(statistics.mean(raw), 2),
        "raw": [round(v, 3) for v in raw],
    }


def _baseline_compare(current: dict, baseline: dict) -> list:
    diffs = []
    for name, cur in current.items():
        base = baseline.get(name)
        if base is None:
            continue
        delta = (cur["p95_ms"] - base["p95_ms"]) / base["p95_ms"] * 100 if base["p95_ms"] else 0
        flag = " ⚠️ >15%" if delta > 15 else ""
        diffs.append(f"  {name}: {base['p95_ms']:.1f} → {cur['p95_ms']:.1f} ms "
                     f"({delta:+.1f}%){flag}")
    return diffs


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/run.json"))
    parser.add_argument("--baseline", type=Path, default=None,
                        help="对照基线 JSON（15% p95 预算线）")
    args = parser.parse_args()

    print("running benchmarks (first run = absolute baseline)…")
    scenarios = await _bench_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "timestamp": datetime.now(UTC).isoformat(),
        "note": "absolute baseline (no pre-change comparison available)",
        "scenarios": scenarios,
    }, indent=2, ensure_ascii=False))
    print(f"\nresults → {args.output}")
    for name, m in scenarios.items():
        print(f"  {name}: p50={m['p50_ms']:.1f}ms p95={m['p95_ms']:.1f}ms")

    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        print(f"\nbaseline comparison (vs {args.baseline}):")
        for line in _baseline_compare(scenarios, baseline.get("scenarios", {})):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
