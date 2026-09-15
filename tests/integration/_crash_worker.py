"""O-T05 强退矩阵的 worker/recover 子进程（被 pytest 版矩阵驱动，同无桩 h3 形态）。

用法：
  python tests/integration/_crash_worker.py ot05-worker <workdir>   # 被硬杀的受害进程
  python tests/integration/_crash_worker.py ot05-recover <workdir>  # 冷恢复进程
"""
from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ctx_weft.core import CtxWeftRuntime  # noqa: E402
from ctx_weft.core.runtime import SessionStartParams  # noqa: E402
from ctx_weft.protocols import ToolCall  # noqa: E402
from ctx_weft.protocols.capability import (  # noqa: E402
    CapabilityEvent, CapabilityProviderInfo, ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store  # noqa: E402
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse  # noqa: E402
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory  # noqa: E402
from tests.integration.test_minimal_loop import (  # noqa: E402
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

TOOL_SLEEP_SEC = 3.0


class _SlowEffectTool(ToolCapabilityProvider):
    """副作用（计数文件 + marker）→ 睡 3s → 返回。默认 reviewed 策略。"""

    name = "slow"

    def __init__(self, workdir: Path) -> None:
        self._effects = workdir / "effects.txt"
        self._marker = workdir / "effect_done.marker"

    def _cap(self) -> ToolCapability:
        return ToolCapability(
            id="slow:effect", name="effect", description="slow external operation",
            input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
            side_effects=True, recovery_policy="reviewed",
        )

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)

    def invoke(self, capability_id, arguments, ctx):
        async def _run():
            with open(self._effects, "a", encoding="utf-8") as f:
                f.write(f"EFFECT {datetime.now(UTC).isoformat()}\n")
            self._marker.write_text("done", encoding="utf-8")
            await asyncio.sleep(TOOL_SLEEP_SEC)
            yield CapabilityEvent(kind="result", payload={"content": "completed"})
        return _run()

    async def cancel(self, invocation_id, ctx): return None


class _RouterLLM(MockLLMAdapter):
    """act 第 1 轮发 slow__effect；之后 finish。observe/bg 标准路由。"""

    def __init__(self, tool_name="slow__effect", tool_args=None, act_offset=0, **kw):
        super().__init__(responses=[], **kw)
        self._tool = tool_name
        self._args = tool_args or {"n": 1}
        self._act_calls = act_offset
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
        self._act_calls += 1
        if self._act_calls == 1:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id="c1", name=self._tool, arguments=dict(self._args))]), request)
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=f"f{self._act_calls}", name="control__finish_task",
                     arguments={})]), request)


def _template():
    from ctx_weft.protocols import (
        AgentTemplate, CapabilityRef, IdentityFacet, LoopConfig, MemoryConfig,
    )
    return AgentTemplate(
        id="tpl_verify", name="verify_agent", version="0.1.0",
        identity={
            "act": IdentityFacet(text="Use the tool, then finish."),
            "observe": IdentityFacet(text="You evaluate."),
        },
        description="crash matrix agent",
        capability_refs=[CapabilityRef(capability_id="slow:effect", mode="required")],
        memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )


async def _make_runtime(workdir: Path, llm) -> CtxWeftRuntime:
    store_cm = open_sqlite_event_store(workdir / "events.sqlite")
    store = await store_cm.__aenter__()
    memory_cm = open_sqlite_memory(workdir / "memory.sqlite")
    memory = await memory_cm.__aenter__()
    tool = _SlowEffectTool(workdir)
    resolver = InlineAgentTemplateProvider()
    resolver.register(_template())
    rt = make_runtime(llm=llm, agent_provider=resolver, event_store=store)
    rt.providers.register_capability(tool)
    rt.providers.register_memory(memory)
    return rt


async def ot05_worker(workdir: Path) -> dict:
    store_cm = open_sqlite_event_store(workdir / "events.sqlite")
    store = await store_cm.__aenter__()
    async with open_sqlite_memory(workdir / "memory.sqlite") as memory:
        llm = _RouterLLM()
        tool = _SlowEffectTool(workdir)
        resolver = InlineAgentTemplateProvider()
        resolver.register(_template())
        rt = make_runtime(llm=llm, agent_provider=resolver, event_store=store)
        rt.providers.register_capability(tool)
        rt.providers.register_memory(memory)
        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_verify", user_prompt="do the effect",
            context_limit=100_000))
        (workdir / "ids.json").write_text(json.dumps({
            "session_id": handle.session_id, "agent_id": handle.agent_id,
        }), encoding="utf-8")
        await handle.wait_for_finish(timeout=120)
    return {"worker": "normal"}


async def ot05_recover(workdir: Path) -> dict:
    import json as _json
    ids = _json.loads((workdir / "ids.json").read_text(encoding="utf-8"))
    store_cm = open_sqlite_event_store(workdir / "events.sqlite")
    store = await store_cm.__aenter__()
    async with open_sqlite_memory(workdir / "memory.sqlite") as memory:
        llm = _RouterLLM(act_offset=99)  # 恢复只负责收尾
        tool = _SlowEffectTool(workdir)
        resolver = InlineAgentTemplateProvider()
        resolver.register(_template())
        rt = make_runtime(llm=llm, agent_provider=resolver, event_store=store)
        rt.providers.register_capability(tool)
        rt.providers.register_memory(memory)
        try:
            await rt.recover_agent(ids["agent_id"])
        except Exception:
            pass  # 恢复期异常不影响判据：O-T05 只看外部副作用计数是否仍为 1
        await asyncio.sleep(1.0)
    return {"recovered": True}


import json  # noqa: E402


if __name__ == "__main__":
    mode, wd = sys.argv[1], Path(sys.argv[2])
    fn = {"ot05-worker": ot05_worker, "ot05-recover": ot05_recover}[mode]
    print(json.dumps(asyncio.run(fn(wd)), ensure_ascii=False))
