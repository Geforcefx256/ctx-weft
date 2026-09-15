"""WP0 基线夹具（H1）——已随 reliability-wp3（spec: event-commit）翻转。

历史：本文件原钉「落库失败仍对外通知成功」的缺陷现状（探针 verify_agent_architecture.py
`persistence_failure` 的 Runtime 级移植）。WP3 落地 required 提交门后翻转：
  - drop-table → 会话进入 storage_unavailable 隔离（健康表可查）
  - wait_for_finish 显式抛 PersistenceUnavailableError（不再等通用超时）
  - 无 committed 通知流出（TaskFinished 不再到达观察者）
  - 落库为零（原本如此）
best_effort 形态保留，钉住旧契约（兼容路径的语义锚）。

回链：docs/plans/2026-09-11-agent-core-reliability-plan.md §1.2/§4.5/§4.7。
"""
from __future__ import annotations

import sqlite3

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio


class _FinishLLM(MockLLMAdapter):
    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
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
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=f"fin{self._n}", name="control__finish_task", arguments={}),
        ]), request)


class FailingEventStore(InMemoryEventStore):
    """append / append_batch 永远失败——模拟存储不可用（读路径保持可用以便断言）。"""

    async def append(self, item):
        raise OSError("simulated storage unavailable")

    async def append_batch(self, *args, **kwargs):
        raise OSError("simulated storage unavailable")


async def test_storage_failure_isolates_session_no_fake_success():
    """required（默认，WP3 翻转后）：存储失败 → 隔离、显式抛错、无 committed 通知。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=_FinishLLM(), agent_provider=resolver,
                           event_store=FailingEventStore())
    runtime.providers.register_memory(InMemoryMemoryProvider())

    notified: list[str] = []

    async def _observe(event) -> None:
        notified.append(event.type if isinstance(event.type, str) else event.type.value)

    runtime.event_bus.subscribe(None, _observe)

    # 存储从出生就死：第一条 emit（SessionCreated）的提交确认即失败——start_session
    # 响亮失败（spec: event-commit「不伪装成功」），而非返回一个注定悬空的句柄。
    with pytest.raises(PersistenceUnavailableError):
        await runtime.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="hello", context_limit=100_000,
        ))

    # 健康表已标记（会话 id 在事件里、由健康表留存；此处断言非空 + 原因含故障类名）
    assert runtime._storage_unavailable, "session must be marked storage_unavailable"
    assert "OSError" in next(iter(runtime._storage_unavailable.values()))
    assert "TaskFinished" not in notified, (
        f"committed notification leaked during storage outage: {notified}"
    )


async def test_sql_drop_table_isolates_and_stops_scheduling(tmp_path):
    """真 SQLite DROP TABLE（无桩 H1 场景）在 required 下的新契约：隔离 + 停推进。"""
    db = tmp_path / "events.sqlite"
    async with open_sqlite_event_store(db) as store:
        resolver = InlineAgentTemplateProvider()
        resolver.register(make_echo_template())
        runtime = make_runtime(llm=_FinishLLM(), agent_provider=resolver, event_store=store)
        runtime.providers.register_memory(InMemoryMemoryProvider())

        notified: list[str] = []

        async def _observe(event) -> None:
            t = event.type if isinstance(event.type, str) else event.type.value
            notified.append(t)
            if t == "TaskStarted":          # 真实存储故障：第二连接 DROP 表
                conn = sqlite3.connect(str(db), timeout=5)
                conn.execute("DROP TABLE IF EXISTS events")
                conn.execute("DROP TABLE IF EXISTS event_session_head")
                conn.execute("DROP TABLE IF EXISTS event_batches")
                conn.commit()
                conn.close()

        runtime.event_bus.subscribe(None, _observe)
        handle = await runtime.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="hello", context_limit=100_000,
        ))
        with pytest.raises(PersistenceUnavailableError):
            await handle.wait_for_finish(timeout=15.0)
        assert runtime.storage_health(handle.session_id) is not None
        # 死后不再有 committed 通知（对照旧行为：死后仍通知 53 条）
        after = notified[notified.index("TaskStarted") + 1:] if "TaskStarted" in notified else []
        assert "TaskFinished" not in after


async def test_best_effort_keeps_old_swallow_contract():
    """best_effort：旧契约锚（兼容路径）——存储失败被吞、会话照常跑完、落库为零。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(
        llm=_FinishLLM(), agent_provider=resolver, event_store=FailingEventStore(),
        config=RuntimeConfig(event_commit_policy="best_effort"))
    runtime.providers.register_memory(InMemoryMemoryProvider())

    notified: list[str] = []

    async def _observe(event) -> None:
        notified.append(event.type if isinstance(event.type, str) else event.type.value)

    runtime.event_bus.subscribe(None, _observe)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hello", context_limit=100_000,
    ))
    state = await handle.wait_for_finish(timeout=10.0)
    assert state is not None and state.task.status == "FINISHED"
    assert any(t == "TaskFinished" for t in notified)
