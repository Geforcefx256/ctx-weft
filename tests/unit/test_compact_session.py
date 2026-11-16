"""Manual compact-only operation over an idle session (Option B + idle-guard)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.errors import SessionBusyError
from ctx_weft.protocols import (
    LoopConfig,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._event_helpers import append_one
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


def _runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY")])
    # 占位 record（手工 materialize 一个从未 instantiate 过的 agent）没有自己的模板——
    # registry 不再借会话那份（2026-09-30）。兜底由 host 声明的母版负责，这里把它配上，
    # 走的是与生产同一条链路：RuntimeConfig.fallback_template_ref → TemplateLookup。
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(fallback_template_ref="agent:tpl_echo"))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


def test_session_busy_error_carries_session_id() -> None:
    err = SessionBusyError("ses_1")
    assert "ses_1" in str(err)


async def test_compact_agent_rejects_busy_session() -> None:
    rt = _runtime()
    sid, aid = "ses_busy", "agt_root"
    rt._agent_lifecycle_manager.register_session(
        sid, tenant_id="default",
    )
    rt._agent_lifecycle_manager.materialize(aid, session_id=sid, tenant_id="default")
    rt._busy_sessions.add(sid)  # simulate an active drain
    with pytest.raises(SessionBusyError):
        await initialize_all(rt)
        await rt.compact_agent(aid)


async def test_compact_agent_unknown_agent_raises() -> None:
    from ctx_weft.core.models.errors import AgentNotFound

    rt = _runtime()
    with pytest.raises(AgentNotFound):
        await initialize_all(rt)
        await rt.compact_agent("agt_missing")


async def test_compact_agent_uses_agent_lifecycle_managers_current_model_not_stale_session_field() -> None:
    """`set_agent_llm` 换模型后手动 compact：送出的 LLMRequest.model 必须是新模型。

    钉住评审 finding：compact_agent 里 `agent.runtime["llm_model"]` 曾经取自
    `session.llm_model`（批次 B 前的真相源，早已停止权威），而不是本次 `materialize()`
    同一处返回的 `ResolvedModel.model`——client 派对了（`rm.client`），模型名却掰旧的。
    """
    resolver = InlineAgentTemplateProvider()
    tmpl = dataclasses.replace(make_echo_template(),
                               loop_config=LoopConfig(compact_keep_last=2))
    resolver.register(tmpl)

    old_llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY-OLD")])
    new_llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY-NEW")])

    class _TwoModelResolver:
        def get_client(self, account=None, model=None, *, ctx):
            return new_llm if model == "model-new" else old_llm

    from ctx_weft.core import ProviderRegistry
    providers = ProviderRegistry()
    rt = make_runtime(llm=old_llm, agent_provider=resolver, providers=providers,
                      config=RuntimeConfig(fallback_template_ref="agent:tpl_echo"))
    rt.providers.register_llm_provider(_TwoModelResolver())
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)

    sid, aid = "ses_m", "agt_root"
    ts = datetime(2026, 6, 16, tzinfo=timezone.utc)
    await append_one(rt.event_store, Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id=sid,
        type=EventType.SESSION_CREATED, timestamp=ts,
        payload={"template_id": f"agent:{tmpl.id}", "user_prompt": "x", "root_agent_id": aid,
                 "llm_model": "mock", "context_limit": 180000},
    ))

    scope = MemoryAddress(session_id=sid, task_id="t_seed", agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id="t_seed", agent_id=aid)
    for i in range(5):
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.USER_PROMPT,
            address=MemoryAddress(session_id=sid, task_id=f"root{i}", agent_id=aid),
            content=f"body {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10)), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"user prompt {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"assistant summary {i}", role="assistant",
            timestamp=ts + timedelta(seconds=i * 10 + 1),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)

    # 先水合 record（生产路径里这一步发生在 root agent 实例化时；此处手工建 session
    # 只走了 event_store，registry 还没见过这个 agent_id）。
    rt._agent_lifecycle_manager.register_session(
        sid, tenant_id="default",
    )
    rt._agent_lifecycle_manager.materialize(aid, session_id=sid, tenant_id="default")

    # host 先经 registry 的真相源换模型……
    await initialize_all(rt)
    changed = await rt.set_agent_llm(aid, llm_model="model-new")
    assert changed is True

    # ……再手动 compact：materialize() 拿到的 client 与 model 必须同源一致。
    await initialize_all(rt)
    await rt.compact_agent(aid)

    assert old_llm.last_request is None, "旧模型不该被调用"
    assert new_llm.last_request is not None, "新模型该被调用（client 派对了）"
    assert new_llm.last_request.model == "model-new", (
        "LLMRequest.model 必须跟 ResolvedModel.model 同源，不能掰回 session.llm_model"
    )


async def test_compact_agent_folds_agent_layer() -> None:
    resolver = InlineAgentTemplateProvider()
    # small keep_last so a handful of dispatch pairs is over budget
    tmpl = dataclasses.replace(make_echo_template(),
                               loop_config=LoopConfig(compact_keep_last=2))
    resolver.register(tmpl)
    # escalating_compact 预算门总开（compact_agent 强制立即压）→ L1 折 agent 层一次调用
    # summarize_for_compact；本例 task_id="" 的当前 task 层无材料可折，L3 guard 拦下、不再空调
    # 第二次 LLM，故只需 1 条 mock 响应。
    llm = MockLLMAdapter(responses=[MockResponse(text="SUMMARY")])
    # 占位 record（手工 materialize 一个从未 instantiate 过的 agent）没有自己的模板——
    # registry 不再借会话那份（2026-09-30）。兜底由 host 声明的母版负责，这里把它配上，
    # 走的是与生产同一条链路：RuntimeConfig.fallback_template_ref → TemplateLookup。
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(fallback_template_ref="agent:tpl_echo"))
    mem = InMemoryMemoryProvider()
    rt.providers.register_memory(mem)

    sid, aid = "ses_c", "agt_root"
    ts = datetime(2026, 6, 16, tzinfo=timezone.utc)
    await append_one(rt.event_store, Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id=sid,
        type=EventType.SESSION_CREATED, timestamp=ts,
        payload={"template_id": f"agent:{tmpl.id}", "user_prompt": "x", "root_agent_id": aid,
                 "llm_model": "mock", "context_limit": 180000},
    ))

    # 新格式：用 AGENT_CONVERSATION_TURN（parent=None）作 root 胶囊触发 agent 层压缩。
    # scope key ignores task_id, uses agent_id.
    scope = MemoryAddress(session_id=sid, task_id="t_seed", agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id="t_seed", agent_id=aid)
    for i in range(5):
        # task 层 body（task_id=root{i}）使每组成为真实 L0 单元（Task-4 §4）
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.USER_PROMPT,
            address=MemoryAddress(session_id=sid, task_id=f"root{i}", agent_id=aid),
            content=f"body {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10)), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"user prompt {i}", role="user",
            timestamp=ts + timedelta(seconds=i * 10),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)
        await mem.ingest(MemoryEvent(
            type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
            content=f"assistant summary {i}", role="assistant",
            timestamp=ts + timedelta(seconds=i * 10 + 1),
            metadata={"origin_task_id": f"root{i}", "parent_task_id": None}), pctx)

    # 先水合 record（生产路径里这一步发生在 root agent 实例化时；此处手工建 session
    # 只走了 event_store，registry 还没见过这个 agent_id——compact_agent 现在先经
    # record_of() 校验存在性，未注册的 agent_id 会被当成 AgentNotFound 挡在门外）。
    rt._agent_lifecycle_manager.register_session(
        sid, tenant_id="default",
    )
    rt._agent_lifecycle_manager.materialize(aid, session_id=sid, tenant_id="default")

    await initialize_all(rt)
    result = await rt.compact_agent(aid)

    assert result.session_id == sid
    assert result.agent_id == aid
    # agent-layer scope key ignores task_id (spec/06 §2), so task_id="" matches the seeded layer
    summaries = await mem.recall_recent(
        scope=MemoryAddress(session_id=sid, task_id="", agent_id=aid),
        types=[MemoryEventType.AGENT_COMPACT_SUMMARY], limit=10, ctx=pctx,
    )
    assert len(summaries) >= 1
    assert sid not in rt._busy_sessions
