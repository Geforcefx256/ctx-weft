"""Shared pytest fixtures for ctx-weft unit tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


def _ts(offset_us: int) -> datetime:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
    return base + timedelta(microseconds=offset_us)


@pytest.fixture
async def fake_state_ctx():
    """Minimal LoopState + LoopContext with a real InMemoryMemoryProvider.

    The task layer is pre-seeded with [USER_PROMPT, LLM_RESPONSE, TOOL_RESULT].
    ctx.task_manager has a no-op track_background.

    async fixture（asyncio_mode=auto）：seed 在测试同一事件循环里跑——
    get_event_loop().run_until_complete 在 pytest-asyncio 清理过循环后会
    RuntimeError('There is no current event loop')，且跨循环 seed 会让 provider
    内部 asyncio.Lock 绑定到错误的循环。
    """
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    async def _seed():
        events = [
            MemoryEvent(
                type=MemoryEventType.USER_PROMPT,
                address=scope,
                content="hello user",
                timestamp=_ts(1),
                role="user",
            ),
            MemoryEvent(
                type=MemoryEventType.LLM_RESPONSE,
                address=scope,
                content="hello llm",
                timestamp=_ts(2),
                role="assistant",
            ),
            MemoryEvent(
                type=MemoryEventType.TOOL_RESULT,
                address=scope,
                content="tool result",
                timestamp=_ts(3),
                role="tool",
            ),
        ]
        for ev in events:
            await mem.ingest(ev, pctx)

    await _seed()

    # Minimal agent with required attributes
    agent = SimpleNamespace(
        id="a1",
        # 字段须与 protocols/template.py::LoopConfig 的默认值一致。
        # 缺字段的后果不是「测试报错」而是「测试假绿」：被测代码读到不存在的属性抛
        # AttributeError，而 _run_background_observe 之类会整段吞掉它，于是用例走的是
        # 异常路径却自称验了成功路径。实测探针确认曾有 1 条如此（详见下）。
        loop_config=SimpleNamespace(
            compact_keep_last=2,
            max_turns_per_observe=5,          # LoopConfig 默认
            short_segment_token_threshold=400,  # LoopConfig 默认
        ),
        runtime={"llm_model": "mock"},
        loop_guard=SimpleNamespace(context_limit=100000, context_tokens=1000),
    )

    # Minimal task
    task = SimpleNamespace(
        id="t1",
        parent_task_id=None,
        title="",
        # 总账 C5（批次二 Task 5）：background recap 现在自己的 run 也发
        # RunStarted/RunFinished，`final_status` 取 `state.task.status`（照抄
        # `_run_loop` 的实际发射点）——缺了这个字段就不是「测试假绿」而是
        # AttributeError 直接炸穿，所以补上，不是行为放宽。
        status="ACTIVE",
        user_prompt="hello user",
        user_prompt_in_memory=True,
        process_report="",
        session_id="s1",
    )

    # Minimal session
    session = SimpleNamespace(id="s1", tenant_id="default")

    # LoopState is a real dataclass (supports dataclasses.replace)
    state = LoopState(
        run_id="run-test-1",
        session=session,
        task=task,
        agent=agent,
        scope=scope,
        extra={"template": None, "bound_capabilities": []},
        # resolve_llm_identity 的真值来源（批次 B）：这里从不真的发 LLM 请求
        # （complete() 被各测试自行打桩），但仍给个占位，防止误触真实调用时 AttributeError。
        resolved_model=SimpleNamespace(model="mock", account=""),
    )

    # Fake task_manager with no-op track_background
    class _FakeTaskManager:
        def track_background(self, t: asyncio.Task) -> None:
            pass  # no-op

        def children_of(self, task_id: str):
            return set()

    # Minimal assembler and LLM (won't be called — summarize_for_compact is monkeypatched)
    class _FakeAssembler:
        async def assemble(self, request: Any) -> Any:
            return SimpleNamespace(system="SYS", messages=[], tools=[])

    class _FakeLLM:
        def __init__(self) -> None:
            from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
            self.tokenizer = HeuristicTokenizer()

        async def complete(self, request: Any, stream: bool = True):
            yield SimpleNamespace(kind="token", text="摘要", usage=None, tool_call=None)

    class _FakeEventBus:
        def __init__(self) -> None:
            self.emitted: list = []

        async def emit(self, event: Any) -> None:
            self.emitted.append(event)

    ctx = LoopContext(
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        memory=mem,
        event_bus=_FakeEventBus(),
        provider_ctx=pctx,
        task_manager=_FakeTaskManager(),
    )

    return state, ctx
