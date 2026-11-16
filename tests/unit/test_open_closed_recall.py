"""OPEN/CLOSED 判据改 task.status — AgentRecallSource 语义验证（spec 2026-06-28 §5）。

覆盖范围（G6 验收要点）：
  (a) 已结束 root task：close 后，AgentRecallSource 召回含 task 层 body AND agent 层 finish 对。
  (b) 运行中/暂停 task（status ≠ FINISHED，未 close）：召回含 body、无 finish 对（活对话）。
  (c) 跨 agent 子任务隔离：子 body 在不同 agent_id 的 task 层；父 agent 召回看不到子 body
      （只见 TASK_DISPATCH_RESULT 黑盒内容）。

判据：task.status / finish 对是否存在，不依赖 task 层是否被 supersede。
"""
from __future__ import annotations
from ctx_weft.core.loop.task_memory_fold import task_record_kind

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair, finalize_task_memory
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.utils.estimate import estimate_tokens
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.protocols.memory_compat import legacy_type_of
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import recalled_blocks


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
SESSION = "s1"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _task_scope(task_id: str, agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=task_id, agent_id=agent_id)


def _agent_scope(agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=None, agent_id=agent_id)


def _ev(type_: MemoryEventType, scope: MemoryAddress, content: str, t: int,
        role: str | None = None, **meta) -> MemoryEvent:
    return MemoryEvent(
        type=type_, address=scope, content=content,
        timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta,
    )


def _make_task(task_id: str = "t1", agent_id: str = "ag1",
               parent_task_id: str | None = None,
               creator_agent_id: str | None = None,
               status: str = "FINISHED",
               outputs: str | None = "答复") -> Task:
    return Task(
        id=task_id, session_id=SESSION, status=status, tenant_id="default",
        assigned_agent_id=agent_id,
        creator_agent_id=creator_agent_id or agent_id,
        parent_task_id=parent_task_id,
        title="测试任务", description="", user_prompt="初始请求",
        settings=NormalTaskSettings(),
        outputs=outputs,
    )


class _FakeTM:
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _state(task: Task, scope: MemoryAddress):
    agent = SimpleNamespace(id=scope.agent_id, loop_config=LoopConfig())
    session = SimpleNamespace(id=SESSION, tenant_id="default")
    return SimpleNamespace(
        run_id="run1", sequence_counter=0,
        session=session, scope=scope, task=task, agent=agent,
    )


def _loop_ctx(mem: InMemoryMemoryProvider):
    return SimpleNamespace(
        memory=mem, provider_ctx=_pctx(),
        task_manager=_FakeTM(),
        llm=SimpleNamespace(tokenizer=HeuristicTokenizer()),
    )


async def _recall_blocks(mem: InMemoryMemoryProvider, agent_scope: MemoryAddress) -> list:
    """Drive AgentRecallSource.fetch and return sorted blocks."""
    deps = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    req = SimpleNamespace(scope=agent_scope, token_counter=estimate_tokens)
    blocks = await recalled_blocks(req, deps)
    blocks.sort(key=lambda b: (b.value.timestamp, b.metadata.get("sequence", 0)))
    return blocks


def _type(block):
    record = block.value
    return record.type or legacy_type_of(record.kind, record.scope, record.role)


# ─── (a) 已结束 root task：召回含 body + finish 对 ─────────────────────────────

@pytest.mark.asyncio
async def test_finished_root_task_recall_yields_body_and_finish_pair() -> None:
    """(a) G6：已结束 task close 后，AgentRecallSource 召回：
    task 层 body（USER_PROMPT + TASK_COMPACT_SUMMARY）AND agent 层 finish 对（2 条 AGENT_CONVERSATION_TURN）。
    判据走 task.status / finish 对存在，不靠 task 层 supersede。
    body 留 task 层（不被 supersede）——两段召回都能找到内容。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1", "ag1")
    asc = _agent_scope("ag1")

    # task 层 body：user 锚点 + 段摘要（留 task 层，不 supersede）
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "把 auth 从 session 改成 JWT", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, tsc, "已改 JWT，3 处改完", 2, role="assistant"), _pctx())

    # close：写 finish 对到 agent 层（task-resident：body 留 task 层）
    task = _make_task(task_id="t1", status="FINISHED", outputs="切换 JWT 完成")
    task_summary_text = "成功。3 处改完，8 测试通过。"
    await _synthesize_dispatch_pair(mem, asc, task, "已完成 JWT 改造", task_summary_text, "success", _pctx())

    blocks = await _recall_blocks(mem, asc)
    types = [_type(b) for b in blocks]

    # ① task 层 body 仍在（USER_PROMPT + TASK_COMPACT_SUMMARY）
    assert T.USER_PROMPT in types, f"finished task body (USER_PROMPT) must be recalled; types={types}"
    assert T.TASK_COMPACT_SUMMARY in types, f"finished task body (TASK_COMPACT_SUMMARY) must be recalled; types={types}"

    # ② agent 层 finish 对（AGENT_CONVERSATION_TURN，2 条）
    act_blocks = [b for b in blocks if _type(b) == T.AGENT_CONVERSATION_TURN]
    assert len(act_blocks) == 2, (
        f"finished task must yield exactly 2 AGENT_CONVERSATION_TURN (finish pair); "
        f"got {[(b.value.role, b.value.content[:40]) for b in act_blocks]}"
    )
    roles = [b.value.role for b in act_blocks]
    assert roles == ["user", "user"], f"finish pair must be [assistant, tool]; got {roles}"

    assert [task_record_kind(b.value) for b in act_blocks] == ["recap", "finish"]
    assert all(not b.value.metadata.get("tool_calls") for b in act_blocks)
    assert task_summary_text in act_blocks[1].value.content

    # 验证判据：body 在 task 层未被 supersede（task-resident 设计）
    task_body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert len(task_body) == 2, (
        f"OPEN/CLOSED 判据 = task.status，不靠 supersede；body 应留 task 层（未被 supersede）, "
        f"got {len(task_body)}"
    )


# ─── (b) 运行中/暂停 task：召回含 body，无 finish 对 ────────────────────────────

@pytest.mark.asyncio
async def test_running_task_recall_yields_body_no_finish_pair() -> None:
    """(b) G6：运行中 task（status=RUNNING，未 close，无 finish 对）：
    AgentRecallSource 召回含 task 层 body（USER_PROMPT + LLM_RESPONSE），无 AGENT_CONVERSATION_TURN。
    活对话用 task.status 判 OPEN，不靠 supersession。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t2", "ag1")
    asc = _agent_scope("ag1")

    # task 层 body（running：尚未 close，无 supersede，无 finish 对）
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "请分析这份代码", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "正在分析…", 2, role="assistant"), _pctx())

    # 不 close（status=RUNNING，无 finish 对写入 agent 层）

    blocks = await _recall_blocks(mem, asc)
    types = [_type(b) for b in blocks]

    # body 存在（task 层）
    assert T.USER_PROMPT in types, f"running task body (USER_PROMPT) must be recalled; types={types}"
    assert T.LLM_RESPONSE in types, f"running task body (LLM_RESPONSE) must be recalled; types={types}"

    # 无 finish 对（无 AGENT_CONVERSATION_TURN）
    act_blocks = [b for b in blocks if _type(b) == T.AGENT_CONVERSATION_TURN]
    assert not act_blocks, (
        f"running task must NOT have finish pair (AGENT_CONVERSATION_TURN); "
        f"got {[(b.value.role, b.value.content[:40]) for b in act_blocks]}"
    )


@pytest.mark.asyncio
async def test_paused_task_recall_yields_body_no_finish_pair() -> None:
    """(b) 变体：SUSPENDED/暂停 task（wait_for_user，status≠FINISHED）：
    召回含 body，无 finish 对——活对话，not finished。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t3", "ag1")
    asc = _agent_scope("ag1")

    await mem.ingest(_ev(T.USER_PROMPT, tsc, "请告知你的选择", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "等待用户输入", 2, role="assistant"), _pctx())

    # 暂停：status=SUSPENDED，未 close，无 finish 对

    blocks = await _recall_blocks(mem, asc)
    types = [_type(b) for b in blocks]

    assert T.USER_PROMPT in types, "paused task body must be recalled"
    act_blocks = [b for b in blocks if _type(b) == T.AGENT_CONVERSATION_TURN]
    assert not act_blocks, (
        f"paused task must NOT have finish pair; got {act_blocks}"
    )


# ─── (c) 跨 agent 子任务隔离：父 agent 召回不到子 body ──────────────────────────

@pytest.mark.asyncio
async def test_cross_agent_child_body_isolated_from_parent() -> None:
    """(c) G4/G6：跨 agent 子任务 body 在子 agent 的 task 层（不同 agent_id）；
    父 agent AgentRecallSource 召回按 agent_id 过滤：看不到子 body（只见 dispatch result 黑盒）。
    验证隔离自然涌现于 recall_recent_by_agent 的 agent_id 过滤。
    """
    mem = InMemoryMemoryProvider()

    parent_agent = "ag_parent"
    child_agent = "ag_child"
    parent_task_id = "ptask"
    child_task_id = "ctask"

    # 子 agent 的 task 层（child_agent）—— parent_agent 的召回不应触及
    child_tsc = _task_scope(child_task_id, child_agent)
    await mem.ingest(_ev(T.USER_PROMPT, child_tsc, "子任务：收集 X 数据", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, child_tsc, "已收集 5 份", 2, role="assistant"), _pctx())
    for i in range(3):
        await mem.ingest(_ev(T.LLM_RESPONSE, child_tsc, "y " * 500, 3 + i, role="assistant"), _pctx())

    # 父 task 层（parent_agent）—— 仅写 parent 自己的 UP
    parent_tsc = _task_scope(parent_task_id, parent_agent)
    await mem.ingest(_ev(T.USER_PROMPT, parent_tsc, "综合分析", 0, role="user"), _pctx())

    # close 子 task（跨 agent）→ bubble TASK_DISPATCH_RESULT 到 parent scope，子 finish 对写 child agent scope
    child_task = _make_task(
        task_id=child_task_id, agent_id=child_agent,
        creator_agent_id=parent_agent, parent_task_id=parent_task_id,
        status="FINISHED", outputs="收集到 5 份数据集",
    )
    child_task.origin_tool_call_id = "oc_cross"
    # Intentionally pre-built mem_content for cross-agent isolation test; actual shape tested elsewhere
    child_mem_content = "收集到 5 份数据集\n\nProcess Report: 成功。"

    await finalize_task_memory(
        mem, _state(child_task, child_tsc),
        child_task, child_mem_content, "success", _loop_ctx(mem),
        act_recap="成功。", task_summary="",
    )

    # 父 agent AgentRecallSource 召回
    parent_asc = _agent_scope(parent_agent)
    parent_blocks = await _recall_blocks(mem, parent_asc)

    # 父应召回自己的 UP
    types = [_type(b) for b in parent_blocks]
    assert T.USER_PROMPT in types, "parent's own USER_PROMPT must be recalled"

    # 父不应看到子 body（USER_PROMPT / TASK_COMPACT_SUMMARY / LLM_RESPONSE 属于子 agent 的 task 层）
    contents = [b.value.content or "" for b in parent_blocks]
    assert not any("子任务：收集 X 数据" in c for c in contents), (
        f"parent recall must NOT surface child's task-layer USER_PROMPT (cross-agent isolation); "
        f"found in: {[c[:60] for c in contents if '子任务' in c]}"
    )
    assert not any("已收集 5 份" in c for c in contents), (
        "parent recall must NOT surface child's TASK_COMPACT_SUMMARY (cross-agent isolation)"
    )

    # 父可见 dispatch result（黑盒 bubble，conversation turn），含 mem_content、配对 oc_cross、归 parent 单元
    parent_task_recs = await mem.recall_recent(parent_tsc, [T.AGENT_CONVERSATION_TURN], 100, _pctx())
    cross_results = [r for r in parent_task_recs
                     if task_record_kind(r) == "result" and r.metadata.get("task_memory", {}).get("group_id") == "oc_cross"]
    assert cross_results, "cross-agent child must bubble dispatch result (conversation turn) to parent"
    assert cross_results[0].metadata.get("origin_task_id") == parent_task_id
    assert "Process Report:" in cross_results[0].content, (
        f"bubble content must be mem_content; got {cross_results[0].content!r}"
    )
    # 不再写 legacy enum
    assert await mem.recall_recent(parent_tsc, [T.TASK_DISPATCH_RESULT], 100, _pctx()) == []

    # 子 body 在子 task 层，不因 finalize 被 supersede（task-resident：body 留原处）
    child_body = await mem.recall_recent(child_tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert any("子任务：收集 X 数据" in (r.content or "") for r in child_body), (
        "child body must stay in child task layer (not superseded by finalize)"
    )
