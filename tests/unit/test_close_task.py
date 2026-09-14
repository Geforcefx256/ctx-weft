"""close(task) — task-resident（spec 2026-06-28 §3/§5）：

每个结束 task = task 层 raw body（不被 supersede、不 GC 子树）+ agent 层 finish 对。
own-root/cross-agent root 写自己的 finish 对；same-agent 子任务无条件 bubble「…scheduled」
TASK_DISPATCH_RESULT + 自己的 finish 对。short 不再 gate 合成/supersede。
"""

from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import finalize_task_memory, _descendant_task_ids, _dispatch_ack
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(task_id: str, agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


class _FakeTM:
    def __init__(self, children: dict[str, set[str]] | None = None) -> None:
        self._children = children or {}

    def children_of(self, task_id: str) -> set[str]:
        return self._children.get(task_id, set())


def _state(task: Task, scope: MemoryAddress, loop_config: LoopConfig, tm: _FakeTM):
    # 字段须满足 make_event：run_id / sequence_counter / session.id / session.tenant_id /
    # task.id / agent.id（finalize_task_memory 内部用 make_event 构造事件）。
    agent = SimpleNamespace(id=scope.agent_id, loop_config=loop_config)
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent)


def _loop_ctx(mem, tm: _FakeTM):
    return SimpleNamespace(memory=mem, provider_ctx=_ctx(), task_manager=tm,
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


async def _seed_conv(mem, scope, n_assistant: int, big: bool = False) -> None:
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello", 0, role="user"), _ctx())
    for i in range(n_assistant):
        text = ("x " * 4000) if big else f"reply {i}"
        await mem.ingest(_ev(T.LLM_RESPONSE, scope, text, i + 1, role="assistant"), _ctx())


def _root_task(task_id="t1", agent="ag1") -> Task:
    return Task(id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
                assigned_agent_id=agent, creator_agent_id=agent, parent_task_id=None,
                title="Root", user_prompt="hello", settings=NormalTaskSettings())


async def _finish_tools(mem, scope, origin: str) -> list:
    """召回 agent 层属于 origin 的 finish-pair tool 回合（content 含 Process Report）。"""
    caps = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    return [r for r in caps if r.role == "tool" and r.metadata.get("origin_task_id") == origin]


async def _seed_delegate(mem, scope, origin: str, tcid: str, parent=None,
                         name="delegate_task", args=None) -> None:
    """模拟 gateway 写的 delegate assistant 回合（新表示：AGENT_CONVERSATION_TURN，spec §2.3）。"""
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, "", 0, role="assistant",
                         origin_task_id=origin, parent_task_id=parent,
                         tool_calls=[{"id": tcid, "name": name, "input": args or {}}]), _ctx())


async def _delegate_turns(mem, scope, tcid: str) -> list:
    """召回 agent 层 delegate assistant 回合（tool_calls 含 tcid）。"""
    caps = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    return [r for r in caps if r.role == "assistant"
            and any(tc.get("id") == tcid for tc in (r.metadata.get("tool_calls") or []))]


async def _dispatch_results(mem, scope, tcid: str) -> list:
    """召回 agent 层 dispatch result tool 回合（tool_call_id==tcid，新表示）。"""
    caps = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 200, _ctx())
    return [r for r in caps if r.role == "tool" and r.metadata.get("tool_call_id") == tcid]


async def test_short_root_leaf_keeps_body_and_synthesizes_finish_pair() -> None:
    """task-resident：短 root 叶子 close 也合成 finish 对（取消 short 延迟），body 原样留。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=2)  # few turns, small → short
    task = _root_task()

    await finalize_task_memory(mem, _state(task, scope, LoopConfig(), _FakeTM()),
                               task, "final out", "success", _loop_ctx(mem, _FakeTM()),
                               act_recap="test recap", task_summary="test summary")

    # body kept (task-resident)
    convs = await mem.recall_recent(scope, [T.USER_PROMPT, T.LLM_RESPONSE], 100, _ctx())
    assert {c.content for c in convs} >= {"hello", "reply 0", "reply 1"}
    # own-root finish pair synthesized (agent layer), no TASK_DISPATCH_RESULT (root never bubbles)
    assert await _finish_tools(mem, scope, "t1"), "short root leaf must synthesize finish pair"
    residues = await mem.recall_recent(scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    assert residues == []


async def test_long_root_leaf_supersedes_final_raw_keeps_anchor_and_finish_pair() -> None:
    """Task 2（spec 2026-06-28 §3.2）：长 root 叶子 close 写 finish 对，且 supersede 末 raw 段
    （active LLM_RESPONSE 不再 recall），USER_PROMPT 锚点保留。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)  # >turn_cap and big → not short
    task = _root_task()
    cfg = LoopConfig()

    await finalize_task_memory(mem, _state(task, scope, cfg, _FakeTM()),
                               task, "final out", "success", _loop_ctx(mem, _FakeTM()),
                               act_recap="test recap", task_summary="test summary")

    # Task 2: 末 raw 段 supersede（active LLM_RESPONSE 不再 recall），USER_PROMPT 锚点留
    llm = await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _ctx())
    assert llm == [], "long task: final raw LLM_RESPONSE must be superseded"
    anchors = await mem.recall_recent(scope, [T.USER_PROMPT], 100, _ctx())
    assert anchors != [], "USER_PROMPT anchor must survive"
    # finish pair written as AGENT_CONVERSATION_TURN (tool role holds Process Report)
    assert await _finish_tools(mem, scope, "t1"), "expected finish-pair tool turn in agent capsule"


async def test_root_close_keeps_subtree_bodies() -> None:
    """task-resident：root close 不 GC 子树——同 agent 子任务 body 留各自 task 层。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)
    # a same-agent sub-task t2 residue already in the agent scope
    await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 10, role="assistant",
                         tool_call_id="d2", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, "t2 out", 11, role="tool",
                         tool_call_id="d2", child_task_id="t2", parent_task_id="t1"), _ctx())
    # t2 also has its own task-layer conversation tagged agent ag1
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("t2"), "t2 work", 12, role="assistant"), _ctx())

    tm = _FakeTM({"t1": {"t2"}})
    task = _root_task()  # non-leaf via tm
    await finalize_task_memory(mem, _state(task, scope, LoopConfig(), tm),
                               task, "final out", "success", _loop_ctx(mem, tm),
                               act_recap="test recap", task_summary="test summary")

    # dispatch pair preserved; t2's task-layer body NOT GC'd (task-resident)
    all_results = await mem.recall_recent(scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    assert any(r.content == "t2 out" for r in all_results), "dispatch pair must be preserved"
    t2_conv = await mem.recall_recent_by_agent(_sc("x", "ag1"), [T.LLM_RESPONSE], 100, _ctx())
    assert any(r.metadata.get("task_id") == "t2" for r in t2_conv), (
        "subtree body must stay (task-resident: no _gc_subtree)"
    )


async def test_cross_agent_child_bubbles_unconditionally_even_when_short() -> None:
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c1", agent_id="ag2")
    await _seed_conv(mem, child_scope, n_assistant=1)  # short leaf
    # parent delegate turn lives in ag1's agent layer (gateway-written, §2.3)
    await _seed_delegate(mem, _sc("p1", "ag1"), origin="p1", tcid="oc1", parent=None)
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag2", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="oc1", title="Child", user_prompt="sub",
                 settings=NormalTaskSettings())

    await finalize_task_memory(mem, _state(child, child_scope, LoopConfig(), _FakeTM()),
                               child, "child out", "success", _loop_ctx(mem, _FakeTM()),
                               act_recap="child recap", task_summary="child summary")

    # parent (ag1) received the dispatch result as a tool conversation turn even though child was short;
    # origin=delegating task(p1) → folds with that unit (§2.3)
    parent_scope = _sc("p1", "ag1")
    res = await _dispatch_results(mem, parent_scope, "oc1")
    assert any(r.content == "child out" and r.metadata.get("origin_task_id") == "p1" for r in res)
    assert await mem.recall_recent(parent_scope, [T.TASK_DISPATCH_RESULT], 100, _ctx()) == []
    # child's own conversation preserved (task-resident: body stays)
    own = await mem.recall_recent(child_scope, [T.LLM_RESPONSE], 100, _ctx())
    assert own != []


async def test_same_agent_short_leaf_no_bubble_supersedes_orphan_dispatch() -> None:
    """§2.5（spec 2026-06-30）：same-agent 子任务 close 不再 supersede delegate 回合，改写一条
    配对的静态 tool result（_DISPATCH_ACK），timestamp back-date 到 delegate 时刻 → 与 delegate
    严格相邻。嵌套 finish 对承载真实内容。body 留 task 层。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t2", "ag1")
    await _seed_conv(mem, scope, n_assistant=1)  # short leaf
    # delegate turn for t2 in the shared ag1 scope (written by gateway at delegate time)
    await _seed_delegate(mem, _sc("t1", "ag1"), origin="t1", tcid="oc2", parent=None)
    child = Task(id="t2", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="t1",
                 origin_tool_call_id="oc2", title="Sub", user_prompt="sub",
                 settings=NormalTaskSettings())

    await finalize_task_memory(mem, _state(child, scope, LoopConfig(), _FakeTM()),
                               child, "t2 out", "success", _loop_ctx(mem, _FakeTM()),
                               act_recap="t2 recap", task_summary="t2 summary")

    # same-agent: static dispatch ack written (not bubbled dispatch result with outcome)
    results = await _dispatch_results(mem, _sc("t1", "ag1"), "oc2")
    assert len(results) == 1, f"expected 1 dispatch ack result; got {results}"
    assert results[0].content == _dispatch_ack(task_ref(child), "success"), (
        f"same-agent dispatch result must be _dispatch_ack(title); got {results[0].content!r}"
    )
    # delegate turn KEPT (not superseded, spec 2026-06-30 §2.5)
    delegates = await _delegate_turns(mem, _sc("t1", "ag1"), "oc2")
    assert len(delegates) == 1, "orphan delegate turn must be KEPT (not superseded)"
    # nested finish pair synthesized (carries real content)
    assert await _finish_tools(mem, scope, "t2"), "nested finish pair must exist"
    # body kept (task-resident)
    own = await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _ctx())
    assert own != []


async def test_same_agent_nonshort_child_no_bubble_supersedes_final_raw_and_orphan() -> None:
    """§2.5 + Task 2（spec 2026-06-30 / 2026-06-28）：长 same-agent 子任务 close——
    (a) 写 _DISPATCH_ACK 配对 result、保留 delegate 回合（不再 supersede）；
    (b) supersede 末 raw 段、保留 USER_PROMPT 锚点；(c) 嵌套 finish 对承载。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t2", "ag1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)  # over turn cap → not short
    # delegate turn for t2 lives in the shared ag1 agent scope (parent t1)
    await _seed_delegate(mem, _sc("t1", "ag1"), origin="t1", tcid="oc2", parent=None)
    child = Task(id="t2", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="t1",
                 origin_tool_call_id="oc2", title="Sub", user_prompt="sub",
                 settings=NormalTaskSettings())
    await finalize_task_memory(mem, _state(child, scope, LoopConfig(), _FakeTM()),
                               child, "t2 out", "success", _loop_ctx(mem, _FakeTM()),
                               act_recap="t2 recap", task_summary="t2 summary")

    # (a) same-agent: dispatch ack written + delegate turn KEPT (not superseded)
    results = await _dispatch_results(mem, _sc("t1", "ag1"), "oc2")
    assert len(results) == 1 and results[0].content == _dispatch_ack(task_ref(child), "success"), (
        f"expected _dispatch_ack(title); got {[r.content for r in results]}"
    )
    delegates = await _delegate_turns(mem, _sc("t1", "ag1"), "oc2")
    assert len(delegates) == 1, "delegate turn must be KEPT (not superseded)"
    # (b) 长任务末 raw 段被 supersede（active LLM_RESPONSE 不再 recall），USER_PROMPT 锚点留
    own = await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _ctx())
    assert own == [], "long task: final raw LLM_RESPONSE must be superseded"
    anchors = await mem.recall_recent(scope, [T.USER_PROMPT], 100, _ctx())
    assert anchors != [], "USER_PROMPT anchor must survive"
    # (c) nested finish pair synthesized
    assert await _finish_tools(mem, scope, "t2"), "nested finish pair must exist"


async def test_root_close_keeps_deep_nested_subtree_bodies() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)
    # A residue (child A, parent t1) + A1 residue (grandchild, child A1, parent A) + A1 leftover conv
    await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 10, role="assistant",
                         tool_call_id="dA", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, "A out", 11, role="tool",
                         tool_call_id="dA", child_task_id="A", parent_task_id="t1"), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 12, role="assistant",
                         tool_call_id="dA1", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, "A1 out", 13, role="tool",
                         tool_call_id="dA1", child_task_id="A1", parent_task_id="A"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("A1", "ag1"), "A1 work", 14, role="assistant"), _ctx())
    tm = _FakeTM({"t1": {"A"}, "A": {"A1"}})
    task = _root_task()
    await finalize_task_memory(mem, _state(task, scope, LoopConfig(), tm),
                               task, "final out", "success", _loop_ctx(mem, tm),
                               act_recap="test recap", task_summary="test summary")
    # dispatch pairs for A and A1 preserved
    results = await mem.recall_recent(scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    contents = {r.content for r in results}
    assert "A out" in contents and "A1 out" in contents, "dispatch pairs must be preserved"
    # root self finish pair written as AGENT_CONVERSATION_TURN
    assert await _finish_tools(mem, scope, "t1"), "root finish pair must exist"
    # grandchild body NOT GC'd (task-resident)
    convs = await mem.recall_recent_by_agent(_sc("x", "ag1"), [T.LLM_RESPONSE], 100, _ctx())
    assert any(r.metadata.get("task_id") == "A1" for r in convs), "grandchild body must stay"


async def test_intermediate_close_keeps_grandchild_body() -> None:
    mem = InMemoryMemoryProvider()
    a_scope = _sc("A", "ag1")
    await _seed_conv(mem, a_scope, n_assistant=5, big=True)  # A's own conversation (not short)
    # parent t1's delegate turn for A (so A's orphan delegate gets superseded)
    await _seed_delegate(mem, _sc("t1", "ag1"), origin="t1", tcid="ocA", parent=None)
    # A1 dispatch pair (delegate + result, origin=A) + A1 conv
    await _seed_delegate(mem, a_scope, origin="A", tcid="dA1", parent="t1")
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, a_scope, "A1 out", 6, role="tool",
                         origin_task_id="A", tool_call_id="dA1"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("A1", "ag1"), "A1 work", 7, role="assistant"), _ctx())
    A = Task(id="A", session_id="s1", status="FINISHED", tenant_id="default",
             assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="t1",
             origin_tool_call_id="ocA", title="A", user_prompt="a", settings=NormalTaskSettings())
    tm = _FakeTM({"A": {"A1"}})
    await finalize_task_memory(mem, _state(A, a_scope, LoopConfig(), tm),
                               A, "A out", "success", _loop_ctx(mem, tm),
                               act_recap="A recap", task_summary="A summary")
    # A1 dispatch result preserved (conversation turn, origin=A)
    assert any(r.content == "A1 out" for r in await _dispatch_results(mem, a_scope, "dA1")), \
        "A1 dispatch pair must be preserved"
    # grandchild body kept (task-resident)
    convs = await mem.recall_recent_by_agent(_sc("x", "ag1"), [T.LLM_RESPONSE], 100, _ctx())
    assert any(r.metadata.get("task_id") == "A1" for r in convs), "grandchild body must stay"
    # §2.5: A (same-agent) writes dispatch ack result + keeps delegate turn (not superseded)
    results_ocA = await _dispatch_results(mem, _sc("t1", "ag1"), "ocA")
    assert len(results_ocA) == 1 and results_ocA[0].content == _dispatch_ack(task_ref(A), "success"), (
        f"same-agent A must write _dispatch_ack(title); got {[r.content for r in results_ocA]}"
    )
    delegates_ocA = await _delegate_turns(mem, _sc("t1", "ag1"), "ocA")
    assert len(delegates_ocA) == 1, "ocA delegate turn must be KEPT (not superseded)"
    # A's own finish pair synthesized (nested, agent layer)
    assert await _finish_tools(mem, a_scope, "A"), "A's nested finish pair must exist"


async def test_root_close_multichild_plan_preserves_dispatch_pairs() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)
    for c in ("A", "B", "C"):
        await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 10, role="assistant",
                             tool_call_id=f"d{c}", tool_name="delegate_task", arguments={}), _ctx())
        await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, f"{c} out", 11, role="tool",
                             tool_call_id=f"d{c}", child_task_id=c, parent_task_id="t1"), _ctx())
    tm = _FakeTM({"t1": {"A", "B", "C"}})
    task = _root_task()
    await finalize_task_memory(mem, _state(task, scope, LoopConfig(), tm),
                               task, "final out", "success", _loop_ctx(mem, tm),
                               act_recap="test recap", task_summary="test summary")
    results = await mem.recall_recent(scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    contents = {r.content for r in results}
    # dispatch pairs preserved
    assert {"A out", "B out", "C out"} <= contents, "all plan-child dispatch pairs must be preserved"
    # root self finish pair written as AGENT_CONVERSATION_TURN
    assert await _finish_tools(mem, scope, "t1"), "root finish pair must exist"


async def test_root_close_mixed_same_and_cross_agent_children_keeps_bodies() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_conv(mem, scope, n_assistant=5, big=True)
    # same-agent child S: residue in ag1 + leftover conv (task_id=S, agent ag1)
    await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 10, role="assistant",
                         tool_call_id="dS", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, "S out", 11, role="tool",
                         tool_call_id="dS", child_task_id="S", parent_task_id="t1"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("S", "ag1"), "S work", 12, role="assistant"), _ctx())
    # cross-agent child X: only a bubbled residue in ag1 (X ran in ag2, black box — no conv in ag1)
    await mem.ingest(_ev(T.TASK_DISPATCH, scope, "", 13, role="assistant",
                         tool_call_id="dX", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, scope, "X out", 14, role="tool",
                         tool_call_id="dX", child_task_id="X", parent_task_id="t1"), _ctx())
    tm = _FakeTM({"t1": {"S", "X"}})
    task = _root_task()
    await finalize_task_memory(mem, _state(task, scope, LoopConfig(), tm),
                               task, "final out", "success", _loop_ctx(mem, tm),
                               act_recap="test recap", task_summary="test summary")
    results = await mem.recall_recent(scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    contents = {r.content for r in results}
    # dispatch pairs preserved
    assert {"S out", "X out"} <= contents, "both children dispatch pairs must be preserved"
    # same-agent child body kept (task-resident)
    convs = await mem.recall_recent_by_agent(_sc("x", "ag1"), [T.LLM_RESPONSE], 100, _ctx())
    assert any(r.metadata.get("task_id") == "S" for r in convs), "same-agent child body must stay"


async def test_descendant_task_ids_multilevel_bfs() -> None:
    tm = _FakeTM({"t1": {"A", "B"}, "A": {"A1"}, "B": {"B1"}, "A1": {"A1a"}})
    assert _descendant_task_ids("t1", tm) == {"A", "B", "A1", "B1", "A1a"}
    assert _descendant_task_ids("A", tm) == {"A1", "A1a"}
    assert _descendant_task_ids("leaf", tm) == set()
