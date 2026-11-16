"""压缩机制全面测试（spec 2026-06-23）：触发 / (a)活跃task折叠 / (b)close短task /
(c)只折已结束root残留 / CompactStep 编排 / 边界 / 幂等。"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import AssembledPrompt
from ctx_weft.core.loop.driver import LoopState
from ctx_weft.core.loop.steps.compact import CompactStep
from tests.compaction_testkit import COLLAPSE_DELIM, _count_root_residues, fold_root_experience
from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1")


def _sc(task_id="t1", agent_id="ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


def _ev(type_, scope, content, t, role=None, **meta):
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


class _FakeTM:
    def __init__(self, tasks: dict[str, Task], children: dict[str, set[str]] | None = None):
        self._tasks = tasks
        self._children = children or {}

    def get_task(self, tid: str) -> Task | None:
        return self._tasks.get(tid)

    def children_of(self, tid: str) -> set[str]:
        return self._children.get(tid, set())


class _FakeAssembler:
    """summarize_for_compact 需要 ctx.assembler.assemble；返回空 prompt（内容由 _FakeLLM 摘要）。"""

    async def compile(self, inputs, **kwargs):
        return await compiled_test_inputs(inputs, **kwargs)

    async def assemble(self, request) -> AssembledPrompt:
        return scripted_inputs(AssembledPrompt(system="", messages=[], tools=[], token_count=0), purpose=request.purpose)


class _FakeLLM:
    """summarize_for_compact 经 stream_llm_resilient 调 LLM 出摘要（compact 硬依赖 LLM，无截断兜底）。"""

    context_limit = 1_000_000  # apply_dynamic_max_tokens ceiling fallback (Task 2 网关接线)

    def __init__(self, text: str = "SUM"):
        self._text = text
        from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
        self.tokenizer = HeuristicTokenizer()

    async def complete(self, request, *, stream: bool = True):
        from ctx_weft.protocols import LLMChunk

        yield LLMChunk(kind="token", text=self._text)


def _agent(cfg: LoopConfig, context_limit: int = 1000) -> Agent:
    a = Agent(id="ag1", session_id="s1", template_id="tpl", loop_config=cfg)
    return a


def _state(active: Task, cfg: LoopConfig, context_limit: int = 1000,
           context_tokens: int | None = None) -> LoopState:
    session = Session(id="s1", user_prompt="u")
    agent = _agent(cfg, context_limit)
    # escalating_compact 的预算门用 loop_guard.context_tokens；缺省=context_limit（视作已满,
    # 门总是打开）——迁移自旧 count-based _compact_scope 的测试显式传本参数模拟"未达预算"。
    agent.loop_guard.context_tokens = context_limit if context_tokens is None else context_tokens
    return LoopState(run_id="run1", session=session, task=active,
                     agent=agent, scope=_sc(active.id, "ag1"), resolved_model=model_snapshot(model="mock", account="", context_limit=context_limit))


def _loop_ctx(mem, tm, *, with_assembler: bool = False):
    return SimpleNamespace(
        memory=mem, provider_ctx=_ctx(), task_manager=tm,
        llm=_FakeLLM() if with_assembler else None,
        assembler=_FakeAssembler() if with_assembler else None,
    )


def _active_task() -> Task:
    return Task(id="t1", session_id="s1", status="ACTIVE", assigned_agent_id="ag1",
                creator_agent_id="ag1", title="Root", user_prompt="root",
                settings=NormalTaskSettings())


def _finished_short(tid: str, parent="t1", oc=None) -> Task:
    return Task(id=tid, session_id="s1", status="FINISHED", assigned_agent_id="ag1",
                creator_agent_id="ag1", parent_task_id=parent, origin_tool_call_id=oc,
                title="Sub", user_prompt="sub", outputs=f"{tid} answer",
                process_report="done", observer_outcome="success", settings=NormalTaskSettings())


async def _seed_root_residues(mem, sc, n: int, start_t: int = 0) -> None:
    """Task-4 task-resident：每个结束单元 = task 层 body（task_id=rt{i} scope 的 USER_PROMPT）
    + agent 层 finish 对（4x AGENT_CONVERSATION_TURN，parent_task_id=None）。L1 删 body，L2 连
    finish 对一并删。"""
    for i in range(n):
        t0 = start_t + 4 * i
        tcid = f"rd{i}"
        body_scope = MemoryAddress(session_id="s1", task_id=f"rt{i}", agent_id=sc.agent_id)
        await mem.ingest(_ev(T.USER_PROMPT, body_scope, f"body rt{i}", t0, role="user"), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, f"user prompt rt{i}", t0,
                             role="user", origin_task_id=f"rt{i}", parent_task_id=None), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, f"root res {i}", t0 + 1,
                             role="assistant", origin_task_id=f"rt{i}", parent_task_id=None), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, "", t0 + 2,
                             role="assistant", origin_task_id=f"rt{i}", parent_task_id=None,
                             tool_calls=[{"id": tcid, "name": "control:finish_task", "input": {}}]), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, f"Process Report: done rt{i}", t0 + 3,
                             role="tool", origin_task_id=f"rt{i}", parent_task_id=None,
                             tool_call_id=tcid), _ctx())


# ═══════════════════ _should_compact triggers ═══════════════════

@pytest.mark.asyncio
async def test_trigger_token_ratio_over_and_under() -> None:
    mem = InMemoryMemoryProvider()
    cfg = LoopConfig(compact_token_ratio=0.8, compact_message_delta=10_000)  # delta off
    state, ctx = _state(_active_task(), cfg), _loop_ctx(mem, _FakeTM({}))
    step = PrepareStep()
    assert await step._should_compact(state, ctx, token_estimate=700) is False
    assert await step._should_compact(state, ctx, token_estimate=900) is True


@pytest.mark.asyncio
async def test_trigger_no_division_when_limit_or_estimate_zero() -> None:
    mem = InMemoryMemoryProvider()
    cfg = LoopConfig(compact_message_delta=10_000)
    step = PrepareStep()
    with pytest.raises(ValueError, match="context_limit"):
        _state(_active_task(), cfg, context_limit=0)
    # token_estimate == 0 → skip token branch
    s1 = _state(_active_task(), cfg, context_limit=1000)
    assert await step._should_compact(s1, _loop_ctx(mem, _FakeTM({})), token_estimate=0) is False


# 消息条数增长触发（compact_message_delta）已废（spec 2026-07-01 §3.6：compact 改纯预算
# 驱动）。旧的 growth-trigger 用例（active task conversation / root residues / finished
# short convs / subtask residues / suspended ancestor / delta==0）随之删除——那些场景现在
# 一律不触发（token 比率之外无其他触发维度），已由 test_trigger_token_ratio_over_and_under
# 与 test_trigger_no_division_when_limit_or_estimate_zero 覆盖纯预算契约。


# ═══════════════════ (b) close_finished_short_tasks — 删除（task-resident） ══════
# spec 2026-06-28 §5：取消「压力下回收 finished 短 task」。结束 task 的 body 留 task 层
# （即胶囊），由跨层 fold 管理。close_finished_short_tasks 函数已删，本节执行测试随之移除。


# ═══════════════════ (c) fold_root_experience ═══════════════════

@pytest.mark.asyncio
async def test_fold_root_residues_keep_last_and_subtask_survive() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 5)
    # active delegating task t1 的在途 dispatch 对（working set，§2.3）：有 dispatch 回合、无 finish
    # 对 → active → 不可折。表示为 delegate assistant + result tool conversation turn（origin=t1）。
    for i in range(2):
        tc = f"sd{i}"
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, "", 200 + i * 2, role="assistant",
                             origin_task_id="t1", parent_task_id=None,
                             tool_calls=[{"id": tc, "name": "control:delegate_task", "input": {}}]), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, f"sub res {i}", 201 + i * 2, role="tool",
                             origin_task_id="t1", tool_call_id=tc), _ctx())

    n = await fold_root_experience(_state(_active_task(), LoopConfig()),
                                   _loop_ctx(mem, _FakeTM({})), keep_last=1, summary_text="SUM")
    assert n > 0
    turns = await mem.recall_recent(sc, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    alive_origins = {r.metadata.get("origin_task_id") for r in turns}
    assert alive_origins == {"rt4", "t1"}                 # newest root kept + active working set t1
    # working set（t1 的 dispatch 结果）survives
    sub_contents = {r.content for r in turns if r.role == "tool"}
    assert {"sub res 0", "sub res 1"} <= sub_contents
    summ = await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
    assert any(r.content == "SUM" for r in summ)


@pytest.mark.asyncio
async def test_fold_retains_delegated_with_result_by_keep_last() -> None:
    """派给别的 agent 的 root task（dispatch + result 已回）是完成胶囊——按 keep_last 保留/折叠，
    不再因 has_dispatch 无 finish 被当在途豁免。当前 task 不在其中,故这些单元正常参与 keep_last。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("cur")  # 当前 task="cur"，不在被折 origin 集内
    for i in range(3):  # 3 个完成的派发型 root task D0,D1,D2
        tc = f"dd{i}"
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, "", 10 * i, role="assistant",
                             origin_task_id=f"D{i}", parent_task_id=None,
                             tool_calls=[{"id": tc, "name": "control:delegate_task", "input": {}}]), _ctx())
        await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, f"res {i}", 10 * i + 1, role="tool",
                             origin_task_id=f"D{i}", tool_call_id=tc), _ctx())
    cur = Task(id="cur", session_id="s1", status="ACTIVE", assigned_agent_id="ag1",
               creator_agent_id="ag1", title="cur", user_prompt="c", settings=NormalTaskSettings())
    n = await fold_root_experience(_state(cur, LoopConfig()), _loop_ctx(mem, _FakeTM({})),
                                   keep_last=1, summary_text="SUM")
    assert n > 0
    turns = await mem.recall_recent(sc, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    alive = {r.metadata.get("origin_task_id") for r in turns}
    assert alive == {"D2"}  # keep_last=1 → 仅最新的派发型完成胶囊保留，D0/D1 折入摘要


@pytest.mark.asyncio
async def test_fold_paired_dispatch_superseded_no_dangling() -> None:
    """新格式：所有已折胶囊的 AGENT_CONVERSATION_TURN 消失，最新 keep_last=1 的 origin=rt3 保留。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 4)
    await fold_root_experience(_state(_active_task(), LoopConfig()),
                               _loop_ctx(mem, _FakeTM({})), keep_last=1, summary_text="SUM")
    # 新格式无 TASK_DISPATCH；检查 AGENT_CONVERSATION_TURN：只剩 rt3 的回合
    turns = await mem.recall_recent(sc, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    alive_origins = {r.metadata.get("origin_task_id") for r in turns}
    assert alive_origins == {"rt3"}  # only the kept root's turns remain


@pytest.mark.asyncio
async def test_fold_rolls_old_summary_into_new() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    # an existing AGENT_COMPACT_SUMMARY from a prior fold
    await mem.ingest(_ev(T.AGENT_COMPACT_SUMMARY, sc, "OLD SUMMARY", 0, role="user"), _ctx())
    await _seed_root_residues(mem, sc, 3, start_t=10)
    await fold_root_experience(_state(_active_task(), LoopConfig()),
                               _loop_ctx(mem, _FakeTM({})), keep_last=1, summary_text="NEW")
    summaries = await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
    contents = {s.content for s in summaries}
    assert "OLD SUMMARY" not in contents and "NEW" in contents  # rolled into one


@pytest.mark.asyncio
async def test_fold_anchor_precedes_surviving_body_capsule() -> None:
    """防御锚点：存活单元的 user 回合被折、task 层 user_prompt 却存活（跨层 split，存量数据）时，
    新摘要仍锚在该 user_prompt 之前——不只看 agent 回合的最早 ts。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 3)  # rt0/rt1/rt2 → 老单元被折
    # 存活单元 rtS：task 层 user_prompt 很早（t=1），但唯一 live 的 agent 回合很晚（t=50）
    body_scope = MemoryAddress(session_id="s1", task_id="rtS", agent_id="ag1")
    early_up = _ev(T.USER_PROMPT, body_scope, "body rtS", 1, role="user")
    await mem.ingest(early_up, _ctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, sc, "late turn rtS", 50, role="assistant",
                         origin_task_id="rtS", parent_task_id=None), _ctx())

    await fold_root_experience(_state(_active_task(), LoopConfig()),
                               _loop_ctx(mem, _FakeTM({})), keep_last=1, summary_text="SUM")
    summ = [s for s in await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
            if s.content == "SUM"]
    assert summ, "new summary written"
    # 锚点须早于存活单元的 task 层胶囊 ts（t=1），否则该胶囊会排到摘要之前
    assert summ[0].timestamp < early_up.timestamp


@pytest.mark.asyncio
async def test_fold_at_or_below_keep_last_is_noop() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 2)
    n = await fold_root_experience(_state(_active_task(), LoopConfig()),
                                   _loop_ctx(mem, _FakeTM({})), keep_last=6, summary_text="SUM")
    assert n == 0
    summaries = await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
    assert summaries == []  # nothing written


@pytest.mark.asyncio
async def test_fold_summary_anchored_before_kept_window() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 3)
    await fold_root_experience(_state(_active_task(), LoopConfig()),
                               _loop_ctx(mem, _FakeTM({})), keep_last=1, summary_text="SUM")
    # summary must be chronologically before all kept AGENT_CONVERSATION_TURN records
    turns = await mem.recall_recent(sc, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    summ = await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
    assert len(summ) == 1
    min_kept_ts = min(r.timestamp for r in turns)
    assert summ[0].timestamp < min_kept_ts            # summary anchored before kept window


@pytest.mark.asyncio
async def test_count_root_residues_excludes_subtask() -> None:
    """新格式：root 胶囊（AGENT_CONVERSATION_TURN, parent=None）计 3；
    TASK_DISPATCH_RESULT(parent=t1) 为 working set，_count_root_residues 不计此类型。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 3)
    # working set: cross-agent TASK_DISPATCH_RESULT with parent_task_id set (不是 AGENT_CONVERSATION_TURN)
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, sc, "sub", 99, role="tool",
                         tool_call_id="sd", parent_task_id="t1"), _ctx())
    assert await _count_root_residues(_state(_active_task(), LoopConfig()), _loop_ctx(mem, _FakeTM({}))) == 3


def _delegate_pair(sc, oid: str, t: int, *, with_result: bool):
    """派发型 root task 的 dispatch 对：delegate assistant 回合（+可选 result tool 回合，同 origin）。"""
    evs = [_ev(T.AGENT_CONVERSATION_TURN, sc, "", t, role="assistant",
               origin_task_id=oid, parent_task_id=None,
               tool_calls=[{"id": f"d_{oid}", "name": "control:delegate_task", "input": {}}])]
    if with_result:
        evs.append(_ev(T.AGENT_CONVERSATION_TURN, sc, f"result {oid}", t + 1, role="tool",
                       origin_task_id=oid, parent_task_id=None, tool_call_id=f"d_{oid}"))
    return evs


@pytest.mark.asyncio
async def test_count_root_residues_counts_delegated_with_result() -> None:
    """派给别的 agent 的 root task：结果已回（result/tool 回合）→ 计入完成；仅 dispatch 无 result → 在途不计。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("cur")  # 当前 task="cur"，不在下面 origin 集内
    for e in _delegate_pair(sc, "P", 0, with_result=True):    # 完成的派发型 root task
        await mem.ingest(e, _ctx())
    for e in _delegate_pair(sc, "Q", 5, with_result=False):   # 在途（结果未回）
        await mem.ingest(e, _ctx())
    cur = Task(id="cur", session_id="s1", status="ACTIVE", assigned_agent_id="ag1",
               creator_agent_id="ag1", title="cur", user_prompt="c", settings=NormalTaskSettings())
    assert await _count_root_residues(_state(cur, LoopConfig()), _loop_ctx(mem, _FakeTM({}))) == 1


@pytest.mark.asyncio
async def test_count_root_residues_excludes_current_task_even_with_result() -> None:
    """当前正在跑的 task 即使 dispatch+result 齐全,也不算可折完成单元（当前任务不能折）。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("cur")
    for e in _delegate_pair(sc, "cur", 0, with_result=True):  # origin==当前 task
        await mem.ingest(e, _ctx())
    cur = Task(id="cur", session_id="s1", status="ACTIVE", assigned_agent_id="ag1",
               creator_agent_id="ag1", title="cur", user_prompt="c", settings=NormalTaskSettings())
    assert await _count_root_residues(_state(cur, LoopConfig()), _loop_ctx(mem, _FakeTM({}))) == 0


# ═══════════════════ CompactStep.execute orchestration ═══════════════════

@pytest.mark.asyncio
async def test_execute_noop_when_budget_gate_closed() -> None:
    """预算门未开（token_estimate 低于 target）时，escalating_compact 空跑，不摸任何记录。"""
    mem = InMemoryMemoryProvider()
    state = _state(_active_task(), LoopConfig(compact_keep_last=1), context_tokens=0)
    outcome = await CompactStep().execute(state, _loop_ctx(mem, _FakeTM({}), with_assembler=True))
    assert outcome.events == []
    assert await mem.recall_recent(_sc("t1"), [T.TASK_COMPACT_SUMMARY, T.AGENT_COMPACT_SUMMARY], 100, _ctx()) == []


@pytest.mark.asyncio
async def test_execute_folds_task_and_root() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    for i in range(4):  # active task conv > keep_last
        await mem.ingest(_ev(T.LLM_RESPONSE, sc, f"turn {i}", i, role="assistant"), _ctx())
    await _seed_root_residues(mem, sc, 4, start_t=100)  # root residues > keep_last

    state = _state(_active_task(), LoopConfig(compact_keep_last=1))
    await CompactStep().execute(state, _loop_ctx(mem, _FakeTM({}), with_assembler=True))

    # (a) task layer collapsed into a USER_PROMPT with collapsed=True metadata and COLLAPSE_DELIM
    task_ups = await mem.recall_recent(sc, [T.USER_PROMPT], 100, _ctx())
    collapsed = [r for r in task_ups if r.metadata.get("collapsed")]
    assert collapsed, "task layer should be collapsed into a USER_PROMPT"
    assert COLLAPSE_DELIM in collapsed[0].content
    agent_recs = await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx())
    assert agent_recs != []                                                  # (c) root folded


@pytest.mark.asyncio
async def test_execute_task_fold_untouches_subtask_residues() -> None:
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    for i in range(4):
        await mem.ingest(_ev(T.LLM_RESPONSE, sc, f"turn {i}", i, role="assistant"), _ctx())
    for i in range(3):  # only sub-task residues (no root residues)
        await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, sc, f"sub {i}", 100 + i, role="tool",
                             tool_call_id=f"sd{i}", parent_task_id="t1"), _ctx())

    state = _state(_active_task(), LoopConfig(compact_keep_last=1))
    await CompactStep().execute(state, _loop_ctx(mem, _FakeTM({}), with_assembler=True))

    # task layer collapsed into a USER_PROMPT with collapsed=True (not a TASK_COMPACT_SUMMARY)
    task_ups = await mem.recall_recent(sc, [T.USER_PROMPT], 100, _ctx())
    assert any(r.metadata.get("collapsed") for r in task_ups), "task layer should be collapsed"
    subs = await mem.recall_recent(sc, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    assert {r.content for r in subs} == {"sub 0", "sub 1", "sub 2"}          # working set survives
    assert await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx()) == []  # no root fold


@pytest.mark.asyncio
async def test_execute_keeps_finished_short_body_then_folds_root() -> None:
    """task-resident（spec 2026-06-28 §5）：CompactStep 不再回收 finished 短 task——其 body 留
    task 层（即胶囊，由跨层 fold 管理）；root residues 仍照常 fold。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    # a finished short same-agent leaf + enough root residues to fold
    await mem.ingest(_ev(T.TASK_DISPATCH, sc, "", 0, role="assistant",
                         tool_call_id="oc2", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("t2"), "sub work", 1, role="assistant"), _ctx())
    await _seed_root_residues(mem, sc, 4, start_t=100)
    tm = _FakeTM({"t1": _active_task(), "t2": _finished_short("t2", oc="oc2")})

    state = _state(_active_task(), LoopConfig(compact_keep_last=1))
    await CompactStep().execute(state, _loop_ctx(mem, tm, with_assembler=True))

    # task-resident：short task t2 的 body 留 task 层（不被 CompactStep 回收）
    conv = await mem.recall_recent_by_agent(_sc("x", "ag1"), [T.LLM_RESPONSE], 100, _ctx())
    assert any(r.metadata.get("task_id") == "t2" for r in conv), (
        "finished short task body must stay (task-resident: no pressure-collapse)"
    )
    # root residues 仍 fold
    assert await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx()) != []


@pytest.mark.asyncio
async def test_execute_idempotent_second_run_noop() -> None:
    """第一次跑（预算门开）真折一轮；折后负载回落，第二次预算门不开（context_tokens 低于
    target）→ 空跑不摸记录（escalating_compact 门未过时连 STARTED 都不发,见 gate 语义）。"""
    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    await _seed_root_residues(mem, sc, 4)
    state = _state(_active_task(), LoopConfig(compact_keep_last=1))
    ctx = _loop_ctx(mem, _FakeTM({}), with_assembler=True)

    out1 = await CompactStep().execute(state, ctx)
    assert out1.events != []  # first run actually did work (root residues > keep_last)
    after_first = await mem.recall_recent(sc, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    # simulate settled load post-compaction: budget gate no longer open
    state.agent.loop_guard.context_tokens = 0
    out2 = await CompactStep().execute(state, ctx)  # nothing new to fold (root residues now ≤ keep_last)
    after_second = await mem.recall_recent(sc, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    assert out2.events == []
    assert {r.id for r in after_first} == {r.id for r in after_second}  # stable


# ═══════════════════ pre-dispatch compaction (派发前压缩) ═══════════════════

@pytest.mark.asyncio
async def test_predispatch_folds_task_and_agent_layers() -> None:
    """派发前压缩越过阈值时，与 CompactStep 同构地同时折 task 层与 agent 层，事件标 pre_dispatch。"""
    from tests.unit.test_predispatch_compact import _compact_before_dispatch

    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    for i in range(4):  # active task conv > keep_last
        await mem.ingest(_ev(T.LLM_RESPONSE, sc, f"turn {i}", i, role="assistant"), _ctx())
    await _seed_root_residues(mem, sc, 4, start_t=100)  # completed root residues > keep_last

    cfg = LoopConfig(compact_keep_last=1, predispatch_compact_token_ratio=0.6)
    state = _state(_active_task(), cfg, context_limit=1000)
    # 950 / 1000 = 0.95 >= 0.6 → 门控通过；留足预算余量使 L1 折后仍在 target(800) 之上，续入
    # L2/L3（escalating_compact 每级折后即测：低于 target 就提前停,此处要打穿到 L3 才折 task 层）。
    events = await _compact_before_dispatch(
        state, _loop_ctx(mem, _FakeTM({}), with_assembler=True), prompt_tokens=950)

    compacted = [e for e in events if e.type == "MemoryCompacted"]
    assert {e.payload.get("layer") for e in compacted} == {"task", "agent"}      # 两层都折
    assert all(e.payload.get("trigger") == "pre_dispatch" for e in events)
    # task layer collapsed into a USER_PROMPT with collapsed=True (not a TASK_COMPACT_SUMMARY)
    task_ups = await mem.recall_recent(sc, [T.USER_PROMPT], 100, _ctx())
    assert any(r.metadata.get("collapsed") for r in task_ups), "task layer should be collapsed"
    assert await mem.recall_recent(sc, [T.AGENT_COMPACT_SUMMARY], 100, _ctx()) != []


@pytest.mark.asyncio
async def test_predispatch_gate_blocks_below_threshold() -> None:
    from tests.unit.test_predispatch_compact import _compact_before_dispatch

    mem = InMemoryMemoryProvider()
    sc = _sc("t1")
    for i in range(4):
        await mem.ingest(_ev(T.LLM_RESPONSE, sc, f"turn {i}", i, role="assistant"), _ctx())
    await _seed_root_residues(mem, sc, 4, start_t=100)

    cfg = LoopConfig(compact_keep_last=1, predispatch_compact_token_ratio=0.6)
    state = _state(_active_task(), cfg, context_limit=1000)
    # 500 / 1000 = 0.5 < 0.6 → 不压
    events = await _compact_before_dispatch(
        state, _loop_ctx(mem, _FakeTM({}), with_assembler=True), prompt_tokens=500)
    assert events == []
    assert await mem.recall_recent(sc, [T.TASK_COMPACT_SUMMARY, T.AGENT_COMPACT_SUMMARY], 100, _ctx()) == []


def test_original_section_splits_on_legacy_and_current_delims() -> None:
    """坍缩分隔标记改成英文后，存量库里带旧中文标记的 USER_PROMPT 仍须能切出「原始消息」节
    ——切不出就会把「原文 + 旧摘要」整体当原文，再坍缩时无界增长。"""
    from tests.compaction_testkit import COLLAPSE_DELIM, LEGACY_COLLAPSE_DELIMS, _original_section
    for delim in (COLLAPSE_DELIM, *LEGACY_COLLAPSE_DELIMS):
        assert _original_section(f"原始消息{delim}执行摘要正文") == "原始消息", \
            f"未按分隔标记 {delim!r} 切出原文节"
    assert _original_section("没有标记的整条消息") == "没有标记的整条消息"
