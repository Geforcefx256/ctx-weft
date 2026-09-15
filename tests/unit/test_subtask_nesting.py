"""Task 9: sub-task bubble split + same-agent nested capsule.

同 agent child close:
  - parent scope TASK_DISPATCH_RESULT content == "Sub-task '<title>' scheduled."
  - parent scope 含 child 的 AGENT_CONVERSATION_TURN (origin_task_id=child.id)
    排在 dispatch pair 之后

跨 agent child close:
  - parent scope TASK_DISPATCH_RESULT content == mem_content (含 "Process Report:")
  - parent scope **无** origin_task_id=child.id 的 AGENT_CONVERSATION_TURN
"""

from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import finalize_task_memory
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
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _state(task: Task, scope: MemoryAddress, loop_config: LoopConfig):
    agent = SimpleNamespace(id=scope.agent_id, loop_config=loop_config)
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent)


def _loop_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_ctx(), task_manager=_FakeTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


async def _seed_conv_nonshort(mem, scope) -> None:
    """Seed enough turns to be non-short (> turn_cap or too many tokens)."""
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello child", 1, role="user"), _ctx())
    for i in range(5):
        big_text = "x " * 4000
        await mem.ingest(_ev(T.LLM_RESPONSE, scope, big_text, i + 2, role="assistant"), _ctx())


async def test_same_agent_child_mints_frame_and_writes_ack() -> None:
    """§2.5(2026-07-03)：同 agent child close → finalize **铸**派发框 + 配对静态 ack，二者**同锚
    task.started_at**（gateway 不再为 delegate_task eager 写框）→ 框与 result 严格相邻、落在「任务开始
    执行」时间线上。框名取 task.origin_tool_name（delegate_task 子 = 真名，保真）。"""
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c1", "ag1")
    await _seed_conv_nonshort(mem, child_scope)
    # 不 seed eager 框：delegate_task 的框由 finalize 铸（见 _ensure_dispatch_frame）

    started = _BASE + timedelta(seconds=5)  # task manager 真正启动子任务的时刻
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="oc1", origin_tool_name=DELEGATE_TASK_NAME,
                 title="My Sub Task", description="do the sub work",
                 user_prompt="do sub", started_at=started, settings=NormalTaskSettings())

    mem_content = "sub outputs\n\nProcess Report: sub summary"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, mem_content, "success", _loop_ctx(mem),
        act_recap="sub summary", task_summary="",
    )

    parent_scope = _sc("p1", "ag1")
    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    from ctx_weft.core.loop.steps.finalize import _dispatch_ack
    # 铸出的派发框（携 oc1，真名 delegate_task），锚 started_at
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "oc1" and tc.get("name") == DELEGATE_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    assert frame, "finalize 须为 delegate_task 子铸一条派发框（真名 delegate_task）"
    assert frame[0].timestamp == started, "框须锚 started_at"
    # 配对静态 ack，同锚 started_at → 与框同时间戳（严格相邻）
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "oc1"]
    assert ack and ack[0].content == _dispatch_ack(task_ref(child), "success"), (
        f"§2.5: static ack content must be {_dispatch_ack(task_ref(child), 'success')!r}; got {[r.content for r in ack]}"
    )
    assert ack[0].timestamp == started == frame[0].timestamp, (
        f"框与 result 须同锚 started_at（相邻）；frame={frame[0].timestamp} ack={ack[0].timestamp} started={started}"
    )


# ── start 时铸框 + running ack（driver.run 钩子）────────────────────────────

def _child_task(started, *, parent="p1", tcid="oc1", agent="ag1") -> Task:
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME
    return Task(id="c1", session_id="s1", status="ACTIVE", tenant_id="default",
                assigned_agent_id=agent, creator_agent_id=agent, parent_task_id=parent,
                origin_tool_call_id=tcid, origin_tool_name=DELEGATE_TASK_NAME,
                title="My Sub Task", description="do the sub work",
                user_prompt="do sub", started_at=started, settings=NormalTaskSettings())


async def _turns(mem, scope):
    return await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())


async def test_start_hook_mints_frame_and_running_ack_at_started_at() -> None:
    """子任务 start 时（driver.run）即铸框 + running ack，同锚 started_at。

    执行期间唯一的读者是子任务自己——同 agent 时父挂起、不装配。有了这一对，子任务才看得见
    自己的来历，那条 task_prompt 的 user 回合不再像用户凭空插话。
    """
    from ctx_weft.core.loop.steps.finalize import ensure_dispatch_frame_at_start
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME

    mem = InMemoryMemoryProvider()
    started = _BASE + timedelta(seconds=5)
    child = _child_task(started)

    await ensure_dispatch_frame_at_start(_state(child, _sc("c1"), LoopConfig()), _loop_ctx(mem))

    turns = await _turns(mem, _sc("p1"))
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "oc1" and tc.get("name") == DELEGATE_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "oc1"]
    assert frame, "start 时须铸出派发框"
    assert ack, "start 时须写配对 running ack（否则框悬挂、被 legalize 剥掉）"
    assert frame[0].timestamp == started == ack[0].timestamp, "框与 ack 须同锚 started_at"
    assert "running" in ack[0].content, f"start 态 ack 须表明在跑；got {ack[0].content!r}"


async def test_start_hook_noop_without_parent() -> None:
    """root task（无 parent / 无 origin_tool_call_id）没有派发方，不铸框。"""
    from ctx_weft.core.loop.steps.finalize import ensure_dispatch_frame_at_start
    mem = InMemoryMemoryProvider()
    root = _child_task(_BASE, parent=None, tcid=None)

    await ensure_dispatch_frame_at_start(_state(root, _sc("c1"), LoopConfig()), _loop_ctx(mem))

    assert await _turns(mem, _sc("p1")) == [], "root task 不得铸框"


async def test_start_hook_idempotent_across_reruns() -> None:
    """retry / resume 会重跑 driver.run：框与 ack 都不得重复，且 ts 不被改写。"""
    from ctx_weft.core.loop.steps.finalize import ensure_dispatch_frame_at_start
    mem = InMemoryMemoryProvider()
    started = _BASE + timedelta(seconds=5)
    child = _child_task(started)
    state, lctx = _state(child, _sc("c1"), LoopConfig()), _loop_ctx(mem)

    await ensure_dispatch_frame_at_start(state, lctx)
    child.started_at = _BASE + timedelta(seconds=99)   # TM 每次派发都刷新 started_at
    await ensure_dispatch_frame_at_start(state, lctx)

    turns = await _turns(mem, _sc("p1"))
    assert len([r for r in turns if r.role == "assistant"]) == 1, f"框不得重复；got {turns}"
    assert len([r for r in turns if r.role == "tool"]) == 1, f"ack 不得重复；got {turns}"
    assert turns[-1].timestamp == started, "重跑不得改写原锚点"


async def test_close_replaces_running_ack_with_terminal() -> None:
    """close 时 running ack 被终态**替换**（不是新增）——同一 tool_call_id 恒只有一条 active
    result，否则 reorder 会把两条都排到框后。"""
    from ctx_weft.core.loop.steps.finalize import _dispatch_ack, ensure_dispatch_frame_at_start

    mem = InMemoryMemoryProvider()
    started = _BASE + timedelta(seconds=5)
    child = _child_task(started)
    child_scope = _sc("c1")
    await _seed_conv_nonshort(mem, child_scope)

    await ensure_dispatch_frame_at_start(_state(child, child_scope, LoopConfig()), _loop_ctx(mem))
    child.status = "FINISHED"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, "sub outputs\n\nProcess Report: sub summary", "success", _loop_ctx(mem),
        act_recap="sub summary", task_summary="",
    )

    acks = [r for r in await _turns(mem, _sc("p1"))
            if r.role == "tool" and r.metadata.get("tool_call_id") == "oc1"]
    assert len(acks) == 1, f"同一 tool_call_id 只能有一条 active result；got {[a.content for a in acks]}"
    assert acks[0].content == _dispatch_ack(task_ref(child), "success")
    assert "running" not in acks[0].content, "running 态须被终态替换掉"
    assert acks[0].timestamp == started, "替换后仍须锚 started_at（与框相邻）"


async def test_close_ack_stays_co_anchored_with_frame_after_retry() -> None:
    """retry 会刷新 task.started_at（TM 每次派发都写）。close 替换 ack 时须沿用**框自己的**
    时间戳，而不是按当下的 started_at 重算——否则框停在首次 started_at、ack 落到末次，
    「框与 result 同锚、严格相邻」的不变量断裂，ack 在原始存储里漂进子 body 中间。
    """
    from ctx_weft.core.loop.steps.finalize import ensure_dispatch_frame_at_start

    mem = InMemoryMemoryProvider()
    first_start = _BASE + timedelta(seconds=5)
    child = _child_task(first_start)
    child_scope = _sc("c1")
    await _seed_conv_nonshort(mem, child_scope)

    await ensure_dispatch_frame_at_start(_state(child, child_scope, LoopConfig()), _loop_ctx(mem))

    # 第一轮撞 max_turns → retry → TM 重新派发，刷新 started_at
    child.started_at = _BASE + timedelta(seconds=900)
    await ensure_dispatch_frame_at_start(_state(child, child_scope, LoopConfig()), _loop_ctx(mem))

    child.status = "FINISHED"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, "sub outputs\n\nProcess Report: sub summary", "success", _loop_ctx(mem),
        act_recap="sub summary", task_summary="",
    )

    turns = await _turns(mem, _sc("p1"))
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "oc1" for tc in (r.metadata.get("tool_calls") or []))]
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "oc1"]
    assert len(frame) == 1 and len(ack) == 1
    assert frame[0].timestamp == first_start, "框须保持首次 started_at，不被 retry 改写"
    assert ack[0].timestamp == frame[0].timestamp, (
        f"终态 ack 须与框同锚；frame={frame[0].timestamp} ack={ack[0].timestamp} "
        f"(retry 后的 started_at={child.started_at})"
    )


async def test_same_agent_child_finish_pair_written_into_parent_scope() -> None:
    """task-resident：同 agent child 的 finish 对（AGENT_CONVERSATION_TURN, origin_task_id=child.id）
    写入 parent agent scope；**不镜像 body**（无 user 锚点），child raw body 留 child task 层。"""
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c1", "ag1")
    await _seed_conv_nonshort(mem, child_scope)

    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="oc1", title="Child Task", user_prompt="do sub",
                 settings=NormalTaskSettings())

    mem_content = "sub outputs\n\nProcess Report: sub summary"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, mem_content, "success", _loop_ctx(mem),
        act_recap="sub summary", task_summary="",
    )

    parent_scope = _sc("p1", "ag1")

    # child finish pair (AGENT_CONVERSATION_TURN, origin_task_id=child.id) must exist in parent scope
    caps = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    child_turns = [r for r in caps if r.metadata.get("origin_task_id") == "c1"]
    assert len(child_turns) == 2, (
        f"expected child finish pair (2 turns) in parent scope; got {[(r.role, r.content[:30]) for r in child_turns]}"
    )

    # task-resident: NO body mirror (no user anchor)
    assert not any(r.role == "user" for r in child_turns), (
        "task-resident: child finish pair must NOT mirror body (no user anchor)"
    )

    # must include finish pair: tool role (content = act_recap fallback since task_summary="")
    finish_tool = [r for r in child_turns if r.role == "tool"]
    assert finish_tool, "expected finish-pair tool turn in parent scope capsule"
    assert finish_tool[0].content, (
        f"finish tool content must be non-empty, got: {finish_tool[0].content!r}"
    )

    # child raw body stays in child task layer (not mirrored/superseded)
    child_body = await mem.recall_recent(child_scope, [T.USER_PROMPT, T.LLM_RESPONSE], 100, _ctx())
    assert any(r.role == "user" and "hello child" in r.content for r in child_body), (
        "child raw body (user anchor) must stay in child task layer"
    )


async def test_cross_agent_child_bubble_is_conversation_turn() -> None:
    """§2.3：跨 agent child 的 dispatch result 写成 AGENT_CONVERSATION_TURN（tool 回合），
    与 gateway 写的 delegate assistant 回合靠 tool_call_id 配对；origin_task_id=delegating
    task（与同单元 finish 对同 origin、同命运一起折）；不再写 TASK_DISPATCH_RESULT enum。"""
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c2", "ag2")
    # cross-agent child: short is fine since cross_agent always bubbles
    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "hello", 1, role="user"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "reply", 2, role="assistant"), _ctx())

    child = Task(id="c2", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag2", creator_agent_id="ag1",  # cross-agent
                 parent_task_id="p1", origin_tool_call_id="oc2",
                 title="Cross Agent Child", user_prompt="do cross",
                 settings=NormalTaskSettings())

    mem_content = "cross outputs\n\nProcess Report: cross summary"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, mem_content, "success", _loop_ctx(mem),
        act_recap="cross summary", task_summary="",
    )

    parent_scope = _sc("p1", "ag1")
    # dispatch result == 普通 conversation turn（tool），配对 oc2、归 delegating task(p1) 单元
    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    result = [r for r in turns
              if r.role == "tool" and r.metadata.get("tool_call_id") == "oc2"]
    assert result, "expected dispatch result as AGENT_CONVERSATION_TURN (tool) paired with oc2"
    assert result[0].metadata.get("origin_task_id") == "p1", (
        "dispatch result 须归 delegating task 单元（与 finish 对同 origin、同命运）"
    )
    assert result[0].content == mem_content, (
        f"expected full mem_content (black box), got: {result[0].content!r}"
    )
    # 不再写 legacy TASK_DISPATCH_RESULT enum
    legacy = await mem.recall_recent(parent_scope, [T.TASK_DISPATCH_RESULT], 100, _ctx())
    assert legacy == [], "cross-agent dispatch result must not write TASK_DISPATCH_RESULT enum"


async def test_same_agent_close_mints_frame_and_ack_co_anchored() -> None:
    """§2.5(2026-07-03)：同 agent close（_close_one）铸派发框 + 配对静态 ack，
    框与 ack 同锚 task.started_at（同时间戳 → 相邻），无 eager 框预置；框名取真名 delegate_task。"""
    from ctx_weft.core.loop.steps.finalize import _close_one, _dispatch_ack
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME

    mem = InMemoryMemoryProvider()
    child_scope = _sc("c1", "ag1")
    parent_scope = _sc("p1", "ag1")
    await _seed_conv_nonshort(mem, child_scope)

    started = _BASE + timedelta(seconds=5)
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="oc1", origin_tool_name=DELEGATE_TASK_NAME,
                 title="My Sub Task", user_prompt="do sub",
                 started_at=started, settings=NormalTaskSettings())
    state = _state(child, child_scope, LoopConfig())

    await _close_one(mem, state, child, "out\n\nProcess Report: r", "success", _loop_ctx(mem),
                     short=True, act_recap="本段做了 X", task_summary="整段总结")

    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())

    # 铸出的派发框（真名 delegate_task）
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "oc1" and tc.get("name") == DELEGATE_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    assert frame, "_close_one 须铸派发框（真名 delegate_task）"

    # 配对静态 result：content=_dispatch_ack(title)、tool_call_id 配对、与框同锚 started_at
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "oc1"]
    assert ack and ack[0].content == _dispatch_ack(task_ref(child), "success"), f"expected static ack {_dispatch_ack(task_ref(child), 'success')!r}, got {[r.content for r in ack]}"
    assert frame[0].timestamp == started == ack[0].timestamp, (
        f"框与 ack 须同锚 started_at；frame={frame[0].timestamp} ack={ack[0].timestamp} started={started}"
    )

    # stray-ack guard：no OTHER tool record carries ack content
    assert _dispatch_ack(task_ref(child), "success") not in {
        r.content for r in turns
        if r.metadata.get("tool_call_id") != child.origin_tool_call_id
    }, "ack content must only appear in the paired tool_call_id record"


async def test_ensure_dispatch_frame_mixed_tz_no_crash() -> None:
    """派发框 timestamp 为 naive（事件重放 / DB 反序列化丢 tz），task.started_at 为 aware：
    _ensure_dispatch_frame 归一 tz 后返回**框自己的** ts，绝不 TypeError（回归 2026-07-03）。

    返回值的唯一用途是给配对 ack 定锚，而要求是「ack 必须贴着框」——故框在哪就返回哪，
    返回 started_at 只在「框恰好就在 started_at」时才成立。框已铸的分支从前是永不触发的防御
    （框与 ack 在 close 同一次调用里写、必然同锚），现在是常态路径（框由子 start 时铸），
    且 TaskManager 每次派发都刷新 started_at，retry 后二者必然分叉——见
    test_close_ack_stays_co_anchored_with_frame_after_retry。对遗留的 eager 框（锚在派发时刻）
    也同理：返回框自己的 ts 才能让这一对重新贴合。
    """
    from ctx_weft.core.loop.steps.finalize import _ensure_dispatch_frame

    mem = InMemoryMemoryProvider()
    parent_scope = _sc("p1", "ag1")
    naive_dispatch = datetime(2026, 1, 1)          # naive（无 tzinfo）
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=parent_scope, content="",
        timestamp=naive_dispatch, role="assistant",
        metadata={"origin_task_id": "p1", "parent_task_id": None,
                  "tool_calls": [{"id": "oc1", "name": "delegate_task", "input": {}}]},
    ), _ctx())

    started = _BASE + timedelta(seconds=5)          # aware
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="oc1", title="My Sub Task", user_prompt="do sub",
                 started_at=started, settings=NormalTaskSettings())

    ts, tcid = await _ensure_dispatch_frame(mem, parent_scope, child, _ctx())
    assert tcid == "oc1", f"须返回框自己携带的配对 id；got {tcid!r}"
    assert ts == naive_dispatch.replace(tzinfo=UTC), (
        f"须返回框自己的 ts（归一为 aware），而非 started_at；got {ts!r}"
    )
    assert ts.tzinfo is not None, "naive 框 ts 须被归一为 aware，否则调用方与 aware 比较时 TypeError"


async def test_concurrent_same_agent_dispatch_pairs_stay_adjacent() -> None:
    """多个同 agent delegate_task 各自 close：每对 frame+ack 同锚各自 started_at → 按 composer 的
    (timestamp, seq_no) 排序严格成对相邻（F0,R0,F1,R1,F2,R2），不再 F,F,F,R,R,R 堆叠错序。
    再过 legalize_messages 确认 provider 合法（每 assistant tool_use 紧跟其 tool result）。"""
    from ctx_weft.core.loop.llm_gateway import legalize_messages
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME
    from ctx_weft.protocols import LLMMessage

    mem = InMemoryMemoryProvider()
    parent_scope = _sc("p1", "ag1")
    for i, secs in enumerate((5, 10, 15)):          # started_at 递增（串行执行）
        cscope = _sc(f"c{i}", "ag1")
        await _seed_conv_nonshort(mem, cscope)
        child = Task(id=f"c{i}", session_id="s1", status="FINISHED", tenant_id="default",
                     assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                     origin_tool_call_id=f"oc{i}", origin_tool_name=DELEGATE_TASK_NAME,
                     title=f"T{i}", user_prompt="x",
                     started_at=_BASE + timedelta(seconds=secs), settings=NormalTaskSettings())
        await finalize_task_memory(
            mem, _state(child, cscope, LoopConfig()), child,
            f"out{i}\n\nProcess Report: r{i}", "success", _loop_ctx(mem),
            act_recap=f"recap{i}", task_summary="")

    recs = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 500, _ctx())
    recs = sorted(recs, key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0)))  # 同 composer 排序

    dispatch_ids = {"oc0", "oc1", "oc2"}

    def _tc_id(r):
        if r.role == "assistant":
            for tc in (r.metadata.get("tool_calls") or []):
                if tc.get("id") in dispatch_ids and tc.get("name") == DELEGATE_TASK_NAME:
                    return tc["id"]
        if r.role == "tool" and r.metadata.get("tool_call_id") in dispatch_ids:
            return r.metadata["tool_call_id"]
        return None

    seq = [(r.role, _tc_id(r)) for r in recs if _tc_id(r) is not None]
    assert seq == [("assistant", "oc0"), ("tool", "oc0"),
                   ("assistant", "oc1"), ("tool", "oc1"),
                   ("assistant", "oc2"), ("tool", "oc2")], f"派发对未成对相邻/未按 started_at 有序: {seq}"

    # 过 legalize：整条(含 finish 对)重建为 LLMMessage 后仍合法（无悬挂/无孤儿/配对紧邻）
    msgs = [LLMMessage(role="user", content="root")]
    for r in recs:
        if r.role == "assistant":
            tcs = [{"id": tc["id"], "name": tc.get("name", ""), "input": tc.get("input", {})}
                   for tc in (r.metadata.get("tool_calls") or [])]
            msgs.append(LLMMessage(role="assistant", content=r.content or "", tool_calls=tcs))
        elif r.role == "tool":
            msgs.append(LLMMessage(role="tool", content=r.content or "(x)",
                                   tool_call_id=r.metadata.get("tool_call_id", "")))
    out = legalize_messages(msgs)
    seen: set[str] = set()
    for m in out:
        if m.role == "assistant":
            seen.update(tc.get("id") for tc in m.tool_calls)
        if m.role == "tool":
            assert m.tool_call_id in seen, f"tool result {m.tool_call_id} 未紧跟其 tool_use"


async def test_cross_agent_result_carries_outputs_and_task_summary() -> None:
    """Task 5: 跨 agent dispatch result（cross_agent bubble, mem_content）须含 outputs + task_summary，
    不掺 act_recap（mem_content 的 report 部分改用 task_summary）。"""
    from ctx_weft.core.loop.steps.finalize import FinalizeStep
    from ctx_weft.core.loop.steps.observe import Verdict

    mem = InMemoryMemoryProvider()
    child_scope = _sc("c99", "ag2")

    # seed minimal conv so FinalizeStep can run
    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "do cross", 1, role="user"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "done", 2, role="assistant"), _ctx())

    child = Task(id="c99", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag2", creator_agent_id="ag1",  # cross-agent
                 parent_task_id="p99", origin_tool_call_id="oc99",
                 title="Cross Agent Child", user_prompt="do cross",
                 settings=NormalTaskSettings())
    child.outputs = "最终产出给 user"

    verdict = Verdict(task_outcome="success", act_recap="本段", task_summary="综合 process report")

    agent = SimpleNamespace(id="ag2", loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    state = SimpleNamespace(
        run_id="run99", sequence_counter=0, session=session,
        scope=child_scope, task=child, agent=agent, verdict=verdict,
    )

    await FinalizeStep().execute(state, _loop_ctx(mem))

    parent_scope = _sc("p99", "ag1")
    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    result = [r for r in turns if r.role == "tool"
              and r.metadata.get("tool_call_id") == "oc99"]
    assert result
    body = result[0].content
    assert "最终产出给 user" in body          # 最终输出
    assert "综合 process report" in body       # task_summary 承载 process report
    assert "本段" not in body                   # 不掺 act_recap


async def test_plan_child_mints_start_task_frame_when_absent() -> None:
    """delegate_plan 子: parent scope 无预置框 → finalize 补铸 start_task 框，bubble/ACK 配对其 id。"""
    from ctx_weft.core.loop.steps.finalize import _close_one, START_TASK_NAME
    from datetime import timedelta

    mem = InMemoryMemoryProvider()
    parent_scope = _sc("p1", "ag1")
    child_scope = _sc("c1", "ag1")
    created = _BASE + timedelta(seconds=0)

    # NOTE: deliberately seed NO delegate frame in parent scope.
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="tcall_plan_c1", title="向 Lily 问好",
                 user_prompt="hi lily", created_at=created, settings=NormalTaskSettings())
    state = _state(child, child_scope, LoopConfig())

    await _close_one(mem, state, child, "out\n\nProcess Report: r", "success", _loop_ctx(mem),
                     short=True, act_recap="本段做了 X", task_summary="整段总结")

    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())

    # a minted start_task frame exists, carrying the child's origin_tool_call_id
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "tcall_plan_c1" and tc.get("name") == START_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    assert frame, "a start_task frame must be minted for the plan child"
    assert frame[0].metadata.get("origin_task_id") == "p1", "frame stays in delegating(p1) unit (留父)"
    assert frame[0].timestamp == created, "frame back-dated to child.created_at for adjacency"

    # the paired ACK result shares the frame's tool_call_id and timestamp
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "tcall_plan_c1"]
    assert ack, "paired result must carry the same tool_call_id"
    assert ack[0].timestamp == frame[0].timestamp, "result adjacent to its frame"


async def test_plan_child_frame_anchors_at_started_at() -> None:
    """时间戳锚 = task.started_at（task manager 真正启动 task 时），优先于 created_at。"""
    from ctx_weft.core.loop.steps.finalize import _close_one, START_TASK_NAME

    mem = InMemoryMemoryProvider()
    parent_scope = _sc("p1", "ag1")
    child_scope = _sc("c1", "ag1")
    created = _BASE + timedelta(seconds=0)
    started = _BASE + timedelta(seconds=5)   # 真正启动晚于创建

    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="tcall_plan_c1", title="向 Lily 问好",
                 user_prompt="hi lily", created_at=created, started_at=started,
                 settings=NormalTaskSettings())
    state = _state(child, child_scope, LoopConfig())

    await _close_one(mem, state, child, "out\n\nProcess Report: r", "success", _loop_ctx(mem),
                     short=True, act_recap="本段做了 X", task_summary="整段总结")

    turns = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "tcall_plan_c1" and tc.get("name") == START_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    assert frame, "a start_task frame must be minted for the plan child"
    assert frame[0].timestamp == started, "frame 须锚在 started_at（task manager 真正启动时），非 created_at"

    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "tcall_plan_c1"]
    assert ack and ack[0].timestamp == started, "paired ack 须与 frame 同锚 started_at"


async def test_cross_agent_child_no_nested_capsule_in_parent_scope() -> None:
    """跨 agent child: parent scope 无 origin_task_id=child.id 的 AGENT_CONVERSATION_TURN。"""
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c2", "ag2")
    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "hello", 1, role="user"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "reply", 2, role="assistant"), _ctx())

    child = Task(id="c2", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag2", creator_agent_id="ag1",
                 parent_task_id="p1", origin_tool_call_id="oc2",
                 title="Cross Agent Child", user_prompt="do cross",
                 settings=NormalTaskSettings())

    mem_content = "cross outputs\n\nProcess Report: cross summary"
    await finalize_task_memory(
        mem, _state(child, child_scope, LoopConfig()),
        child, mem_content, "success", _loop_ctx(mem),
        act_recap="cross summary", task_summary="",
    )

    parent_scope = _sc("p1", "ag1")
    caps = await mem.recall_recent(parent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    child_turns = [r for r in caps if r.metadata.get("origin_task_id") == "c2"]
    assert not child_turns, (
        f"cross-agent child should NOT write capsule into parent scope, found: {child_turns}"
    )
