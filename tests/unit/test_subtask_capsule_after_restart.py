"""同 agent 子任务 close 后，父的上下文里必须留下子任务胶囊——重启/恢复后不成立（bug）。

现象（用户报告）：root task A 派发同 agent 子任务 B；B 正常调 finish_task、有 outputs、
自己没再派发子任务。按设计 B 结束时应折成胶囊内联进 A 的对话
（`[assistant delegate_task][tool 终态 ack][B 的 body][B 的 finish 对]`），实际只剩
`[assistant delegate_task]` + 一条「子任务执行如下」的 tool 文本，胶囊不见了。

根因链（本文件两条断言分别钉在链的两端）：
  `Task.origin_tool_call_id` 是纯瞬态字段——`task_payload`(TASK_CREATED) 不带它、
  `TaskView` 不存它、`task_from_projection` 不还原它。于是任何跨进程重启 / 崩溃恢复 /
  冷 resume 之后重建出来的子任务，`origin_tool_call_id is None`：
    - `finalize._close_one` 的 bubble 分支 `if task.parent_task_id and task.origin_tool_call_id
      and mem_content` 整段跳过 → 既不把父的 running ack 换成终态，也不合成 B 的 finish 对；
    - `is_own_root`（parent 非空且同 agent）为 False → 自身 scope 也不合成 finish 对；
    - 而 `_supersede_final_raw_segment` 照常执行 → B 的末段 raw 被删。
  结果：父只剩「派发框 + 停在 running 的 ack」，子任务的执行与胶囊双双消失。

对照组 test_capsule_present_without_restart 走同一条生产路径但不重启，胶囊正常 → 差异变量
只有 origin_tool_call_id。
"""

from __future__ import annotations

from ctx_weft.core.loop.task_memory_fold import task_record_kind, task_record_group

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME
from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.loop.steps.finalize import (
    ensure_dispatch_frame_at_start,
    finalize_task_memory,
)
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.orchestrator.task.manager import task_payload
from ctx_weft.core.utils.estimate import estimate_tokens
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
    load_records,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import history_messages, recalled_blocks


T = MemoryEventType
SESSION = "s1"
AGENT = "ag1"
PARENT = "tsk_parent"
CHILD = "tsk_child"
DISPATCH_TCID = "tcall_dispatch"
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


# ─── 辅助 ────────────────────────────────────────────────────────────────────

def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _task_scope(task_id: str) -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=task_id, agent_id=AGENT)


def _ts(t: float) -> datetime:
    return _BASE + timedelta(seconds=t)


async def _ingest(mem, type_, scope, content, t, role=None, **meta):
    await mem.ingest(
        MemoryEvent(type=type_, address=scope, content=content,
                    timestamp=_ts(t), role=role, metadata=meta),
        _pctx(),
    )


class _LeafTM:
    """子任务是叶子：没有后代。"""

    def children_of(self, task_id: str) -> set[str]:
        return set()


def _state(task: Task, scope: MemoryAddress):
    return SimpleNamespace(
        run_id="run1", sequence_counter=0,
        session=SimpleNamespace(id=SESSION, tenant_id="default"),
        scope=scope, task=task,
        agent=SimpleNamespace(id=AGENT, loop_config=LoopConfig()),
    )


def _loop_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx(), task_manager=_LeafTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _child_task() -> Task:
    """delegate_task 造出来的同 agent 子任务（见 control_capability.delegate_task）。"""
    return Task(
        id=CHILD, session_id=SESSION, status="PENDING", tenant_id="default",
        assigned_agent_id=AGENT, creator_agent_id=AGENT, parent_task_id=PARENT,
        title="子任务", description="去做这件小事",
        user_prompt="子任务：去做这件小事",
        settings=NormalTaskSettings(),
        origin_tool_call_id=DISPATCH_TCID,
        origin_tool_name=DELEGATE_TASK_NAME,
        created_at=_ts(3),
    )


def _restart_roundtrip(task: Task) -> Task:
    """生产重启链：Task → task_payload(TASK_CREATED) → reducer(TaskView) → task_from_projection。"""
    ev = Event(
        id=generate_id("evt"), run_id=None, sequence=1, session_id=SESSION,
        type=EventType.TASK_CREATED, timestamp=_ts(3), tenant_id="default",
        task_id=task.id, payload=task_payload(task, user_prompt_jsonable=task.user_prompt),
    )
    view = reduce_events([ev], run_id="run1")
    return task_from_projection(view.tasks[task.id])


async def _compose_parent_messages(mem):
    """REAL AgentRecallSource → REAL PromptBuilder：父续跑时装配出的 message 序列。"""
    deps = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    req = SimpleNamespace(scope=_task_scope(PARENT), token_counter=estimate_tokens)
    blocks = await recalled_blocks(req, deps)
    return history_messages(blocks, task_id=PARENT)


def _labels(messages) -> list[str]:
    out = []
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        name = tcs[0].get("name", "") if tcs else ""
        out.append(f"{m.role}:{name or (m.content or '')[:44]}")
    return out


async def _run_dispatch_and_close(mem, *, restart: str | None):
    """A 派发同 agent 子任务 B → B 跑几轮 → B 正常 finish（有 outputs）。

    restart="before_close"：在 B 已 start（派发框 + running ack 已落库）之后、close 之前，
      按生产重启链重建 B 的 Task 对象——进程重启 / 崩溃恢复 / 冷 resume 后继续跑完子任务。
    restart="before_start"：连框都还没铸就先重启（派发与 start 之间崩），重建后才 start。
    """
    ctx = _loop_ctx(mem)
    child = _child_task()

    # ① 父 body
    await _ingest(mem, T.USER_PROMPT, _task_scope(PARENT), "帮我做一件事，必要时派子任务", 1,
                  role="user", task_id=PARENT)
    await _ingest(mem, T.LLM_RESPONSE, _task_scope(PARENT), "我来派个子任务", 2, role="assistant")

    if restart == "before_start":
        child = _restart_roundtrip(child)
        child.assigned_agent_id = AGENT

    # ② 子任务真正 start（driver.run）：铸派发框 + running ack
    child.started_at = _ts(3)
    await ensure_dispatch_frame_at_start(_state(child, _task_scope(CHILD)), ctx)

    # ③ 子 body：task_prompt + 若干轮执行
    await _ingest(mem, T.USER_PROMPT, _task_scope(CHILD), child.user_prompt, 4,
                  role="user", task_id=CHILD)
    for i in range(5):
        filler = "x " * 300
        await _ingest(mem, T.LLM_RESPONSE, _task_scope(CHILD), f"子任务干活 {filler}#{i}",
                      5 + i, role="assistant")

    if restart == "before_close":
        child = _restart_roundtrip(child)
        child.started_at = _ts(3)          # TaskManager 每次 run 重置
        child.assigned_agent_id = AGENT

    # ④ 子任务正常收尾：finish_task 有产出、叶子无后代
    child.status = "FINISHED"
    child.outputs = "子任务最终产出：做完了"
    mem_content = f"{child.outputs}\n\nProcess Report: 子任务过程报告"
    await finalize_task_memory(
        mem, _state(child, _task_scope(CHILD)), child, mem_content, "success", ctx,
        act_recap="子任务 act 复述", task_summary="子任务过程报告",
    )


def _assert_capsule_present(messages) -> None:
    labels = _labels(messages)
    assert all("<task_final_reply" not in (m.content or "") for m in messages), labels
    assert any(m.role == "user" and "<task_finish" in (m.content or "") for m in messages), labels
    assert any(m.role == "assistant" and m.content == "子任务最终产出：做完了" for m in messages), labels
    assert any("子任务 act 复述" in (m.content or "") for m in messages), labels
    assert not any("is running now" in (m.content or "") for m in messages), labels
    assert all(not m.tool_calls and m.role != "tool" for m in messages)



async def _parent_agent_turns(mem):
    """父分区（agent 层）的全部幸存对话回合。"""
    from ctx_weft.protocols import MemoryKind, MemoryScope
    return await load_records(mem,
        MemoryAddress(session_id=SESSION, agent_id=AGENT), MemoryScope.AGENT, _pctx(),
        kinds=[MemoryKind.CONVERSATION_TURN])


# ─── 对照组：不重启 → 胶囊正常 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_capsule_present_without_restart() -> None:
    mem = InMemoryMemoryProvider()
    await _run_dispatch_and_close(mem, restart=None)
    _assert_capsule_present(await _compose_parent_messages(mem))


# ─── 重启后子任务 close，胶囊仍在 ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_capsule_survives_restart_before_close() -> None:
    """已 start 的子任务经重启重建后 close：父的上下文里仍须有子任务胶囊。"""
    mem = InMemoryMemoryProvider()
    await _run_dispatch_and_close(mem, restart="before_close")
    _assert_capsule_present(await _compose_parent_messages(mem))


@pytest.mark.asyncio
async def test_capsule_survives_restart_before_start() -> None:
    """派发与 start 之间崩溃：重建后才铸框（配对 id 现生成），整对照样闭合。"""
    mem = InMemoryMemoryProvider()
    await _run_dispatch_and_close(mem, restart="before_start")
    _assert_capsule_present(await _compose_parent_messages(mem))


@pytest.mark.asyncio
async def test_single_active_ack_after_restart_close() -> None:
    """重启后 close 必须**替换**那条 running ack，而不是并列新增一条终态。"""
    mem = InMemoryMemoryProvider()
    await _run_dispatch_and_close(mem, restart="before_close")

    acks = [r for r in await _parent_agent_turns(mem)
            if task_record_kind(r) == "result" and r.metadata.get("child_task_id") == CHILD]
    assert len(acks) == 1, (
        f"本子任务的派发 ack 应恰有一条 active；got {[r.content for r in acks]}"
    )
    assert "ran here" in acks[0].content, f"ack 应是终态文案；got {acks[0].content!r}"


# ─── 新不变量：派发框/ack 自带 child_task_id ──────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_pair_records_child_task_id() -> None:
    """框与 ack 都必须带 child_task_id——「这次派发对应哪个子任务」是 memory 里的一等事实，
    close 认框不再依赖任何进程内状态。"""
    mem = InMemoryMemoryProvider()
    ctx = _loop_ctx(mem)
    child = _child_task()
    child.started_at = _ts(3)
    await ensure_dispatch_frame_at_start(_state(child, _task_scope(CHILD)), ctx)

    turns = await _parent_agent_turns(mem)
    frames = [r for r in turns if task_record_kind(r) == "start"]
    acks = [r for r in turns if task_record_kind(r) == "result"]
    assert len(frames) == 1 and len(acks) == 1, f"start 应恰好铸一对；got {turns!r}"
    assert frames[0].metadata.get("child_task_id") == CHILD
    assert acks[0].metadata.get("child_task_id") == CHILD
    # 配对仍然成立：框的 tool_calls[].id == ack 的 tool_call_id
    assert task_record_group(frames[0]) == task_record_group(acks[0])


# ─── 根因：派发来源在投影/恢复链上丢失 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_dispatch_origin_survives_projection() -> None:
    """origin_tool_call_id/origin_tool_name 须跨 TASK_CREATED → TaskView → 恢复保留。

    它们是 finalize 认出「我是被谁派发的」的唯一凭据：丢了 → 子任务 close 时既不闭合父的
    派发 ack、也不合成 finish 对（见 finalize._close_one）。
    """
    restored = _restart_roundtrip(_child_task())
    assert restored.parent_task_id == PARENT
    assert restored.origin_tool_call_id == DISPATCH_TCID, (
        "TASK_CREATED payload / TaskView / task_from_projection 丢了 origin_tool_call_id"
    )
    assert restored.origin_tool_name == DELEGATE_TASK_NAME, (
        "TASK_CREATED payload / TaskView / task_from_projection 丢了 origin_tool_name"
    )
