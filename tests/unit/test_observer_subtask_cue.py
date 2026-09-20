"""Observer cue lists the agent's own sub-tasks from task_manager (spec Phase 3 2026-06-30),
independent of any blackboard subscription.

自 2026-09-19 起这份清单纯是**信息**：observer 据它在 next_step_hint 里指名哪个子任务的
产出不合格，由下一轮 actor 自己决定重派还是自己做（`task_reviews` 参数连同 reopen 一并删除）。
"""
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.assembler import ContextRequest, ContextBlock
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress


def _req(extra):
    task = Task(id="p1", session_id="s1", status="RUNNING", tenant_id="default",
                title="parent", settings=NormalTaskSettings())
    agent = type("A", (), {"id": "ag1"})()
    session = type("S", (), {"id": "s1"})()
    return ContextRequest(
        purpose="observe", scope=MemoryAddress(session_id="s1", task_id="p1", agent_id="ag1"),
        task=task, agent=agent, session=session, template=None, bound_capabilities=[],
        extra=extra,
    )


def test_observer_cue_lists_subtasks_from_extra():
    composer = DefaultComposer()
    # minimal history block so there is a user turn to anchor the trailing cue
    blocks = [ContextBlock(id="b1", source="t", kind="history", target="messages",
                           content="do the work", priority=3, token_estimate=3,
                           metadata={"role": "user", "type": "user_prompt", "timestamp": "2026-06-30T00:00:00+00:00"})]
    req = _req({"subtasks": [
        {"task_id": "tsk_amy", "title": "向 Amy 问好", "outcome": "finished"},
        {"task_id": "tsk_lily", "title": "向 Lily 问好", "outcome": "failed"},
    ]})
    msgs = composer._build_observer_messages(blocks, req)
    cue = msgs[-1].content
    assert "tsk_amy" in cue and "tsk_lily" in cue, f"cue must list child task_ids; got: {cue}"
    assert "向 Amy 问好" in cue and "failed" in cue
    assert "next_step_hint" in cue, (
        f"cue must tell the observer where to flag a bad result; got: {cue}")


def test_observer_cue_no_subtask_section_when_no_children():
    composer = DefaultComposer()
    blocks = [ContextBlock(id="b1", source="t", kind="history", target="messages",
                           content="do the work", priority=3, token_estimate=3,
                           metadata={"role": "user", "type": "user_prompt", "timestamp": "2026-06-30T00:00:00+00:00"})]
    req = _req({"subtasks": []})
    msgs = composer._build_observer_messages(blocks, req)
    assert "tsk_" not in msgs[-1].content


def test_observer_cue_renders_blocked_cancel_note():
    """spec: task-handoff——依赖阻塞取消的子任务在子任务清单里带解释性 note。"""
    composer = DefaultComposer()
    blocks = [ContextBlock(id="b1", source="t", kind="history", target="messages",
                           content="do the work", priority=3, token_estimate=3,
                           metadata={"role": "user", "type": "user_prompt", "timestamp": "2026-06-30T00:00:00+00:00"})]
    req = _req({"subtasks": [
        {"task_id": "tsk_a", "title": "produce", "outcome": "failed"},
        {"task_id": "tsk_b", "title": "final", "outcome": "canceled",
         "note": "blocked by failed/canceled predecessor tsk_a"},
    ]})
    cue = composer._build_observer_messages(blocks, req)[-1].content
    assert "tsk_b" in cue and "canceled" in cue
    assert "blocked by failed/canceled predecessor tsk_a" in cue
