"""带外收尾：park 之后 `FinalizeStep` 没跑过，那三件事得有人补上（2026-09-27）。

park 在 `ActStep` 里抛 `HitlPark`，driver 到不了 observe/finalize。于是 S5/S6 之后每个
「park → 后台判 success」的 task 都跳过了整个 `FinalizeStep`，而它是三件事的唯一发生地：

  finalize_task_memory → _close_one   bubble 给 parent / 派发 ack 终态化 / 写 finish 对
  BLACKBOARD_PUBLISHED                parent `recall_topic(task.id)` 的正路
  TASK_FINALIZED                      host 的 tasks.outputs_json / error 两列

**子任务漏得最重**：parent 醒来只看到「派发框 + 停在 running 的 ack」，拿不到任何产出。

单测那一层把 `apply_task_close` 桩掉了（那里测的是组装与调用顺序）。这份用例反过来——
用**真** memory / 真 event bus 跑一遍，证明这条路上那些写入真的落得下去。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.control_tools import ControlMetaKey as K
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.steps.observe import _out_of_band_finalize
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HitlAsk, UserTurnDelivery
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compiled_test_inputs
from tests.hitl_env import make_hitl


_SID = "s1"


class _Recorder:
    def __init__(self) -> None:
        self.events: list = []

    async def __call__(self, event) -> None:
        self.events.append(event)

    def types(self) -> list:
        return [e.type for e in self.events]

    def one(self, type_):
        return next(e for e in self.events if e.type == type_)


async def _fixture(*, parent: str | None = "t_parent"):
    """一个 park 过、刚被带外判 success 的子任务的现场。"""
    bus = InProcessEventBus()
    rec = _Recorder()
    bus.subscribe(None, rec)   # None = 全部类型
    mem = InMemoryMemoryProvider()
    hitl_service, hitl_registry = make_hitl(bus)

    session = Session(id=_SID, tenant_id="default", user_prompt="hi")
    task = Task(
        id="t_child", session_id=_SID, status="AWAITING_HUMAN",
        title="查资料", description="查一下甲的资料",
        parent_task_id=parent, creator_agent_id="ag_parent", assigned_agent_id="ag_child",
        outputs="甲的资料是这样的", settings=NormalTaskSettings(),
    )
    agent = Agent(id="ag_child", session_id=_SID, template_id="t")
    scope = MemoryAddress(session_id=_SID, task_id=task.id, agent_id="ag_child")
    state = LoopState(
        run_id="r_bg", session=session, task=task, agent=agent, scope=scope,
        prompt_inputs=None, resolved_model=model_snapshot(model="mock", account=""),
    )
    state.extra["final_body"] = "甲的资料是这样的"
    state.extra["final_summary"] = "查了三处来源"
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=MockLLMAdapter(responses=[]), memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(session_id=_SID, tenant_id="default",
                                     task_id=task.id, agent_id="ag_child"),
        hitl=hitl_service,
    )
    # 它 park 时开的那个「等你说话」气泡。
    bubble = await hitl_service.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=task.id)),
        session_id=_SID, task_id=task.id, agent_id="ag_child", tenant_id="default",
        stage="tool", unattended=False,
    )
    rec.events.clear()          # 只看收尾发的那些
    return state, ctx, mem, hitl_registry, rec, bubble.id


def _meta() -> dict:
    return {
        K.OBSERVER_OUTCOME: "success",
        K.OBSERVER_ACT_RECAP: "查了三处来源，交叉核对一致",
        K.OBSERVER_TASK_SUMMARY: "资料已核实",
        K.OBSERVER_NEXT_STEP_HINT: "",
        K.OBSERVER_FAILURE_REASON: "",
    }


@pytest.mark.asyncio
async def test_task_finalized_carries_the_deliverable() -> None:
    """host 的 `tasks.outputs_json` 就靠这一条。S5 起它一直是空的。"""
    state, ctx, _mem, _hitl, rec, _bid = await _fixture()
    await _out_of_band_finalize(state, ctx, _meta())

    assert EventType.TASK_FINALIZED in rec.types()
    payload = rec.one(EventType.TASK_FINALIZED).payload
    assert payload["task_id"] == "t_child"
    assert payload["outcome"] == "success"
    # 两段分开出核：output 是答复本身，summary 是给 reviewer 的自评清单。
    assert payload["outputs"]["output"] == "甲的资料是这样的"
    assert payload["outputs"]["summary"] == "查了三处来源"


@pytest.mark.asyncio
async def test_the_result_is_published_for_the_parent_to_recall() -> None:
    """`recall_topic(task_id)` 是 parent 读子任务结果的正路——没有这一条它读到空。"""
    state, ctx, mem, _hitl, rec, _bid = await _fixture()
    await _out_of_band_finalize(state, ctx, _meta())

    assert EventType.BLACKBOARD_PUBLISHED in rec.types()
    # SESSION 视图只按 session 取，禁带 task_id/agent_id。
    sess_scope = MemoryAddress(session_id=_SID, task_id=None, agent_id=None)
    pubs = await load_records(mem, sess_scope, MemoryScope.SESSION, ctx.provider_ctx,
                               kinds=[MemoryKind.PUBLICATION])
    mine = [r for r in pubs if (r.metadata or {}).get("task_id") == "t_child"]
    assert mine, "没有发布 = parent 按 topic 召回时读到空"
    assert "甲的资料是这样的" in (mine[0].content or "")


@pytest.mark.asyncio
async def test_the_parent_scope_gets_the_dispatch_result() -> None:
    """bubble 给 parent：不写这一份，parent 醒来只看到「派发框 + 停在 running 的 ack」。

    本用例是跨 agent 子任务（creator=ag_parent ≠ assigned=ag_child），走
    `_close_one` 的 cross_agent 分支——产出写进 parent scope 的 tool 回合。
    """
    state, ctx, mem, _hitl, _rec, _bid = await _fixture()
    await _out_of_band_finalize(state, ctx, _meta())

    # AGENT 视图按 agent 取、禁带 task_id（那是 TASK 视图的维度）。
    parent_scope = MemoryAddress(session_id=_SID, task_id=None, agent_id="ag_parent")
    recs = await load_records(mem, parent_scope, MemoryScope.AGENT, ctx.provider_ctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    blob = " ".join(str(r.content or "") for r in recs)
    assert "甲的资料是这样的" in blob, "parent 那边看不到子任务的产出"


@pytest.mark.asyncio
async def test_the_park_bubble_is_closed() -> None:
    """让位入口用掉了 → 收。留着会把会话钉死在 PAUSED（气泡在折叠里优先于 task 终态）。"""
    state, ctx, _mem, hitl, _rec, bid = await _fixture()
    await _out_of_band_finalize(state, ctx, _meta())
    assert [r.id for r in hitl.list_pending(_SID) if not r.resolved] == []
    assert bid is not None


@pytest.mark.asyncio
async def test_a_root_task_is_closed_and_finalized_the_same_way() -> None:
    """root（`parent_task_id is None`）走的是同一条路：收尾做完，气泡也收。

    人后面还想说话就再发一条消息——`send_message` 看到 `current_task_id` 已终态自己会开新
    的一轮，不需要留着这个气泡当入口。
    """
    state, ctx, _mem, hitl, rec, _bid = await _fixture(parent=None)
    await _out_of_band_finalize(state, ctx, _meta())

    assert EventType.TASK_FINALIZED in rec.types()
    assert [r.id for r in hitl.list_pending(_SID) if not r.resolved] == []
