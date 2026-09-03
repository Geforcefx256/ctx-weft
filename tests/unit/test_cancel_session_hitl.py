"""用户取消会话时，未决的 ask_user 必须一并终局（总账 A10）。

不终局的代价不在当下——当下那个 task 已经被 cancel_all 清掉了——
而在**重启之后**：`rebuild_hitl` 按「有 HitlOpened 无终局事件」折 pending，
于是一个已取消会话的提问会被当成未决恢复出来。
"""

from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HITL_OUTCOME_CANCELLED, HitlAsk, UserTurnDelivery
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit.test_runtime_agent_api import _plant

pytestmark = pytest.mark.asyncio


class _RecordingBus:
    """挂在 runtime 真实事件总线上的记录桩——不替换总线，只旁听。"""

    def __init__(self) -> None:
        self.events: list = []

    async def _record(self, ev) -> None:
        self.events.append(ev)


def _runtime():
    return make_runtime(agent_provider=InlineAgentTemplateProvider())


async def _wire_pending_session(
    rt, session_id: str, task_id: str, *, agent_id: str = "ag1",
    tenant_id: str = "default",
) -> None:
    """手搭一个「有 task 正等 ask_user」的会话：不真跑 TaskManager.drain，只登记
    runtime 侧的两处状态（`_task_managers` 映射 + SessionRegistry 的会话状态），
    使 `cancel_session` 的守卫（`per` 或 `task_manager` 非空）与 `cancel_all`
    内部对 `_session_registry` 的透传都落在真实组件上。

    额外在 `AgentLifecycleManager` 里种一个该 session 下的 agent（R23 需要它：
    `cancel_session` 现在要把这个 session 下每个 agent 显式转 `terminated`，
    没有 agent 记录就没有可观测的 `AgentTerminated`）。`_plant` 是纯 dict 注入，
    不发事件，不影响其余只盯 `HitlResolved` 的既有测试。

    `tenant_id` 可覆盖（F7）：默认场景全程 "default" 掩盖了「非 default 租户是否
    也被正确透传」这条——机制上 `HitlService._emit` 的 `tenant_id=req.tenant_id`
    已经是对的，只是此前没有测试固定过非 default 的取值。
    """
    session = Session(
        id=session_id, user_prompt="hi", status="RUNNING",
        tenant_id=tenant_id, created_at=now_utc(),
    )
    tm = TaskManager(session_id=session_id, event_bus=rt._event_bus, max_concurrent=0)
    tm.set_session(session)
    task = Task(id=task_id, session_id=session_id, status="SUSPENDED", tenant_id=tenant_id)
    tm.register_task(task)
    rt._task_managers[session_id] = tm
    rt._session_registry.register_session(session_id, tenant_id=tenant_id)
    _plant(rt, agent_id, None, session_id=session_id, status="waiting_human")


@pytest.fixture
async def runtime_with_pending_hitl():
    rt = _runtime()
    bus = _RecordingBus()
    rt._event_bus.subscribe(None, bus._record)  # type: ignore[attr-defined]

    session_id, task_id = "s1", "t1"
    await _wire_pending_session(rt, session_id, task_id)

    req = await rt.hitl.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=task_id)),
        session_id=session_id, task_id=task_id, stage="tool", unattended=False,
    )
    return rt, bus, session_id, req.id


@pytest.fixture
async def runtime_with_pending_hitl_non_default_tenant():
    """同 `runtime_with_pending_hitl`，但 tenant 不是 "default"（F7）。"""
    rt = _runtime()
    bus = _RecordingBus()
    rt._event_bus.subscribe(None, bus._record)  # type: ignore[attr-defined]

    session_id, task_id, tenant_id = "s1", "t1", "acme-corp"
    await _wire_pending_session(rt, session_id, task_id, tenant_id=tenant_id)

    req = await rt.hitl.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=task_id)),
        session_id=session_id, task_id=task_id, stage="tool", tenant_id=tenant_id,
    )
    return rt, bus, session_id, req.id, tenant_id


@pytest.fixture
async def runtime_without_hitl():
    rt = _runtime()
    bus = _RecordingBus()
    rt._event_bus.subscribe(None, bus._record)  # type: ignore[attr-defined]

    session_id, task_id = "s1", "t1"
    await _wire_pending_session(rt, session_id, task_id)
    return rt, bus, session_id


async def test_cancel_session_resolves_pending_hitl(runtime_with_pending_hitl):
    rt, bus, session_id, hitl_id = runtime_with_pending_hitl

    await rt.cancel_session(session_id)

    resolved = [e for e in bus.events if e.type == EventType.HITL_RESOLVED]
    assert len(resolved) == 1, "未决的 ask_user 没有被终局"
    assert resolved[0].payload["hitl_id"] == hitl_id
    assert resolved[0].payload["outcome"] == HITL_OUTCOME_CANCELLED
    assert rt.hitl_registry.list_pending(session_id=session_id) == []


async def test_cancel_session_resolves_pending_hitl_with_non_default_tenant(
        runtime_with_pending_hitl_non_default_tenant):
    """F7：取消路径把 `PendingHitl.tenant_id` 正确透传到 `HitlResolved`——全程非
    default tenant，钉住 `service.py:200` `tenant_id=req.tenant_id` 这条透传。"""
    rt, bus, session_id, hitl_id, tenant_id = runtime_with_pending_hitl_non_default_tenant

    await rt.cancel_session(session_id)

    resolved = [e for e in bus.events if e.type == EventType.HITL_RESOLVED]
    assert len(resolved) == 1, "未决的 ask_user 没有被终局"
    assert resolved[0].payload["hitl_id"] == hitl_id
    assert resolved[0].payload["outcome"] == HITL_OUTCOME_CANCELLED
    assert resolved[0].tenant_id == tenant_id, "取消发出的 HitlResolved 没带对 tenant"
    assert resolved[0].tenant_id != "default"


async def test_hitl_cancelled_before_session_terminal(runtime_with_pending_hitl):
    """HitlResolved 必须先于该 session 下 agent 的终态事件——与熔断 trip 序列同一条纪律。

    观测点由 `SessionFinished` 换成 `AgentTerminated`（Task 20 R23）：
    `SessionFinished` 事件本身已经停发（`TaskManager.cancel_all()` 只发
    `TASK_CANCELED` 并直接改写内存态 `self._session.status`，不再有任何『会话终态』
    事件可供排序，见 `task_manager.py:cancel_all` 与本任务报告 §6）；`cancel_session`
    现在改为额外广播 `cancel_agent`，把该 session 下每个 agent 显式转
    `terminated` 并发 `AgentTerminated`——这条纪律因此有了新的、真实存在的下游锚点。
    """
    rt, bus, session_id, _ = runtime_with_pending_hitl

    await rt.cancel_session(session_id)

    types = [e.type for e in bus.events]
    assert EventType.HITL_RESOLVED in types
    assert EventType.AGENT_TERMINATED in types
    assert types.index(EventType.HITL_RESOLVED) < types.index(EventType.AGENT_TERMINATED), (
        "已取消会话的 HITL 终局事件晚于该 session 下 agent 的终态事件"
    )
    # 事件本身即是终态化的证据——`_release_session`（会话此刻已空闲挂起，随
    # `cancel_session` 一并触发）随后会把 registry 里的 agent record 摘掉，
    # `status_of` 事后查不再可靠，不作为观测点。
    terminated = [e for e in bus.events if e.type == EventType.AGENT_TERMINATED]
    assert any(e.agent_id == "ag1" for e in terminated)


async def test_cancel_session_without_pending_hitl_is_noop(runtime_without_hitl):
    """没有未决 HITL 时不该多发任何 HITL 事件。"""
    rt, bus, session_id = runtime_without_hitl
    await rt.cancel_session(session_id)
    assert [e for e in bus.events if e.type == EventType.HITL_RESOLVED] == []
