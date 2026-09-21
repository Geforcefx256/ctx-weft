from __future__ import annotations

import pathlib
from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.control.types import AgentView
from ctx_weft.core.models.errors import AgentBusyError, AgentNotFound, AgentTerminatedError
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager, _AgentRecord
from ctx_weft.core.orchestrator.lifecycle.agent_state import AgentInput
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.task import Task
from ctx_weft.protocols.events import (
    EVENT_TYPES,
    L_TIER_EVENT_TYPES,
    Event,
    EventStore,
    EventType,
)

pytestmark = pytest.mark.asyncio

_NEW_AGENT_TYPES = {
    EventType.AGENT_RUNNING,
    EventType.AGENT_IDLE,
    EventType.AGENT_WAITING_HUMAN,
    EventType.AGENT_INTERRUPTED,
    EventType.AGENT_TERMINATED,
}


class _SpyBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)

    def subscribe(self, _flt, _handler, *, provisional: bool = False,
                  required: bool = False) -> None:
        pass


async def test_task_events_carry_agent_id_on_envelope():
    """V2 §0：envelope 管身份。ALM 靠 ev.agent_id 定位 agent。"""
    bus = _SpyBus()
    tm = TaskManager("s1", event_bus=bus)
    task = Task(id="t1", session_id="s1", status="PENDING", assigned_agent_id="a1")
    await tm.push_task(task)

    created = [e for e in bus.events if e.type == EventType.TASK_CREATED]
    assert created, "没有 TaskCreated"
    assert created[0].agent_id == "a1"


async def test_task_event_agent_id_prefers_running_agent_over_assigned():
    """在跑登记（真实执行者）优先于 task 自己的 assigned_agent_id。"""
    bus = _SpyBus()
    tm = TaskManager("s1", event_bus=bus)
    task = Task(id="t1", session_id="s1", status="PENDING", assigned_agent_id="a1")
    await tm.push_task(task)
    # 模拟：另一个 agent 实际在跑这个 task（登记与 assigned 不一致）
    tm._running_agents["t1"] = "a2"

    await tm._emit(EventType.TASK_STARTED, task_id="t1", payload={})

    started = [e for e in bus.events if e.type == EventType.TASK_STARTED]
    assert started
    assert started[-1].agent_id == "a2"


async def test_queue_level_event_has_no_agent_id():
    """无 task_id 的事件不应乱填 agent_id——`_emit` 的通用规则，与事件类型本身无关。

    这里借用 SESSION_FINISHED 只是因为它是会话级、天然「无 task_id」的一个例子
    （此前借的 TASK_QUEUE_DRAINED 已于 2026-09-05 连枚举一并删除）：测的不是这个
    类型本身，而是直接调 `tm._emit(...)` 时的通用填充逻辑。
    """
    bus = _SpyBus()
    tm = TaskManager("s1", event_bus=bus)
    await tm._emit(EventType.SESSION_FINISHED, payload={})

    emitted = [e for e in bus.events if e.type == EventType.SESSION_FINISHED]
    assert emitted
    assert emitted[0].agent_id is None
    assert emitted[0].task_id is None


def test_new_agent_types_registered():
    """新增的 5 个 AGENT_* 状态事件应在 EVENT_TYPES 中注册。"""
    assert _NEW_AGENT_TYPES <= set(EVENT_TYPES)


def test_new_agent_types_are_s_tier():
    """不变式 2：加一个枚举值就得显式选边。ALM 状态被 reducer 折叠 -> S 档。"""
    assert not (_NEW_AGENT_TYPES & set(L_TIER_EVENT_TYPES))


def test_agent_event_wire_values_are_pascal_case():
    """验证 agent 状态事件的字符串值为 PascalCase。"""
    assert EventType.AGENT_RUNNING == "AgentRunning"
    assert EventType.AGENT_IDLE == "AgentIdle"
    assert EventType.AGENT_WAITING_HUMAN == "AgentWaitingHuman"
    assert EventType.AGENT_INTERRUPTED == "AgentInterrupted"
    assert EventType.AGENT_TERMINATED == "AgentTerminated"


def _reg() -> AgentLifecycleManager:
    return AgentLifecycleManager(
        template_lookup=None, event_bus=_SpyBus(), model_resolver=lambda a, m: None
    )


def _plant(reg: AgentLifecycleManager, agent_id: str, parent: str | None, session_id: str = "s1") -> None:
    """直接种记录，绕开 instantiate 的模板依赖。"""
    reg._agents[agent_id] = _AgentRecord(
        session_id=session_id, tenant_id="default", template_id="tpl",
        parent_agent_id=parent, spawn_depth=0 if parent is None else 1,
        memory_config=None, loop_config=None,
    )
    if parent is not None:
        reg._children.setdefault(parent, set()).add(agent_id)


def test_record_defaults_to_idle_with_no_task():
    reg = _reg()
    _plant(reg, "root", None)
    assert reg.status_of("root") == "idle"
    assert reg._agents["root"].current_task_id is None


def test_children_and_descendants():
    reg = _reg()
    _plant(reg, "root", None)
    _plant(reg, "kid1", "root")
    _plant(reg, "kid2", "root")
    _plant(reg, "grandkid", "kid1")

    assert reg.children_of("root") == {"kid1", "kid2"}
    assert set(reg.descendants_of("root")) == {"kid1", "kid2", "grandkid"}
    assert reg.descendants_of("grandkid") == []


def test_descendants_tolerates_cycle():
    """防御性：父子关系理论上无环，索引损坏时也不能死循环。"""
    reg = _reg()
    _plant(reg, "a", None)
    _plant(reg, "b", "a")
    reg._children.setdefault("b", set()).add("a")
    assert set(reg.descendants_of("a")) == {"b"}


def test_agent_ids_of_session():
    reg = _reg()
    _plant(reg, "a", None, session_id="s1")
    _plant(reg, "b", None, session_id="s2")
    assert reg.agent_ids_of_session("s1") == ["a"]


def test_forget_session_cleans_children_index():
    """forget_session 之后 _children 不能留悬垂键，也不能留悬垂值。

    root/kid1/kid2/grandkid 都在 s1；额外种一个 s2 的 other，其 _children
    指向 s1 的 kid1（模拟索引本不该出现、但要能被安全清理的悬垂引用来源）。
    释放 s1 后：s1 的 agent 不能再作为键出现在 _children 里；也不能作为值
    残留在任何还活着的 agent（other）的 children 集合里。
    """
    reg = _reg()
    _plant(reg, "root", None, session_id="s1")
    _plant(reg, "kid1", "root", session_id="s1")
    _plant(reg, "kid2", "root", session_id="s1")
    _plant(reg, "grandkid", "kid1", session_id="s1")
    _plant(reg, "other", None, session_id="s2")
    reg._children.setdefault("other", set()).add("kid1")

    reg.forget_session("s1")

    removed = {"root", "kid1", "kid2", "grandkid"}
    assert not (removed & set(reg._children.keys()))
    for children in reg._children.values():
        assert not (removed & children)
    assert reg.children_of("other") == set()
    assert reg.agent_ids_of_session("s1") == []
    assert reg.agent_ids_of_session("s2") == ["other"]


def test_forget_agent_cleans_children_index_on_both_sides():
    """单点逐出与 forget_session 同一纪律：既摘键，也从每个父的值集合里摘掉。

    只 pop 键不摘值的话，父的 children_of 会指向一个 _agents 里已经不存在的 id，
    descendants_of 遍历到它时仍把它当活的吐出来（级联 cancel/pause 会去操作一个
    幽灵）。
    """
    reg = _reg()
    _plant(reg, "root", None, session_id="s1")
    _plant(reg, "kid", "root", session_id="s1")
    _plant(reg, "grandkid", "kid", session_id="s1")

    assert reg.forget_agent("kid") is True

    assert not reg.has("kid")
    assert reg.has("root") and reg.has("grandkid"), "只逐出这一个，不牵连别人"
    assert "kid" not in reg._children            # 键
    assert "kid" not in reg.children_of("root")  # 值
    assert reg.forget_agent("kid") is False, "已经不在了 → False，幂等"


def test_forget_agent_is_pure_mechanism_and_does_not_judge_status():
    """ALM 侧的 forget_* 是**纯机制**：只管摘干净，不判断该不该摘。

    「这个 agent 现在能不能被忘掉」要同时看 task 队列、未决 HITL、以及同 session 其他
    agent 的状态——ALM 一样都不认识。判据住在 `CtxWeftRuntime.forget_agent` /
    `forget_session`（组合根，那里才看得全），行为由
    `tests/integration/test_session_lifecycle_forget_rebuild.py` 覆盖。

    这条用例钉的是"策略别再爬回 ALM 里"：一旦有人在这里加状态检查，两处判据就会各说
    各话，而 runtime 那份才是被调用方依赖的那份。
    """
    reg = _reg()
    _plant(reg, "busy", None, session_id="s1")
    reg._agents["busy"].status = "running"

    assert reg.forget_agent("busy") is True, "ALM 不看状态——把关是 runtime 的事"
    assert not reg.has("busy")


async def test_load_rebuilds_children_index_for_cold_recovery():
    """冷恢复（load）必须在没有任何 instantiate() 调用的情况下重建 _children。

    这钉住的性质：进程重启后，级联 cancel/pause（Task 19/20）依赖 _children
    对父子树做遍历——如果恢复期不重建索引，父被 load() 灌回来之后 cascade
    就是静默失效的（父存在、子存在，但父找不到子）。
    """
    reg = _reg()
    views = {
        "root": AgentView(id="root"),
        "kid1": AgentView(id="kid1", parent_agent_id="root", spawn_depth=1),
        "kid2": AgentView(id="kid2", parent_agent_id="root", spawn_depth=1),
        "grandkid": AgentView(id="grandkid", parent_agent_id="kid1", spawn_depth=2),
    }

    n = await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")

    assert n == 4
    assert reg.children_of("root") == {"kid1", "kid2"}
    assert set(reg.descendants_of("root")) == {"kid1", "kid2", "grandkid"}
    assert reg.descendants_of("grandkid") == []


async def test_load_tolerates_parent_outside_batch():
    """view.parent_agent_id 指向本次 load() 批次之外的 id（幽灵父/另一 session
    尚未加载）——不该崩，也不该让 descendants_of 在真实 agent 上查出脏结果。
    """
    reg = _reg()
    views = {"orphan": AgentView(id="orphan", parent_agent_id="not-in-this-batch", spawn_depth=1)}

    n = await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")

    assert n == 1
    assert reg.has("orphan")
    assert reg.children_of("not-in-this-batch") == {"orphan"}
    assert reg.descendants_of("orphan") == []


async def test_load_is_idempotent_for_children_index():
    """load() 对同一批 views 重复调用（比如 recover() 内部的重试路径），
    _children 不应重复计数或产生副本——set 语义天然幂等。
    """
    reg = _reg()
    views = {
        "root": AgentView(id="root"),
        "kid1": AgentView(id="kid1", parent_agent_id="root", spawn_depth=1),
    }
    await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")
    await reg.load(views, session_id="s1", tenant_id="default", fallback_template_id="tpl")

    assert reg.children_of("root") == {"kid1"}


# ── Task 12: ALM 订阅 TASK_* 驱动转移并发 AGENT_* ──────────────────────────


def _task_ev(t: str, agent_id: str, payload: dict | None = None) -> Event:
    return Event(
        id="evt_x", run_id=None, sequence=0, session_id="s1", type=t,
        timestamp=datetime.now(UTC), task_id="t1", agent_id=agent_id,
        payload=payload or {},
    )


async def test_task_started_drives_agent_to_running_and_emits():
    reg = _reg()
    _plant(reg, "a1", None)
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1", {"assigned_agent_id": "a1"}))

    assert reg.status_of("a1") == "running"
    emitted = [e for e in reg.event_bus.events if e.type == EventType.AGENT_RUNNING]
    assert len(emitted) == 1
    assert emitted[0].agent_id == "a1"
    assert emitted[0].session_id == "s1"
    assert emitted[0].payload["trigger"] == "task_started"


async def test_awaiting_human_then_resolved():
    reg = _reg()
    _plant(reg, "a1", None)
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))
    await reg.handle_event(_task_ev(EventType.TASK_AWAITING_HUMAN, "a1", {"hitl_id": "h1"}))
    assert reg.status_of("a1") == "waiting_human"

    await reg.handle_event(_task_ev(EventType.TASK_HUMAN_RESOLVED, "a1", {"hitl_id": "h1"}))
    assert reg.status_of("a1") == "running"


async def test_task_terminal_returns_to_idle_not_terminated():
    """spec 3.1：task 终态不是 agent 终态。"""
    for t in (
        EventType.TASK_FINISHED,
        EventType.TASK_FAILED,
        EventType.TASK_CANCELED,
        EventType.TASK_FINALIZED,
        EventType.TASK_REQUEUED,
        EventType.TASK_SUSPENDED,
    ):
        reg = _reg()
        _plant(reg, "a1", None)
        await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))
        await reg.handle_event(_task_ev(t, "a1"))
        assert reg.status_of("a1") == "idle", f"{t} 应回 idle"


async def test_unknown_agent_id_is_ignored():
    reg = _reg()
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "ghost"))
    assert reg.event_bus.events == []


async def test_current_task_id_tracked():
    reg = _reg()
    _plant(reg, "a1", None)
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))
    assert reg._agents["a1"].current_task_id == "t1"


async def test_no_duplicate_event_on_same_state_input():
    reg = _reg()
    _plant(reg, "a1", None)
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))
    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))
    running = [e for e in reg.event_bus.events if e.type == EventType.AGENT_RUNNING]
    assert len(running) == 1


async def test_apply_input_is_direct_entry_point_not_only_via_events():
    """apply_input 是独立可调用的转移入口——Task 19/20 的 cancel/pause 要走它，
    不经由 handle_event/事件总线也必须能驱动转移并发事件。"""
    reg = _reg()
    _plant(reg, "a1", None)
    changed = await reg.apply_input("a1", AgentInput.TASK_STARTED, task_id="t1")
    assert changed is True
    assert reg.status_of("a1") == "running"
    changed_again = await reg.apply_input("a1", AgentInput.TASK_STARTED, task_id="t1")
    assert changed_again is False  # 同态输入，不转移


async def test_handle_event_does_not_recurse_on_its_own_agent_events():
    """同步 drain 下，ALM 在 handle_event 里发出的 AGENT_* 事件会在 emit() 返回前
    回流给全部订阅者（包括 ALM 自己，因为 attach_to_bus 用 subscribe(None, ...)
    订阅了全部类型）。_INPUT_BY_EVENT 只含 TASK_*，AGENT_* 类型查表落空直接
    return，不会对自己发的事件再喂一次状态机——这里用会把自己也接进总线的
    真实 handler 验证「不会递归/不会对 AGENT_* 二次转移」。
    """
    reg = _reg()
    _plant(reg, "a1", None)

    # 模拟总线同步 drain：emit 时把事件也喂回 reg.handle_event 自己，
    # 与 InProcessEventBus 的真实行为同构（_SpyBus 本身不做 drain，这里补上）。
    async def _emit_and_redrain(ev: Event) -> None:
        reg.event_bus.events.append(ev)
        await reg.handle_event(ev)

    reg.event_bus.emit = _emit_and_redrain  # type: ignore[method-assign]

    await reg.handle_event(_task_ev(EventType.TASK_STARTED, "a1"))

    # 若发生递归/自触发，AGENT_RUNNING 之后 AgentInput.SETTLED 等输入不会凭空
    # 出现——真正要钉住的是：没有抛出 RecursionError，且状态、事件数都符合
    # 「只转移一次」的预期（AGENT_* 不在 _INPUT_BY_EVENT 里，二次投递必然是 no-op）。
    assert reg.status_of("a1") == "running"
    running = [e for e in reg.event_bus.events if e.type == EventType.AGENT_RUNNING]
    assert len(running) == 1


async def test_attach_to_bus_registers_one_handler():
    """对等 SessionRegistry 的同名先例（test_session_registry_state.py 的
    test_attach_to_bus_registers_one_handler，Task 16 起该测试从
    test_session_registry_inputs.py 迁来——原文件随会话状态机一并整体退役）：
    `_SpyBus.subscribe` 是空实现，
    验证不了订阅是否真的发生——这里换成会记录 handler 的 `RecordingBus`，
    直接断言 `attach_to_bus()` 确实调用了一次 `subscribe`（review Important #2）。
    """
    from tests.unit._session_helpers import RecordingBus

    bus = RecordingBus()
    reg = AgentLifecycleManager(template_lookup=None, event_bus=bus, model_resolver=lambda a, m: None)
    reg.attach_to_bus()
    assert len(bus.handlers) == 1


async def test_attach_to_bus_with_real_event_bus_drives_transition_without_recursion():
    """走真实 `InProcessEventBus`（而不是简化 mock）的集成验证（review Important #2）：
    `emit()` 内同步 drain——handler 在同一次 `emit()` 调用里被直接 await，不经调度器
    让出。真实注册一次订阅、真实发一条 `TASK_STARTED` 进总线，断言 ALM 收到、
    转移到 `running`、发出恰一条 `AGENT_RUNNING`（没有因为总线把 ALM 自己发的
    `AGENT_RUNNING` 回流给它自己而递归/重复转移）。
    """
    from ctx_weft.providers.events import InProcessEventBus

    bus = InProcessEventBus()
    reg = AgentLifecycleManager(template_lookup=None, event_bus=bus, model_resolver=lambda a, m: None)
    _plant(reg, "a1", None)
    reg.attach_to_bus()

    received: list[Event] = []

    async def _spy(ev: Event) -> None:
        received.append(ev)

    bus.subscribe(None, _spy)

    await bus.emit(_task_ev(EventType.TASK_STARTED, "a1", {"assigned_agent_id": "a1"}))

    assert reg.status_of("a1") == "running"
    running = [e for e in received if e.type == EventType.AGENT_RUNNING]
    assert len(running) == 1
    assert running[0].agent_id == "a1"
    assert running[0].session_id == "s1"


def test_status_of_unknown_agent_raises_agent_not_found():
    """R18 收口：`status_of` 对不存在的 agent 抛 `AgentNotFound`，而不是裸 KeyError。"""
    reg = _reg()
    with pytest.raises(AgentNotFound):
        reg.status_of("ghost")


def test_guard_allows_idle_and_waiting_human():
    reg = _reg()
    _plant(reg, "a1", None)
    reg.assert_can_receive("a1")
    reg._agents["a1"].status = "waiting_human"
    reg.assert_can_receive("a1")


def test_guard_rejects_running():
    """spec 4.1：忙碌直接报错，不排队。"""
    reg = _reg()
    _plant(reg, "a1", None)
    reg._agents["a1"].status = "running"
    with pytest.raises(AgentBusyError):
        reg.assert_can_receive("a1")


def test_guard_rejects_terminated_and_unknown():
    reg = _reg()
    _plant(reg, "a1", None)
    reg._agents["a1"].status = "terminated"
    with pytest.raises(AgentTerminatedError):
        reg.assert_can_receive("a1")
    with pytest.raises(AgentNotFound):
        reg.assert_can_receive("ghost")


def test_guard_allows_interrupted():
    """interrupted 是可恢复态，不拒收——resume 后继续处理。"""
    reg = _reg()
    _plant(reg, "a1", None)
    reg._agents["a1"].status = "interrupted"
    reg.assert_can_receive("a1")


def test_new_exception_codes_follow_existing_naming_style():
    """host 侧错误处理按 code 分流，新增异常必须带 code（跟随既有 SCREAMING_SNAKE 风格）。"""
    assert AgentNotFound.code == "AGENT_NOT_FOUND"
    assert AgentBusyError.code == "AGENT_BUSY"
    assert AgentTerminatedError.code == "AGENT_TERMINATED"


# ── Task 14: reducers 折叠 AGENT_* ──────────────────────────────────────────


class _MemStore(EventStore):
    """只读 fake：喂一段事件给 `rebuild_view`，不模拟快照。

    有序提交是 `EventStore` 的必需部分（spec: event-log），恢复路径无条件走
    `committed_head` + `read_range`，所以 fake 也得提供这两个——position 直接用
    列表下标（1-based），语义上等价于「这些事件按此顺序提交过」。

    **显式继承协议**，不用鸭子类型：分批重放（`EventStore.replay`）在协议里有能用的默认
    实现，只调这两个必需方法，继承下来就白拿。不继承就得自己再写一遍那段区间切分，那正是
    「照抄一份必然和 core 分叉」。顺带还进了 ABC 检查——少一个必需方法在实例化时就炸。
    """

    def __init__(self, events):
        self._events = events

    async def append(self, event):  # pragma: no cover - 只读 fake
        raise NotImplementedError

    async def read_by_session(self, session_id, **_kw):
        return list(self._events)

    async def committed_head(self, session_id):
        return len(self._events)

    async def read_range(self, session_id, *, after_position=0,
                         through_position=None, exclude_types=()):
        from ctx_weft.protocols.events import StoredEvent
        end = len(self._events) if through_position is None else through_position
        return [
            StoredEvent(event=e, position=i)
            for i, e in enumerate(self._events, start=1)
            if after_position < i <= end
        ]

    async def read_last_of_type(self, session_id, type_):
        """没有快照——恢复因此走全量重放。那是合法配置（正确，只是慢）。"""
        return None

    async def append_batch(self, session_id, batch_id, events):  # pragma: no cover - 只读 fake
        raise NotImplementedError

    async def load_latest_snapshot(self, session_id):
        # `rebuild_view` 把 NotImplementedError 当「这个 store 不支持快照」处理，
        # 落到全量 `read_range(0..head)` 重放——这里没有快照机制要模拟，直接选这条
        # 契约化的退路，而不是让 `_MemStore` 假装有快照。
        raise NotImplementedError


async def test_rebuild_view_folds_agent_status():
    """不变式 3：S 档事件必须被 reducer 折叠，冷重建与内存态等价。"""
    evs = [
        _task_ev(EventType.AGENT_INSTANTIATED, "a1", {"template_id": "tpl"}),
        _task_ev(EventType.AGENT_RUNNING, "a1", {"from_status": "idle", "trigger": "task_started"}),
        _task_ev(EventType.AGENT_WAITING_HUMAN, "a1", {"from_status": "running", "hitl_id": "h1"}),
    ]
    view = await rebuild_view(_MemStore(evs), "s1")
    assert view.agents["a1"].status == "waiting_human"


async def test_rebuild_view_terminated_is_sticky():
    evs = [
        _task_ev(EventType.AGENT_INSTANTIATED, "a1", {"template_id": "tpl"}),
        _task_ev(EventType.AGENT_TERMINATED, "a1", {"from_status": "idle", "reason": "user"}),
        _task_ev(EventType.AGENT_RUNNING, "a1", {"from_status": "idle"}),
    ]
    view = await rebuild_view(_MemStore(evs), "s1")
    assert view.agents["a1"].status == "terminated"


async def test_rebuild_view_terminated_freezes_current_task_id():
    """粘滞的反向验证，且比 `test_rebuild_view_terminated_is_sticky` 多钉一维：
    迟到事件换了个不同的 task_id 也不能挪动 current_task_id——不只是 status 冻结，
    「正在处理哪个 task」这一维同样冻结在终局那一刻。"""
    def mk(t: str, task_id: str, payload: dict) -> Event:
        return Event(
            id="evt_x", run_id=None, sequence=0, session_id="s1", type=t,
            timestamp=datetime.now(UTC), task_id=task_id, agent_id="a1", payload=payload,
        )

    evs = [
        mk(EventType.AGENT_INSTANTIATED, "t1", {"template_id": "tpl"}),
        mk(EventType.AGENT_RUNNING, "t1", {"from_status": "idle"}),
        mk(EventType.AGENT_TERMINATED, "t1", {"from_status": "running", "reason": "user"}),
        mk(EventType.AGENT_RUNNING, "t2", {"from_status": "idle"}),  # 迟到，换了个 task_id
    ]
    view = await rebuild_view(_MemStore(evs), "s1")
    assert view.agents["a1"].status == "terminated"
    assert view.agents["a1"].current_task_id == "t1"


async def test_rebuild_view_and_load_agree_on_status():
    """端到端：钉住「冷重建与运行时内存态等价」这个核心性质。

    一串 AGENT_* 事件 → `rebuild_view` 折叠出 view → `load(views)` 灌进
    registry → `registry.status_of()` 必须与折出的 `view.agents[id].status`
    一致——否则冷恢复后 `assert_can_receive` 会错误放行、`send_message`
    的路由判断也会失准（控制方在本任务追加的验收点）。
    """
    evs = [
        _task_ev(EventType.AGENT_INSTANTIATED, "a1", {"template_id": "tpl"}),
        _task_ev(EventType.AGENT_RUNNING, "a1", {"from_status": "idle", "trigger": "task_started"}),
        _task_ev(EventType.AGENT_WAITING_HUMAN, "a1", {"from_status": "running", "hitl_id": "h1"}),
    ]
    view = await rebuild_view(_MemStore(evs), "s1")

    reg = _reg()
    n = await reg.load(view.agents, session_id="s1", tenant_id="default", fallback_template_id="tpl")

    assert n == 1
    assert view.agents["a1"].status == "waiting_human"
    assert reg.status_of("a1") == view.agents["a1"].status
    assert reg._agents["a1"].current_task_id == view.agents["a1"].current_task_id == "t1"


def test_legacy_reducer_branches_still_present():
    """确认 L 档类型的 reducer 读取分支未被动过——存量日志重放全靠它们（events-v2.md §5）。

    HITL 那 6 个（`HITL_REQUIRED` + 5 个终态镜像）、`SESSION_STATUS_CHANGED` /
    `SESSION_PAUSED_HITL`、以及 `SESSION_FINISHED` 都在册。同批退役的
    `SESSION_INTERRUPTED`/`WAITING`/`RUNNING` 已于 2026-09-05 连枚举一并删除
    （`master` 从未有过这三个类型，不可能出现在任何存量流里），故不在此列。

    源码扫描而非只查 `L_TIER_EVENT_TYPES`：要钉住的是 reducers.py 这份源码本身，
    不是登记表。
    """
    import inspect

    from ctx_weft.core.control import reducers as reducers_mod

    src = inspect.getsource(reducers_mod)
    legacy_refs = [
        "EventType.SESSION_STATUS_CHANGED",
        "EventType.SESSION_PAUSED_HITL",
        "EventType.SESSION_FINISHED",
        "EventType.HITL_REQUIRED",
        "EventType.HITL_APPROVED",
        "EventType.HITL_MODIFIED",
        "EventType.HITL_ANSWERED",
        "EventType.HITL_REJECTED",
        "EventType.HITL_CANCELLED",
    ]
    missing = [ref for ref in legacy_refs if ref not in src]
    assert missing == [], f"reducer 分支缺失: {missing}"


# ── Task 16：SESSION_* 运行态停发进 L 档 ──────────────────────────────────────
# 同批的 SessionRunning / SessionWaiting / SessionInterrupted 已于 2026-09-05 连枚举
# 一并删除（见 `test_intra_branch_session_types_are_gone`），只剩这一个留在 L 档。

_RETIRED_SESSION_TYPES = {"SessionFinished"}


def test_retired_session_types_in_l_tier():
    assert _RETIRED_SESSION_TYPES <= set(L_TIER_EVENT_TYPES)


def test_retired_session_types_still_in_enum():
    """§5：只删发射，不删枚举。"""
    assert _RETIRED_SESSION_TYPES <= set(EVENT_TYPES)


def test_retired_session_types_not_emitted_in_core():
    """外加条：L 档 ∩ 实际发射集合 = ∅。"""
    members = ["SESSION_FINISHED"]
    hits = []
    for py in pathlib.Path("src/ctx_weft/core").rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for m in members:
            if f"EventType.{m}" in text and "reducers.py" not in str(py):
                hits.append(f"{py}:{m}")
    assert hits == [], f"停发类型仍在被发射：{hits}"


def test_reducers_still_understands_retired_session_types():
    """reducer 分支必须保留——存量日志靠它重建。"""
    src = pathlib.Path("src/ctx_weft/core/control/reducers.py").read_text(encoding="utf-8")
    for m in ("SESSION_FINISHED",):
        assert f"EventType.{m}" in src, f"reducers 丢了 {m} 的重放分支"


def test_intra_branch_session_types_are_gone():
    """2026-09-05：分支内部生死的 3 个会话运行态类型已连枚举一并删除。

    退役闸门（events-v2.md §5 第 2 级「确认没有任何回放会碰到这些字符串」）对它们
    天然成立：`master` 的 `EventType` 里从来没有这三个名字，它们只在 2026-09-02→09-03
    之间的分支内部存在过，任何从 master 迁移来的事件流都不可能含有它们。
    """
    for m in ("SESSION_RUNNING", "SESSION_WAITING", "SESSION_INTERRUPTED"):
        assert not hasattr(EventType, m), f"{m} 应已删除"
    for v in ("SessionRunning", "SessionWaiting", "SessionInterrupted"):
        assert v not in set(EVENT_TYPES)
        assert v not in set(L_TIER_EVENT_TYPES)
