"""CommitGate conformance（spec: event-commit；change reliability-wp3）。

required 语义的组件级钉子：确认后通知、失败先标记后抛、自定义 bus 显式失败、
瞬态跳过、整批提交与缓冲保留重试。Runtime 级（隔离传播、drain 停摆）见
test_runtime_storage_failure.py 与 test_observer_backpressure.py。
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.protocols.events import (
    EventType,
    PersistenceUnavailableError,
    StoredEvent,
)
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus
from ctx_weft.protocols.events import Event

_T0 = datetime(2026, 9, 11, tzinfo=UTC)


def _ev(n: int, *, session: str = "s1", type_=EventType.RUN_STARTED,
         task_id: str = "") -> Event:
    return Event(id=f"evt_{n:04d}", run_id="r1", sequence=n, session_id=session,
                 type=type_, timestamp=_T0, task_id=task_id or None, payload={"n": n})


class _FailingStore(InMemoryEventStore):
    async def append_batch(self, *args, **kwargs):
        raise OSError("simulated storage unavailable")


# ── required：确认 → 通知顺序 ────────────────────────────────────────────────


async def test_commit_confirms_before_notification():
    """确认先于通知：观察者 handler 读 store 时事件必然已在（E-T01 组件级）。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))

    seen_in_store_at_notify: list[int] = []

    async def _observer(ev: Event) -> None:
        seen_in_store_at_notify.append(len(await store.read_by_session("s1")))

    bus.subscribe(None, _observer)
    await bus.emit(_ev(1))
    assert seen_in_store_at_notify == [1], "observer must see event only AFTER commit"


async def test_storage_failure_raises_and_marks_session_first():
    """失败：先标记健康（回调）再抛 PersistenceUnavailableError；无 committed 通知。"""
    store = _FailingStore()
    marked: list[tuple[str, BaseException]] = []
    gate = CommitGate(store, on_unavailable=lambda sid, exc: marked.append((sid, exc)))
    bus = InProcessEventBus()
    bus.attach_commit_gate(gate)

    notified: list[Event] = []
    bus.subscribe(None, notified.append)

    with pytest.raises(PersistenceUnavailableError):
        await bus.emit(_ev(1))
    assert marked and marked[0][0] == "s1"           # 先标记
    assert notified == []                            # 未通知（不伪装成功）
    assert await store.committed_head("s1") == 0


async def test_transient_events_skip_commit_but_notify():
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(InMemoryEventStore()))
    got: list[Event] = []
    bus.subscribe(None, got.append)
    from ctx_weft.protocols.events import TRANSIENT_EVENT_TYPES
    transient_type = next(iter(TRANSIENT_EVENT_TYPES))
    await bus.emit(_ev(1, type_=transient_type))
    assert len(got) == 1                              # 实时通知照发


# ── 自定义 bus 不支持 → 构造期显式失败 ────────────────────────────────────────


class _ForeignBus:
    """鸭子类型的自定义总线：实现 EventBus 抽象面但没有 attach_commit_gate。

    subscribe 给最小可用实现（best_effort 路径构造期要真订阅 persister）。
    """

    def __init__(self):
        self.subscriptions = []

    async def emit(self, event): raise NotImplementedError

    def subscribe(self, event_type, handler, *, provisional=False, required=False):
        from ctx_weft.protocols.events import SubscriptionHandle
        self.subscriptions.append(handler)
        return SubscriptionHandle(subscriber_id=f"foreign_{len(self.subscriptions)}",
                                  _bus=self)
    def begin_provisional(self, task_id): return None
    async def commit_provisional(self, task_id): return None
    def discard_provisional(self, task_id): return None
    async def stream(self, filter): raise NotImplementedError
    async def _unsubscribe(self, sid): return None


def test_required_rejects_unsupported_bus():
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template
    from ctx_weft.core import CtxWeftRuntime

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    with pytest.raises(ValueError, match="attach_commit_gate"):
        CtxWeftRuntime(
            providers=None, event_bus=_ForeignBus(),
        ) if False else _runtime_with_bus(resolver, _ForeignBus())


def _runtime_with_bus(resolver, bus):
    from ctx_weft.core import CtxWeftRuntime, ProviderRegistry
    providers = ProviderRegistry()
    providers.register_capability(resolver)
    return CtxWeftRuntime(providers=providers, event_bus=bus)


def test_best_effort_accepts_unsupported_bus_with_warning(caplog):
    from ctx_weft.core.models.config import RuntimeConfig
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    import logging
    with caplog.at_level(logging.WARNING):
        rt = _runtime_with_bus_cfg(resolver, _ForeignBus(),
                                   RuntimeConfig(event_commit_policy="best_effort"))
    assert rt.persistence is not None                  # 旧观察者路径
    assert any("best_effort" in r.message for r in caplog.records)


def _runtime_with_bus_cfg(resolver, bus, config):
    from ctx_weft.core import CtxWeftRuntime, ProviderRegistry
    providers = ProviderRegistry()
    providers.register_capability(resolver)
    return CtxWeftRuntime(providers=providers, event_bus=bus, config=config)


# ── provisional 整批：原子 + 失败缓冲保留重试 ────────────────────────────────


class _FlakyStore(InMemoryEventStore):
    """第一次 append_batch 失败，之后正常——模拟确认丢失后重试成功。"""

    def __init__(self):
        super().__init__()
        self.fail_next = True

    async def append_batch(self, session_id, batch_id, events):
        if self.fail_next:
            self.fail_next = False
            raise OSError("simulated transient failure")
        return await super().append_batch(session_id, batch_id, events)


async def test_window_commit_atomic_and_retryable():
    store = _FlakyStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))

    observers: list[str] = []
    bus.subscribe(None, lambda ev: observers.append(ev.id))

    bus.begin_provisional("t1")
    await bus.emit(_ev(1, task_id="t1"))          # 窗口内：只给 required/推测态
    assert observers == []
    with pytest.raises(PersistenceUnavailableError):
        await bus.commit_provisional("t1")       # 第一次提交失败
    assert await store.read_by_session("s1") == []
    assert observers == []                       # 观察者仍未收到任何事件

    await bus.commit_provisional("t1")           # 重试：同 batch_id，成功
    stored = await store.read_by_session("s1")
    assert [e.id for e in stored] == ["evt_0001"]   # 恰好一次、不双写
    assert observers == ["evt_0001"]                # 成功后才补投


async def test_window_positions_are_batch_contiguous():
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    bus.begin_provisional("t1")
    await bus.emit(_ev(1, task_id="t1"))
    await bus.emit(_ev(2, task_id="t1"))
    await bus.commit_provisional("t1")
    stored = await store.read_range("s1")
    assert [se.position for se in stored] == [1, 2]
    assert isinstance(stored[0], StoredEvent)


# ── 派生事件窗口归属（contextvar 继承，方案 §4.6）────────────────────────────


async def test_derived_session_event_inherits_round_and_discarded():
    """required 消费者在窗口内派生的无 task_id 事件：随窗口一起丢弃，不逃逸。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))

    async def _state_machine(ev: Event) -> None:
        # 模拟 ALM：消费窗口事件时同步派生一条无 task_id 的会话级事件
        if ev.type == EventType.TASK_STARTED:
            await bus.emit(_ev(99, type_=EventType.AGENT_RUNNING))

    bus.subscribe(None, _state_machine, provisional=True, required=True)
    observers: list[str] = []
    bus.subscribe(None, lambda ev: observers.append(ev.id))

    bus.begin_provisional("t1")
    await bus.emit(Event(id="evt_t1", run_id="r", sequence=1, session_id="s1",
                         type=EventType.TASK_STARTED, timestamp=_T0, task_id="t1"))
    bus.discard_provisional("t1")                # 整窗丢弃：派生事件随之消失

    assert await store.read_by_session("s1") == []   # 无逃逸落库
    assert observers == []                           # 宿主可见流无逃逸


async def test_required_consumer_failure_propagates_and_observers_unnotified():
    """required 消费者异常穿出 emit（E-T11）；观察者未收到该事件的通知。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))

    async def _bad(ev: Event) -> None:
        raise RuntimeError("state machine bug")

    bus.subscribe(None, _bad, provisional=True, required=True)
    observers: list[str] = []
    bus.subscribe(None, lambda ev: observers.append(ev.id))

    with pytest.raises(RuntimeError, match="state machine bug"):
        await bus.emit(_ev(1))
    assert observers == []                       # committed 通知未流出
    # 事件本身已提交（store 有）——提交与通知是两件事；恢复由持久日志收口
    assert len(await store.read_by_session("s1")) == 1
