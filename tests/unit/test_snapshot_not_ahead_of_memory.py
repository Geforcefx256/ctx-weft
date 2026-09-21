"""快照不得领先于 memory：**有快照 ⟹ 到它的 `committed_head` 为止，memory 效果都已落地。**

这条不变式是恢复路径「以快照为准」的前提。立不起来的话，一张领先的快照就是**静默的数据丢失**
——它说某件事做完了（所以 `pending_recap` / HITL `resolved` 里没有它），而 memory 里其实没有。

违规是真实可达的，不是假想。未提交窗口下的顺序是：

    TaskManager.commit_round:
      1. ROUND_COMMITTED
      2. bus.commit_provisional(task_id)   ← 缓冲里的事件**在这里**补投给 rest 订阅者
      3. commit_round 钩子：
           a. HitlResolved
           b. 暂存的 memory 写入落盘        ← memory 在这里才落地
      4. _rounds.pop

`SnapshotWriter` 是 rest 订阅者，所以它在第 2 步就收到了缓冲里的 `RunFinished` / `SessionFinished`
——那一刻 `committed_head` 已经包含整批事件，而第 3b 步还没跑。

这也是为什么「从事件里找一个『已落 memory』的信号」那条路走不通：缓冲里的**每一条**事件都排在
它自己的 memory 效果之前。`ACT_TURN_COMPLETED`、`CapabilityFinished` 都试过，都是这个原因。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._snapshot_helpers import latest_snapshot

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID, _TID = "s1", "t1"


def _ev(n: int, type_: str) -> Event:
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="acme", task_id=_TID, agent_id="ag1",
        payload={"outcome": "completed"},
    )


async def _bus():
    from ctx_weft.providers.events.bus.in_process.bus import InProcessEventBus

    return InProcessEventBus()


async def test_buffered_events_reach_rest_subscribers_at_flush_time() -> None:
    """缓冲里的事件在 `commit_provisional` 里就投给 rest 订阅者了。

    这是违规的机制本身：`SnapshotWriter` 正是 rest 订阅者（`subscribe` 的 `provisional`
    默认 False，那个默认「刻意」把落盘类订阅者放在安全一侧——但对快照来说，「看不到未提交的
    东西」不等于「不会在未提交的东西落盘时被叫醒」）。
    """
    bus = await _bus()
    seen: list[str] = []

    async def _rest(ev: Event) -> None:
        seen.append(ev.type)

    bus.subscribe(None, _rest)                    # provisional=False → rest 订阅者

    bus.begin_provisional(_TID)
    await bus.emit(_ev(1, EventType.RUN_FINISHED))
    assert seen == [], "窗口里 rest 订阅者不该收到——这一半是对的"

    await bus.commit_provisional(_TID)

    assert seen == [str(EventType.RUN_FINISHED)], (
        "补投发生在 commit_provisional 里；而暂存的 memory 写入要等 commit_round 钩子"
    )


async def test_snapshot_writer_is_a_rest_subscriber() -> None:
    """`SnapshotWriter` 走的就是上面那条路——它不声明 `provisional`。

    钉这一条是因为「它是哪一类订阅者」决定了它会不会在那个不安全的窗口被叫醒，而那件事
    从类名上完全看不出来。
    """
    import inspect

    from ctx_weft.providers.events import persister

    src = inspect.getsource(persister.attach_persistence)
    i = src.index("SnapshotWriter")
    wiring = src[i:i + 400]
    assert "provisional=True" not in wiring, (
        "快照订阅者不该声明 provisional——但它因此会在 commit_provisional 的补投里被叫醒"
    )


async def test_head_already_covers_the_batch_when_the_writer_is_woken() -> None:
    """被叫醒的那一刻，`committed_head` 已经涵盖整批缓冲事件。

    所以快照的边界（写那一刻的 head）会把「memory 还没落」的那些事件算进去——快照因此领先。
    这条用真 store 量出来，不靠读代码推断。
    """
    from ctx_weft.core.events.commit_gate import CommitGate

    store = InMemoryEventStore()
    bus = await _bus()
    bus.attach_commit_gate(CommitGate(store))

    heads: list[int] = []

    async def _rest(ev: Event) -> None:
        heads.append(await store.committed_head(_SID))

    bus.subscribe(None, _rest)

    bus.begin_provisional(_TID)
    for n in (1, 2, 3):
        await bus.emit(_ev(n, EventType.RUN_FINISHED))
    await bus.commit_provisional(_TID)

    assert heads, "rest 订阅者应当被叫醒"
    assert heads[0] == 3, (
        f"第一次被叫醒时 head 已经是 {heads[0]}——整批都在里面了，"
        "而这批事件的 memory 效果还在暂存区"
    )


# ── 修复：写入前置条件 ────────────────────────────────────────────────────────


async def test_writer_skips_while_memory_is_unsettled() -> None:
    """谓词说「没落定」→ 不写快照。

    跳过的代价在**安全的那一侧**：这一刻不写，下一个安全边界再写；快照偏旧只意味着恢复多
    重放一段、多做一次幂等写。而快照偏新是不可挽回的。
    """
    from ctx_weft.providers.events.snapshot import SnapshotWriter

    store = InMemoryEventStore()
    settled = {"v": False}
    w = SnapshotWriter(store, None, every_n_events=1,
                       memory_settled=lambda _sid: settled["v"])

    await store.append(_ev(1, EventType.SESSION_CREATED))
    await w.on_event(_ev(2, EventType.RUN_FINISHED))
    assert await latest_snapshot(store, _SID) is None, "没落定就不该写"

    settled["v"] = True
    await w.on_event(_ev(3, EventType.RUN_FINISHED))
    assert await latest_snapshot(store, _SID) is not None, "落定之后要写得出来"


async def test_skipping_still_counts_toward_the_threshold() -> None:
    """跳过的那些**仍然计数**——否则跳过会拖长重放窗口。

    计数不累加的话，一条一直开着窗的会话会把「每 N 事件一张」拖成「从不写」，恢复退回
    O(全部事件)。跳过只该延后这一张，不该取消它。
    """
    from ctx_weft.providers.events.snapshot import SnapshotWriter

    store = InMemoryEventStore()
    settled = {"v": False}
    w = SnapshotWriter(store, None, every_n_events=3,
                       memory_settled=lambda _sid: settled["v"])

    await store.append(_ev(1, EventType.SESSION_CREATED))
    for n in (2, 3, 4):                       # 三条，但都在「没落定」期间
        await w.on_event(_ev(n, EventType.RUN_FINISHED))
    assert await latest_snapshot(store, _SID) is None

    settled["v"] = True
    await w.on_event(_ev(5, EventType.RUN_FINISHED))   # 第一个安全边界
    assert await latest_snapshot(store, _SID) is not None, (
        "跳过期间的计数没累加 → 安全边界到了也不写 → 重放窗口被拖长"
    )


async def test_a_throwing_predicate_blocks_the_write() -> None:
    """谓词自己炸了 → 当作「不安全」。宁可少写一张，不可写一张领先的。"""
    from ctx_weft.providers.events.snapshot import SnapshotWriter

    def _boom(_sid: str) -> bool:
        raise RuntimeError("no task manager")

    store = InMemoryEventStore()
    w = SnapshotWriter(store, None, every_n_events=1, memory_settled=_boom)

    await store.append(_ev(1, EventType.SESSION_CREATED))
    await w.on_event(_ev(2, EventType.RUN_FINISHED))    # 不抛

    assert await latest_snapshot(store, _SID) is None


async def test_no_predicate_means_no_gate() -> None:
    """不注入谓词 → 行为与从前逐字一致（极简接线 / 单测不因此挡住快照）。"""
    from ctx_weft.providers.events.snapshot import SnapshotWriter

    store = InMemoryEventStore()
    w = SnapshotWriter(store, None, every_n_events=1)

    await store.append(_ev(1, EventType.SESSION_CREATED))
    await w.on_event(_ev(2, EventType.RUN_FINISHED))

    assert await latest_snapshot(store, _SID) is not None


def test_the_predicate_comes_from_core_not_the_provider() -> None:
    """判据必须由 core 注入——provider 结构上无从知道。

    provider 只看得见 EventBus，而缓冲里的**每一条**事件都排在它自己的 memory 效果之前。
    留在 provider 手里，它只能拿「收到某个事件」当代理，而那个代理恰好是错的。这条钉住
    「core 真的接上了」，以及 core 的判据真的读了窗口状态。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime)
    # **两个接线点都要传**：`required` 策略那条直接构造 SnapshotWriter，`best_effort` 那条
    # 走 attach_persistence。只钉「出现过」护不住漏掉一个——漏掉的那条策略下快照又会领先。
    sites = src.count("SnapshotWriter(") + src.count("attach_persistence(")
    assert sites >= 2, f"接线点数变了（{sites}），这条守卫要跟着更新"
    assert src.count("memory_settled=self._memory_settled") == sites, (
        f"{sites} 个接线点，只有 {src.count('memory_settled=self._memory_settled')} 个传了判据"
    )

    pred = inspect.getsource(CtxWeftRuntime._memory_settled)
    assert "any_round_open" in pred, "判据得读未提交窗口状态，那是它唯一的依据"
