"""注入消息的崩溃窗口补写：读取范围有上界，且上界的依据是那条快照不变式。

`Runtime._restore_appended_messages` 补的是这个窗口：`send_message` 注入的消息在回合提交时
**先**随事件落盘（`TaskMessageAppended`），**后**才从暂存区写进 memory。崩在两者之间 →
日志说这条消息来过、host 也已经显示出来了、memory 里却没有 → 模型永远看不见它。

它从前读整条会话的 `TaskMessageAppended`，而那是一条随会话长度线性增长的读。收窄的依据是
把「快照不得领先于 memory」反过来用：

    存在一张切面为 P 的可用快照 ⟹ position ≤ P 的每一条事件，其 memory 效果已落盘

所以 P 之前的那些必然撞上「视图里已有这个 id」而跳过。这个文件分两组钉：
`settled_memory_floor` 自己的判据（第一组），以及补写真的只看 floor 之后（第二组）。

**为什么这组测试值得存在**：收窄的失效方向是不对称的。多读几条只是浪费；少读一条就是一条
用户消息永久消失，而且**不报错**——正是这套设计一路在防的那个方向。
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from types import SimpleNamespace

from ctx_weft.core.control.reducers import (
    _PROJECTION_VERSION,
    reduce_events,
    settled_memory_floor,
)
from ctx_weft.protocols import (
    AgentCapability,
    AgentCapabilityProvider,
    CapabilityProviderInfo,
    ProviderContext,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._snapshot_helpers import seed_snapshot

_T0 = datetime(2026, 9, 20, tzinfo=UTC)


def _ev(seq: int, type_: str = "TaskMessageAppended", **payload) -> Event:
    task_id = payload.pop("task_id", None)
    return Event(id=f"evt_{seq:04d}", run_id="r", sequence=seq, session_id="s1",
                 type=type_, timestamp=_T0, task_id=task_id, payload=payload)


async def _store_with(n: int) -> InMemoryEventStore:
    store = InMemoryEventStore()
    await store.append(_ev(1, EventType.SESSION_CREATED,
                           template_id="t", root_agent_id="a"))
    for i in range(2, n + 1):
        await store.append(_ev(i, task_id="tsk_1", memory_id=f"mem_{i}"))
    return store


# ── 第一组：floor 的判据 ─────────────────────────────────────────────────────


async def test_floor_is_zero_without_a_snapshot() -> None:
    """没有快照 = 什么都不敢保证 → 退回全量（与收窄前行为一致）。"""
    assert await settled_memory_floor(await _store_with(5), "s1") == 0


async def test_floor_is_the_snapshot_cut() -> None:
    store = await _store_with(10)
    view = reduce_events(
        [se.event for se in await store.read_range("s1", through_position=6)],
        run_id="s1")
    await seed_snapshot(store, "s1", view, cut=6)
    assert await settled_memory_floor(store, "s1") == 6


async def test_floor_is_zero_when_the_snapshot_is_not_usable() -> None:
    """判据与恢复路径共用 `snapshot_is_usable`——版本不匹配的快照不得当 floor 用。

    这一条的方向很要紧：一张判不可用的快照**不保证**它切面之前的 memory 都落了（它可能是
    上一代投影语义写下的），拿它当 floor 就会跳过该补的消息。所以「不可用」必须退回 0，
    不能退回「就先信着吧」。
    """
    store = await _store_with(10)
    view = reduce_events(
        [se.event for se in await store.read_range("s1", through_position=6)],
        run_id="s1")
    await seed_snapshot(store, "s1", view, cut=6,
                        overrides={"projection_version": _PROJECTION_VERSION + 1})
    assert await settled_memory_floor(store, "s1") == 0


async def test_floor_is_zero_when_the_snapshot_points_past_the_head() -> None:
    """切面超前于 `committed_head` = 数据异常，同样退回 0。"""
    store = await _store_with(5)
    view = reduce_events([se.event for se in await store.read_range("s1")], run_id="s1")
    await seed_snapshot(store, "s1", view, cut=999)
    assert await settled_memory_floor(store, "s1") == 0


async def test_floor_reads_exactly_one_event() -> None:
    """取 floor 本身不许是一条随会话长度增长的读。

    它用 `read_last_of_type`（索引直取一条）。若哪天改成「按类型全取再挑最后一张」，那就
    把历史上每一张快照连 blob 一起捞了回来——这条会红。
    """
    store = await _store_with(60)
    view = reduce_events([se.event for se in await store.read_range("s1")], run_id="s1")
    for cut in (10, 20, 30, 40, 50):
        await seed_snapshot(store, "s1", view, cut=cut)

    calls: list[str] = []
    real_last, real_range = store.read_last_of_type, store.read_range

    async def spy_last(*a, **k):
        calls.append("read_last_of_type")
        return await real_last(*a, **k)

    async def spy_range(*a, **k):
        calls.append("read_range")
        return await real_range(*a, **k)

    store.read_last_of_type, store.read_range = spy_last, spy_range  # type: ignore[method-assign]
    assert await settled_memory_floor(store, "s1") == 50
    assert calls == ["read_last_of_type"], f"取 floor 多读了东西：{calls}"


# ── 第二组：补写的行为 ───────────────────────────────────────────────────────
#
# 这一组是真的跑一遍 `_restore_appended_messages`。写它的直接原因是：收窄之前**全仓没有
# 任何行为测试碰过这个方法**，只有源码断言护不住 off-by-one（`after_position` 是不是
# 排他的、floor 该不该 +1），而这里少读一条就是一条用户消息永久消失、且不报错。


class _StubAgents(AgentCapabilityProvider):
    """runtime 构造期硬校验要求至少一个 AgentCapabilityProvider。"""

    name = "stub_agents"

    async def list(self, ctx):
        return [AgentCapability(id="stub_agents:a", name="a", kind="agent")]

    async def get_template(self, template_id, version, ctx):
        return None

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name)


async def _runtime_with(store: InMemoryEventStore):
    from ctx_weft.core.registry import ProviderRegistry
    from ctx_weft.core.runtime import CtxWeftRuntime
    from ctx_weft.providers.memory.in_memory.provider import InMemoryMemoryProvider

    registry = ProviderRegistry()
    registry.register_capability(_StubAgents())
    memory = InMemoryMemoryProvider()
    registry.register_memory(memory)
    return CtxWeftRuntime(providers=registry, event_store=store), memory


async def _appended(seq: int, *, mem_id: str, task_id: str = "tsk_1") -> Event:
    """一条 `TaskMessageAppended`——注入消息在事件日志里的那一份。"""
    return Event(
        id=f"evt_{seq:04d}", run_id="r", sequence=seq, session_id="s1",
        type=EventType.TASK_MESSAGE_APPENDED, timestamp=_T0, task_id=task_id,
        agent_id="agt_1",
        payload={"memory_id": mem_id, "agent_id": "agt_1", "content": f"msg {mem_id}",
                 "source": "send_message", "timestamp": _T0.isoformat()})


async def _restore_and_collect(store, session_id="s1"):
    """跑一遍补写，返回 memory 里出现的 record id。"""
    from ctx_weft.protocols import MemoryAddress, MemoryScope

    rt, memory = await _runtime_with(store)
    session = SimpleNamespace(id=session_id, tenant_id="default")
    await rt._restore_appended_messages(session)  # type: ignore[arg-type]
    records = await memory.load_view(
        MemoryAddress(session_id=session_id, task_id="tsk_1", agent_id="agt_1"),
        MemoryScope.TASK,
        ProviderContext(session_id=session_id, tenant_id="default",
                        task_id="tsk_1", agent_id="agt_1"))
    return {r.id for r in records}


async def test_restores_a_message_that_never_made_it_into_memory() -> None:
    """崩溃窗口的正路：事件在、memory 里没有 → 补回来。"""
    store = InMemoryEventStore()
    await store.append(_ev(1, EventType.SESSION_CREATED,
                           template_id="t", root_agent_id="agt_1"))
    await store.append(await _appended(2, mem_id="mem_lost"))

    assert "mem_lost" in await _restore_and_collect(store)


async def test_restores_the_message_sitting_exactly_at_the_floor_plus_one() -> None:
    """**off-by-one**：position 恰好 = floor + 1 的那条必须被补。

    `after_position` 是排他的，floor = 切面 P，所以 P+1 要读到。写成 `floor + 1` 就会漏掉
    它，而在生产里那表现为「崩溃前最后那条消息丢了」——最难被注意到的那种。

    ⚠️ **构造是刻意的**，第一版没钉住：常态下快照事件自己就落在 `cut + 1`（writer 取
    `cut = committed_head` 之后紧接着 append 那条 `StateSnapshot`），于是 floor+1 永远是
    一条会被类型过滤掉的快照事件，下界差一位也看不出来。这里让 `cut` 落在快照事件自己的
    position **之前**（cut=3，快照事件在 6），使 position 4 是一条真实的
    `TaskMessageAppended`。这不是臆造的形状：`snapshot_is_usable` 只要求 `cut <= head`，
    而并发 append 完全可以在「取 head」与「append 快照事件」之间插进来。
    """
    store = InMemoryEventStore()
    await store.append(_ev(1, EventType.SESSION_CREATED,
                           template_id="t", root_agent_id="agt_1"))
    await store.append(await _appended(2, mem_id="mem_old"))
    await store.append(await _appended(3, mem_id="mem_at_cut"))
    await store.append(await _appended(4, mem_id="mem_floor_plus_one"))
    await store.append(await _appended(5, mem_id="mem_later"))
    view = reduce_events(
        [se.event for se in await store.read_range("s1", through_position=3)],
        run_id="s1")
    await seed_snapshot(store, "s1", view, cut=3)       # 快照事件落在 position 6

    assert await settled_memory_floor(store, "s1") == 3
    restored = await _restore_and_collect(store)
    assert "mem_floor_plus_one" in restored, (
        "position == floor + 1 那条没被补——下界写成 floor + 1 了吗")
    assert "mem_later" in restored


async def test_does_not_reread_what_the_snapshot_already_guarantees() -> None:
    """切面之前的那些不再读回来——收窄的全部意义。

    它们**本来也不会**被重复写进 memory（id 幂等），所以这条不能拿 memory 内容当判据，
    只能看实际读了哪个区间。
    """
    store = InMemoryEventStore()
    await store.append(_ev(1, EventType.SESSION_CREATED,
                           template_id="t", root_agent_id="agt_1"))
    for i in range(2, 12):
        await store.append(await _appended(i, mem_id=f"mem_{i}"))
    view = reduce_events(
        [se.event for se in await store.read_range("s1")], run_id="s1")
    await seed_snapshot(store, "s1", view, cut=11)

    seen: list[int] = []
    real_range = store.read_range

    async def spy(session_id, **k):
        seen.append(int(k.get("after_position", 0)))
        return await real_range(session_id, **k)

    store.read_range = spy  # type: ignore[method-assign]
    rt, _memory = await _runtime_with(store)
    await rt._restore_appended_messages(  # type: ignore[arg-type]
        SimpleNamespace(id="s1", tenant_id="default"))
    assert seen and min(seen) == 11, f"读的下界不是切面：{seen}"


# ── 第三组：源码形态与耦合守卫 ───────────────────────────────────────────────


def test_the_restore_reads_from_the_floor_not_the_whole_session() -> None:
    """补写的读取以 floor 为下界，且排除快照事件本身（与恢复侧同口径）。"""
    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._restore_appended_messages)
    assert "settled_memory_floor" in src, "补写又读全会话了"
    # 起点是 floor，然后按区间往前走（`cursor = floor` + while 循环）。分批是必要的：
    # floor==0 时（首张快照之前 / 存量库 / projection_version 刚 bump）区间就是整条会话，
    # 一次性物化那一段的代价与重锚同源（实测 20 万事件 604.9MB）。
    assert "cursor = floor" in src, src
    assert "while cursor < head" in src, "没有分批——floor==0 时会一次读完整条会话"
    assert "load_events_of_types" not in src, (
        "`load_events_of_types` 没有下界参数——用它就是又读全会话")
    # 刻意**不**断言 `exclude_types=REPLAY_EXCLUDE_TYPES`。代码里传了它，但那纯粹是省
    # 传输/反序列化（快照 blob 不必从库里搬回来再丢掉）——下面那道
    # `type == TASK_MESSAGE_APPENDED` 的过滤已经把快照事件挡掉了，所以拿掉 exclude_types
    # 行为完全不变。验证过：去掉它这一整个文件仍然全绿。断言一件没有行为后果的事，只会
    # 让人误以为它是正确性保障。


def test_the_invariant_this_narrowing_depends_on_is_still_enforced() -> None:
    """收窄的依据是「快照不得领先于 memory」，所以那道门必须还在。

    这条是**耦合守卫**，不是重复测试。`test_snapshot_not_ahead_of_memory.py` 钉的是那道门
    本身；这里钉的是「补写收窄依赖它」这层关系：哪天有人把 `memory_settled` 那道门去掉、
    只跑快照那套测试，可能只看到「快照偶尔偏新」这种听起来可以接受的退化，而不会意识到
    上游还有一个补写把 floor 当成了保证——后果是用户消息静默消失。

    所以这里直接断言门还在，并在失败信息里把这条关系写出来。

    ⚠️ **这是指路牌，不是保护**。它只看源码里那个名字还在不在，挡不住「门看着还在、实际
    被改成恒真」。真正在行为上拦住那种改法的是
    `test_snapshot_not_ahead_of_memory.py::{test_writer_skips_while_memory_is_unsettled,
    test_skipping_still_counts_toward_the_threshold, test_a_throwing_predicate_blocks_the_write}`
    ——验证过：把门改成 `if True or ...`，那三条会红，本条不会。所以本条的作用是让读到这里
    的人知道「上游还有一个补写依赖这道门」，改门之前先看这条关系。
    """
    from ctx_weft.core.control.snapshot_writer import SnapshotWriter
    from ctx_weft.core.runtime import CtxWeftRuntime

    gate = inspect.getsource(SnapshotWriter._is_safe_to_write)
    assert "_memory_settled" in gate, (
        "写快照的安全门不见了。`_restore_appended_messages` 把快照切面当作"
        "「到这里 memory 都落了」的保证，门一去掉那个保证就不成立，它会跳过该补的消息。")
    assert "any_round_open" in inspect.getsource(CtxWeftRuntime._memory_settled), (
        "判据不再读未提交窗口状态——那是它唯一的依据")
