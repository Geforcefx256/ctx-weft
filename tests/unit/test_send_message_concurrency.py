"""回归：并发 `send_message` 撞上「ALM 有 record、TaskManager 没了」时，不得重建出
第二个 TaskManager 顶替第一个。

`_start_task_for_agent` 探到 `tm is None` 会调 `recover_agent(keep_alive=True)`。
`recover_agent` 拿 per-session 锁串行化，但 `_recover_session_locked` 的「复用活
owner」快路径此前要求 `resumed_task_id is not None`——而 `keep_alive` 这条路径恒为
None，于是第二个调用即便在锁内看见第一个刚建好的活 TM，也照样再建一个顶掉它。

顶替之后，第一个调用手上的 `tm` 局部变量指向已被顶替的 TM_A（`is_current()` 为
False）：它 push 进去的 task 不会被派发，`drain()` 也被 `_is_current` 守卫挡住。
`inflight` 那道护栏救不了它——那个集合只覆盖「已派发、正在跑」的 task
（`running_task_ids()`），刚 push 还没派发的不在里面。

复现要点：纯内存 provider 下所有 await 都同步完成，`gather` 的第一个协程会一口气
跑完、根本不交错。真实部署里 `_recover_session_locked`（读事件库重建投影）和
`_validate_and_normalize_content`（内容校验/外部化，带图时还要写 blob store）都是
真 IO。测试在这两处各插一个让出点，把那个真实的交错还原出来。
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio
_TS = datetime(2026, 6, 13, tzinfo=UTC)


def _ev(seq, sid, type_, *, agent_id=None, task_id=None, **payload):
    # id 前缀刻意用 `evt_0000…`：`InMemoryEventStore.read_by_session` 按**事件 id**
    # （ULID 字典序）排序，不按 sequence。手造 id 若排在真实 ULID（`evt_01M2…`）之后，
    # 本用例后面由 runtime 真发出来的事件会被折在种子事件**之前**，回放顺序颠倒。
    return Event(id=f"evt_0000{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                 type=type_, timestamp=_TS, task_id=task_id, agent_id=agent_id, payload=payload)


async def _seed_idle_agent(rt, sid: str, aid: str) -> None:
    """最小事件集 + `recover()`：装填 ALM 但**不建 TM**（`recover()` 自己的纪律：
    "startup runs nothing"）——即「agent 在、TM 不在」这个状态，与会话跑完被
    `_release_session` 回收后的状态同形。"""
    await append_one(rt.event_store, _ev(1, sid, EventType.SESSION_CREATED,
                                    template_id="agent:tpl_echo", root_agent_id=aid))
    await append_one(rt.event_store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                    template_id="agent:tpl_echo"))
    await append_one(rt.event_store, _ev(3, sid, EventType.AGENT_IDLE, agent_id=aid))
    await rebuild_all_active(rt)
    assert sid not in rt._task_managers, "前提条件：recover() 不该建 TM"


async def test_concurrent_send_message_does_not_supersede_the_task_manager(monkeypatch):
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.runtime import CtxWeftRuntime
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    async def _noop_drain(self):        # 不真的派发到 LLM
        return None
    monkeypatch.setattr(TaskManager, "drain", _noop_drain)

    # 让出点①：重建时读事件库
    orig_locked = CtxWeftRuntime._recover_session_locked
    async def _slow_locked(self, session_id, **kw):
        await asyncio.sleep(0)
        return await orig_locked(self, session_id, **kw)
    monkeypatch.setattr(CtxWeftRuntime, "_recover_session_locked", _slow_locked)

    # 让出点②：push 之前的内容校验/外部化。第一条多等一会儿，确保第二条的重建
    # （含读事件库）完整跑完——这正是「push 落到已被顶替的 TM 上」那个最坏交错。
    orig_validate = CtxWeftRuntime._validate_and_normalize_content
    seen = {"n": 0}
    async def _slow_validate(self, *a, **kw):
        seen["n"] += 1
        await asyncio.sleep(0.1 if seen["n"] == 1 else 0)
        return await orig_validate(self, *a, **kw)
    monkeypatch.setattr(CtxWeftRuntime, "_validate_and_normalize_content", _slow_validate)

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    sid, aid = "A", "agt_1"
    await _seed_idle_agent(rt, sid, aid)

    built: list[int] = []
    orig_reg = CtxWeftRuntime._register_and_drain
    def _spy(self, session, tm):
        built.append(id(tm))
        return orig_reg(self, session, tm)
    monkeypatch.setattr(CtxWeftRuntime, "_register_and_drain", _spy)

    h1, h2 = await asyncio.gather(
        rt.send_message(aid, "first", session_id=sid),
        rt.send_message(aid, "second", session_id=sid),
    )

    assert len(built) == 1, f"同一个 session 被重建了 {len(built)} 个 TaskManager"
    owner = rt._task_managers[sid]
    orphans = [h.task_id for h in (h1, h2) if owner.get_task(h.task_id) is None]
    assert not orphans, f"这些 task 落在已被顶替的 TM 上，永远不会被派发: {orphans}"
