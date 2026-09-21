"""测试侧的「全量装填」等价物——生产里那个 `CtxWeftRuntime.recover()` 已删。

`recover()` 于 2026-09-09 移除，两个理由：

  ① 它的循环体逐字就是 `rebuild_session`（解 tenant → 装填 HITL → 登记成员 → 装填 ALM），
     同一件事写在两处；
  ② 它扫的那个「active session」集合是坏的——判据只会 add、不会 discard，两条 discard
     依据 `SessionFinished` / `SessionStatusChanged` 早已随会话状态机退役而停发。于是
     「active 集」= 这台机器历史上跑过的**全部**会话，启动开销与历史会话数线性增长且
     永不收敛。

生产的恢复因此改成用户驱动：用到哪条会话，就 `rebuild_session` 哪条。host 自己有会话清单
（它建的会话、它的会话表），不必向 core 打听——`EventStore.list_active_session_ids` 与
`providers/events/_lifecycle` 那台判据状态机已于 2026-09-21 一并删除。

**但装填机制本身一行没变**，而下面这些用例测的正是那套机制。给它们一个测试侧的等价物，
比把每条用例改写成「点名装填哪几个 session」更贴近各自的题意——那样会把「验装填」的用例
悄悄变成「验我记得种了哪几个 session」。

枚举源因此改成 **store 的分区键**（`_stored` 的键集合）而不是从前那个活跃判据：用例要的
本来就是「把种下去的每一条会话都装填一遍」，分区键逐字就是这个意思，而活跃判据还要先答
一个这里根本没人问的问题（哪些会话还活着）。生产侧对应的是 host 自己的会话清单。

⚠ 新代码不要模仿它。要装填一条会话请直接 `await rt.rebuild_session(sid)`。
"""

from __future__ import annotations

from typing import Any


async def rebuild_all_active(rt: Any) -> int:
    """把 store 里出现过的每条 session 都 `rebuild_session` 一遍，返回装填的 agent 总数。

    只支持 `InMemoryEventStore`（读它的 `_stored` 分区键）——测试夹具用的就是它，而生产
    侧的等价物是 host 自己的会话清单，不在 core 的面里。
    """
    store = rt.event_store
    session_ids = getattr(store, "_stored", None)
    assert session_ids is not None, (
        f"rebuild_all_active 只支持 InMemoryEventStore，收到 {type(store).__name__}")
    total = 0
    for sid in list(session_ids.keys()):
        total += await rt.rebuild_session(sid)
    return total
