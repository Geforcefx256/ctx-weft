"""测试侧读事件的小工具。

`EventStore.read_by_session`（「把一个会话的全部事件读成一个 list」）在 2026-09-21 从协议里
删掉了——它就是本仓一直在清的那个形状，而删它的时候 src 里已经零调用者。理由写在
`protocols/events.py` 里它原来的位置。

**但测试里「把这个会话的事件全拿出来」是正当需求**：夹具只有几条到几十条事件，而断言要看
的正是「都发了些什么、什么顺序」。所以这里把它收成一个函数，而不是在 70 多个断言里各展开一
遍 `[se.event for se in await store.read_range(sid)]`——各展开一遍的问题不是难看，是下次要改
口径（比如又要排除某个类型）时得改 70 处，改漏的那几处不会红。

放在 `tests/` 下而不是 `src/` 下是刻意的：它的安全性来自「调用方是测试、数据量由夹具决定」，
那个前提出了 `tests/` 就不成立。
"""

from __future__ import annotations

from typing import Any

from ctx_weft.protocols.events import Event


async def all_events(store: Any, session_id: str) -> "list[Event]":
    """该会话的全部已提交事件，按 position（提交序）升序。

    只看得见已提交（position 非空）的事件。存量未回填的 NULL position 行**读不到**——那是
    `read_range` 的口径，也是删掉 `read_by_session` 时顺带失去的唯一能力。那个混合状态现在
    没有读者：`open_sqlite_event_store` 拒绝打开未回填的库，宿主侧 m020 在任何读之前回填完。
    """
    return [se.event for se in await store.read_range(session_id)]


async def append_one(store: Any, event: Event) -> None:
    """落一条事件——`store.append(event)` 的替代。

    `EventStore.append` 在 2026-09-21 从协议删掉了：两个实现逐字相同
    （`append_batch(event.session_id, event.id, [event])`），src 里只有一个调用者，而
    `batch_id = event.id` 那一句是**策略**（确定性取值，好让原样重试撞上幂等账而不是写出第二
    份），由每个实现各写一遍正是本仓被咬过的那个形状。理由写在协议里它原来的位置。

    夹具里「落一条事件」仍然是最常见的动作，所以收成这个函数——**策略只写在这一处**，和
    `all_events` 同一个理由、同一个位置。
    """
    await store.append_batch(event.session_id, event.id, [event])
