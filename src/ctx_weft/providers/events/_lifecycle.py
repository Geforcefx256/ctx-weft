"""会话活跃判据——**两个 EventStore 实现共用的同一台状态机**。

`list_active_session_ids()` 决定崩溃恢复要捞哪些会话。两个实现各写一遍判据必然分叉，
而分叉的表现是「重启后某些会话不弹恢复」或「已结束的会话反复被恢复」——都极难归因。
故判据只此一份，`in_memory` 增量调用它，`sql` 把查出来的生命周期事件重放一遍。

**重放顺序必须是提交序。** `in_memory` 用的是到达序（= 提交序，它就是在 append 里增量
调用 `apply_lifecycle`），所以 `sql` 侧那条查询按 `position` 排。从前它按 `id` 排——那是
ULID 铸造序，并发提交下与提交序分叉，于是同一台状态机在两个实现上吃到不同顺序的输入，
`SessionResumed` 与 `SessionFinished` 的先后就可能反。把判据收成一份是为了不分叉，
输入顺序不同口径等于把分叉从判据挪到了喂料口。

**为什么 SQL 侧可以「先把所有 session 置为 active，再重放生命周期事件」**：
`in_memory` 是在每个 session 的**首次出现**时把它加进 active 的。由于操作只有
add/discard 且逐 session 独立，「在 -∞ 处 add」与「在该 session 首个事件处 add」
对最终结果完全等价——首个事件必然先于该 session 的其余事件。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "LIFECYCLE_EVENT_TYPES",
    "TERMINAL_STATUSES",
    "apply_lifecycle",
    "replay_lifecycle",
]

#: `SessionStatusChanged.payload["new_status"]` 取这些值时视为会话已终结。
#:
#: **刻意不复用 `core.domain.status.TERMINAL_SESSION_STATUSES`**：这里多一个
#: `INTERRUPTED`。两者语义不同——那个是「会话终态」（到达后任何输入都不再引发转移），
#: 这个是「不必再当作活跃会话查询」的**活跃性**判据，被打断的会话虽非终态，但在
#: SQL 侧收窄查询时同样不该算活跃。看着像第五份复制，其实不是；合并会让
#: INTERRUPTED 会话被误判成终态。
TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "INTERRUPTED"})

#: 会改变活跃性的事件类型。SQL 侧据此收窄查询范围；`SessionCreated` 本身在
#: `apply_lifecycle` 里是 no-op（它的效果由「该 session 出现过」这个种子覆盖），
#: 但仍列在这里——收窄查询时漏掉它没有坏处，列上它让集合的语义自解释。
LIFECYCLE_EVENT_TYPES = (
    "SessionCreated",
    "SessionResumed",
    "SessionFinished",
    "SessionStatusChanged",  # L 档：只为读存量日志
)


def apply_lifecycle(active: set[str], event: Any) -> None:
    """把一条事件应用到活跃集合上。非生命周期事件一律 no-op。

    ⚠️ **不负责「首次出现即 active」那条种子规则**——那条由调用方提供：
    `in_memory` 在首次见到某 session 时 add，`sql` 用 `SELECT DISTINCT session_id`
    一次性 add。两者等价，见模块 docstring。
    """
    sid = event.session_id
    t = event.type
    if t == "SessionFinished":
        # 已结束，不必在下次重启时再捞。
        active.discard(sid)
    elif t == "SessionResumed":
        # 续跑 = 还活着，重启后要重新装填它的未决 HITL。
        active.add(sid)
    elif t == "SessionStatusChanged":
        if (event.payload or {}).get("new_status", "") in TERMINAL_STATUSES:
            active.discard(sid)


def replay_lifecycle(session_ids: Iterable[str], events: Iterable[Any]) -> set[str]:
    """种子（全部出现过的 session）+ **按提交序**的生命周期事件 → 活跃集合。

    `events` 必须按 position（提交序）给，不是 `id`（铸造序）——见模块 docstring。
    """
    active = set(session_ids)
    for event in events:
        apply_lifecycle(active, event)
    return active
