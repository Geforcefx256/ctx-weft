"""Control plane: CancelToken, PauseToken, reducers。

`ReplayEngine`（按事件 id 回放到任意历史点）2026-09-20 删除：全仓零调用，而它是
src 里最后一处 `read_by_session` 全量读。留着是地雷——「重建到某个历史事件点」
读起来是个合理需求，下一个人照它写就会把整条流读进内存（实测 3 万事件 ≈ 130MB）。
真需要的话按 position 区间做，`read_range` 现成。
"""

from ctx_weft.core.control.converters import session_from_projection, task_from_projection
from ctx_weft.core.control.tokens import CancelToken, PauseToken, RunTokens
from ctx_weft.core.control.types import RunStateView, SessionView, TaskView
from ctx_weft.protocols.events import EventStore

__all__ = [
    "CancelToken",
    "PauseToken",
    "RunTokens",
    "EventStore",
    "RunStateView",
    "SessionView",
    "TaskView",
    "session_from_projection",
    "task_from_projection",
]
