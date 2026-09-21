"""SQL-backed EventStore（SQLAlchemy async；默认 SQLite，postgres 同源）。

**可选依赖**：需要 ``sqlalchemy>=2.0``（SQLite 后端另需 ``aiosqlite``），
装法 ``pip install ctx-weft[sql]``。没有任何上层包 eager import 本子包。
"""

from ctx_weft.providers.events.store.sql.models import Base, EventModel
from ctx_weft.providers.events.store.sql.store import (
    SqlEventStore,
    open_sqlite_event_store,
)

__all__ = [
    "Base",
    "EventModel",
    "SqlEventStore",
    "open_sqlite_event_store",
]
