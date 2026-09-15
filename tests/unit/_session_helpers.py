"""Task 4/5 共用的事件总线替身。

单独成模块而不是放在某个 test_*.py 里：跨测试文件 import 会让删掉一个文件
连带打断另外两个，而 pytest 的收集顺序不保证被 import 的那个先被收集。
"""

from __future__ import annotations

from ctx_weft.protocols.events import Event


class RecordingBus:
    """记录所有 emit 的事件。`subscribe` 记下 handler 供测试手动投递。"""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self.handlers: list = []

    async def emit(self, event: Event) -> None:
        self.events.append(event)

    def subscribe(self, event_type, handler, *, provisional=False,
                  required=False):  # noqa: ANN001 - 测试替身
        self.handlers.append(handler)
        return None

    def stream(self, filter):                        # noqa: A002 - 对齐协议签名
        raise NotImplementedError

    async def _unsubscribe(self, subscriber_id: str) -> None:
        return None

    def types(self) -> list[str]:
        return [e.type for e in self.events]
