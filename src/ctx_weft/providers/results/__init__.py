"""结果存储实现包（spec: tool-result-recovery）。内存默认 / 宿主可注册持久实现。"""

from ctx_weft.providers.results.in_memory import InMemoryToolResultStore

__all__ = ["InMemoryToolResultStore"]
