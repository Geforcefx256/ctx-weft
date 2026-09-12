"""ToolResultStore：工具长输出的可回取结果存储（spec: tool-result-recovery）。

收敛契约分工：上下文（对话与 memory TOOL_RESULT）只承载收敛版（引用 + 全长 + 头尾
预览），全文入本存储——键为本次执行身份（invocation_id），支持窗口读取（offset/limit
分页与从末尾直读）。与操作账本（tool-operations）互补：账本持收敛前全文供恢复重建，
本存储是 live 会话的低成本回取通路（可逐出）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# 回读工具的 qualified 名（results provider 注册；收敛文本里的引用与此同源——
# 由 tests 钉死两侧一致，改 provider 名必须同步这里）。
READ_TOOL_QUALIFIED_NAME = "results__read_tool_output"


class ToolResultStore(ABC):
    """全文存取协议。窗口语义：``offset``（0 基字符偏移）+ ``limit`` 分页；
    ``tail`` 直取末尾 N 字符。未命中/已逐出 → ``None``（调用方转显式不可用信息，
    可区分于空输出）。"""

    @abstractmethod
    async def put(self, invocation_id: str, text: str, ctx: Any = None) -> None:
        """写入/覆盖一次执行的全文（同键重写以最新执行为准）。失败 SHALL 抛出。"""
        ...

    @abstractmethod
    async def get(
        self,
        invocation_id: str,
        *,
        offset: int | None = None,
        limit: int | None = None,
        tail: int | None = None,
        ctx: Any = None,
    ) -> str | None:
        """按窗口读取；键不存在或已逐出返回 None。"""
        ...
