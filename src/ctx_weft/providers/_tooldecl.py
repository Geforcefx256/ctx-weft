"""共享的 @tool 装饰器工厂：进程内 capability provider 用它声明工具。

每个 provider 调用 make_tool_registry(provider_name) 得到独立的 (decorator, tools, impls)
三元组，互不串扰——这样文件系统工具与其他内置工具可以分属不同 provider、各自 PROVIDER 前缀。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from ctx_weft.core.capabilities.schema import extract_schema
from ctx_weft.protocols.capability import normalize_recovery_policy, Purpose, ToolCapability


def make_tool_registry(provider_name: str):
    """返回 (tool 装饰器, tools dict, impls dict)，三者绑定同一 provider 前缀。

    用法::

        tool, _TOOLS, _IMPLS = make_tool_registry("fs")

        @tool(purposes=["act"], side_effects=True)
        async def write_file(...): ...
    """
    tools: dict[str, ToolCapability] = {}
    impls: dict[str, Callable] = {}

    def tool(
        *,
        purposes: list[Purpose],
        side_effects: bool = False,
        spillable: bool = True,
        description: str | None = None,
        recovery_policy: str = "reviewed",
    ):
        """声明并注册工具：提取 schema，存函数体为实现。

        description 缺省取完整 docstring（inspect.cleandoc 去缩进）；传入则覆盖（用于运行时动态生成的描述）。
        """
        def decorator(fn: Callable) -> Callable:
            doc = inspect.cleandoc(fn.__doc__ or "")
            cap = ToolCapability(
                id=f"{provider_name}:{fn.__name__}",
                name=fn.__name__,
                kind="tool",
                purposes=list(purposes),
                description=description or doc,
                input_schema=extract_schema(fn),  # ctx 已在 _SCHEMA_SKIP_DEFAULT 中
                side_effects=side_effects,
                # 崩溃后该工具的在途调用能不能自动重跑。默认 reviewed（保守）；
                # 只读、可重新派生的工具（read_file / read_tool_output）声明 idempotent
                # ——重跑它们既安全又是唯一合理的恢复动作。
                recovery_policy=normalize_recovery_policy(recovery_policy),
                spillable=spillable,
            )
            tools[fn.__name__] = cap
            impls[fn.__name__] = fn
            return fn
        return decorator

    return tool, tools, impls
