    # 「来源」的真正钉子：用 AST 读 runtime.py 里 import 语句的 `module` 字段，
    # 钉的是**源码里写的路径**而不是运行时对象。
    #
    # 历史理由（2026-08-27）：当时 core.events.bus / core.state.event_store 两个 shim
    # 转发的是同一个类对象，`is` 身份比较因此钉不住「从哪条路径 import」，恒真。
    # 那两个 shim 已于 2026-08-29 删除，`is` 比较今天确实能区分了——但本用例仍用
    # AST，因为它钉的是**不许再冒出第二条来源**：任何人日后新加一个转发层，`is`
    # 会重新失效，而 AST 断言的集合会立刻多出一个元素、转红。
"""event 契约的分层守卫：契约在 protocols、实现在 providers、投影逻辑留 core。

spec: docs/superpowers/specs/2026-08-27-protocols-layer-event-contracts-design.md

原先本文件还钉「两条 import 路径指向同一批对象」——`core/events/` 与
`core/state/event_store.py` 两个 re-export shim 在 2026-08-29 删除后，那批用例
全部退化成同义反复（同一个模块 import 两次再断言 `is`），已一并删除；只剩下真正
还钉得住东西的：层序（protocols 不 import core）、归属（实现不进 protocols、
TASK_STATUS_BY_EVENT 不进 protocols）、包根导出、以及 shim 不得复活。
"""

from __future__ import annotations





def test_task_status_map_stays_in_core() -> None:
    """TASK_STATUS_BY_EVENT 是 core 的投影逻辑（依赖 core 的 TaskStatus），不进 protocols。

    shim 删除后它落在 `core/control/reducers.py`（唯一的生产消费者就在那个文件里）。
    """
    import ctx_weft.protocols.events as pe
    from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT

    assert TASK_STATUS_BY_EVENT
    assert not hasattr(pe, "TASK_STATUS_BY_EVENT")


def _assert_module_does_not_import_core(module: object, label: str) -> None:
    """层序守卫的公共实现：解析 `module` 的源码，确认它不 import `ctx_weft.core`。

    用 `ast` 解析实际的 import 语句，而不是对源码整体做子串扫描：字符串扫描
    误报（docstring/注释里提到字面路径 `ctx_weft.core` 也会被判违规）也漏报
    （`from ctx_weft import core`、`importlib.import_module(...)`、拼接/折行
    的字符串都扫不出来）。别为了"简化"把这条改回字符串匹配。
    """
    import ast
    import inspect

    source = inspect.getsource(module)
    tree = ast.parse(source)

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "ctx_weft.core" or mod.startswith("ctx_weft.core."):
                violations.append(f"line {node.lineno}: from {mod} import ...")
            elif mod == "ctx_weft":
                for alias in node.names:
                    if alias.name == "core":
                        violations.append(f"line {node.lineno}: from ctx_weft import core")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "ctx_weft.core" or name.startswith("ctx_weft.core."):
                    violations.append(f"line {node.lineno}: import {name}")

    assert not violations, f"{label} 不得 import core，违规语句：" + "; ".join(violations)


def test_protocols_events_does_not_import_core() -> None:
    """层序守卫：protocols 不得依赖 core。

    这条不变量一旦破掉，`protocols/context.py` 那个刻意的惰性绑定就白做了，
    且会在某些 import 顺序下变成真实的循环导入。
    """
    import ctx_weft.protocols.events as pe

    _assert_module_does_not_import_core(pe, "protocols/events.py")


def test_providers_events_does_not_import_core() -> None:
    """同一条层序守卫，扩展到 `providers/events/`。

    `providers/*` 整体是允许 import core 的（`providers/llm/*` 等 7 个模块确实
    这么做），但 `providers/events/bus/in_process/bus.py` /
    `providers/events/store/in_memory/store.py` 被 `core/events/bus.py`
    等兼容层反向 import——若它们也 import core，加上 `providers/__init__.py`
    未来若不再是平凡的 3 行，就会拼出一条包内真实的循环 import
    （见 2026-08-28 final-fix 计划 I3）。这条钉住前一半：这两个模块本身 core-free。
    """
    import ctx_weft.providers.events.bus as bus_module
    import ctx_weft.providers.events.store as store_module

    _assert_module_does_not_import_core(
        bus_module, "providers/events/bus/in_process/bus.py")
    _assert_module_does_not_import_core(
        store_module, "providers/events/store/in_memory/store.py")



def test_implementations_do_not_enter_protocols() -> None:
    """实现不进 protocols——协议与实现分居是本次划界的全部意义。

    本任务只保证「不在 protocols」；Task 3 会把它们从 core 搬进 providers。
    """
    import ctx_weft.protocols.events as pe
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.events import InMemoryEventStore

    assert InProcessEventBus is not None and InMemoryEventStore is not None
    assert not hasattr(pe, "InProcessEventBus")
    assert not hasattr(pe, "InMemoryEventStore")


def test_implementations_still_satisfy_the_relocated_protocols() -> None:
    """re-export 若产生了同名副本，这条会红——runtime_checkable 认的是具体类对象。"""
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.events import InMemoryEventStore
    from ctx_weft.protocols.events import EventBus, EventStore

    assert isinstance(InProcessEventBus(), EventBus)
    assert isinstance(InMemoryEventStore(), EventStore)


def test_event_store_contract_is_exported_from_package_root() -> None:
    """host 必须实现 EventStore，却一直只能从 core 里 import 它（spec §1）。"""
    import ctx_weft
    from ctx_weft.protocols.events import EventStore

    assert ctx_weft.EventStore is EventStore
    assert "EventStore" in ctx_weft.__all__



def test_package_root_store_comes_from_providers() -> None:
    """host 的 `from ctx_weft import InMemoryEventStore` 拿到的仍是同一个类。"""
    import ctx_weft
    from ctx_weft.providers.events import InMemoryEventStore

    assert ctx_weft.InMemoryEventStore is InMemoryEventStore


def test_dead_simplified_store_is_gone() -> None:
    """`core/control/replay.py` 的同名简化版是死代码（零使用者），已删。

    留着的危害是「同名不同实现、都对外可见」：
    `from ctx_weft.core.control import InMemoryEventStore` 会拿到一个缺快照方法的对象。
    """
    import ctx_weft.core.control as cc
    import ctx_weft.core.control.replay as replay

    assert not hasattr(replay, "InMemoryEventStore")
    assert not hasattr(cc, "InMemoryEventStore")
    assert "InMemoryEventStore" not in getattr(cc, "__all__", [])


async def test_runtime_still_gets_a_working_default_store() -> None:
    """接线换了来源，`CtxWeftRuntime` 的默认 event_store 仍要能用。

    ⚠️ 修订记录：本测试原先按 brief 的 fallback 直接构造
    `InMemoryEventStore(event_bus=InProcessEventBus())`，绕开了
    `CtxWeftRuntime()` 无参构造会因「未注册 AgentCapabilityProvider」抛
    ValueError 这一既有硬校验（core/runtime.py:498-503，与本次搬迁无关）。
    但那样一来测试从未真正碰到 `core/runtime.py` 里本任务实际改动的两处接线
    （模块级 `InProcessEventBus` 改从 providers 取、函数内 `InMemoryEventStore`
    改从 providers 取），且与同文件的
    `test_implementations_still_satisfy_the_relocated_protocols` 重复。code
    review 后改为：照抄 `tests/unit/test_media_get_image.py` 里
    `_StubAgents` 的最小构造方式，真正实例化 `CtxWeftRuntime`；换源本身则用
    AST 读 runtime.py 的 import 语句直接钉住（对象身份比较会被 re-export
    链路掩盖，验证过，见下方注释与 task-3-report.md）。
    """
    from datetime import datetime, timezone

    from ctx_weft.core.registry import ProviderRegistry
    from ctx_weft.core.runtime import CtxWeftRuntime
    from ctx_weft.protocols.capability import (
        AgentCapability,
        AgentCapabilityProvider,
        CapabilityProviderInfo,
    )
    from ctx_weft.protocols.events import Event
    from ctx_weft.providers.events import InMemoryEventStore

    class _StubAgents(AgentCapabilityProvider):
        """`CtxWeftRuntime` 构造期硬校验要求至少一个 AgentCapabilityProvider——
        照抄 `tests/unit/test_media_get_image.py::_StubAgents`，只为满足这个
        构造期前置条件，与 event 体系无关。
        """

        name = "stub_agents"

        async def list(self, ctx): return [AgentCapability(id="stub_agents:a", name="a", kind="agent")]
        async def get_template(self, template_id, version, ctx): return None
        async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)

    registry = ProviderRegistry()
    registry.register_capability(_StubAgents())
    rt = CtxWeftRuntime(providers=registry)

    # 行为面的最低限：默认 event_store 得是能用的 providers 版实现。
    assert isinstance(rt.event_store, InMemoryEventStore)

    # 「来源」的真正钉子：core/runtime.py 现在两处都被 re-export 链路环绕
    # （core.events.bus → providers.events.bus、core.state.event_store →
    # providers.events.store 都转发同一个类对象），所以 `is` 身份比较钉不住
    # 「从哪条路径 import」——不管 runtime.py 写 `from ctx_weft.core.events import
    # InProcessEventBus` 还是 `from ctx_weft.providers.events import
    # InProcessEventBus`，运行时拿到的都是同一个类对象，`is` 恒真（已用注入
    # 验证过，见 task-3-report.md 的「注入验证」小节）。故改用 AST 直接读
    # runtime.py 里两条 import 语句的 `module` 字段，钉的是源码里写的路径本身，
    # 不是运行时对象——这不是 Task 1 否决的"扫源码字符串"，是结构化解析。
    import ast
    import inspect

    import ctx_weft.core.runtime as runtime_module

    tree = ast.parse(inspect.getsource(runtime_module))
    # 按名字收集*全部*来源（而非 last-wins 的单值 dict）：若日后 runtime.py 里
    # 又冒出一条从 core 路径 import 同名符号的语句，last-wins 会被后写的正确
    # 语句悄悄盖掉、测试仍然全绿；collect-all 后断言集合，任何一条杂质来源都会
    # 让集合里多出一个不该有的元素，测试转红。
    import_sources: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                if alias.name in ("InProcessEventBus", "InMemoryEventStore"):
                    import_sources.setdefault(alias.name, []).append(node.module)

    assert import_sources.get("InProcessEventBus") == ["ctx_weft.providers.events"], (
        "core/runtime.py 模块级 InProcessEventBus 必须从 ctx_weft.providers.events 导入，"
        f"且只能有这一条来源，实际：{import_sources.get('InProcessEventBus')}"
    )
    assert import_sources.get("InMemoryEventStore") == ["ctx_weft.providers.events"], (
        "core/runtime.py 函数内 InMemoryEventStore 必须从 ctx_weft.providers.events 导入，"
        f"且只能有这一条来源，实际：{import_sources.get('InMemoryEventStore')}"
    )

    # 订阅链路仍要能用：emit 之后 event_store 收得到（自动订阅 bus 后异步投递）。
    await rt._event_bus.emit(Event(
        id="evt_0001", run_id="run_1", sequence=1, session_id="ses_1",
        type="SessionCreated", timestamp=datetime(2026, 8, 28, tzinfo=timezone.utc),
    ))
    # 订阅是异步投递的，给它一次调度机会
    import asyncio
    await asyncio.sleep(0.05)
    assert await rt.event_store.read_by_session("ses_1")


def test_blob_ref_prefix_lives_in_context() -> None:
    """前缀属于内容形态（ImagePart.data 的前缀），不属于 memory，也不属于 events。

    移动的动机是让两个 blob 协议共用它而不互相 import——见 spec §3。
    """
    import ast
    import inspect

    import ctx_weft.protocols.memory as memory_module
    from ctx_weft.protocols import BLOB_REF_PREFIX as PkgPrefix
    from ctx_weft.protocols.context import BLOB_REF_PREFIX
    from ctx_weft.protocols.memory import BLOB_REF_PREFIX as MemPrefix

    assert BLOB_REF_PREFIX == "blob:"
    assert MemPrefix == BLOB_REF_PREFIX

    # `"blob:" is "blob:"` 恒真——短字符串字面量会被 CPython interned，`is` 在这里
    # 分不清「re-export 同一个对象」和「memory.py 自己又写了一份同样的字面量」，
    # 承重的是上面的 `==`。真正有区分力的检查是：memory.py 里不该有一条给
    # `BLOB_REF_PREFIX` 赋值的语句——它必须只是从 context.py import 进来。
    tree = ast.parse(inspect.getsource(memory_module))
    own_definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "BLOB_REF_PREFIX" for t in node.targets)
    ]
    assert not own_definitions, (
        "protocols/memory.py 不该自己定义 BLOB_REF_PREFIX，应只从 context.py re-export"
    )
    assert PkgPrefix is BLOB_REF_PREFIX


def test_memory_blob_store_is_renamed() -> None:
    """两个 blob 协议并存后，`BlobStore` 这个名字不再自明——见 spec §6。"""
    import ctx_weft.protocols as p
    from ctx_weft.protocols import MemoryBlobStore, NullMemoryBlobStore

    assert issubclass(NullMemoryBlobStore, MemoryBlobStore)
    assert NullMemoryBlobStore().can_externalize is False
    assert not hasattr(p, "BlobStore"), "旧名不得残留，否则两个名字并存更糊涂"
    assert not hasattr(p, "NullBlobStore")


def test_registry_methods_are_renamed() -> None:
    from ctx_weft.core.registry import ProviderRegistry

    reg = ProviderRegistry()
    assert hasattr(reg, "register_memory_blob_store")
    assert hasattr(reg, "get_memory_blob_store")
    assert not hasattr(reg, "register_blob_store")
    assert not hasattr(reg, "get_blob_store")
    assert reg.get_memory_blob_store().can_externalize is False


def test_deleted_shims_do_not_come_back() -> None:
    """`core/events/` 与 `core/state/event_store.py` 两个 re-export shim 已删（2026-08-29）。

    它们在 2026-08-27 的三层划界里是刻意留下的过渡层，让 130+ 处既有 import 不必
    同时改。过渡期结束后删除，本用例防的是「有人为了少改一行又把转发层加回来」——
    那会让「一个符号只有一条 import 路径」这条刚建立的性质悄悄失效，且不会有任何
    别的测试变红。
    """
    import importlib

    # 2026-09-11（reliability-wp3，spec: event-commit）有意恢复 `ctx_weft.core.events`
    # **作为编排子包**（仅 commit_gate.py——提交门，方案 §8-WP3 明令「位于编排层」）。
    # 本用例防的仍是 re-export shim 复活：core/events/__init__.py 不得转发任何契约
    # 符号（契约唯一路径 protocols.events），shim 子模块（types/bus）仍然必须不存在。
    for mod in ("ctx_weft.core.events.types",
                "ctx_weft.core.events.bus", "ctx_weft.core.state.event_store"):
        try:
            importlib.import_module(mod)
            raise AssertionError(f"{mod} 应当已删除——re-export shim 禁止复活")
        except ImportError:
            pass
    import ctx_weft.core.events as _ce  # noqa: F401 —— 编排子包本身存在（仅提交门）
    assert not [n for n in getattr(_ce, "__all__", []) or []
                if n not in ()], "core/events 不得 re-export 契约符号"
    _banned = ("Event", "EventBus", "EventStore", "EventType", "RunSnapshot")
    _leaked = [n for n in dir(_ce) if n in _banned]
    assert not _leaked, f"core/events 不得转发契约符号: {_leaked}"
