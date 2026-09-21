"""分层边界：`providers/` 不许反向 import `core` 的域逻辑。

方向是 **core → providers**，不是反过来。`core/runtime.py` 一直在 import providers（它要
把默认实现接上），那是对的；反向那条边一旦出现，就意味着某个领域决定跑进了实现层。

**这条测试是有来历的，不是洁癖。** `SnapshotWriter` 在 `providers/events/snapshot.py` 待过
很久，而它决定的五件事——写在哪个边界、隔多少事件写一张、这一刻写安不安全、切面取在哪、
增量还是全量重锚——全是领域知识。代价是真的出过事：`PROJECTION_VERSION` 曾经在那个文件里
是个独立字面量，与 `reducers._PROJECTION_VERSION` 两处各写一遍。只改一个的后果是**所有**
快照永远判不可用（写侧盖 1、读侧要 2），恢复退回全量回放、writer 每次全量重锚，
O(delta) 变 O(n)，**且不报任何错**——只能从日志里 `mode=anchor` 一直出现才看得出来。

2026-09-20 把那个类整个搬进 `core/control/snapshot_writer.py`，这条边就此清空。刻意**没有**
在 `providers/events/__init__.py` 留兼容 re-export：留了的话这条测试就钉不住，等于白搬。
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "ctx_weft"

#: 允许 providers import 的 core 子包：**纯工具**，不含任何领域决定。
#: 往这里加东西之前先问一句：它是「怎么算」还是「该不该这么做」。后者不属于 providers。
_ALLOWED_CORE_SUBPACKAGES = {
    "ctx_weft.core.utils",          # ids / content / event / estimate —— 纯函数
    "ctx_weft.core.capabilities.schema",  # 工具签名抽取，provider 声明工具时要用
}


def _core_imports_of(path: pathlib.Path) -> "list[str]":
    """该文件 import 了哪些 `ctx_weft.core.*`（含函数内的延迟 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "ctx_weft.core"):
            found.append(node.module or "")
        elif isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.startswith("ctx_weft.core")]
    return found


def _is_allowed(module: str) -> bool:
    return any(module == ok or module.startswith(ok + ".")
               for ok in _ALLOWED_CORE_SUBPACKAGES)


def test_providers_do_not_import_core_domain_logic() -> None:
    offenders: dict[str, list[str]] = {}
    for path in sorted((_SRC / "providers").rglob("*.py")):
        bad = sorted({m for m in _core_imports_of(path) if not _is_allowed(m)})
        if bad:
            offenders[str(path.relative_to(_SRC))] = bad
    assert offenders == {}, (
        "providers 反向 import 了 core 的域逻辑。要么那段逻辑本就属于 core（搬过去），"
        f"要么它是纯工具（加进 _ALLOWED_CORE_SUBPACKAGES 并说明理由）：{offenders}")


def test_the_snapshot_writer_lives_in_core() -> None:
    """具体钉住上面那次搬家的结果，免得有人「顺手」把它搬回去。"""
    import ctx_weft.providers.events as pe
    from ctx_weft.core.control.snapshot_writer import SnapshotWriter

    assert SnapshotWriter.__module__ == "ctx_weft.core.control.snapshot_writer"
    assert not (_SRC / "providers" / "events" / "snapshot.py").exists()
    assert not hasattr(pe, "SnapshotWriter"), (
        "providers.events 又 re-export 了 SnapshotWriter——那条 import 会让上面那条边界"
        "测试重新通不过，或者（更糟）被当成「特例」加进白名单")


def test_attach_persistence_knows_nothing_about_snapshots() -> None:
    """provider 侧的接线函数只接 persister；「接不接快照」是领域决定，归 core。"""
    import inspect

    from ctx_weft.providers.events import attach_persistence

    sig = inspect.signature(attach_persistence)
    assert "snapshot_every_n" not in sig.parameters, sig
    assert "memory_settled" not in sig.parameters, sig
    # 看**函数体**，不看 docstring——那里写着这段历史（为什么它不再认识快照），是该留的。
    src = inspect.getsource(attach_persistence)
    body = src.rsplit('"""', 1)[-1]
    assert "Snapshot" not in body, body
