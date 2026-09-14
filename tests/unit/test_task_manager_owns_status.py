"""task 状态与状态事件只归 TaskManager（Task 4）。

两条静态守卫 + 一组行为测试：
- 守卫 A：loop 侧（steps/ 与 runtime.py）一条 task 状态事件都不许发。
  **按 AST 扫 `EventType.TASK_*` 的属性访问**，而不是按行匹配 `emit(`——真实代码里
  emit 调用经常跨行（`await bus.emit(make_event(\n state, EventType.TASK_CANCELED, ...))`），
  按行匹配会漏掉，守卫就成了摆设。注释里的字面量不进 AST，天然不误报。
- 守卫 B：判决归 loop、状态归 TM——loop 侧不许出现 `task.status` 的赋值。
  **同样扫全树 + 白名单**（不是「列举几个要看的文件」）：开集判据在这道守卫上已经
  出事两次——第一次它认不出裸变量形态（`task.status=`），第二次它只看硬编码的 6 个
  文件，往 `steps/prepare.py` 注入裸赋值照样绿。新文件天然被守到，是这道守卫唯一
  能长久成立的形态。
- 行为：五种 run 结局（含崩溃就地构造那一条）经同一张处置表落到状态 + 事件。
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.runner import AgentBinding, effective_agent_id
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols.events import EventType

TASK_STATUS_EVENTS = (
    "TASK_FINISHED", "TASK_FAILED", "TASK_REQUEUED", "TASK_SUSPENDED",
    "TASK_AWAITING_HUMAN", "TASK_INTERRUPTED", "TASK_CANCELED",
    "TASK_HUMAN_RESOLVED", "TASK_RESUMED",
)
#: 比 `reducers.TASK_STATUS_BY_EVENT` 少一个 `TASK_STARTED`，是刻意的、不是漏记：
#: 下面的判据匹配**任意** `EventType.X` 属性引用，不分「发射」还是「查表读」；
#: 而 `lifecycle/session_registry.py` 里有 `EventType.TASK_STARTED: SessionInput.TASK_STARTED`
#: 这样一条纯读的映射条目，加进这张清单会把它当误报抓出来。TASK_STARTED 自己的
#: 唯一发射点在 `task/manager.py`，不需要这道守卫再管。

#: 锚定本文件位置，**不吃调用 cwd**：写成相对路径 `pathlib.Path("src/ctx_weft")`
#: 时，从别的 cwd 下跑 rglob 会命中 0 个文件 → offenders 恒空 → 守卫报绿。
#: 零扫描即通过，正是本批次在治的那类假绿灯（与 golden `_GOLDEN_DIR`、
#: test_discriminators.py 那次同源）。
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src" / "ctx_weft"

#: 允许出现 `EventType.TASK_*` 属性引用、或 task 状态事件 wire 字符串字面量的
#: 文件：唯一发射者 + 事件表 + 投影映射表 + 处置表。
#: **按仓根相对路径认，不按 basename**——按 basename 放行会连带豁免 `src/` 下任何
#: 同名新文件（再出现一个 `reducers.py` / `events.py` 就白白开了个口子）。
_ALLOWED = {
    "src/ctx_weft/core/orchestrator/task/manager.py",
    "src/ctx_weft/protocols/events.py",
    "src/ctx_weft/core/control/reducers.py",
    "src/ctx_weft/core/orchestrator/task/disposition.py",
    # Task 12：AgentLifecycleManager（ALM）的 `_INPUT_BY_EVENT` / `_SETTLE_REASON` 把全部
    # 11 种 TASK_* 当纯读的映射键——翻译成 AgentInput、驱动 agent 五态机，自己只
    # 发 AGENT_*，不发一条 TASK_* 事件。与上面 TASK_STARTED 被逐出 TASK_STATUS_EVENTS
    # 清单本身是同一类豁免（session_registry.py 那条），只是这里读的类型更多、判据
    # 不区分「发射」与「查表读」，只能靠按文件放行，不能靠收窄清单（清单收窄到
    # 只剩 TASK_STARTED 会连 task_manager.py 自己的发射点都放过）。
    "src/ctx_weft/core/orchestrator/lifecycle/agent_manager.py",
}

#: 上面 `TASK_STATUS_EVENTS` 里每个名字对应的 wire 字符串值（`EventType` 的值），
#: 供字符串形态判据用。
TASK_STATUS_EVENT_VALUES = tuple(EventType[name].value for name in TASK_STATUS_EVENTS)


def test_only_task_manager_emits_task_status_events() -> None:
    """loop 侧（steps/ 与 runtime.py）一条 task 状态事件都不许发。"""
    assert _task_status_event_sites(_SRC) == []


def test_guard_a_scans_a_real_tree_not_an_empty_one() -> None:
    """守卫必须真的扫到文件——零扫描也会报绿，那是假绿灯。"""
    scanned = list(_SRC.rglob("*.py"))
    assert len(scanned) > 50, f"守卫只扫到 {len(scanned)} 个文件，疑似路径解析错误"
    assert (_SRC / "core" / "orchestrator" / "task" / "manager.py").exists(), \
        "_SRC 没指向真的源码树"


def _task_status_event_sites(root: pathlib.Path) -> list[str]:
    """AST 扫描：非白名单文件里所有 `EventType.<TASK_状态事件>` 的出现点。

    比「同一行里同时有 emit 和 EventType.X」强：跨行的 emit 调用、藏在 helper
    里的构造、`payload=` 后面才换行的写法，一个都跑不掉。
    """
    offenders: list[str] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(_REPO_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and node.attr in TASK_STATUS_EVENTS
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "EventType"):
                offenders.append(f"{rel}:{node.lineno}:{node.attr}")
    return offenders


def _task_status_string_sites(root: pathlib.Path) -> list[str]:
    """AST 扫描：非白名单文件里以字符串字面量形式出现的 task 状态事件值（总账 B2）。

    上面 `_task_status_event_sites` 只认 `EventType.<NAME>` 属性访问；
    `task_disposition.py` 自己就用这些事件的 wire 字符串（`"TaskFinished"` 等，
    TM 发射前才 `EventType(...)` 转回来）表达处置结果——模仿这种写法的新模块能
    绕过属性判据。`task_disposition.py` 是处置表本体，正当使用，随 `_ALLOWED`
    放行。
    """
    offenders: list[str] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(_REPO_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in TASK_STATUS_EVENT_VALUES):
                offenders.append(f"{rel}:{node.lineno}:{node.value}")
    return offenders


def test_only_task_manager_emits_task_status_events_as_string_literals() -> None:
    """守卫 A 的字符串形态补充判据：非白名单文件不许以字符串字面量表达 task 状态事件。"""
    assert _task_status_string_sites(_SRC) == []


def test_loop_side_does_not_write_task_status() -> None:
    """判决归 loop，状态归 TM——loop 侧不许出现 task.status 的赋值。

    与守卫 A 同构：**扫全树 + 白名单**。上一版列举 6 个「要看的文件」是开集——往
    任何不在列表里的文件（评审实测：`core/loop/steps/prepare.py`）注入裸
    `task.status = "FINISHED"`，守卫照样绿。
    """
    assert _task_status_write_sites(_SRC) == []


def test_guard_b_scans_a_real_tree_not_an_empty_one() -> None:
    """守卫必须真的扫到文件——零扫描也会报绿，那是假绿灯。"""
    scanned = list(_SRC.rglob("*.py"))
    assert len(scanned) > 50, f"守卫只扫到 {len(scanned)} 个文件，疑似路径解析错误"
    assert (_SRC / "core" / "orchestrator" / "task" / "manager.py").exists(), \
        "_SRC 没指向真的源码树"


#: 整份文件放行的写入者，按**仓根相对路径**认（不按 basename——同名新文件不该白拿豁免）：
#: - `task_manager.py`：task 状态的唯一所有者，本守卫存在的目的就是把写入收进它。
#: - `reducers.py`：投影层，按事件重放 task 状态，不产生判决。
#: - `session_registry.py`：只写 `sess.status`，被判据的**裸变量**形态误伤（`<name>.status`
#:   一律算命中是刻意的——被删掉的那批 loop 侧写入绝大多数长成 `task.status=` / `t.status=`，
#:   收窄判据比列几个白名单文件危险得多）。
_ALLOWED_STATUS_WRITE_FILES: frozenset[str] = frozenset({
    "src/ctx_weft/core/orchestrator/task/manager.py",
    "src/ctx_weft/core/control/reducers.py",
    "src/ctx_weft/core/orchestrator/lifecycle/session_registry.py",
    # 不放 lifecycle/agent_manager.py：它近 700 行、职责杂、随 Task 19/20 还会继续长——文件级
    # 豁免会让守卫对它整体失明。它唯一一处 `.status` 赋值走下面 _ALLOWED_STATUS_WRITES
    # 的函数级豁免（review 2026-09-03，Task 12 修复轮）。
})

#: 判据里精确放过的写入点，按 (path, 所在函数) 认——行号会漂，函数名不会。
#: ⚠️ path 是**绝对路径**（`_task_status_writes` 里比较的是 `path.as_posix()`，
#: 而调用方 `_task_status_write_sites` 传进来的 `p` 来自 `root.rglob(...)`，
#: `root` 是本文件顶部的 `_SRC`——已经是绝对路径，rglob 不会把它转回相对）。
#: 上面 `_ALLOWED_STATUS_WRITE_FILES` 的「按仓根相对路径认」是那张表自己的判据，
#: 两张表判据不同、**不要混用**——照着仓根相对路径的写法在这里写一条，会因为
#: 永远比较不相等而静默豁免失效（测试照样绿，只是没起到豁免的作用）。
#:
#: `runtime._inject_user_reply` 曾在这里就地写 `task.status = "PENDING"` 并自行
#: 发事件（Task 6 之前），豁免是为它开的。Task 6 把状态重置连同事件发射一并收拢进
#: `TaskManager.mark_human_resolved`，该函数自此不再写 task 状态——豁免随之撤销。
#: 撤销后守卫是否仍能抓人，由下面的 `test_removed_exemption_still_catches_a_reinstated_write`
#: 常驻钉住——这张表不空不代表它是摆设，它下面的全树扫描仍然覆盖 runtime.py。
#:
#: Task 12（review 2026-09-03 修复轮）新增一条：`lifecycle/agent_manager.py` 的 `apply_input`
#: 写 `rec.status = tr.status`——agent 五态机状态（spec 3.1），不是 task 状态，判据
#: 认裸变量形态（`<name>.status=`）撞上同一属性名纯属误伤，与 `lifecycle/session_registry.py`
#: 被 `_ALLOWED_STATUS_WRITE_FILES` 放行的原因相同；但 `lifecycle/agent_manager.py` 体量大、
#: 还会随 Task 19/20 继续长，改用这里的函数级豁免——只放 `apply_input` 这一个函数，
#: 该文件里任何其它函数新写一行 `xxx.status = ...` 仍然会被守卫抓到——已用一份
#: 临时补丁手工验证过（review 2026-09-03 修复轮，见 task-12-report.md），未固化
#: 成常驻测试：这条豁免只锁 (path, func) 两个值，`_task_status_writes` 本身逐函数
#: 独立判断（见其内部 `_walk` 按 `func` 传参），新函数不命中这条豁免元组是判据的
#: 结构性质，不依赖额外测试维持。
_ALLOWED_STATUS_WRITES: frozenset[tuple[str, str]] = frozenset({
    ((_SRC / "core" / "orchestrator" / "lifecycle" / "agent_manager.py").as_posix(), "apply_input"),
})


def _task_status_write_sites(root: pathlib.Path) -> list[str]:
    """全树扫描：非白名单文件里所有对 task 状态的赋值。"""
    offenders: list[str] = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(_REPO_ROOT).as_posix()
        if rel in _ALLOWED_STATUS_WRITE_FILES:
            continue
        offenders += [f"{rel}:{lineno}" for lineno in _task_status_writes(p)]
    return offenders


def _task_status_writes(path: pathlib.Path) -> list[int]:
    """AST 扫描：对 task 状态的赋值（含增量赋值）。

    判据是 `<base>.status = ...`，其中 base 既认**裸名字**（`task.status`、
    `t.status`——被删掉的那批写入绝大多数长这样）也认 `<...>.task`
    （`state.task.status` / `ctx.task.status`）。只认后者是本守卫第一版的洞：
    负向对照恰好只注入了被覆盖的那一种形态，于是「碰巧过」。
    读（`if task.status != ...`）不算，只抓写。

    `runtime.py` 里 TM 之外的模块也在此列——`_run_loop` 的 except 链正是被搬走的
    那批写入；确属编排层的写入走 `_ALLOWED_STATUS_WRITES` 精确放行。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []

    def _is_task_status(n: ast.expr) -> bool:
        if not (isinstance(n, ast.Attribute) and n.attr == "status"):
            return False
        base = n.value
        return (isinstance(base, ast.Name)
                or (isinstance(base, ast.Attribute) and base.attr == "task"))

    def _walk(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk(child, child.name)
                continue
            targets: list[ast.expr] = []
            if isinstance(child, ast.Assign):
                targets = list(child.targets)
            elif isinstance(child, (ast.AugAssign, ast.AnnAssign)):
                targets = [child.target]
            if (any(_is_task_status(t) for t in targets)
                    and (path.as_posix(), func) not in _ALLOWED_STATUS_WRITES):
                hits.append(child.lineno)
            _walk(child, func)

    _walk(tree, "<module>")
    return sorted(hits)


# ── 行为：处置表在 TM 落地 ────────────────────────────────────────────────────


class _CapturingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


class _OutcomeRunner:
    """execute 直接交回一个 RunOutcome（或抛异常，走崩溃入口）。"""

    def __init__(self, tm: TaskManager, outcome: RunOutcome | None = None,
                 exc: BaseException | None = None) -> None:
        self._tm = tm
        self._outcome = outcome
        self._exc = exc

    async def assemble(self, task_id: str) -> AgentBinding | None:
        t = self._tm.get_task(task_id)
        if t is None:
            return None
        root = self._tm.session.root_agent_id if self._tm.session else ""
        return AgentBinding(agent_id=effective_agent_id(t, root))

    async def execute(self, binding: AgentBinding, task_id: str) -> RunOutcome | None:
        if self._exc is not None:
            raise self._exc
        return self._outcome


def _setup(bus: _CapturingBus, **task_kw) -> tuple[TaskManager, Task]:
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    t = Task(id="A", session_id="s1", status="PENDING", **task_kw)
    tm.register_task(t)
    return tm, t


async def _run(tm: TaskManager) -> None:
    await tm._run_task("A")


def _of(bus: _CapturingBus, et: EventType):
    return next(e for e in bus.events if e.type == et)


async def test_retry_exhausted_now_decided_by_task_manager() -> None:
    """重试耗尽的降级判断从 FinalizeStep 搬到了 TM，结果必须一模一样。"""
    bus = _CapturingBus()
    tm, t = _setup(bus, retry_count=3, max_retries=3)
    tm.set_runner(_OutcomeRunner(tm, RunOutcome(
        kind=RunOutcomeKind.COMPLETED, verdict="retry", error="stuck")))

    await _run(tm)

    ev = _of(bus, EventType.TASK_FAILED)
    assert ev.payload["error_code"] == "TASK_FAILED_RETRY_EXHAUSTED"
    assert ev.payload["error_message"] == "stuck"
    assert t.status == "FAILED"


async def test_retry_within_budget_writes_back_retry_count() -> None:
    """处置表不 mutate；新的 retry_count 必须由 TM 写回 task，否则预算永不消耗。"""
    bus = _CapturingBus()
    tm, t = _setup(bus, retry_count=1, max_retries=3)
    tm._max_concurrent = 0                       # drain 空转，不再次派发
    tm.set_runner(_OutcomeRunner(tm, RunOutcome(
        kind=RunOutcomeKind.COMPLETED, verdict="retry", summary="again")))

    await _run(tm)

    ev = _of(bus, EventType.TASK_REQUEUED)
    assert ev.payload["retry_count"] == 2
    assert t.retry_count == 2                    # ← 写回
    assert t.status == "PENDING"


async def test_success_finishes_from_task_manager() -> None:
    bus = _CapturingBus()
    tm, t = _setup(bus)
    tm.set_runner(_OutcomeRunner(tm, RunOutcome(
        kind=RunOutcomeKind.COMPLETED, verdict="success", summary="done", outputs="x")))

    await _run(tm)

    ev = _of(bus, EventType.TASK_FINISHED)
    assert ev.payload == {"outcome": "success", "summary": "done", "outputs": "x"}
    assert ev.run_id is None                     # TM 在 run 外面
    assert t.status == "FINISHED"


async def test_awaiting_human_and_suspended_come_from_task_manager() -> None:
    bus = _CapturingBus()
    tm, t = _setup(bus)
    tm.set_runner(_OutcomeRunner(tm, RunOutcome(
        kind=RunOutcomeKind.AWAITING_HUMAN, hitl_id="h1")))
    await _run(tm)
    assert _of(bus, EventType.TASK_AWAITING_HUMAN).payload == {"hitl_id": "h1"}
    assert t.status == "AWAITING_HUMAN"

    bus2 = _CapturingBus()
    tm2, t2 = _setup(bus2)
    tm2.set_runner(_OutcomeRunner(tm2, RunOutcome(
        kind=RunOutcomeKind.SUSPENDED_ON_CHILDREN, summary="s", spawn_titles=("a",))))
    await _run(tm2)
    assert _of(bus2, EventType.TASK_SUSPENDED).payload == {
        "summary": "s", "spawn_titles": ["a"]}
    assert t2.status == "SUSPENDED"


async def test_cancel_comes_from_task_manager_and_payload_is_empty() -> None:
    """取消走正常返回（CancelledError 在 _run_loop 里被吞、不重抛），并入处置表。"""
    bus = _CapturingBus()
    tm, t = _setup(bus)
    tm.set_runner(_OutcomeRunner(tm, RunOutcome(kind=RunOutcomeKind.CANCELED)))

    await _run(tm)

    assert _of(bus, EventType.TASK_CANCELED).payload == {}
    assert t.status == "CANCELED"


async def test_crash_entry_builds_outcome_in_place() -> None:
    """崩溃：`_run_loop` 重抛 → `_run_task` 的 except 就地构造 RunOutcome 喂同一张表。"""

    class _Boom(Exception):
        retriable = False

    bus = _CapturingBus()
    tm, t = _setup(bus, retry_count=0, max_retries=3)
    tm.set_runner(_OutcomeRunner(tm, exc=_Boom("kaboom")))

    await _run(tm)

    ev = _of(bus, EventType.TASK_INTERRUPTED)
    assert ev.payload["reason"] == "run_crash"
    assert ev.payload["error_message"] == "kaboom"
    assert ev.payload["retry_count"] == 0
    assert t.status == "INTERRUPTED"
    assert t.error == "kaboom"


async def test_crash_retriable_requeues_and_consumes_budget() -> None:
    bus = _CapturingBus()
    tm, t = _setup(bus, retry_count=0, max_retries=2)
    tm._max_concurrent = 0
    tm.set_runner(_OutcomeRunner(tm, exc=RuntimeError("transient")))

    await _run(tm)

    ev = _of(bus, EventType.TASK_REQUEUED)
    assert ev.payload == {"reason": "run_crash", "retry_count": 1}
    assert t.retry_count == 1
    assert t.status == "PENDING"


async def test_tm_terminal_status_wins_over_run_outcome() -> None:
    """熔断已把 root 判死（TM 自己写的终态）→ run 的取消结局不得把 FAILED 盖回 CANCELED。"""
    bus = _CapturingBus()
    tm, t = _setup(bus)

    class _TripRunner(_OutcomeRunner):
        async def execute(self, binding, task_id):
            self._tm.get_task(task_id).status = "FAILED"      # 熔断 trip 的先手
            return RunOutcome(kind=RunOutcomeKind.CANCELED)

    tm.set_runner(_TripRunner(tm))
    await _run(tm)

    assert EventType.TASK_CANCELED not in [e.type for e in bus.events]
    assert t.status == "FAILED"


@pytest.mark.parametrize("kind", list(RunOutcomeKind))
def test_every_run_outcome_kind_is_handled(kind: RunOutcomeKind) -> None:
    """处置表对五个结局都给得出状态——新增结局忘了接会在这里红。"""
    from ctx_weft.core.orchestrator.task.disposition import disposition_for

    disp = disposition_for(RunOutcome(kind=kind), retry_count=0, max_retries=3)
    assert disp.status and disp.event_type


def test_removed_exemption_still_catches_a_reinstated_write(tmp_path):
    """撤销豁免不是空集合摆设：真有人把写入加回来，守卫必须报出来。

    Task 6 撤掉了 `runtime._inject_user_reply` 的精确豁免。若判据其实扫不到
    那个文件（或 `_ALLOWED_STATUS_WRITE_FILES` 把它整份放过了），空集合就成了
    一块遮羞布——本测试用一份合成源码钉死「扫得到 + 报得出」。
    """
    pkg = tmp_path / "src" / "ctx_weft" / "core"
    pkg.mkdir(parents=True)
    (pkg / "reinstated.py").write_text(
        'def _cold_reply(task):\n    task.status = "PENDING"\n',
        encoding="utf-8",
    )
    offenders = _task_status_writes(pkg / "reinstated.py")
    assert offenders == [2], f"守卫漏掉了重新加回来的写入: {offenders}"


def test_exemption_table_is_empty_by_design():
    """`_ALLOWED_STATUS_WRITES` 不是想加就能加的摆设——改动它需要一条明确理由，
    这条测试把「当前理由」钉成断言：谁想再加一条，得同时改这里，等于逼着他把
    理由写进 PR。Task 6 之后曾经是空集合；Task 12（review 2026-09-03 修复轮）
    为 `agent_lifecycle_manager.py::apply_input` 开了唯一一条函数级豁免（见上方大段注释），
    不再是空的，但依旧只精确放行这一个 (path, 函数) 组合。
    """
    assert _ALLOWED_STATUS_WRITES == frozenset({
        ((_SRC / "core" / "orchestrator" / "lifecycle" / "agent_manager.py").as_posix(), "apply_input"),
    })
