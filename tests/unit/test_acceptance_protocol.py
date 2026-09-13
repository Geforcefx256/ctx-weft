"""spec: delivery-acceptance——协议、注册表与隔离执行（任务 1.1/1.2）。"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from ctx_weft.core.acceptance.executor import AcceptanceExecutor
from ctx_weft.core.acceptance.protocol import (
    BUILTIN_STRUCTURE_ID,
    AcceptanceFinding,
    AcceptanceRegistry,
    InvalidAcceptanceSpec,
    acceptance_spec_version,
    normalize_acceptance_spec,
    required_checks,
)


def _reg() -> AcceptanceRegistry:
    return AcceptanceRegistry()


# ── 1.1 注册表与声明 ──────────────────────────────────────────────────────────


def test_registry_resolves_by_id_and_version():
    reg = _reg()
    reg.register("acme.total", "1", lambda c, p: ("passed", []))
    assert reg.has("acme.total", "1")
    assert reg.resolve("acme.total", "1")["fn"] is not None
    with pytest.raises(InvalidAcceptanceSpec, match="not registered"):
        reg.resolve("acme.total", "2")  # 同 id 不同版本 = 未注册


def test_registry_rejects_bad_entries():
    reg = _reg()
    with pytest.raises(InvalidAcceptanceSpec):
        reg.register("", "1", lambda c, p: ("passed", []))
    with pytest.raises(InvalidAcceptanceSpec):
        reg.register("x", "1", "not-callable")


def test_normalize_spec_validates_and_orders():
    spec = normalize_acceptance_spec([
        {"checker_id": "acme.total", "checker_version": "1", "params": {"k": 1}, "required": True},
        {"checker_id": BUILTIN_STRUCTURE_ID, "checker_version": "1", "required": True},
    ])
    # 内置结构检查排在前（先结构、后领域）
    assert required_checks(spec)[0]["checker_id"] == BUILTIN_STRUCTURE_ID
    with pytest.raises(InvalidAcceptanceSpec):
        normalize_acceptance_spec([{"checker_id": "x"}])  # 缺版本
    with pytest.raises(InvalidAcceptanceSpec):
        normalize_acceptance_spec("nope")


def test_spec_version_includes_checker_version():
    base = [{"checker_id": "c", "checker_version": "1", "params": {}, "required": True}]
    v1 = acceptance_spec_version(base)
    v2 = acceptance_spec_version([dict(base[0], checker_version="2")])
    v3 = acceptance_spec_version([dict(base[0], params={"a": 1})])
    assert len({v1, v2, v3}) == 3  # 版本/参数任一变化 → 指纹变化


def test_builtin_structure_check():
    from ctx_weft.core.acceptance.protocol import builtin_structure_check
    cand = {"final_body": '{"a": [1], "b": "x"}', "final_summary": "note"}
    assert builtin_structure_check(cand, {"fields": {"a": "array", "b": "string"}})[0] == "passed"
    verdict, findings = builtin_structure_check(
        {"final_body": '{"a": [1]}', "final_summary": "s"}, {"fields": {"a": "array", "b": "string"}})
    assert verdict == "failed" and findings
    # 摘要附着不参与正文结构判定（spec：摘要附着不污染结构检查）
    assert builtin_structure_check({"final_body": '{"a": []}', "final_summary": "not json ["},
                                   {"fields": {"a": "array"}})[0] == "passed"


# ── 1.2 隔离执行 ─────────────────────────────────────────────────────────────


def _ok(candidate, params):
    return ("passed", [])


def _fail(candidate, params):
    return ("failed", [AcceptanceFinding(field="total", condition="sum mismatch",
                                         evidence_ref="rows", suggestion="recompute")])


async def test_executor_pass_fail_raise():
    ex = AcceptanceExecutor()
    reg = _reg()
    reg.register("ok", "1", _ok)
    reg.register("bad", "1", _fail)
    reg.register("boom", "1", lambda c, p: 1 / 0)
    r1 = await ex.run(reg.resolve("ok", "1"), "ok", "1", {}, {})
    assert r1.verdict == "passed"
    r2 = await ex.run(reg.resolve("bad", "1"), "bad", "1", {}, {})
    assert r2.verdict == "failed" and r2.findings[0].field == "total"
    r3 = await ex.run(reg.resolve("boom", "1"), "boom", "1", {}, {})
    assert r3.verdict == "error" and "ZeroDivision" in r3.error  # 异常 → unverified，不 passed


async def test_executor_timeout_degrades_and_late_result_dropped():
    ex = AcceptanceExecutor(timeout_sec=0.05)
    started = threading.Event()

    def _block(candidate, params):
        started.set()
        time.sleep(1.5)  # 远超超时；迟到结果必须被丢弃
        return ("passed", [])

    reg = _reg()
    reg.register("slow", "1", _block)
    r = await ex.run(reg.resolve("slow", "1"), "slow", "1", {}, {})
    assert r.verdict == "error" and "timed out" in r.error
    assert ("slow", "1") in ex.degraded
    # degraded 后续直接 unverified，不再执行
    r2 = await ex.run(reg.resolve("slow", "1"), "slow", "1", {}, {})
    assert r2.verdict == "error" and "degraded" in r2.error
    assert started.is_set() and ex.stats["executed"] == 1


async def test_executor_backpressure_does_not_queue():
    release = threading.Event()

    def _block(candidate, params):
        release.wait(2)
        return ("passed", [])

    ex = AcceptanceExecutor(timeout_sec=5.0, max_in_flight=1)
    reg = _reg()
    reg.register("block", "1", _block)
    reg.register("healthy", "1", _ok)
    t1 = asyncio.create_task(ex.run(reg.resolve("block", "1"), "block", "1", {}, {}))
    await asyncio.sleep(0.05)  # 让第一个占住在飞槽
    r2 = await ex.run(reg.resolve("healthy", "1"), "healthy", "1", {}, {})
    assert r2.verdict == "error" and "cap reached" in r2.error  # 不排队：直接 unverified
    assert ex.stats["backpressure"] == 1
    release.set()
    r1 = await t1
    assert r1.verdict == "passed"  # 健康路径本身不受影响


async def test_executor_healthy_checker_available_after_degraded():
    """阻塞者降级后，健康检查器仍可执行（评审 P2 五轮：不排队拖垮可用性）。"""
    ex = AcceptanceExecutor(timeout_sec=0.05)
    reg = _reg()
    reg.register("blk", "1", lambda c, p: (time.sleep(1.0), ("passed", []))[1])
    reg.register("good", "1", _ok)
    await ex.run(reg.resolve("blk", "1"), "blk", "1", {}, {})  # 超时降级，但线程是独立的
    r = await ex.run(reg.resolve("good", "1"), "good", "1", {}, {})
    assert r.verdict == "passed"


def test_executor_daemon_threads_do_not_block_exit():
    """含阻塞检查器的执行器：daemon 线程不阻止进程退出（子进程级验证）。"""
    import subprocess
    import sys
    code = (
        "import sys, time, threading; sys.path.insert(0, {root!r}); "
        "from ctx_weft.core.acceptance.executor import AcceptanceExecutor; "
        "from ctx_weft.core.acceptance.protocol import AcceptanceRegistry; "
        "import asyncio; "
        "reg = AcceptanceRegistry(); "
        "reg.register('blk', '1', lambda c, p: (time.sleep(5), ('passed', []))[1]); "
        "ex = AcceptanceExecutor(timeout_sec=0.05); "
        "asyncio.run(ex.run(reg.resolve('blk', '1'), 'blk', '1', {{}}, {{}})); "
        "print('done')"
    ).format(root="src")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0 and "done" in proc.stdout  # 阻塞线程随主线程退出被回收
