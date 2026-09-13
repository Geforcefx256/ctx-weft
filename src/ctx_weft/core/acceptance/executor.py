"""spec: delivery-acceptance——检查器的隔离执行（评审 P2 四/五轮定案）。

每次检查在独立 **daemon 线程**执行（不阻止进程退出）；在飞线程数硬上限
（达限对新检查直接判 unverified，**不排队**——单个阻塞检查者不得拖垮健康检查器
的可用性）；超时判 unverified 并把该 (id, version) 标记 degraded（后续调用直接
unverified、不再执行）；迟到结果丢弃不采用。契约如实收窄：不承诺回收被阻塞的
检查线程（要求检查器时间有界，注册时申报 expected_max_ms）；泄漏计数进入指标。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from ctx_weft.core.acceptance.protocol import AcceptanceCheckResult, AcceptanceFinding

#: 单检查超时上限（秒）。常量起步；检查器注册时的 expected_max_ms 仅供申报与观测。
CHECK_TIMEOUT_SEC = 5.0
#: 在飞检查线程硬上限：达限即背压（新检查判 unverified，不排队）。
MAX_IN_FLIGHT = 2


class AcceptanceExecutor:
    def __init__(self, *, timeout_sec: float = CHECK_TIMEOUT_SEC, max_in_flight: int = MAX_IN_FLIGHT):
        self._timeout = timeout_sec
        self._max_in_flight = max_in_flight
        self._in_flight = 0
        self._lock = threading.Lock()
        #: degraded 的 (id, version) 集合：熔断后不再执行，直接 unverified。
        self.degraded: set[tuple[str, str]] = set()
        #: 观测指标：degraded 次数 / 背压次数 / 超时（泄漏）次数。
        self.stats = {"degraded": 0, "backpressure": 0, "timed_out": 0, "executed": 0}

    def _unverified(self, cid: str, ver: str, reason: str) -> AcceptanceCheckResult:
        return AcceptanceCheckResult(checker_id=cid, checker_version=ver,
                                     verdict="error", error=reason)

    async def run(self, entry: dict, checker_id: str, checker_version: str,
                  candidate: dict, params: dict) -> AcceptanceCheckResult:
        key = (checker_id, checker_version)
        if key in self.degraded:
            self.stats["degraded"] += 1
            return self._unverified(checker_id, checker_version, "checker degraded (previously timed out)")
        with self._lock:
            if self._in_flight >= self._max_in_flight:
                self.stats["backpressure"] += 1
                return self._unverified(checker_id, checker_version,
                                        "in-flight checker thread cap reached; check judged unverified")
            self._in_flight += 1
        started = False
        try:
            fut: concurrent.futures.Future = concurrent.futures.Future()

            def _worker() -> None:
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(entry["fn"](candidate, params))
                    except BaseException as exc:  # noqa: BLE001 - 宿主检查器任意异常都兜住
                        fut.set_exception(exc)

            thread = threading.Thread(target=_worker, daemon=True,
                                      name=f"acceptance-{checker_id}")
            started = True
            thread.start()
            self.stats["executed"] += 1
            try:
                verdict, findings = await asyncio.wait_for(asyncio.wrap_future(fut), self._timeout)
            except asyncio.TimeoutError:
                self.stats["timed_out"] += 1
                self.degraded.add(key)
                return self._unverified(checker_id, checker_version,
                                        f"checker timed out after {self._timeout}s (marked degraded)")
            except Exception as exc:  # 检查器抛异常 → unverified，不判 passed
                return self._unverified(checker_id, checker_version,
                                        f"checker raised {type(exc).__name__}: {exc}")
            if not isinstance(verdict, str) or verdict not in ("passed", "failed"):
                return self._unverified(checker_id, checker_version,
                                        "checker returned invalid verdict (want 'passed'/'failed')")
            normalized = tuple(
                f if isinstance(f, AcceptanceFinding) else AcceptanceFinding(**f)
                for f in (findings or []))
            return AcceptanceCheckResult(checker_id=checker_id, checker_version=checker_version,
                                         verdict=verdict, findings=normalized)
        finally:
            if started:
                with self._lock:
                    self._in_flight -= 1
