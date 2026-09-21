"""三层各发各的：HITL→task，loop→run，TM→会话级信号（Task 6）。"""

from __future__ import annotations

import pathlib
import re


def test_no_component_emits_session_status_changed_any_more():
    """SessionStatusChanged 进 L 档：只读存量，不得再发射
    （docs/events-v2.md §5.2、§6 不变式 6）。"""
    src = pathlib.Path("src/ctx_weft")
    offenders = [
        str(p) for p in src.rglob("*.py")
        if "SESSION_STATUS_CHANGED" in p.read_text(encoding="utf-8")
        and p.name not in ("reducers.py", "events.py")
    ]
    assert offenders == []


def test_no_component_dispatches_on_a_task_suspended_reason():
    """判据只能是类型，不能是 payload 里的字符串（Global Constraints）。

    **测的是「分流」，不是「出现」**：约束的原话是「不许拿这两个字面量做路由」，
    而同一份 Global Constraints 另一条明说 `reason` 字段只作溯源、不作路由——
    发射端把它写进 payload 是显式允许的（`RunInterrupted{reason:"run_crash"}`
    还是对外契约的一部分）。故只在字面量与比较/成员运算符**同现的那一行**上告警。
    reducers 读存量事件时可以认这两个字符串；别处不许拿它们做路由。
    """
    src = pathlib.Path("src/ctx_weft")
    routing = re.compile(r"==|!=|\bnot in\b|\bin\b|\bmatch\b|\bcase\b")
    offenders = []
    for p in src.rglob("*.py"):
        if p.name == "reducers.py":       # 读存量日志，允许认这两个旧值
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for needle in ('"hitl_park"', '"run_crash"'):
                if needle in line and routing.search(line):
                    offenders.append(f"{p}:{lineno}:{needle}")
    assert offenders == []
