"""SDK 公开面（2026-09-04 spec §10）。

改动前只导出 CtxWeftRuntime / RunHandle / SessionStartParams，agent-centric 的
类型一个都不在顶层，host 必须深挖 ctx_weft.protocols.*。
"""

from __future__ import annotations

import ctx_weft


def test_agent_centric_types_are_exported():
    for name in (
        "CtxWeftRuntime", "TurnHandle", "SessionStartParams",
        "SessionForkResult", "MemorySeedConflictError", "ForkCheckpointUnavailableError",
        "AgentSummary", "AgentDetail", "CompactReceipt",
        "HitlReply", "HitlRequestView",
        "AgentNotFound", "AgentNotRunningError",
        "AgentBusyError", "AgentTerminatedError", "SessionBusyError",
    ):
        assert hasattr(ctx_weft, name), f"{name} 不在顶层导出面上"
        assert name in ctx_weft.__all__, f"{name} 不在 __all__ 里"


def test_run_handle_is_gone():
    """不留 shim（spec §1）。"""
    assert not hasattr(ctx_weft, "RunHandle")


def test_all_entries_are_importable():
    """__all__ 里不能有写错的名字。"""
    for name in ctx_weft.__all__:
        assert hasattr(ctx_weft, name), f"__all__ 里的 {name} 实际不存在"
