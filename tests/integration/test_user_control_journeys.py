"""Complete user operations, real process restarts and subsequent work."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("_user_control_journey_worker.py")


async def run_phase(path, scenario, stage, variant):
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), scenario, stage, variant],
        cwd=WORKER.parents[2], capture_output=True, text=True, timeout=60)
    assert result.returncode == (73 if stage == "prepare" else 0), result.stdout + result.stderr


async def journey(path, scenario, variant, restart):
    if restart:
        await run_phase(path, scenario, "prepare", variant)
        await run_phase(path, scenario, "continue", variant)
    else:
        await run_phase(path, scenario, "hot", variant)
    await run_phase(path, scenario, "verify", variant)


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["act", "observe"])
@pytest.mark.parametrize("restart", [False, True], ids=["same_process", "restart_before_resume"])
async def test_pause_continue_preserves_real_child_question_and_allows_next_tasks(tmp_path, pause_at, restart):
    await journey(tmp_path, "resume", pause_at, restart)


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["hot", "cold", "restart"])
async def test_rejected_approval_reaches_model_without_running_tool_and_allows_next_tasks(tmp_path, delivery):
    await journey(tmp_path, "approval", delivery, delivery == "restart")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["agent", "task"])
@pytest.mark.parametrize("restart", [False, True], ids=["same_process", "restart_after_compact"])
async def test_manual_compact_preserves_real_decisions_across_restart_and_next_tasks(tmp_path, scope, restart):
    await journey(tmp_path, "compact", scope, restart)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["agent", "session"])
@pytest.mark.parametrize("restart", [False, True], ids=["same_process", "restart_before_resume"])
async def test_model_switch_after_pause_preserves_report_context_and_allows_next_tasks(tmp_path, scope, restart):
    await journey(tmp_path, "model", scope, restart)
