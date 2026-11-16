"""Complete plans, accepted approvals, independent questions, and image followups."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("_remaining_user_journey_worker.py")


async def phase(path, scenario, variant, stage, *, crash=False):
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), scenario, variant, stage],
        cwd=WORKER.parents[2], capture_output=True, text=True, timeout=60)
    assert result.returncode == (73 if crash else 0), result.stdout + result.stderr


async def journey(path, scenario, variant, restart):
    await phase(path, scenario, variant, "prepare" if restart else "hot", crash=restart)
    if restart:
        await phase(path, scenario, variant, "continue")
    await phase(path, scenario, variant, "verify")


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["same_success", "same_fail", "cross_success", "cross_fail"])
@pytest.mark.parametrize("restart", [False, True], ids=["same_process", "restart_at_step_two_question"])
async def test_ordered_plan_runs_real_children_and_parent_then_allows_more_work(tmp_path, variant, restart):
    await journey(tmp_path, "plan", variant, restart)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["hot", "cold", "restart", "before", "unknown", "decision_crash", "tool_crash"])
async def test_accepted_modified_approval_executes_once_and_survives_restart(tmp_path, variant):
    await journey(tmp_path, "approval", variant, variant in {"restart", "decision_crash", "tool_crash"})


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["main_first", "side_first", "main_first_hot", "side_first_hot"])
@pytest.mark.parametrize("restart", [False, True], ids=["same_process", "restart_between_answers"])
async def test_independent_agent_questions_keep_answers_and_followups_separate(tmp_path, variant, restart):
    await journey(tmp_path, "ports", variant, restart)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["hot", "restart", "message_crash"])
async def test_image_followup_withdraw_retype_and_restart_preserves_actual_bytes(tmp_path, variant):
    await journey(tmp_path, "image", variant, variant != "hot")
