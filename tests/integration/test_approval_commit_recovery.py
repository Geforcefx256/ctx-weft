"""Hot approval durability, real crashes, and an actual pre-fix journal."""
import asyncio
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from tests.integration.test_user_control_journeys import run_phase

WORKER = Path(__file__).with_name("_approval_commit_worker.py")


async def approval_phase(path, mode, stage, *, crashes=False):
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), mode, stage],
        cwd=WORKER.parents[2], capture_output=True, text=True, timeout=60)
    assert result.returncode == (73 if crashes else 0), result.stdout + result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["before", "unknown", "before_recover", "unknown_recover", "concurrent"])
async def test_hot_rejection_retries_original_decision_before_waking_actor(tmp_path, mode):
    await approval_phase(tmp_path, mode, "hot")
    await run_phase(tmp_path, "approval", "verify", "hot")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["before_crash", "after_crash"])
@pytest.mark.parametrize("crash_during_recovery", [False, True])
async def test_hot_rejection_commit_crashes_recover_without_reviving_approval(tmp_path, mode, crash_during_recovery):
    await approval_phase(tmp_path, mode, "source", crashes=True)
    if crash_during_recovery:
        await approval_phase(tmp_path, "recrash", "recrash", crashes=True)
    await approval_phase(tmp_path, "normal", "continue")
    await run_phase(tmp_path, "approval", "verify", "hot")


@pytest.mark.asyncio
async def test_real_legacy_closed_before_resolved_journal_stays_closed_after_restart(tmp_path):
    # Exact databases captured by the old public hot-rejection journey, including
    # its completed followup task. No events or memory records are synthesized.
    fixture = Path(__file__).with_name("fixtures") / "legacy_hot_rejection.zip"
    with zipfile.ZipFile(fixture) as archive:
        assert set(archive.namelist()) == {"events.db", "memory.db", "receipt.json"}
        for name in archive.namelist():
            (tmp_path / name).write_bytes(archive.read(name))
    await run_phase(tmp_path, "approval", "verify", "hot")
