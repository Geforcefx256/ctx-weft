"""spec: delivery-acceptance——冻结 18 答复重放回归（方案 §5.1 验收条款）。

以 docs 六类验证器作为宿主检查器（仅测试侧注册），重放 live-deepseek-diverse 冻结的
A 组 18 份真实答复：10 份正确不受影响、8 份已知错误全部被拦、0 误报——与实验
replay.json 基线逐项一致。证据文件缺失时跳过（不阻断无文档环境的 CI）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs/plans/verification/live-deepseek-diverse"
GAP_SCRIPT = ROOT / "docs/plans/verification/compare_acceptance_gap.py"

pytestmark = pytest.mark.skipif(
    not FROZEN.exists() or not GAP_SCRIPT.exists(),
    reason="frozen evidence not present in this checkout")


def _load_gap_module():
    sys.path.insert(0, str(GAP_SCRIPT.parent))
    import compare_acceptance_gap as gap  # noqa: N813
    return gap


def test_frozen_replay_ten_untouched_eight_caught_zero_fp():
    gap = _load_gap_module()
    results = json.loads((FROZEN / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in manifest["cases"]}
    rows = [r for r in results["runs"] if r["arm"] == "A_current"]
    assert len(rows) == 18

    passes_untouched = known_caught = false_positives = 0
    for run in rows:
        case = cases[run["case_id"]]
        value = run.get("parsed_output")
        if value is None and isinstance(run["output"], str):
            value = gap.diverse.parse_output(run["output"])
        found = gap.findings(value, case)
        if run["passed"] and not found:
            passes_untouched += 1
        elif (not run["passed"]) and found:
            known_caught += 1
        elif run["passed"] and found:
            false_positives += 1
    assert passes_untouched == 10, "10 份正确答复必须不受影响"
    assert known_caught == 8, "8 份已知业务错误必须全部被拦截"
    assert false_positives == 0, "正确答复零误伤"
