"""spec: task-handoff——delegate 输入规整的分级收敛（探针对齐的边界数据）。"""

from __future__ import annotations

import pytest

from ctx_weft.core.capabilities.task_inputs import (
    INPUTS_MAX_BYTES,
    InvalidTaskInputs,
    _measure,
    normalize_task_inputs,
)


def test_valid_inputs_pass_through_unchanged():
    raw = {"file": "a.csv", "mode": "strict", "retries": 2}
    assert normalize_task_inputs(raw) == raw


def test_none_passes_through():
    assert normalize_task_inputs(None) is None


def test_non_dict_rejected():
    with pytest.raises(InvalidTaskInputs, match="JSON object"):
        normalize_task_inputs(["a", "b"])  # type: ignore[arg-type]


def test_unserializable_rejected():
    with pytest.raises(InvalidTaskInputs, match="JSON-serializable"):
        normalize_task_inputs({"cb": object()})


def test_big_numeric_array_converges():
    """探针实测形态：10,000 元素数字数组 → 30,012 字节，无字符串可截，靠容器裁剪收敛。"""
    raw = {"values": list(range(10_000))}
    out = normalize_task_inputs(raw)
    assert _measure(out) <= INPUTS_MAX_BYTES
    arr = out["values"]
    # 前 64 项保留 + 丢弃计数哨兵
    assert arr[:5] == [0, 1, 2, 3, 4]
    assert any(isinstance(x, str) and "dropped" in x for x in arr)
    assert "_truncated" in out and out["_truncated"].get("pruned_containers") is True


def test_huge_key_converges():
    """探针实测形态：9,000 字符的键 → 9,007 字节，键截断收敛。"""
    raw = {"K" * 9_000: 1}
    out = normalize_task_inputs(raw)
    assert _measure(out) <= INPUTS_MAX_BYTES
    key = next(k for k in out if k != "_truncated")
    assert len(key) < 500
    assert key.endswith("…[truncated]")
    assert out["_truncated"]["truncated_strings"] == 1


def test_long_string_value_truncated_with_marker():
    raw = {"note": "x" * 5_000}
    out = normalize_task_inputs(raw)
    assert _measure(out) <= INPUTS_MAX_BYTES
    assert out["note"].endswith("…[truncated]")
    assert out["_truncated"]["truncated_strings"] == 1


def test_huge_scalar_rejected():
    """巨型数字标量：截数字=伪造数值，不截 → 响亮拒绝（序列化检查或复核任一关拦下）。"""
    raw = {"n": 10**20_000}
    with pytest.raises(InvalidTaskInputs):
        normalize_task_inputs(raw)


def test_cyclic_structure_rejected():
    cyc: list = []
    cyc.append(cyc)
    with pytest.raises(InvalidTaskInputs):
        normalize_task_inputs({"c": cyc})
