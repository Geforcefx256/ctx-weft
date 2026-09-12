"""spec: task-handoff——delegate 输入的规整：纯 JSON 校验 + 尺寸分级收敛。

控制工具侧（delegate_task / delegate_plan）的唯一入口 `normalize_task_inputs`：

1. ``None`` 透传（未声明输入）；必须为 dict；
2. JSON 可序列化校验——失败响亮报错（与 push_task 对 part 列表的响亮拒绝同一先例，
   「事件库恒不含字节」的输入侧门卫）；
3. UTF-8 字节计量（``ensure_ascii=False``），超 ``INPUTS_MAX_BYTES`` 走分级收敛：
   a. 长字符串（值**与键**）截断到 ``INPUTS_STRING_MAX`` 字符；
   b. 容器裁剪：数组保前 ``INPUTS_LIST_MAX`` 项 + 丢弃计数哨兵元素、字典按插入序
      保前 ``INPUTS_DICT_MAX`` 键 + ``_dropped_keys`` 计数——纯数字大数组、海量键
      这类「没有字符串可截」的形态靠这一级收敛（探针实测：万元素数字数组
      30,012 字节、九千字符键 9,007 字节）；
   c. 加标记后**复核**：仍超限（如单个巨大数字标量——截数字会伪造数值，不截）→ 拒绝。

截断/裁剪信息集中在顶层 ``_truncated`` 键（保留键，与用户数据撞名时以标记为准），
对模型可见——拿到的可能是规整后的数据这件事不允许静默。
"""

from __future__ import annotations

import json

__all__ = [
    "INPUTS_MAX_BYTES",
    "InvalidTaskInputs",
    "normalize_task_inputs",
]

#: 序列化后的字节上限（UTF-8 计量）。常量可调，不做配置面——先跑起来再看实际分布。
INPUTS_MAX_BYTES = 8 * 1024
#: 单个字符串（值或键）超过此长度即截断。
INPUTS_STRING_MAX = 400
#: 数组保留下限项数（超出部分丢弃并以计数哨兵标注）。
INPUTS_LIST_MAX = 64
#: 字典保留下限键数（超出部分按插入序丢弃并以计数标注）。
INPUTS_DICT_MAX = 32

_TRUNCATED_KEY = "_truncated"
_DROPPED_KEYS = "_dropped_keys"
_LIST_SENTINEL_FMT = "[+{n} items dropped]"


class InvalidTaskInputs(ValueError):
    """inputs 不合格（非 JSON / 无法收敛至上限）。message 面向 LLM，直接进工具回执。"""


def _measure(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _trunc_strings(value):
    """递归截断超长字符串（值与键）。返回 (新值, 截断计数)。"""
    if isinstance(value, str):
        if len(value) > INPUTS_STRING_MAX:
            return value[:INPUTS_STRING_MAX] + "…[truncated]", 1
        return value, 0
    if isinstance(value, list):
        out: list = []
        count = 0
        for item in value:
            new, c = _trunc_strings(item)
            out.append(new)
            count += c
        return out, count
    if isinstance(value, dict):
        out: dict = {}
        count = 0
        for k, v in value.items():
            if isinstance(k, str) and len(k) > INPUTS_STRING_MAX:
                k = k[:INPUTS_STRING_MAX] + "…[truncated]"
                count += 1
            new, c = _trunc_strings(v)
            out[k] = new
            count += c
        return out, count
    return value, 0


def _prune_containers(value):
    """递归裁剪容器规模。返回 (新值, 是否发生任何裁剪)。"""
    if isinstance(value, list):
        pruned = len(value) > INPUTS_LIST_MAX
        out = [_prune_containers(item)[0] for item in value[:INPUTS_LIST_MAX]]
        if pruned:
            out.append(_LIST_SENTINEL_FMT.format(n=len(value) - INPUTS_LIST_MAX))
        return out, pruned
    if isinstance(value, dict):
        pruned = len(value) > INPUTS_DICT_MAX
        items = list(value.items())[:INPUTS_DICT_MAX]
        out: dict = {}
        for k, v in items:
            new, p = _prune_containers(v)
            out[k] = new
            pruned = pruned or p
        if len(value) > INPUTS_DICT_MAX:
            out[_DROPPED_KEYS] = len(value) - INPUTS_DICT_MAX
        return out, pruned
    return value, False


def normalize_task_inputs(raw: dict | None) -> dict | None:
    """规整 delegate 声明的输入；不合格抛 `InvalidTaskInputs`（调用方转工具错误回执）。"""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InvalidTaskInputs(
            "inputs must be a JSON object (dict); got "
            f"{type(raw).__name__}. Re-send with inputs as a JSON object, or omit it."
        )
    try:
        json.dumps(raw, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as e:
        raise InvalidTaskInputs(
            f"inputs must be pure JSON-serializable data (no bytes/objects/cycles; "
            f"error: {e}). Re-send with JSON data only, or omit inputs."
        ) from e

    value, n_strings = _trunc_strings(raw)
    marker: dict[str, object] = {}
    if n_strings:
        marker["truncated_strings"] = n_strings

    if _measure(value) > INPUTS_MAX_BYTES:
        value, pruned = _prune_containers(value)
        if pruned:
            marker["pruned_containers"] = True

    result = dict(value)
    if marker:
        result[_TRUNCATED_KEY] = marker

    size = _measure(result)
    if size > INPUTS_MAX_BYTES:
        raise InvalidTaskInputs(
            f"inputs serialize to {size} bytes, still over the {INPUTS_MAX_BYTES}-byte "
            "limit after truncation/pruning (e.g. a single huge number cannot be safely "
            "truncated). Pass a smaller payload — reference large data via tool results "
            "or memory instead."
        )
    return result
