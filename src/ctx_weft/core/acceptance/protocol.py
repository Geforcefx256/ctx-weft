"""spec: delivery-acceptance——检查协议与注册表。

内核只提供协议、流程挂点与内置结构检查；业务规则一律由宿主按 (id, version)
注册（方案 §3/§7：内核不从任务名称猜业务规则）。声明固定检查器版本（评审 P2
三轮），版本纳入声明指纹——同 id 升版即指纹变化，旧验收结论随之失效。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

#: 内核内置结构检查器（非业务规则：只验证交付正文的顶层形状）。
BUILTIN_STRUCTURE_ID = "builtin.structure"
BUILTIN_STRUCTURE_VERSION = "1"

#: 声明条目合法键（task.acceptance_spec 的每项）。
SPEC_KEYS = {"checker_id", "checker_version", "params", "required"}


class InvalidAcceptanceSpec(ValueError):
    """声明不合格（缺键/类型错/引用未知检查器）。message 面向调用方，直接回显。"""


@dataclass(frozen=True)
class AcceptanceFinding:
    """结构化缺口：失败条件、涉及字段、数据依据、可修正建议。"""

    field: str = ""
    condition: str = ""
    evidence_ref: str = ""
    suggestion: str = ""

    def as_dict(self) -> dict:
        return {"field": self.field, "condition": self.condition,
                "evidence_ref": self.evidence_ref, "suggestion": self.suggestion}


@dataclass(frozen=True)
class AcceptanceCheckResult:
    """单检查器一次执行的结论。verdict 三态：passed / failed / error。

    error（检查器异常/超时/degraded/背压）→ 验收 unverified，绝不判 passed。
    """

    checker_id: str
    checker_version: str
    verdict: str
    findings: tuple[AcceptanceFinding, ...] = ()
    error: str = ""

    def as_dict(self) -> dict:
        return {"checker_id": self.checker_id, "checker_version": self.checker_version,
                "verdict": self.verdict, "findings": [f.as_dict() for f in self.findings],
                "error": self.error}


def builtin_structure_check(candidate: dict, params: dict) -> tuple[str, list[AcceptanceFinding]]:
    """内置结构检查：final_body 按声明的顶层字段集判形（先结构、后领域的前半段）。

    params: {"fields": {name: "array"|"object"|"string"|"integer"}}。正文非 JSON 对象、
    字段集不符或类型不符即 failed，缺口逐字段列出。
    """
    fields = params.get("fields") or {}
    types = {"array": list, "object": dict, "string": str, "integer": int}
    body = candidate.get("final_body", "")
    try:
        value = json.loads(body) if isinstance(body, str) else body
    except (TypeError, ValueError):
        return "failed", [AcceptanceFinding(
            field="final_body", condition="must be a JSON object",
            evidence_ref="final_body", suggestion="Return one JSON object as the final body.")]
    if not isinstance(value, dict):
        return "failed", [AcceptanceFinding(
            field="final_body", condition="must be a JSON object",
            evidence_ref=type(value).__name__, suggestion="Return one JSON object as the final body.")]
    findings: list[AcceptanceFinding] = []
    if set(value) != set(fields):
        findings.append(AcceptanceFinding(
            field="final_body", condition="exactly these top-level fields: " + ", ".join(sorted(fields)),
            evidence_ref="got " + ", ".join(sorted(value)),
            suggestion="Return exactly the requested fields."))
    for name, typename in fields.items():
        want = types.get(typename)
        if want is not None and type(value.get(name)) is not want:
            findings.append(AcceptanceFinding(
                field=name, condition=f"must be {typename}",
                evidence_ref=f"got {type(value.get(name)).__name__}",
                suggestion=f"Fix the type of {name}."))
    return ("failed", findings) if findings else ("passed", [])


class AcceptanceRegistry:
    """宿主检查器注册表：(id, version) 双键；派发前校验；恢复缺版本由分档处置。"""

    def __init__(self) -> None:
        self._checkers: dict[tuple[str, str], dict] = {}
        # 内置结构检查器先注册（每张表都有；非业务规则）。
        self.register(BUILTIN_STRUCTURE_ID, BUILTIN_STRUCTURE_VERSION, builtin_structure_check,
                      expected_max_ms=1000, builtin=True)

    def register(self, checker_id: str, version: str,
                 fn: Callable[[dict, dict], tuple[str, list[AcceptanceFinding]]],
                 *, expected_max_ms: int = 5000, builtin: bool = False) -> None:
        if not isinstance(checker_id, str) or not checker_id or not isinstance(version, str) or not version:
            raise InvalidAcceptanceSpec("checker_id/checker_version must be non-empty strings")
        if not callable(fn):
            raise InvalidAcceptanceSpec("checker must be callable(candidate, params)")
        self._checkers[(checker_id, version)] = {"fn": fn, "expected_max_ms": expected_max_ms,
                                                 "builtin": builtin}

    def has(self, checker_id: str, version: str) -> bool:
        return (checker_id, version) in self._checkers

    def resolve(self, checker_id: str, version: str):
        entry = self._checkers.get((checker_id, version))
        if entry is None:
            raise InvalidAcceptanceSpec(
                f"checker ({checker_id!r}, {version!r}) is not registered; "
                "register it on the AcceptanceRegistry or fix the declaration")
        return entry


def normalize_acceptance_spec(raw: Any) -> list[dict]:
    """校验并规整任务声明：[{checker_id, checker_version, params, required}]。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InvalidAcceptanceSpec("acceptance_spec must be a list of check declarations")
    out: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) - SPEC_KEYS or not isinstance(item.get("checker_id"), str) \
                or not isinstance(item.get("checker_version"), str) or not item["checker_id"] \
                or not item["checker_version"]:
            raise InvalidAcceptanceSpec(
                f"acceptance_spec[{i}] must be an object with exactly {sorted(SPEC_KEYS)}; "
                "checker_id/checker_version are required non-empty strings")
        params = item.get("params")
        if params is not None and not isinstance(params, dict):
            raise InvalidAcceptanceSpec(f"acceptance_spec[{i}].params must be an object")
        out.append({"checker_id": item["checker_id"], "checker_version": item["checker_version"],
                    "params": params or {}, "required": bool(item.get("required", False))})
    return out


def acceptance_spec_version(spec: list[dict] | None) -> str:
    """声明指纹：覆盖 checker_id、checker_version、params、required（含版本，评审 P2 三轮）。"""
    canonical = json.dumps(spec or [], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def required_checks(spec: list[dict]) -> list[dict]:
    """执行顺序：内置结构检查在前（先结构、后领域），其余按声明顺序。"""
    items = [c for c in (spec or []) if c.get("required")]
    items.sort(key=lambda c: 0 if c["checker_id"] == BUILTIN_STRUCTURE_ID else 1)
    return items


def advisory_checks(spec: list[dict]) -> list[dict]:
    return [c for c in (spec or []) if not c.get("required")]
