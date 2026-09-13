"""spec: delivery-acceptance——验收执行的原语：指纹、候选、缺口、输入回合标识。

设计定案（见 design.md D2/D3）：候选 = 分开的 {final_body, final_summary} 二元组
（非 task.outputs 拼接串）；绑定三元组 = 输出指纹 + 输入快照标识 + 声明版本指纹；
有效用户回合标识 = 单独持久化字段 effective_input_turn_id，由四个写点与恢复对账
维护（评审 P1 三/五轮），恢复对账与首次启用初始化共用同一"从 memory 重算并幂等
落盘"原语。
"""

from __future__ import annotations

import hashlib
import json

from ctx_weft.core.acceptance.protocol import (
    AcceptanceRegistry,
    advisory_checks,
    required_checks,
)

#: 维护输入标识事件的充要条件：验收模式非 off 且任务存在检查声明（评审 P2 五轮）。
def maintenance_active(task, mode: str) -> bool:
    return mode != "off" and bool(getattr(task, "acceptance_spec", None))


def candidate_of(task, extra: dict) -> dict:
    """分开的交付二元组：检查、指纹与提交共用同一份（评审 P2-5）。"""
    deliverable = extra.get("final_body") if extra is not None else None
    if deliverable is None:
        deliverable = task.outputs or ""
    return {"final_body": deliverable, "final_summary": (extra or {}).get("final_summary", "") or ""}


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()


def output_fingerprint(candidate: dict) -> str:
    return _sha(candidate)


def input_snapshot_id(task) -> str:
    return _sha({"user_prompt": task.user_prompt, "inputs": getattr(task, "inputs", None),
                 "effective_input_turn_id": getattr(task, "effective_input_turn_id", None)})


def compose_gap_hint(findings: list[dict]) -> str:
    """缺口 → next_step_hint 一次性投递文本（结构化事实，非泛化失败文案）。"""
    lines = []
    for f in findings:
        parts = [p for p in (f.get("field"), f.get("condition"),
                             (f"evidence: {f['evidence_ref']}" if f.get("evidence_ref") else ""),
                             (f"fix: {f['suggestion']}" if f.get("suggestion") else "")) if p]
        lines.append(" - " + "; ".join(parts))
    return ("Acceptance findings (deterministically derived from the current data; "
            "recheck and correct the final deliverable):\n" + "\n".join(lines)) if lines else ""


def findings_equal(a: list[dict], b: list[dict]) -> bool:
    return _sha(a) == _sha(b)


async def run_acceptance_pass(*, registry: AcceptanceRegistry, executor, spec: list,
                              candidate: dict, task, mode: str,
                              unavailable_checkers: set | None = None) -> dict:
    """执行一次验收：必需检查（先结构后领域）+ 提示性检查（只记录）。

    返回记录 dict：{overall, results[], required_failed, required_unverified,
    findings[], fingerprint, input_snapshot, spec_version, attempt}。off 模式不出现在
    此入口（调用方先判 maintenance/required 门禁）；shadow 模式同样执行并返回记录，
    但调用方不得据此改变交付决策（行为矩阵）。
    """
    from ctx_weft.core.acceptance.protocol import acceptance_spec_version

    unavailable = unavailable_checkers or set()
    results: list[dict] = []
    findings: list[dict] = []
    required_failed = required_unverified = False
    for check in required_checks(spec):
        key = (check["checker_id"], check["checker_version"])
        if key in unavailable:
            result = {"checker_id": check["checker_id"], "checker_version": check["checker_version"],
                      "verdict": "error", "findings": [],
                      "error": "checker version unavailable after recovery (judged unverified)"}
        else:
            entry = registry.resolve(check["checker_id"], check["checker_version"])
            r = await executor.run(entry, check["checker_id"], check["checker_version"],
                                   candidate, check["params"])
            result = r.as_dict()
        results.append(result)
        if result["verdict"] == "failed":
            required_failed = True
            findings.extend(result["findings"])
        elif result["verdict"] == "error":
            required_unverified = True
    for check in advisory_checks(spec):
        key = (check["checker_id"], check["checker_version"])
        if key in unavailable:
            results.append({"checker_id": check["checker_id"], "checker_version": check["checker_version"],
                            "verdict": "error", "findings": [],
                            "error": "checker version unavailable after recovery (advisory, recorded only)"})
            continue
        entry = registry.resolve(check["checker_id"], check["checker_version"])
        r = await executor.run(entry, check["checker_id"], check["checker_version"],
                               candidate, check["params"])
        results.append(r.as_dict())
    if required_failed:
        overall = "failed"
    elif required_unverified:
        overall = "unverified"
    else:
        overall = "passed"
    return {"overall": overall, "results": results, "required_failed": required_failed,
            "required_unverified": required_unverified, "findings": findings,
            "fingerprint": output_fingerprint(candidate),
            "input_snapshot": input_snapshot_id(task),
            "spec_version": acceptance_spec_version(spec),
            "attempt": getattr(task, "retry_count", 0)}


async def recompute_effective_turn(memory, provider_ctx, scope) -> "str | None":
    """从 memory 视图重算应有标识：task scope 内最新存活 user_prompt 回合的记录 id。

    被 fold 的回合不在视图里 → 撤销窗口的回退值天然正确。无存活回合返回 None。
    """
    from ctx_weft.protocols.memory import MemoryKind, MemoryScope
    records = await memory.load_view(scope, MemoryScope.TASK, provider_ctx,
                                     kinds=[MemoryKind.CONVERSATION_TURN])
    for record in reversed(records):
        if (record.role or "user") == "user":
            return record.id
    return None


async def reconcile_acceptance_inputs(task_manager, tasks, memory, provider_ctx,
                                      *, session_id: str, root_agent_id: str,
                                      mode: str) -> list[str]:
    """恢复对账原语（design D3；评审 P1 五轮）：重建后、任何验收判定前调用。

    对每个维护中的任务（mode≠off 且有声明）：memory 重算应有标识 ↔ 投影/任务值比对，
    不一致幂等补发 TASK_INPUT_ADVANCED（最后写生效，重复安全）。同一原语覆盖两个
    崩溃窗口：消息已落库/事件未落库；撤销已完成/回退事件未落库。返回补发的 task_id。
    """
    advanced: list[str] = []
    for t in tasks:
        if not maintenance_active(t, mode):
            continue
        from ctx_weft.protocols.memory import MemoryAddress
        should = await recompute_effective_turn(
            memory, provider_ctx,
            MemoryAddress(session_id=session_id, task_id=t.id,
                          agent_id=t.assigned_agent_id or root_agent_id or ""))
        if should != t.effective_input_turn_id:
            await task_manager.advance_input_turn(t.id, should)
            t.effective_input_turn_id = should
            advanced.append(t.id)
    return advanced


def acceptance_event_payload(record: dict) -> dict:
    """TASK_ACCEPTANCE_CHECKED payload：绑定三元组 + 候选引用 + 全部检查结果。"""
    return {"verdict": record["overall"], "results": record["results"],
            "findings": record["findings"],
            "output_fingerprint": record["fingerprint"],
            "input_snapshot_id": record["input_snapshot"],
            "acceptance_spec_version": record["spec_version"],
            "attempt": record["attempt"],
            "repairs_used_after": record.get("repairs_used_after")}
