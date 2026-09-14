"""ReconcileStep：resume 后、任何 LLM turn 之前，补完 dangling tool_call。

被 park（HITL）或崩溃中途打断时，最近一个 assistant turn 的部分 tool_call 没有对应
TOOL_RESULT。直接把含 dangling 的消息序列喂给 LLM 会非法报错。本步对账（spec: tool-operations，wp6 改造）：

  完成判定（双通道，spec §5.4 / design D1）：
    done(op_id) = 账本 get(op_id).status == COMPLETED
              或 task view 存在 id == operation_memory_result_id(op_id) 的 tool 记录
    ——判据是**逻辑身份**（op_id），与 wire id 无关：call_1 复用不再串扰。

  未完成的 → 按「状态 × recovery_policy」分派（design D2）。策略只有两类：
    None（存量无身份）        → 作结「无从查证」（不用随机 id 去碰副作用工具）
    PREPARED                  → gateway.invoke（首执；此前无副作用，授权链自然重查）
    STARTED + idempotent      → invoke（同 op_id 重跑安全）
    STARTED + reviewed        → 问裁决者，它只回答**该不该重跑**：
                                  rerun=True   → invoke（权威判定没跑成）
                                  rerun=False  → 作结：把它给的 result 写成工具结果
                                provider 未实现裁决接口 → core 代为作结「无从查证」
    WAITING_HUMAN             → 既有 HITL 恢复路径（不 invoke）
    COMPLETED                 → 不应到达（完成判定已滤）；防御性跳过

  「不重跑」的两种由来——查到了真结果 / 查不到——对 core 是**同一条路**，区别全在
  result 文本里。结果不确定是一种工具结果，不是一种控制流：不停机、不发专属事件、
  不需要宿主介入，agent 下一轮读到它自行决定。

  置 unknown = 账本 CAS unknown + task INTERRUPTED(TOOL_OUTCOME_UNKNOWN)
  + OperationUncertain 事件（revision 供 resolve_operation）+ 短路停止后续 dangling。
→ next_step="prepare"：assembler 重建出完整 turn，LLM 续跑。
"""

from __future__ import annotations

import logging

from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome
from ctx_weft.core.loop.steps._capabilities import resolve_and_bind

logger = logging.getLogger(__name__)


class ReconcileStep(Step):
    name = "reconcile"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        # → "prepare"（非 "act"）：填完 dangling 后须由 PrepareStep 用补齐的 memory 重装 assembled_prompt
        # 并绑定 capability,再 act 调 LLM。直接 "act" 会因缺 assembled_prompt 报错（spec/07 §6）。
        dangling, tool_record_ids = await _dangling_tool_calls(
            ctx.memory, state.scope, ctx.provider_ctx)
        if not dangling:
            logger.info("ReconcileStep: no dangling tool_calls for task %s", state.task.id)
            return StepOutcome(next_step="prepare")

        gateway = ctx.capability_gateway
        if gateway is None:
            raise RuntimeError("ReconcileStep requires a CapabilityGateway")

        # reconcile 跑在 prepare 之前 → 须自行绑定 capability,否则 gateway.invoke 命中空 cache
        # 找不到 dangling 工具（spec/07 §6 端到端缺陷修复）。
        await resolve_and_bind(state, ctx)

        from ctx_weft.protocols.operations import (
            OperationStatus, OperationUpdate, operation_id_for,
            operation_memory_result_id, Adjudication, normalize_recovery_policy,
            RecoveryPolicy,
        )

        for tc in dangling:
            if ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
                ctx.cancel_token.raise_if_cancelled()
            # 同一逻辑调用跨重启同 id：用原回合的 record_id + ordinal 派生（spec: tool-operations）
            op_id = operation_id_for(
                getattr(state.session, "tenant_id", "default"), state.session.id,
                state.agent.id, tc.get("_record_id", ""), tc.get("_ordinal", 0))

            # ── 双通道完成判定（D1）：账本 / 确定性 memory id ─────────────────────
            ledger = getattr(gateway, "_operation_store", None)
            rec = await ledger.get(op_id, ctx.provider_ctx) if ledger is not None else None
            if (rec is not None and rec.status == OperationStatus.COMPLETED) or (
                    operation_memory_result_id(op_id) in tool_record_ids):
                logger.info("ReconcileStep: op %s completed — reusing, not re-running", op_id)
                continue

            # ── 策略分派（D2）：两类——idempotent 重跑 / reviewed 交裁决链 ─────
            # 取值在启动期已由 resolver 校验过，这里再归一一次兜住鸭子类型 cap。
            _cache = getattr(gateway, "_cache", None)
            cap = _cache.get_by_qualified_name(
                state.agent.id, tc["name"], state.task.id) if _cache is not None else None
            policy = normalize_recovery_policy(
                getattr(cap, "recovery_policy", None) if cap is not None else None)
            if rec is None:
                from ctx_weft.core.capabilities.control_tools import PROVIDER_NAME as _CTL
                from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
                hitl = getattr(ctx, "hitl", None)
                from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ
                _sid = getattr(state.session, "id", "")
                _tcid = tc.get("id", "")
                decision_on_record = hitl is not None and (
                    hitl.registry.decision_for(_sid, _tcid, HITL_STAGE_TOOL) is not None
                    or hitl.registry.decision_for(_sid, _tcid, HITL_STAGE_AUTHZ) is not None
                )
                if tc["name"].startswith(f"{_CTL}__"):
                    # 控制工具（方案 §5.4「单独核验」）：core 自有的幂等状态迁移——
                    # finish/metadata 同身份幂等、ask_user 复用既有请求（HITL 决定缓存
                    # 门控）、delegate 经 gateway completed 短路防双建。无账本记录按
                    # 首执处理（PREPARED 语义），不进 unknown。
                    pass
                elif decision_on_record:
                    # 冷 HITL 重入：该 tool_call 的人工决定已在案（crash 发生在等待处、
                    # provider 从未启动）= 「prepared 且从未进入 started」——首执安全。
                    pass
                else:
                    # 存量无身份（WP5 之前的数据）或账本未接线的外部副作用工具：
                    # 不重跑，作结说明无从查证——不用随机 id 去碰副作用工具。
                    await self._conclude_unverified(
                        state, ctx, op_id, tc, ledger, rec,
                        f"[Operation outcome unverified] 工具 {tc['name']} 的这次调用在"
                        f"崩溃前没有留下账本记录，无法确定副作用是否已发生。不要重试这次调用。")
                    continue
            if rec is None:
                pass  # 控制工具 / 冷 HITL 首执 carve-out（上方已分流其它去作结）
            elif rec.status == OperationStatus.WAITING_HUMAN:
                logger.info("ReconcileStep: op %s waiting_human — HITL path resumes", op_id)
                continue  # 既有 HITL 恢复路径处理
            elif rec.status == OperationStatus.PREPARED:
                pass  # 首次执行（此前无副作用）
            elif rec.status == OperationStatus.STARTED:
                from ctx_weft.core.capabilities.control_tools import PROVIDER_NAME as _CTL2
                if tc["name"].startswith(f"{_CTL2}__"):
                    pass  # 控制工具 started：幂等状态迁移（同上 carve-out 理由）
                elif policy == RecoveryPolicy.IDEMPOTENT:
                    pass  # 重跑安全 → 同 op_id 重试
                else:  # RecoveryPolicy.REVIEWED —— core 绝不自行决定
                    verdict = await self._adjudicate(gateway, op_id, tc, state, ctx)
                    if not verdict.rerun:
                        # 作结：裁决者查到了真结果、或只能说明查不到——对 core 同一件事。
                        await self._conclude_unverified(
                            state, ctx, op_id, tc, ledger, rec, verdict.result)
                        continue
                    # 权威判定没跑成 → 同 op_id 重跑安全

            logger.info(
                "ReconcileStep: executing op %s (tool %s, ledger=%s, policy=%s)",
                op_id, tc["name"],
                rec.status if rec is not None else "no-record", policy)
            ctx.provider_ctx.operation_id = op_id
            await gateway.invoke(
                tool_name=tc["name"],
                arguments=tc.get("input", {}) or {},
                state=state,
                ctx=ctx,
                tool_call_id=tc["id"],
            )

        return StepOutcome(next_step="prepare")

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _adjudicate(self, gateway, op_id, tc, state, ctx):
        """裁决链第一环：provider 若实现 OperationAdjudicator 就交它裁。

        **恒返回 Adjudication**：provider 找不到、未实现接口、或裁决抛错时，core 代为
        作结「无从查证」——同样不重跑。没有裁决者不是错误配置而是默认形态：绝大多数
        provider 无从查证外部真值。

        这里是 core 唯一还替人组织措辞的地方，但没有别的选择；它说的也是实话——
        「没人能查证」，而不是「查了查不到」。
        """
        from ctx_weft.protocols.operations import Adjudication, OperationAdjudicator
        _cache = getattr(gateway, "_cache", None)
        cap = _cache.get_by_qualified_name(
            state.agent.id, tc["name"], state.task.id) if _cache is not None else None
        _find = getattr(gateway, "_find_provider", None)
        provider = _find(cap.id) if (cap is not None and _find is not None) else None
        unverified = Adjudication.conclude(
            f"[Operation outcome unverified] 工具 {tc['name']} 在执行中被中断，"
            f"provider 未提供结果查证能力，无法确定副作用是否已发生。"
            f"operation_id: {op_id}。不要重试这次调用。")
        if provider is None or not isinstance(provider, OperationAdjudicator):
            logger.info("ReconcileStep: op %s has no adjudicator — concluding unverified", op_id)
            return unverified
        try:
            return await provider.adjudicate(op_id, ctx.provider_ctx)
        except Exception:
            logger.exception(
                "ReconcileStep: adjudicate failed for %s — concluding unverified", op_id)
            return unverified

    async def _conclude_unverified(self, state, ctx, op_id, tc, ledger, rec, result):
        """作结一次不重跑的操作：账本 completed + 把 ``result`` 写成这次调用的工具结果。

        与「裁决者查到了真结果」走的是**同一条路**——对 core 而言两者没有区别，差的
        只是 result 的内容。不发专属事件、不改 task 状态、不停机：**结果不确定是一种
        工具结果，不是一种控制流**。agent 下一轮读到它，在任务上下文里决定怎么办。
        """
        from ctx_weft.protocols.operations import OperationStatus, OperationUpdate

        if ledger is not None and rec is not None:
            try:
                await ledger.compare_and_set(
                    rec.operation_id, rec.revision,
                    OperationUpdate(status=OperationStatus.COMPLETED,
                                    result=result, result_set=True),
                    ctx.provider_ctx)
            except Exception:
                # 账本写不进去不该拦住「把结果告诉 agent」——后者才是这一步的产出。
                logger.exception("ReconcileStep: ledger conclude failed for %s", op_id)
        await self._backfill_memory(state, ctx, op_id, result, tc)
        logger.info("ReconcileStep: op %s concluded without re-running", op_id)

    async def _backfill_memory(self, state, ctx, op_id, result, tc):
        """query completed：按确定性 id 补写 TOOL_RESULT。幂等由 memory 的 id 契约保证
        （同 id ingest = no-op，见 InMemoryMemoryProvider.ingest）。"""
        from ctx_weft.protocols import MemoryEvent, MemoryKind, MemoryScope
        from ctx_weft.protocols.operations import operation_memory_result_id
        from ctx_weft.core.utils.clock import now_utc
        await ctx.memory.ingest(MemoryEvent(
            id=operation_memory_result_id(op_id),
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope, content=str(result), timestamp=now_utc(),
            role="tool",
            metadata={"tool_call_id": tc.get("id"), "operation_id": op_id,
                      "recovered_via": "adjudicator"},
        ), ctx.provider_ctx)


async def _dangling_tool_calls(memory, scope, provider_ctx) -> tuple[list[dict], set[str]]:
    """最近一个 assistant turn 里未完成的 tool_call（按逻辑身份判定）。

    返回 (dangling, tool_record_ids)：前者带 _record_id/_ordinal/_recovery_policy，
    后者是 task view 里全部 tool 记录的 id 集合（memory 通道完成判据的输入）。
    """
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope

    view = await memory.load_view(
        MemoryAddress(session_id=scope.session_id, task_id=scope.task_id,
                      agent_id=scope.agent_id),
        MemoryScope.TASK, provider_ctx,
    )
    # 升序视图："最近一个 assistant turn" = 末条 role=assistant 的 CONVERSATION_TURN。
    # 必须按 kind 排除 SUMMARY——task 层段摘要 role 同为 assistant（自述体），会被误认。
    last_asst = next(
        (r for r in reversed(view)
         if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "assistant"),
        None,
    )
    if last_asst is None:
        return [], set()
    tool_calls = last_asst.metadata.get("tool_calls") or []
    if not tool_calls:
        return [], set()
    # tool 记录集合（双通道之一：确定性 id 存在 = 已完成）。刻意**不看 wire id**：
    # call_1 复用是常态，wire 配对会把新调用误判为完成（串扰——正是 H3 判据切换要
    # 根治的形态）。旧数据的 tool 记录（无确定性 id）经 wire 配对曾判 done 的场景，
    # 由账本通道（COMPLETED）或 unknown 分支兜底。
    tool_records = [r for r in view
                    if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "tool"]
    tool_record_ids = {r.id for r in tool_records}

    from ctx_weft.protocols.operations import operation_id_for, operation_memory_result_id
    dangling: list[dict] = []
    for i, tc in enumerate(tool_calls):
        op_id = operation_id_for(
            getattr(provider_ctx, "tenant_id", "default"),
            scope.session_id, scope.agent_id or "", last_asst.id, i)
        if operation_memory_result_id(op_id) in tool_record_ids:
            continue
        dangling.append(dict(tc, _record_id=last_asst.id, _ordinal=i,
                             _op_id=op_id))
    return dangling, tool_record_ids
