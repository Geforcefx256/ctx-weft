"""ReconcileStep：resume 后、任何 LLM turn 之前，补完 dangling tool_call。

被 park（HITL）或崩溃中途打断时，最近一个 assistant turn 的部分 tool_call 没有对应
TOOL_RESULT。直接把含 dangling 的消息序列喂给 LLM 会非法报错。本步对账（spec: tool-operations，wp6 改造）：

  完成判定（双通道，spec §5.4 / design D1）：
    done(op_id) = 账本 get(op_id).status == COMPLETED
              或 task view 存在 id == tool_result_record_id(op_id) 的 tool 记录
    ——判据是**逻辑身份**（op_id），与 wire id 无关：call_1 复用不再串扰。

  未完成的 → 按「账本状态 × recovery_policy」分派（design D2）。策略**只有两类**，
  且取自 capability 的活声明（不是账本行上那一列，见 `OperationRecord.recovery_policy`）：

    无账本记录 + 控制工具      → invoke（core 自有的幂等状态迁移，见下方 carve-out）
    无账本记录 + HITL 决定在案 → invoke（崩溃在等人处，provider 从未启动 = 首执安全）
    无账本记录 + 其它          → 作结「无账本记录，无从查证」（不用随机 id 碰副作用工具）
    PREPARED                   → invoke（首执；此前无副作用，授权链自然重查）
    WAITING_HUMAN              → 既有 HITL 恢复路径（不 invoke）
    STARTED + 控制工具         → invoke（同上 carve-out）
    STARTED + idempotent       → invoke（同 op_id 重跑安全）
    STARTED + reviewed         → 交给 gateway 问**重跑授权**（宿主注册的
                                 RerunAuthorizer），它只回答该不该再跑一遍：
                                   allowed=True  → invoke（权威判定没跑成）
                                   allowed=False → 作结：把它的 message 写成工具结果
                                   needs_human   → 既有 HITL park
                                 未注册 → core 代为作结「无从查证」
    COMPLETED                  → 不应到达（完成判定已滤）

  **作结 = 账本 CAS completed + 把 result 经收敛写成 TOOL_RESULT，然后照常续跑。**
  「不重跑」的两种由来——授权方查到了真结果 / 谁也查不到——对 core 是**同一条路**，
  区别全在 result 文本里。结果不确定是一种工具结果，不是一种控制流：不停机、不改
  task 状态、不发专属事件、不设专属错误码或处置 API，agent 下一轮读到它自行决定。
→ next_step="prepare"：assembler 重建出完整 turn，LLM 续跑。
"""

from __future__ import annotations

import logging

from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome
from ctx_weft.core.loop.steps._capabilities import resolve_and_bind
from ctx_weft.core.utils.ids import is_internal_call_id

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

        from ctx_weft.protocols.capability import OperationStatus, tool_result_record_id

        # 鸭子类型的 gateway（测试替身 / 宿主直构）没有这个属性 → 账本旁路。
        ledger = getattr(gateway, "operation_store", None)
        for tc in dangling:
            if ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
                ctx.cancel_token.raise_if_cancelled()
            # 账本键 = 该调用的内部标识：它随 assistant 回合落库，恢复时**读出来**
            # 而不是重算（铸造之前才需要五元组派生）。裸 wire id → None → 账本旁路。
            op_id = tc.get("id", "") if is_internal_call_id(tc.get("id", "")) else ""

            # ── 双通道完成判定（D1）：账本 / 确定性 memory id ─────────────────────
            rec = await ledger.get(op_id, ctx.provider_ctx) if ledger is not None else None
            if (rec is not None and rec.status == OperationStatus.COMPLETED) or (
                    tool_result_record_id(op_id) in tool_record_ids):
                if (
                    rec is not None and rec.status == OperationStatus.COMPLETED
                    and tool_result_record_id(op_id) not in tool_record_ids
                ):
                    # spec: tool-result-recovery——「账本完成而 memory 缺失」（completed 与
                    # TOOL_RESULT 写入之间崩溃）：以账本全文经收敛补写（原执行 id 引用），
                    # 兑现 tool-operations 的补写要求；此前此处直接 continue，dangling 兜底
                    # 会把这次调用的结果整个剥掉。
                    await self._backfill_memory(
                        state, ctx, op_id, tc, rec.result,
                        gateway=gateway, rec=rec, via="ledger-completed")
                logger.info("ReconcileStep: op %s completed — reusing, not re-running", op_id)
                continue

            # ── 未完成的三条出路 ────────────────────────────────────────────
            # started 的重跑判断**不在这里**：gateway 自己从账本读得到 started 记录，
            # 策略与重跑授权的解析也都在它手上（`_authorize_rerun`）。reconcile 只要
            # 把它交出去，由 gateway 决定这次 invoke 到底打不打 provider。
            if rec is not None and rec.status == OperationStatus.WAITING_HUMAN:
                logger.info("ReconcileStep: %s waiting_human — HITL path resumes", op_id)
                continue  # 既有 HITL 恢复路径处理

            if rec is None and not self._safe_first_execution(tc, op_id, ledger, state, ctx):
                # 真·无身份：账本未接线，或存量数据（账本建立之前写入）。不重跑，
                # 作结说明无从查证——不用随机 id 去碰副作用工具。
                await self._conclude_no_record(state, ctx, op_id, tc, gateway)
                continue

            logger.info(
                "ReconcileStep: handing %s (tool %s, ledger=%s) to gateway",
                op_id, tc["name"], rec.status if rec is not None else "no-record")
            await gateway.invoke(
                tool_name=tc["name"],
                arguments=tc.get("input", {}) or {},
                state=state,
                ctx=ctx,
                tool_call_id=tc["id"],
            )

        return StepOutcome(next_step="prepare")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _safe_first_execution(self, tc, op_id, ledger, state, ctx) -> bool:
        """无账本记录时，这次调用是不是「确定还没跑过」——是则首执安全。

        三种成立情形：

        1. **持久账本查不到记录**：gateway 的 prepare 在授权与参数校验**之后**，所以
           「有内部标识 + 账本跨进程活着 + 无记录」只能是崩在 prepare 之前——provider
           确定没被调用过，副作用确定没发生。这比 PREPARED 还早一格。

           判据必须是 `durable`，不能只是「账本非 None」：内存账本重启后一片空白，
           「没跑过」和「跑过但账本随进程没了」长得一模一样（O-T05 的真子进程强退就
           踩在这里——副作用会被执行两次）。
        2. **控制工具**：core 自有的幂等状态迁移——finish/metadata 同身份幂等、
           ask_user 复用既有请求（HITL 决定缓存门控）、delegate 经 gateway 的
           completed 短路防双建（方案 §5.4「单独核验」）。
        3. **冷 HITL 重入**：该 tool_call 的人工决定已在案，说明崩在等人处、
           provider 从未启动。

        都不成立 = 账本未接线，或存量数据（账本建立之前写入）——无从判断，保守作结。
        """
        from ctx_weft.core.capabilities.control_tools import PROVIDER_NAME as _CTL
        from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL

        if op_id and getattr(ledger, "durable", False):
            return True
        if tc["name"].startswith(f"{_CTL}__"):
            return True
        hitl = getattr(ctx, "hitl", None)
        if hitl is None:
            return False
        sid, tcid = getattr(state.session, "id", ""), tc.get("id", "")
        return (hitl.registry.decision_for(sid, tcid, HITL_STAGE_TOOL) is not None
                or hitl.registry.decision_for(sid, tcid, HITL_STAGE_AUTHZ) is not None)

    async def _conclude_no_record(self, state, ctx, op_id, tc, gateway):
        """作结一条无从查证的 dangling：把说明写成这次调用的工具结果，不重跑。

        与 gateway 的 `_conclude_without_rerun` 是同一件事的两个入口——那边处理「账本
        里有 started 记录」，这边处理「压根没有记录」，后者没有可 CAS 的行。两边都不
        发专属事件、不改 task 状态、不停机：**结果不确定是一种工具结果，不是一种
        控制流**。agent 下一轮读到它，在任务上下文里决定怎么办。
        """
        await self._backfill_memory(
            state, ctx, op_id, tc,
            f"[Operation outcome unverified] 工具 {tc['name']} 的这次调用在崩溃前没有"
            f"留下账本记录，无法确定副作用是否已发生。不要重试这次调用。",
            gateway=gateway, via="no-ledger-record")
        logger.info("ReconcileStep: %s concluded without re-running (no ledger record)", op_id)

    async def _backfill_memory(self, state, ctx, op_id, tc, result, *,
                               gateway=None, rec=None, via="ledger-completed"):
        """补写 TOOL_RESULT（spec: tool-result-recovery——补写统一过收敛，禁全文直灌）。

        幂等由 memory 的 id 契约保证（同 id ingest = no-op）。收敛引用沿用账本原执行
        invocation_id（attempts 尾项）；store 逐出后以账本全文重新入库（restore 语义）。
        """
        from ctx_weft.core.loop.capability_gateway import ingest_tool_result
        content = str(result)
        attempts = getattr(rec, "attempts", None)
        ref_inv = attempts[-1] if attempts else tc.get("id", "")
        if gateway is not None:
            try:
                content = await gateway._converge_result(  # noqa: SLF001 —— 与本文件既有 gateway 私有件同口径
                    content, ctx, ref_inv, tc.get("name", ""), spillable=True)
            except Exception:
                logger.exception("ReconcileStep: converge on backfill failed for %s", op_id)
        await ingest_tool_result(
            ctx.memory, ctx.provider_ctx, state.scope,
            tool_call_id=tc.get("id", ""), content=content,
            invocation_id=ref_inv, tool_name=tc.get("name", ""), via=via)


async def _dangling_tool_calls(memory, scope, provider_ctx) -> tuple[list[dict], set[str]]:
    """最近一个 assistant turn 里未完成的 tool_call（按逻辑身份判定）。

    返回 (dangling, tool_record_ids)：前者每项是原 tool_call dict 加 `_record_id` /
    `_ordinal` 两个伴随键（策略不在这里取——见 execute 里读 capability 活声明那一处）；
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
    # 完成判据分两个平面，取决于这个调用有没有内部标识。
    #
    # **内部标识（新数据）**：确定性记录 id 在视图里 = 已完成。刻意**不看 wire id**：
    # call_1 复用是常态，wire 配对会把新调用误判为完成（串扰——正是 H3 判据切换要根治
    # 的形态）。
    #
    # **裸 wire id（存量数据）**：不存在确定性派生（同一个 call_1 可属于任意多个回合），
    # 只能退回 wire 配对，且**限定在锚定回合之后写入的 tool 记录**里找。不这样限定的话
    # 同 id 的旧回合结果会把本回合的调用误判成已完成；不做这个判定的话（此前如此）裸 id
    # 的 dangling 每轮 reconcile 都会被重新作结，对话里每轮多一份结果。跨回合歧义仍在
    # ——那是存量数据的既定形态（spec: conversation-integrity 明列为不在保证范围内），
    # 命中时留痕，不静默。
    tool_records = [r for r in view
                    if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "tool"]
    tool_record_ids = {r.id for r in tool_records}
    anchor_at = next((i for i, r in enumerate(view) if r.id == last_asst.id), -1)
    legacy_paired = {
        r.metadata.get("tool_call_id")
        for r in view[anchor_at + 1:]
        if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "tool"
    } - {None, ""}

    from ctx_weft.protocols.capability import tool_result_record_id
    dangling: list[dict] = []
    for i, tc in enumerate(tool_calls):
        call_id = tc.get("id", "")
        rid = tool_result_record_id(call_id)
        if rid is not None:
            if rid in tool_record_ids:
                continue
        elif call_id in legacy_paired:
            logger.info(
                "dangling check: bare wire id %r paired by metadata (no deterministic "
                "record id available; cross-turn reuse of this id would be ambiguous)",
                call_id)
            continue
        dangling.append(dict(tc, _record_id=last_asst.id, _ordinal=i))
    return dangling, tool_record_ids
