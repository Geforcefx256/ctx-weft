# Proposal: reliability-remaining-defects

> 上游方案：`docs/plans/2026-09-11-agent-core-reliability-plan.md`（WP3–WP9）。
> ⚠️ **本文是 2026-09-11 立项时的跟踪面，其后多处被推翻，不可当作当前契约读。**
> 当前契约以 `openspec/specs/` 下的 capability spec 为准；被推翻的清单见
> `docs/plans/2026-09-11-agent-core-reliability-plan.md` 顶部的状态表。要点三条：
> WP5 的 `operation_id` 五元组派生已合并进摄入点铸造的 tool_call 标识；
> WP6 的四值策略收成两值、`TOOL_OUTCOME_UNKNOWN` / `resolve_operation` 一整套处置面已删除；
> WP7 的 `ExecutionLimits`（本文记分板标为「H5 ✅」）已整体撤销，H5 现状是**未解决且不打算解决**。

> **角色：总纲/跟踪面（2026-09-11 裁定）**——实施按 WP 拆子 change（一会话一个，如 reliability-wp0-wp1 / reliability-wp2 的节奏），本 change 不直接 apply；每 WP 子 change 归档时回到本表勾对应任务组，tasks 组号即子 change 的映射单位。
> 子 change 映射：WP2→reliability-wp2（已建）；WP3→reliability-wp3；WP4→reliability-wp4（WP3+WP4 合并宣布 H1/H2 解决）；WP5/6→H3 单元；WP7→reliability-wp7；WP8→验收；WP9→可选。
> 已修复不在本 change：H4（reliability-wp0-wp1，真链路验证）；WP2 有序日志地基已单独立 change（reliability-wp2，待实施）——本 change 的 WP3/WP4 依赖其先行完成。

## Why

可靠性方案六项假设中五项尚未修复（H1/H2/H3/H5/H6，另有验证附带发现两项），全部有可复现证据（WP0 夹具 + 探针 + 不打桩端到端验证 `verify_no_stubs_e2e.py`）：

1. **H1 落库失败伪装成功**：persister 吞存储异常，store 全程失败时会话照常跑完、死后仍对外通知事件、落库为零（无桩实测：通知 53 / 落库 0）。
2. **附① 丢弃不可见**：bus 背压丢事件只有 warning 日志；代码注释宣称的 `EventsDropped` 元事件不存在，订阅者无从感知缺口。
3. **H2 快照漏事件**：快照游标取触发事件 ID + 增量按 ID 过滤；延迟提交的旧 ID 永久丢失（无桩实测：全量 [a,b] vs 快照恢复 [b]）。
4. **H3 恢复盲重跑副作用**：dangling tool_call 一律重执行，无 operation_id/幂等账本（无桩实测：真硬杀 + recover_agent 冷恢复，副作用 1→2）。
5. **H5 死配置限制**（截至 2026-09-11 记分板：H1 ✅ wp3、H2 ✅ wp4、H4 ✅ wp0-wp1——**H1/H2 发布单元（WP2+WP3+WP4）仓内齐备，按方案 §3 宣布解决**；剩 H6）——**H3 ✅ wp5+wp6 · H5 ✅ wp7**，六项假设仅余 H6（wp9 可选拆分，无收益证据可整体否决）：`max_turns_per_agent` / `Task.timeout_ms` / `timeout_per_step_sec` 公开声明却零执行点。
6. **H6 Runtime 耦合**：编排/恢复/交互/装配共存 4173 行，sources/budget/composer 无注入口（定性，方案允许独立否决）。

## What Changes

按方案的发布单元组织（依方案 §3 依赖序，前者未完成不进后者）：

- **WP3 提交门与通知分离（H1 + 附①）**：`CommitGate` 接入事件通道替代 persister 观察者接线；`event_commit_policy="required"|"best_effort"`（默认 required，自定义 bus 不支持时构造失败不静默退化）；必要状态消费者（ALM/SessionRegistry）异常上报不吞；观察者独立有界队列 + 真正实现丢弃可观测（补 `EventsDropped` 或等价机制）；provisional 批次原子提交（round_id 因果归属，COMMITTING 竞态处理）；`PersistenceUnavailableError` 让会话进入 `storage_unavailable` 隔离——停止调度、不再伪装成功、`wait_for_finish` 明确抛错。翻转 WP0 夹具 `test_runtime_storage_failure`。
- **WP4 快照一致切面与迁移（H2）**：快照边界 = `committed_head` 截断（C = head → S = 最新可用快照 → read_range(S.cursor+1, C) → apply）；快照触发事件只是请求不是边界；projection_version 不匹配/legacy 无位置 → 忽略快照全量回放重造；迁移工具实装（依赖 WP2 的 position 回填）。翻转夹具 `test_snapshot_commit_interleaving` 与无桩验证 h2。
- **WP5 稳定操作身份与账本（H3 前半）**：`operation_id` 由 `(tenant, session, agent, assistant_record_id, tool_ordinal)` 确定性生成（跨重启稳定，区别于单次执行的 invocation_id 与可复用的 tool_call_id）；`OperationStore` 协议（prepare/CAS/completed 状态机）内存+SQL 实现；完整结果入账本（非审计截断文本）；`_ingest_assistant_turn` 返回结构化 record_id。
- **WP6 恢复策略与 unknown 处置（H3 后半）**：`ToolCapability.recovery_policy`（retry_safe/idempotent/queryable/manual，默认 manual）；Reconcile 完成匹配改用 operation_id（防旧 call_1 覆盖新 call_1）；unknown → task INTERRUPTED + `TOOL_OUTCOME_UNKNOWN` + 宿主接口 `resolve_operation(supply_result|retry_confirmed|cancel_task)`（revision 乐观锁）；控制工具（delegate/finish/ask_user）按 operation_id 幂等核验。翻转夹具 `test_tool_outcome_unknown` 与无桩验证 h3。
- **WP7 ExecutionLimits（H5）**：opt-in 独立限制对象（step/task/provider deadline + actor turns，monotonic clock，持久化已消费量）；超限 INTERRUPTED 带专用错误码；三个死配置字段发去重 DeprecationWarning + README 迁移说明，**不激活旧默认值**。
- **WP8 全量验收**：方案的验收矩阵（E-T01..14 / O-T01..16 / L/X 系列）+ 性能基准脚本（p95 回退预算 ≤15% 待验证）。
- **WP9 Runtime 拆分（H6，可选可否决）**：Recovery/Interaction/ContextPolicy 三边界 + 公开注入；默认行为轨迹等价为验收；无收益证据可整体撤销。

## Capabilities

### New Capabilities

- `event-commit`: 事件提交确认与通知分离契约——required 提交、必要消费者异常上报、观察者背压与丢弃可观测、provisional 批次原子、存储不可用隔离。
- `snapshot-recovery`: 快照恢复一致切面契约——committed_head 截断、全量与快照+增量等价、legacy/版本不匹配降级。
- `tool-operations`: 工具执行身份与恢复契约——operation_id 稳定性、操作账本状态机、恢复策略表、结果未知处置与宿主决策接口。
- `execution-limits`: 执行限制契约——opt-in 语义（active time/turns 计量、deadline 错误码）、旧配置弃用不激活。

### Modified Capabilities

（无——`capability-gateway` 与 `event-log`（reliability-wp2）的既有需求不变；本 change 在其之上新增行为。）

## Impact

- **代码**：bus/persister/runtime 构造与 TaskManager 异常链（WP3）；snapshot writer/reducers/恢复读取（WP4）；ProviderContext/ToolCapability/Gateway/ActStep/Registry（WP5）；ReconcileStep/HITL 恢复/runtime 公开接口（WP6）；RuntimeConfig/StepDriver/TaskManager（WP7）。
- **测试**：翻转 WP0 三份夹具 + 两个验证脚本判定（H1/H2/H3 → 修复后语义）；方案的验收矩阵大量新测试；有意修改「persister 吞异常」旧契约测试。
- **破坏性**：required 默认值使「自定义 EventBus 不支持提交门」构造失败（原静默）；`resolve_operation` 等新公开接口；旧限制字段仅警告不删除。
- **顺序硬约束**：WP3/WP4 依赖 reliability-wp2 先落地；WP5 依赖 WP1（已完成）；单 change 内按 WP 序渐进提交，每个 WP 独立可回滚。
