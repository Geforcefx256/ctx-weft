# delivery-acceptance 验收矩阵

spec 场景 → 测试锚点。全量基线：`pytest tests` = 3250 过 / 2 失败（既知预存：compact L3 e2e、observe role prompt 环境性）/ 1 skip。相对 f49c831 基线 3204 → 本变更 +46 条、零新增失败。实现期实测并修正两处：delegate 工具 schema 膨胀触发媒体窗口预算（收紧描述后回基线——tool-schema-budget 守卫按设计拦截）；`_write_hitl_reply_turn` 刻意无状态，回合推进注入移至活跃消费方。

## Requirement 1：宿主注册检查器与任务级声明

| Spec 场景 | 测试锚点 |
|---|---|
| 声明随创建事件持久化并恢复可用 | `tests/unit/test_delivery_acceptance.py::test_projection_folds_acceptance_domain`（TASK_CREATED 携带 + converter 往返）；`test_view_serialization_coverage`（快照守卫） |
| 未注册或版本缺失的声明被响亮拒绝 | `tests/unit/test_acceptance_protocol.py::test_registry_resolves_by_id_and_version`；集成 `test_delivery_acceptance_e2e.py::test_unregistered_checker_rejects_fresh_task_loudly`（compat 入口 ValueError；队列路径经 push_task 注入校验器，同判据） |
| 恢复中版本缺失按模式分档处置 | `test_delivery_acceptance.py::test_gate_unavailable_version_after_recovery_unverified`（required 阻止成功 / shadow 记录不改决策）；off 档 = `maintenance_active(off)=False`（矩阵测试） |
| 阻塞检查器不影响健康检查器与退出 | `test_acceptance_protocol.py::test_executor_timeout_degrades_and_late_result_dropped`、`::test_executor_backpressure_does_not_queue`、`::test_executor_healthy_checker_available_after_degraded`、`::test_executor_daemon_threads_do_not_block_exit`（子进程级） |
| 未声明任何检查的任务零行为变化 | `test_delivery_acceptance.py::test_off_mode_no_events_at_all` + `::test_mode_matrix_nine_cells`（none 格）；全量回归 3204→3250 无新增失败 |

## Requirement 2：候选结构与全路径提交边界

| Spec 场景 | 测试锚点 |
|---|---|
| 摘要附着不污染结构检查 | `test_acceptance_protocol.py::test_builtin_structure_check`；`test_delivery_acceptance.py::test_candidate_uses_separated_pair_not_concatenated_outputs` |
| finish_task 收尾必经验收 | `test_delivery_acceptance.py::test_gate_required_fail_reserves_one_repair`（gate）+ `test_tm_reserve_writes_budget_and_delivers_gap_then_requeues`（处置链）——finish/observe 收尾同经 FinalizeStep→apply_run_outcome |
| 交互中间交流不触发验收 | `test_delivery_acceptance_e2e.py::test_park_turn_never_reaches_acceptance`（park 抛 HitlPark 不经 finalize，结构判定 + disposition 无验收分支）；额度不动由 gate 不执行保证 |
| 交互最终答复经收尾验收 | 同 finish_task 路径（收尾 run 形态）；`test_compat_entry_performs_actual_repair_and_returns_final` 证明收尾 run 候选受验 |
| 子任务结果上抛前经验收 | `test_observer_subtask_cue.py::test_observer_cue_renders_acceptance_annotation`（验收状态随结果上抛带标注） |
| 兼容入口实际完成一次修正 | `test_delivery_acceptance_e2e.py::test_compat_entry_performs_actual_repair_and_returns_final`（FINISHED + repairs=1 + 全量复检 calls==2 + CHECKED 序列 [failed, passed] + 返回修正后输出）；`::test_compat_entry_terminates_when_repair_still_fails` |

## Requirement 3：结构化缺口与一次受控修正

| Spec 场景 | 测试锚点 |
|---|---|
| 一次修正后全量复检 | `test_gate_repair_reruns_all_required_checks_and_second_fail_terminates`（结构+领域检查器各两次） |
| 修正引入的新损坏被复检拦截 | 同上（second fail → failed，e2e terminate 测试同判据） |
| 重复缺口终止自动修正 | `test_gate_repeated_identical_findings_skip_repair_even_with_budget` |
| 预算不足时不修正 | `test_gate_budget_exhausted_skips_repair` |
| 预占先于生成落盘 | `test_tm_reserve_writes_budget_and_delivers_gap_then_requeues`（apply_run_outcome 内先发 RESERVED 事件再重排；TM `_settle` PENDING 出口复用现有重排） |
| 预占后生成前崩溃的续跑语义 | `test_delivery_acceptance.py::test_crash_after_reserve_recovery_completes_that_repair`（计数不变交付）/ `::test_crash_after_reserve_recovery_still_failing_terminates`；预占前崩溃 → `::test_crash_before_reserve_recovery_may_retry_once` |

## Requirement 4：验收状态独立于执行状态

| Spec 场景 | 测试锚点 |
|---|---|
| 检查器异常判未验证 | `test_acceptance_protocol.py::test_executor_pass_fail_raise`；`test_delivery_acceptance.py::test_gate_checker_error_is_unverified_and_terminal` |
| 提示性检查不阻塞 | `test_mode_matrix_nine_cells`（advisory 格：required 下 gate None） |
| 未启用验收的任务不产生输入标识事件 | `test_reconcile_ignores_unmaintained_tasks` + 矩阵 none/off 格 |
| 首次启用时以重算值初始化 | `test_gate_first_enable_initializes_turn_id_from_memory`（缺标识 → memory 重算 + TASK_INPUT_ADVANCED） |
| 用户经真实入口追加修改使旧结论失效 | `test_input_snapshot_distinguishes_new_turn_same_output`（输出同、回合变 → 指纹失效）；注入点接线：runtime send_message/HITL 消费方调 `tm.advance_input_turn`（代码锚 `runtime.py` 两处防御式调用） |
| 无输入变化的重启指纹不变 | `test_reconcile_covers_both_crash_windows`（第二次对账零补发 = 标识稳定） |
| 新增输入后重启仍能识别变化 | 同上窗口一（memory 新回合 → 补发 → 标识更新） |
| 声明变化使旧结论失效 | `test_acceptance_protocol.py::test_spec_version_includes_checker_version`（版本/参数任一变 → 指纹变） |

## Requirement 5：验收记录持久化与恢复可审计

| Spec 场景 | 测试锚点 |
|---|---|
| 重启后验收历史可审计 | `test_projection_folds_acceptance_domain`（三事件折叠 + 重复 RESERVED 幂等 + converter 重建） |
| 重启后取回失败候选 | `test_gate_oversize_candidate_spills_to_memory`（超限外存 + load_view 解引用全文取回） |
| 恢复后缺口上下文重建 | `test_restore_rebuilds_gap_hint_for_interrupted_repair`（restore 据投影重建 next_step_hint） |

## Requirement 6：成本归集与模式行为矩阵

| Spec 场景 | 测试锚点 |
|---|---|
| 影子模式不改变交付 | `test_delivery_acceptance.py::test_gate_shadow_records_but_never_decides`；集成 `test_off_vs_shadow_same_input_same_delivery`（同输入 off/shadow 终态与 outputs 全同，差异仅 CHECKED 记录） |
| off 模式下声明不产生任何副作用 | `test_off_mode_no_events_at_all`；e2e off 臂零事件断言 |
| 修正调用计入任务成本 | `test_gate_budget_exhausted_skips_repair`（限额码拒修正）；修正= 经 TM 重排的常规 run，天然进入 ExecutionBudget/usage 口径（execution-limits 套件守卫既有预算判定） |
| 行为矩阵九格 | `test_mode_matrix_nine_cells`（3 模式 × 无/提示/必需逐一断言） |

## 恢复对账两窗口（故障注入）

| 窗口 | 测试锚点 |
|---|---|
| 消息已落库、输入事件未落库 | `test_reconcile_covers_both_crash_windows`（窗口一：补发至新回合；幂等复跑零补发） |
| 撤销已完成、回退事件未落库 | 同上窗口二（fold 后重算值即回退值 → 补发回退） |

## 宿主成本单列（方案 §6 口径）

- 检查器执行：daemon 线程 + 在飞上限 2 + 超时 5s——时延与 degraded/背压计数在 `AcceptanceExecutor.stats`（宿主可观测），不进 token 口径。
- 实现期一次实测教训已记录：控制工具 schema 增长会直接侵蚀宿主上下文预算（媒体套件拦截），参数描述保持最小。
