# 可靠性方案验收矩阵映射（2026-09-11，reliability-wp8 产出）

> 方案 §9 四个表逐条对照：已有测试锚（`file::test_name`）/ 本 change 补齐 / 未落地（注明原因）。
> 这是**审计对照表**——不是新测试集，是「每条矩阵条目都有归宿」的证明。

## E-T：事件、恢复与通知（14 条）

| 条目 | 故障/交错 | 必须观察到 | 测试锚 |
|---|---|---|---|
| E-T01 | append 前失败 | 无 committed 通知，Runtime 停副作用 | `test_commit_gate.py::test_storage_failure_raises_and_marks_session_first` |
| E-T02 | SQL 批次中第 k 条插入失败 | 整批不存在，可重试，既存批次不受影响 | `test_ordered_event_store_conformance.py::test_batch_atomicity_kth_failure_leaves_no_trace` |
| E-T03 | COMMIT 后 receipt 丢失 | 同 batch_id 重试得原位置，无重复 | `test_ordered_event_store_conformance.py::test_idempotent_retry_returns_original_receipt` |
| E-T04 | A 延迟提交，B 触发快照 | A 不丢；全量与增量等价 | `test_snapshot_consistent_cut.py::test_late_committed_event_not_lost` |
| E-T05 | head 后并发提交 | blob 不混入 > C；delta 应用一次 | `test_snapshot_consistent_cut.py::test_concurrent_commit_after_cut_not_in_blob` |
| E-T06 | 快照损坏/版本不匹配 | 降级或报错 | `test_snapshot_consistent_cut.py::test_legacy_and_corrupt_snapshots_ignored` |
| E-T07 | 回调嵌套发派生事件 | 不死锁，父先于子 | `test_commit_gate.py::test_derived_session_event_inherits_round_and_discarded`（round 归属 + 不逃逸）+ `test_observer_backpressure.py`（主循环不挂死）——**完整死锁验证未单独立项**（contextvar + 短临界区设计使死锁不可达，见 wp3 design D4） |
| E-T08 | provisional 派生不同 task_id | round 归属保持 | `test_commit_gate.py::test_derived_session_event_inherits_round_and_discarded` |
| E-T09 | 多窗口一提交一撤销 | live/full/snapshot 等价 | `test_commit_gate.py::test_window_commit_atomic_and_retryable`（提交侧）+ `test_snapshot_consistent_cut.py`（快照侧）——**完整三态等价由 E5 逐字段测试覆盖** |
| E-T10 | 观察者不消费、队列满 | 主循环不挂死；缺口可补读 | `test_observer_backpressure.py::test_slow_stream_observer_drops_visible_and_backfillable` |
| E-T11 | required consumer 抛错 | 健康故障，不吞 | `test_observer_backpressure.py`（required 上抛）+ `test_commit_gate.py::test_required_consumer_failure_propagates` |
| E-T12 | 提交后内存状态更新前退出 | 冷恢复收敛 | `test_crash_recovery_reconcile.py`（真 runtime 恢复）+ `test_outage_resume.py` |
| E-T13 | legacy 数据迁移 | 业务字段等价 | `test_event_position_migration.py::test_migrated_db_recovers_via_position` |
| E-T14 | 会话 A 故障 B 正常 | A 隔离 B 继续 | `test_runtime_storage_failure.py::test_sql_drop_table_isolates_and_stops_scheduling` |

## O-T：工具执行（16 条）

| 条目 | 故障/场景 | 测试锚 |
|---|---|---|
| O-T01 | 认证头原值 | `test_gateway_argument_channels.py::test_provider_receives_original_authorization_header` |
| O-T02 | 人类修改参数 | `test_gateway_argument_channels.py::test_provider_receives_hitl_modified_header_unredacted` |
| O-T03 | prepared 前崩溃 | `test_operation_recovery_policy.py::test_prepared_runs_first_execution`（prepared→started 转移）|
| O-T04 | started 后 reviewed（默认） | `test_operation_recovery_policy.py::test_reviewed_started_concludes_without_rerun` + `::test_reviewed_without_adjudicator_concludes_not_errors` |
| O-T05 | 外部成功 completed 前崩溃 reviewed | `test_operation_crash_matrix.py::test_ot05_reviewed_started_real_exit_side_effect_once`（**真子进程退出**）|
| O-T06 | completed 后 memory 写前崩溃 | `test_operation_crash_matrix.py::test_ot06_completed_memory_write_crash_ledger_backfills` |
| O-T07 | memory 后 Finished 前崩溃 | memory 幂等 + 通知补发——`test_gateway_operation_ledger.py::test_full_execution_order_records_completed`（顺序含 Finished）；完整崩溃间隙由 O-T05 的真退出覆盖 |
| O-T08 | 多次重入同 operation_id | `test_gateway_operation_ledger.py::test_completed_reentry_short_circuits_no_reinvoke` |
| O-T09 | 两条 assistant 同 call_1 同参 | `test_conversation_pairing.py::test_mint_deterministic_and_differentiating`（铸造面：同 raw 不同回合 → 不同 id）+ `test_operation_recovery_policy.py::test_call1_reuse_no_cross_talk`（账本面：不误判完成）|
| O-T10 | HITL 热/冷竞态 | `test_hitl_e2e_v2.py::test_cold_approval_reconciles_and_invokes_the_tool_exactly_once` |
| ~~O-T11~~ | ~~unknown 后 recover_agent~~ | **条目作废**——「结果不确定」不再是控制流，没有停机闸门可测。现契约（无专属错误码/事件/处置 API）由 `test_operation_recovery_policy.py::test_no_uncertainty_control_plane` 反向钉住 |
| ~~O-T12~~ | ~~两宿主并发 resolve~~ | **条目作废**——`resolve_operation` 已删除。账本自身的 revision 乐观锁由 `test_operation_store_conformance.py::test_cas_serializes_and_rejects_stale` 覆盖 |
| O-T13 | 结果超 8000 字符或含图片 | `test_operation_store_conformance.py::test_full_roundtrip_with_parts_and_blob_ref`（blob ref 往返）|
| O-T14 | delegate 确认丢失 | `test_operation_crash_matrix.py::test_ot14_delegate_completed_reentry_no_duplicate_children` |
| O-T15 | 授权撤销后重试 | `test_operation_recovery_policy.py::test_legacy_no_ledger_record_concludes_without_rerun`（保守面）；完整授权重查由 gateway 既有授权链覆盖 |
| O-T16 | 远端取消失败 | **未落地**——原锚在 wp7 的 uncooperative provider 标记上，wp7 的 `ExecutionLimits` 已整体移除（计时与打断判定为宿主策略，不属 SDK 职责），该语义面当前无 core 侧实现可测 |

## L-T：限制与扩展（6 条）——**整表作废**

WP7 的 `ExecutionLimits`（step/task/provider 超时、actor 轮数、预算持久化）连同 WP8 的
budget persisted 接线已整体移除：计时与打断是宿主策略，不是 SDK 职责。L-T01–L-T06 六条
全部失去被测对象，对应的 `test_execution_limits.py` / `test_execution_budget.py` /
`test_budget_persisted.py` 均已不在仓内。**不要按本表去找这些文件。**

现存的有效限制（act 轮数上限、LLM 自愈预算、内置 shell 超时）是既有机制，不在本矩阵范围内。

## X-T：扩展（2 条）——归 WP9

| 条目 | 状态 |
|---|---|
| X-T01 两个示例 ContextPolicy | **未落地**——WP9 可选（H6），无收益证据可整体否决 |
| X-T02 默认策略轨迹等价 | 同上 |
