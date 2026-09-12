# task-handoff 验收矩阵

spec 场景 → 测试锚点（对齐仓库验收矩阵惯例）。全量基线：`pytest tests` = 3204 过 / 2 失败（均为既知预存：compact L3 e2e、observe role prompt 环境性）/ 1 skip（memory-conformance 协议面）。基线 3110 → 本变更 +94 条、零新增失败。

## Requirement: 子任务输入的保存与传递

| Spec 场景 | 测试锚点 |
|---|---|
| delegate_task 输入到达子任务（契约/事件/回执三处一致） | `tests/unit/test_delegate_inputs_and_ids.py::test_delegate_task_inputs_reach_child_and_ack_carries_id`、`::test_push_task_emits_inputs_in_created_payload` |
| delegate_plan 每任务输入到达 | `::test_delegate_plan_per_task_inputs_and_ordered_id_ack` |
| 不可序列化输入被响亮拒绝 | `::test_delegate_task_invalid_inputs_rejects_without_dispatch`、`tests/unit/test_task_inputs_normalization.py::test_unserializable_rejected`、`::test_non_dict_rejected`、`::test_cyclic_structure_rejected` |
| 超限输入分级收敛（探针对齐：数字大数组 30,012B / 长键 9,007B） | `test_task_inputs_normalization.py::test_big_numeric_array_converges`、`::test_huge_key_converges`、`::test_long_string_value_truncated_with_marker` |
| 无法收敛的超限输入被拒绝 | `::test_huge_scalar_rejected` |
| 崩溃恢复后输入仍可用 | `tests/integration/test_task_handoff_recovery.py::test_inputs_survive_crash_and_reach_reassembled_context`、`tests/unit/test_task_handoff_projection.py::test_created_folds_inputs_and_dep_conditions` |

## Requirement: 子任务输入的上下文投递

| Spec 场景 | 测试锚点 |
|---|---|
| 执行上下文包含输入区块 | `tests/unit/test_task_inputs_delivery.py`（fresh 双测 + 回退 + 截断标记可见） |
| 恢复后续跑同样投递（四路径：首次/memory/恢复/reopen） | 同上 `test_inmemory_path_prefixes_inputs_section` + 集成 `test_inputs_survive_crash_and_reach_reassembled_context` |
| 无 inputs 任务零行为变化 | `test_task_inputs_delivery.py::test_fresh_path_without_inputs_zero_change`、`::test_inmemory_path_without_inputs_zero_change`、`::test_task_spec_source_without_inputs_keeps_metadata_none` |

## Requirement: 任务操作以稳定 ID 配对

| Spec 场景 | 测试锚点 |
|---|---|
| 同名子任务审核对象不漂移 | `tests/unit/test_reopen.py::test_collect_reviews_same_named_children_resolve_by_id` |
| 改名不影响既定操作 | `::test_collect_reviews_survives_child_rename` |
| 越权引用被拒绝 | `::test_collect_reviews_only_accepts_own_children` |
| 旧式条目被显式拒绝（其余照常 + 回执引导） | `::test_collect_reviews_rejects_legacy_title_entries_others_proceed`、集成 `test_legacy_title_entries_rejected_with_guidance_through_tool` |
| 派发回执返回子任务 id | `test_delegate_inputs_and_ids.py`（task ack / plan 有序列表）、集成 `test_delegate_ack_id_drives_review_reopen_by_id` |
| review 面指令行改引 task_id | `tests/unit/test_observer_subtask_cue.py::test_observer_cue_lists_subtask_reviews_from_extra` |

## Requirement: 依赖条件区分前序成败

| Spec 场景 | 测试锚点 |
|---|---|
| 前序失败不放行成果依赖 | `tests/unit/test_task_dependency_conditions.py::test_failed_predecessor_disposes_success_dependent`、集成 `test_plan_break_cancels_successors_and_runs_cleanup` |
| 清理任务在前序失败后仍执行 | `::test_blocked_disposal_cascades_and_spares_any_dependents`（级联 + on_any 豁免）、集成同上 |
| 前序成功时两种条件均放行 | `::test_finished_releases_both_kinds`、`::test_restore_finished_dep_releases_success_dep` |
| 依赖取消不终结会话（豁免 + 父唤醒 + 不计阈值） | `::test_failed_predecessor_disposes_success_dependent`、`::test_blocked_cancel_does_not_touch_failure_counter`、`::test_blocked_cancel_still_wakes_parent` |
| 崩溃窗口的恢复期补扫 | `::test_recovery_scan_disposes_crash_window_leftover`（runtime 首次 drain 前接线：`core/runtime.py` `dispose_blocked_dependents` 调用点） |
| 重启后阻塞原因可解释 | 集成 `test_blocked_reason_survives_replay_and_reaches_review_face`、投影 `tests/unit/test_task_handoff_projection.py::test_canceled_folds_blocked_reason_into_view`、面渲染 `test_observer_subtask_cue.py::test_observer_cue_renders_blocked_cancel_note` |
| 条件断裂不滞留会话 | `::test_failed_predecessor_disposes_success_dependent`（队列断言）、`::test_blocked_disposal_cascades_and_spares_any_dependents`（幂等 + 队列清空） |
| 存量回放保真（无字段 → any） | `::test_restore_legacy_deps_replay_as_any`、投影 `test_created_without_new_keys_keeps_legacy_defaults`、快照 `tests/unit/test_view_serialization_coverage.py`（新字段全覆盖 + 旧快照缺省） |
| 重开后条件存活 | `::test_reopen_chain_preserves_conditions` |
| 清理步的显式声明（run_if） | `test_delegate_inputs_and_ids.py::test_delegate_plan_run_if_materialized_into_dep_conditions`、集成 `test_plan_break_cancels_successors_and_runs_cleanup` |

## 快照序列化补全（评审 P2-6 引出）

`serialize_view` / `deserialize_view` 补 inputs / dep_conditions / error_code / blocked_by_task_id 四字段（旧快照缺键 → 缺省语义）；守卫测试 `test_view_serialization_coverage.py::test_serialize_covers_every_task_view_field` 全绿。
