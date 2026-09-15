# Proposal: reliability-wp8

> 上游方案：`docs/plans/2026-09-11-agent-core-reliability-plan.md` **WP8（故障及性能验收）**，§9/§10/§8-WP8。
> 前置：wp2–wp7 全部归档（H1–H5 ✅）；总纲任务组 1-7 已回勾。
> 本 change 是**收口单元**：补齐验收矩阵、移植子进程强退、接线 wp7 遗留的 budget persisted、建立性能基准——不引入任何新行为。

> ⚠️ **本 change 的两项产出其后被撤销**：budget persisted 接线随 wp7 的 `ExecutionLimits`
> 整体移除（L-T 整表作废，见 `docs/plans/acceptance-matrix.md`）；验收矩阵中的 O-T11/O-T12
> 随 `resolve_operation` 删除而作废。子进程强退矩阵与性能基准仍在仓内有效。

## Why

可靠性方案的可信度取决于验收。四个假设的修复各自有测试锚，但方案的验收矩阵（§9 的 E-T×14 / O-T×16 / L-T×6 / X-T×2）从未逐条对照——部分条目散落在各 WP 的测试里、部分从未落地（子进程强退、性能基准）。此外 wp6 的 3.3（强退矩阵）和 wp7 的 budget persisted 接线均诚实标注移入本单元。没有这一步，「六项假设中五项修复」的声明缺最后一层数据背书。

## What Changes

- **子进程强退矩阵**（`tests/integration/test_operation_crash_matrix.py`）：O-T05（manual started → 真子进程退出 → 新 Runtime → 副作用 1 次 + INTERRUPTED）与 O-T06（completed 后 memory 写前崩溃 → 新实例从账本补写 TOOL_RESULT，不重执行）各至少一条真退出。无桩 h3 的形态移植为 pytest 夹具。
- **budget persisted 接线**（wp7 遗留）：task 投影加 `budget_consumed` 字段（reducers/converters 透传）；TM 派发时 `ExecutionBudget.restore()`（而非空 budget）——跨 retry / 跨重启续用剩余预算（算术已证，这里接上面）。
- **性能基准**（`scripts/benchmark_runtime_commit.py`）：固定负载五场景（单会话 100 工具调用 5ms 延迟 / 4 agent 并发 / 10 会话 / 恢复 1k/10k/100k 含快照开关 / 阻塞观察者并存），输出机器可读 JSON + 原始数据留存。15% p95 是待验证预算不是门禁——超了报告原因。
- **验收矩阵映射与补齐**：方案 §9 的四个表逐条标注已有测试锚（各 WP 测试的映射清单）；无锚的补上（预计：E-T02 批次中第 k 条失败回滚 / E-T03 同 batch 重试 / E-T07 回调嵌套发事件 / O-T14 delegate 找回子任务）。
- **验证脚本终核**：探针 `--expect fixed` 四 True（已达成）；无桩四项 fixed:true（h1-h4 已达成——终核作为门禁记录）；**2 个既知预存失败如实报告**（非 PostgreSQL）。
- **核验表**（方案 §10.3）：逐项填写实测数据（H1-H5 各行 + 性能预算行），空项写「未验证」。总纲任务组 8 回勾 + 记分板终态（H1-H5 ✅ / H6 ⬜ 可选）。

## 不做什么

- 不修两个既知预存失败（test_compact_flow 的 L3 坍缩 / test_observe_outcomes 的 ROLE.md 外部路径——均非可靠性范畴）。
- 不做 PostgreSQL 真库验证（本地环境无 PG 实例——如实标记「未验证」）。
- 不实现 WP9（Runtime 拆分——H6 可选）。
- 不改任何生产代码行为（budget persisted 是接线修复不是新逻辑）。

## Capabilities

### New Capabilities

（无——本 change 无新行为契约，只有验收与收口。）

### Modified Capabilities

（无。）

## Impact

- **代码**：`core/control/reducers.py` + `core/control/converters.py`（budget_consumed 投影透传）、`core/orchestrator/task/manager.py`（派发时 restore 一行）。
- **测试**：新建 `test_operation_crash_matrix.py` + `test_budget_persisted.py`；验收矩阵映射清单（markdown 或测试 docstring）。
- **脚本**：新建 `scripts/benchmark_runtime_commit.py`。
- **文档**：核验表（写入总纲或独立文件）；ARCHITECTURE.md 无需更新（各 WP 已更新过）。
