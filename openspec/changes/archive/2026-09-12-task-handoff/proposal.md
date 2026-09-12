# Proposal: task-handoff

## Why

任务交接存在三项已在代码层面坐实的正确性缺陷：`delegate_task` 声明的 `inputs` 参数被静默丢弃（`core/capabilities/control_tools.py:187` 起的 `TaskModel(...)` 构造不消费它，`delegate_plan` 的 spec schema 甚至未声明该字段）；子任务审核按标题配对（`_collect_reviews` 的 `{t.title: t}` 字典，`control_tools.py:350`），同名子任务的目标随登记顺序漂移；依赖解锁不区分成败（`TaskQueue.mark_failed` 把失败任务也塞进 `_completed` 解阻塞，`core/orchestrator/task/queue.py:96`），前序失败会误放行依赖其成果的后继任务。三者都属于确定性行为缺陷，不依赖模型评测即可验收，应先于更大的架构改造修复。

## What Changes

- **任务输入传递**：`Task` 新增 `inputs` 字段；`delegate_task` / `delegate_plan` 的输入数据写入子任务并随 TASK_CREATED 事件持久化；`TaskSpecSource` 将输入渲染进执行上下文。崩溃恢复后输入仍可用。纯 JSON、有尺寸上限，不扩展为通用数据流平台。
- **任务操作改用 ID**：`report_task_outcome` 的 `task_reviews` 配对键从 `task_title` 改为 `task_id`（**BREAKING**，控制工具 schema 变更）；review 面指令行同步改为引用 task_id（id 本就已在展示面打印）；`delegate_task` / `delegate_plan` 的 ack 回传子任务 id，使模型后续操作有稳定句柄。
- **区分依赖条件**：依赖条件支持「成功后执行」（on_success）与「结束后执行」（on_any）两种；新派发默认 on_success，清理类任务显式声明 on_any；on_success 依赖因前序失败永不可满足时，后继任务落 CANCELED 终态（专用 error_code），不得滞留队列；存量事件回放按历史语义（on_any）解释，不虚构事件未声明的条件。

## Capabilities

### New Capabilities

- `task-handoff`: 任务交接契约——输入数据从派发到执行的保存/传递/上下文投递、任务操作基于稳定 ID 配对、依赖条件区分前序成败。

### Modified Capabilities

（无——三项修复的行为均未被现有 capability spec 覆盖；`capability-gateway` 管的是参数校验管线而非单个控制工具的语义，不需要 delta。）

## Impact

- **控制工具**：`core/capabilities/control_tools.py`（delegate_task / delegate_plan / report_task_outcome 的 schema 与实现，ack 文案变更对模型可见）。
- **任务模型与投影**：`core/models/task.py`（inputs 字段、依赖条件表示）、`core/control/types.py` / `reducers.py` / `converters.py`（TaskView 投影链）、TASK_CREATED payload。
- **调度**：`core/orchestrator/task/queue.py`（双完成集与解锁判据、永久阻塞善后）、`manager.py`（push_task / restore / reopen_chain 的条件重建）。
- **上下文装配**：`core/assembler/sources/task_spec.py`（inputs 渲染）、`composer.py`（review 面指令行）。
- **测试**：上述各层的单元与集成测试；恢复链测试覆盖 inputs 与依赖条件重建。
