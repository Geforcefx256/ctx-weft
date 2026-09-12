# Tasks: task-handoff

## 1. 模型与投影地基

- [x] 1.1 `Task` 增 `inputs: dict | None` 与 `dep_conditions: dict[str, str] | None` 字段，`TaskView` 增 `error_code` / `blocked_by_task_id` 字段，TaskView / reducers / converters 全链透传；单测覆盖投影往返与各字段缺省（存量事件形态）不炸
- [x] 1.2 控制工具侧输入规整工具函数：纯 JSON 校验（不可序列化即报错）+ UTF-8 字节计量 + 8 KiB 上限分级收敛（长字符串值截断 → 数组/字典键数裁剪带丢弃计数标记 → 加标记后复核，仍超限则响亮拒绝）；单测覆盖合法 / 不可序列化 / 纯数字大数组（探针实测 30,012 字节）/ 超长键（9,007 字节）/ 巨型标量拒绝五类

## 2. 输入传递

- [x] 2.1 `delegate_task` 消费 `inputs` 参数写入子任务契约，TASK_CREATED payload 带 inputs，ack 回执改为含子任务 id；单测断言契约、事件 payload、回执三处一致
- [x] 2.2 `delegate_plan` 任务 spec schema 增 `inputs` 与 `run_if: "success" | "any"` 字段声明与读取，ack 回执含与 spec 顺序对应的 id 列表；单测覆盖多任务 spec 的 inputs 各归各、id 有序
- [x] 2.3 输入投递的 Composer 双路径接线：`TaskSpecSource` 将 inputs 写入 block `metadata["inputs"]`（content 仍为快照，遵守其契约），`_task_spec_fields` 扩读 inputs，fresh 实时构建分支与 `_frame_current_message` 装饰分支均渲染 `## Inputs` 小节（截断/裁剪标记原样可见）；单测覆盖四条执行路径——首次执行、已有 memory 续跑、崩溃恢复后重建执行、reopen 后重跑，并断言无 inputs 任务零行为变化
- [x] 2.4 恢复链集成测试：携带 inputs 的子任务派发落盘 → 进程重启恢复 → 子任务重跑时装配上下文仍含输入区块（对齐 spec 场景「崩溃恢复后输入仍可用 / 恢复后续跑同样投递」）

## 3. ID 配对

- [x] 3.1 `_collect_reviews` 配对键从 title 改 id；`report_task_outcome` 的 `task_reviews` 条目 schema 从 `task_title` 改 `task_id`（直接移除旧字段）并补 items 描述；**条目级显式校验**（gateway 严格校验不覆盖嵌套层）：缺 `task_id` / 非字符串 / 携带未知字段（含旧 `task_title`）→ 该条目拒绝且回执说明，其余条目照常；单测：同名双子任务定向生效、审核前改名仍命中、越权 id 拒绝、旧式条目拒绝且其余生效
- [x] 3.2 composer review 面指令行改为引用 task_id；全仓检索 `task_title` / 旧 ack 文案断言并同步更新，`pytest tests` 相关文件全绿
- [x] 3.3 端到端集成测试：delegate 派发拿到回执 id → observer 以 task_id 发起 reopen → 链路按预期重排（对齐 spec 场景「派发回执返回子任务 id / 同名不漂移」）

## 4. 依赖条件

- [x] 4.1 `TaskQueue` 增 `_succeeded` 完成集，`QueueEntry.blocked_by` 拆 `blocked_any` / `blocked_success`，`mark_complete` / `mark_failed` / push·pop 惰性刷新按两集判据释放；单测：FINISHED 放行两集、FAILED/CANCELED 只放行 any
- [x] 4.2 `push_task` / `restore` / `reopen_chain` 写入与重建 `dep_conditions`（写入时物化缺省：新派发全量落 `success`，存量无字段回放按 `any`）；单测覆盖两种缺省路径与 reopen 后条件存活
- [x] 4.3 永久阻塞扫描双入口（同一套幂等实现）：运行期（`mark_failed` / 协作取消后定点迭代至不动点）与恢复期（`restore` 重建依赖后、首次 drain 前）；on_success 依赖判明永不可满足时落 `CANCELED` + `error_code=BLOCKED_BY_FAILED_DEP`（含级联），出队并发终态事件；单测：链式级联、菱形依赖、判定后会话无永久 PENDING、**崩溃窗口**（A FAILED 已落盘、B 级联取消未落盘 → 恢复期补扫补齐）
- [x] 4.4 `delegate_plan` 的 `run_if` 接线到 `dep_conditions` 物化；集成测试：三步计划第二步失败 → 第三步（success）取消且带阻塞原因、清理步（any）照常执行
- [x] 4.5 阻塞原因持久化投影全链：TASK_CANCELED payload 带 `error_code` + `blocked_by_task_id` → reducer 回填 TaskView → converter 回填 `Task.error` / `error_code`（快照随 TaskView 自动覆盖）；集成测试：重启回放后父任务 review 面仍可见阻塞源与原因
- [x] 4.6 依赖取消的会话语义豁免：`on_task_finished` 的 CANCELED 分支按 `error_code == BLOCKED_BY_FAILED_DEP` 豁免 session-CANCELED 改写（与 `_pause_abandon` / `_threshold_tripped` 同型），父任务唤醒（`_try_resume_parent`）与 on_any 后继调度保留；集成测试：子任务被阻塞取消后父任务照常唤醒、会话不被标 CANCELED、不计入 failure_counter

## 5. 回归与收尾

- [x] 5.1 全量 `pytest tests` 对照既有基线（3110 过 / 2 既知预存失败 / 1 skip）无新增失败
- [x] 5.2 行为变更专项回归：delegate_plan 失败链中断语义、旧式 `task_title` 条目被**条目级校验**拒绝且回执可引导重发、ack 文案消费方全部更新——各一条集成测试
- [x] 5.3 将 spec 全部场景映射到测试文件与用例清单，落 `openspec/changes/task-handoff/acceptance-matrix.md`（对齐仓库既有验收矩阵惯例）
