# Tasks: delivery-acceptance

## 1. 协议与数据地基

- [x] 1.1 定义 `AcceptanceValidator` 协议、`AcceptanceCheckResult`（passed/failed/error + 结构化缺口 `{field, condition, evidence_ref, suggestion}`）与 `AcceptanceRegistry`（**(id, version) 双键注册** + expected_max_ms 申报、未注册/版本缺失查询报错）；单测：通过/失败/抛异常三类输入的协议行为、按版本解析与版本缺失拒绝
- [x] 1.2 隔离执行：每次检查独立 daemon 线程 + 在飞线程数硬上限（达限新检查直接 unverified，不排队），内核 `asyncio.wait_for` 超时 → unverified + degraded 熔断（后续不再执行）+ 迟到丢弃；单测四件：阻塞检查器超时判 unverified、**降级后健康检查器仍可执行**、并发另一任务不被阻塞、**含阻塞检查器的 runtime 可正常退出**（子进程级测试，对齐 spec"阻塞检查器不影响健康检查器与退出"场景）
- [x] 1.3 `Task` 增 `acceptance_spec`（含 `checker_version`）/ `acceptance_repairs_used` 字段（含声明指纹 `acceptance_spec_version` 计算，版本入指纹），随 TASK_CREATED payload 落盘；派发前校验声明 `(id, version)` 可用；**恢复中版本缺失 × 三模式分档测试**（off 不影响交付 / shadow 记 unverified 不改决策 / required 阻止必需检查任务成功，对齐 spec 分档场景）；TaskView / reducers / converters / 快照序列化全链透传（`test_view_serialization_coverage` 守卫过）；单测投影往返与存量缺省
- [x] 1.4 runtime 配置 `acceptance_mode: off | shadow | required`（缺省 off）；行为矩阵单测：3 模式 × 无声明/仅提示性/含必需 九格逐一断言（零变化、记录不改交付、门禁三档语义，对齐 spec 行为矩阵）

## 2. 交付边界接线

- [x] 2.1 FinalizeStep 验收挂点：候选 = 分开另存的 `{final_body, final_summary}` 二元组（非 task.outputs 拼接串），检查/指纹/提交同一份；按声明顺序、required 分级执行"先结构后领域"；产出 `TASK_ACCEPTANCE_CHECKED` 事件（verdict/检查器版本/输出指纹/输入快照标识/声明版本/attempt/缺口/候选引用）；单测：required 失败不落 FINISHED、提示性失败照常交付、error 判 unverified、摘要附着不污染结构检查
- [x] 2.2 成功路径全覆盖测试（均限 required 模式）：finish_task 收尾、observe 判成功收尾 run、交互任务 park 回合（负向：不产生验收记录不动额度）、交互任务 finish_task 收尾（正向：候选受验）、子任务结果上抛（父侧收到带验收标注的结果）——各一集成测试；另加守卫测试"**required 模式下**声明必需检查而无验收记录落成功终态 → 断言失败；off/shadow 模式下同样情形允许成功"（对齐 spec 模式矩阵与守卫限定）
- [x] 2.3 有效用户回合标识持久化链：新事件 `TASK_INPUT_ADVANCED`（task_id + 回合记录 id），**维护限定**「mode≠off 且有检查声明」的任务（未启用任务零事件变化，单测锚定），**首次启用**（含中途切换/补挂声明）经对账原语以 memory 重算值初始化后才可验收判定（单测锚定）；写点（仅维护中的任务）：driver 起步/新一轮落库、send_message 注入、HITL 回复注入、撤销/弃轮重算，reducer 折叠 `TaskView.effective_input_turn_id`、converter 回填、快照随链覆盖；**恢复对账**：重建后、任何验收判定前从 memory 视图重算应有标识与投影值比对、不一致幂等补发（最后写生效，重复无副作用），**两条故障注入测试**：消息已落库/输入事件未落库、撤销已完成/回退事件未落库（对齐 spec 两个对账场景）；两条恢复指纹测试：无输入变化重启标识与指纹不变（不退化哨兵）、真实入口新增输入后重启仍识别变化；三元组绑定 = 候选 outputs 规范化哈希 + 输入快照标识（user_prompt + inputs + effective_input_turn_id）+ 声明版本指纹（含检查器版本），单测”输入变化输出相同””id/参数不变仅检查器版本升级”两类失效
- [x] 2.4 候选引用与外存：小候选直存事件、超限 spill 外存可解引用；单测含"重启后按引用取回失败候选"

## 3. 缺口反馈与一次修正（跨崩溃保证）

- [x] 3.1 额度预占先行：落盘顺序定案——①TASK_ACCEPTANCE_CHECKED（failed+缺口+候选引用）→②TASK_ACCEPTANCE_RETRY_RESERVED（repairs_used 0→1、任务回 PENDING）→③修正生成；缺口经 `next_step_hint` 投递一次性提示；单测重排、计数与预占事件时序（预占先于任何修正生成）
- [x] 3.2 修正轮全量复检：下一轮 finalize 对新候选重跑全部必需检查；集成测试含"修正修复 A 破坏 B 被复检拦截"
- [x] 3.3 终止条件：缺口重复（findings 全同）、预算耗尽、检查器 degraded/故障、额度已用 → 落 `FAILED + TASK_ACCEPTANCE_FAILED`，验收记录保留候选与剩余缺口；单测四分支
- [x] 3.4 三点崩溃测试：预占前崩溃（恢复后可正常发起一次修正）/ 预占后生成前崩溃（恢复续跑=完成该次修正、计数不变、再失败落 FAILED）/ 生成中崩溃（恢复走 run 恢复链、额度不重置）；断言任何路径重启后修正总次数 ≤1
- [x] 3.5 缺口持久化与恢复：TASK_ACCEPTANCE_CHECKED 投影折叠（TaskView.acceptance 含绑定三元组与历史），修正轮中途中断后恢复重建缺口上下文并注入续跑；集成测试覆盖重启续跑
- [x] 3.6 依赖传播：验收 FAILED 任务对 on_success 下游不放行（复用 task-handoff 依赖条件），父任务 review 面透出验收失败缺口（note 通道）；集成测试
- [x] 3.7 compat 入口内联续跑：`run_single_task` 处置为 ACCEPTANCE_RETRY 时内联再执行一轮（上限 1）后返回最终结果；集成测试断言"首次检查失败 → 实际发生一次修正生成 → 返回修正后结果"（对齐 spec 兼容入口场景）

## 4. 成本与影子模式

- [x] 4.1 修正调用计入任务成本累计并受 ExecutionBudget 约束；预算耗尽时修正被拒直接走失败处置；单测预算边界
- [x] 4.2 影子对照集成测试：同一输入分别以 off 与 shadow 运行，交付决策与任务终态完全一致、差异仅存在于验收记录（对齐 spec 影子场景；required 与 shadow 的差异 = 门禁，由 1.4 矩阵覆盖）

## 5. 回归与验收矩阵

- [x] 5.1 冻结 18 答复重放回归：以仓库六类验证器注册为宿主检查器（仅测试侧），重放断言 10 份正确不受影响、8 份已知错误全拦、0 误报（对齐实验 replay.json 基线）
- [x] 5.2 全量 `pytest tests` 对照基线（3204 过 / 2 既知预存 / 1 skip）无新增失败
- [x] 5.3 验收矩阵 `acceptance-matrix.md`：spec 全部场景逐条映射测试锚点 + 检查器 CPU 时延与 degraded 次数单列（方案 §6 成本口径）
