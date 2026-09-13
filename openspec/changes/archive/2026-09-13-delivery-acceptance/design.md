# Design: delivery-acceptance

## Context

见 proposal（Why）与演进方案 §3/§4/§5.1/§7。代码基线 `f49c831`（task-handoff 已落地：inputs/dep_conditions/error_code 均随 TASK_CREATED 持久化，投影与恢复链现成）。两条实验结论约束本设计：留出集上"结构检查→领域检查→缺口→一次修正→全量复检"把 14/30 提到 30/30；生命周期轮证明"再生/修正环节"是模型弱点的放大器——因此缺口必须**结构化、带数据引用**（实验中修正成功的全部案例都依赖具体缺口，而非泛化失败文案）。

路径事实（评审核实，代码锚点）：

- interactive 任务的纯文本回合在 act 内 `_park_await_user` 抛 `HitlPark`（`act.py:754` 起），**该回合不经过 FinalizeStep**；
- `run_single_task` compat 入口无队列、不经 `_run_task`（`runtime.py:1531` 起注释自证），状态改 PENDING 不会产生下一轮；
- `task.outputs` 是正文与摘要**拼好的单串**（`act.py:178`，6 处读取方的既有契约），但收尾时两段**分开另存**（`final_body`/`final_summary`，TASK_FINALIZED 亦按分开形态出核，`finalize.py:793`）。

方案 §5.1 三条边界要求逐条落实：接入边界见 D1（含上述两条路径的处置），状态边界见 D4/D5，流式边界如实声明不撤回。

## Goals / Non-Goals

**Goals：**

- 一个交付边界覆盖全部**以任务收尾为形态**的成功提交路径；检查协议最小（隔离执行的同步纯函数）；一次修正 + 全量复检 + 跨崩溃额度保证；验收三态独立持久化并绑定输入/输出/声明三元组；影子模式与成本归集。

**Non-Goals：**

- 不做阶段 2（修订作用域与证据过滤——另立 change）；不做阶段 3（确定性计算前置为工具）；不做自然语言通用验收器；**不验收交互任务的中间交流回合**（见 D1——过程输出非交付候选，且流式已展示无法撤回）；不承诺流式文本撤回；不处理外部副作用幂等（阶段 5）；不把六类实验算法写进内核（方案 §7：算法在宿主）。

## Decisions

### D1 挂点：FinalizeStep 交付边界；中间交流与交付候选显式区分

**决策**：验收检查挂在 `FinalizeStep`（候选形成之后、TASK_FINALIZED/终态落定之前），结果经 RunOutcome 交给 TaskManager 处置表——校验层不写状态，无第二状态机。检查仅在 required 模式下构成门禁（D4 矩阵）；shadow 只执行并记录、off 不执行。

**中间交流 vs 交付候选**（评审 P1-1 的处置）：交互任务 park 回合的文本是**中间交流**——它在 act 内即让位用户、不经 finalize，且已流式展示。阶段 1 规定：

- park 回合**不构成交付候选**、不触发必需验收、不产生验收记录；
- 声明了必需检查的交互任务，其**最终交付**必须以收尾 run 为形态（actor 调 `finish_task`，或 observe 判成功的收尾 run）——该 run 的 FinalizeStep 执行验收；宿主经模板约束"最终答复须以 finish_task 收尾"；
- 澄清、追问、进度说明属于中间交流，不被当成待验收产物——这正是"检查交付物而非过程"的边界。

成功提交路径映射（实现期以测试逐条锚定）：

```
路径                          候选形成点                验收挂点
──────────────────────────────────────────────────────────────────────
finish_task 收尾（自治）       act 收尾 final_body/     FinalizeStep
                              final_summary 分开另存
任务收尾 run（observe 判成功） 同上                      FinalizeStep
交互任务中间交流（park 回合）   ——非交付候选——            不验收（D1 边界）
父任务接收子任务结果           子任务 finalize close     子任务自己的
                                                      FinalizeStep；验收
                                                      状态随结果上抛
run_single_task（compat）      同上                      FinalizeStep 检查；
                                                      修正经 D4 内联续跑
```

守卫（限 **required 模式**）：required 模式下声明了必需检查的任务若在无验收记录的情况下落成功终态，视为实现缺陷（断言/测试捕获）；off/shadow 不门禁——off 不执行检查、shadow 只记录不改变交付决策（见 D4 矩阵），二者允许无验收记录的成功。

### D2 候选结构与检查协议

**候选 = 分开的交付二元组，检查/指纹/提交同一份**（评审 P2-5 的处置）：检查对象是收尾时分开另存的 `{final_body, final_summary}`（含 finish_task 的 deliverables_summary），**不是** `task.outputs` 拼接串——拼接串是 memory 契约、6 处读取方不动。结构检查按交付 schema（正文可声明 JSON 面则解析判型）作用于 `final_body`；摘要与附件的范围在检查声明中显式圈定，杜绝"合法 JSON 正文被附着的自评摘要污染"的误判。指纹、验收记录、TASK_FINALIZED 提交使用同一份二元组。

**检查协议**：`AcceptanceValidator` = `check(candidate, params, ctx) -> AcceptanceCheckResult`，同步纯函数、无工具/HITL/网络；宿主按稳定 id + version 注册到 `AcceptanceRegistry`。结果三态 `passed / failed / error`（error → 验收 unverified），failed 携带结构化缺口 `[{field, condition, evidence_ref, suggestion}]`。未知 id 在任务派发前拒绝（同 inputs 响亮拒绝先例）。

**隔离执行**（评审 P2-6 二轮的处置）：每次检查在**独立 daemon 线程**中执行（`daemon=True` 保证不阻止进程退出），内核以 `asyncio.wait_for` 包裹；在飞检查线程数有硬上限（常量，如 2），达到上限时新检查直接判 unverified（背压，不排队——单个阻塞检查者不得拖垮后续健康检查器的可用性）。超时上限（常量，如 5s）→ 判 unverified + 该检查器标记 degraded（后续调用直接 unverified、不再执行），迟到结果**丢弃不采用**。如实收窄契约：**不承诺支持无限阻塞的检查器**——超时判 unverified 是兜底而非回收保证，被占用的 daemon 线程随进程退出回收，泄漏数进入验收矩阵指标；宿主对检查器执行时间负有申报责任（注册时申报 expected_max_ms）。测试须证明：某检查器阻塞降级后健康检查器仍可正常执行；含阻塞检查器的 runtime 可正常关闭退出。

### D3 数据最小表达：三元组绑定 + 候选可恢复引用

- `Task.acceptance_spec: list | None`——`[{checker_id, checker_version, params, required}]`，**声明固定检查器版本**（评审 P2 三轮的处置）：注册表按 `(id, version)` 注册与解析，版本纳入声明指纹——检查器升级（同 id 新版本）即声明指纹变化，旧验收结论随之失效。派发前校验声明的 `(id, version)` 可用，缺失即响亮拒绝（同未注册 id 先例）。**恢复中版本缺失按模式分档处置**（评审 P2 四轮的处置，与 D4 矩阵对齐）：off 模式不执行验收、不影响原交付流程；shadow 模式该检查记录 unverified 但不改变交付决策；required 模式该检查判 unverified 且必需检查任务不得落成功终态。声明随 TASK_CREATED 持久化；投影/快照链复用 task-handoff 铺好的路（含旧数据缺省 None）。声明指纹 `acceptance_spec_version` = 声明规范化哈希（含版本）。
- `TASK_ACCEPTANCE_CHECKED` 事件 payload（评审 P2-4 的补齐）：`{verdict, checker_id, checker_version, required, output_fingerprint, input_snapshot_id, acceptance_spec_version, attempt, repairs_used_after, findings[], candidate_ref}`：
  - `input_snapshot_id` = 规范化哈希（`task.user_prompt + task.inputs + 有效用户回合标识`）。**有效用户回合标识 = 单独持久化的任务字段 `effective_input_turn_id`**（评审 P1 三轮的处置：瞬态 `user_prompt_memory_id` 不可恢复——converters 不回填、已落库 HITL 回复恢复时跳过补写、初始记录 id 来自 memory 写入返回，实测恢复后为 None，指纹会漂移或退化为同一哨兵）。维护规则为定案：
    - **维护条件**（评审 P2 五轮的处置）：仅当「验收模式 ≠ off 且任务存在检查声明」时维护 `TASK_INPUT_ADVANCED` 事件与该字段——未启用验收的任务事件流完全不变（与零变化承诺一致）；**首次启用**（创建时已启用，或运行中由 off 切换/补挂声明）时，以下述对账原语从 memory 视图重算当前有效回合并落盘初始化，完成前不进行该任务的验收判定；
    - **写点**（仅对维护中的任务）：driver 起步落库初始/新一轮 user_prompt、`send_message` 注入、HITL 回复注入、撤销/弃轮重算（回合被 fold 后重算为最近一条仍存活回合的记录 id，无存活回合才为哨兵值）；
    - **恢复对账原语**（评审 P1 五轮的处置，堵"落库已发生、事件未落盘"的崩溃窗口）：恢复链在重建后、该任务任何验收判定前，从 memory 视图重算应有标识（task scope 内最新存活 user_prompt 回合的记录 id）与投影值比对；不一致 → **幂等补发** `TASK_INPUT_ADVANCED`（重算值；事件折叠为最后写生效，重复补发无副作用），补齐落盘后才允许验收判定。对账同时覆盖撤销窗口（fold 已完成、回退事件未落 → 重算值即回退值）。两条故障注入测试锚定：消息已落库/输入事件未落库、撤销已完成/回退事件未落库；
    - reducer 折叠进 `TaskView.effective_input_turn_id`，converter 回填 Task，快照序列化随链覆盖——恢复后与崩溃前同值，**无输入变化重启指纹不变；新增输入后重启仍能识别变化**；
    - 指纹在该字段上的取值即"当前有效回合"，追加/重答换新 id、撤销回退到上一存活 id，不依赖任何瞬态字段；
  - `candidate_ref` = 候选正文引用：小候选（≤ 常量上限）随事件直存，超限 spill 到 memory blob、事件存 record id——恢复后可解引用取回候选（评审"重启后取回失败候选"要求）；
- **绑定三元组**：`passed` 记录仅当 `output_fingerprint + input_snapshot_id + acceptance_spec_version` 三者与提交时一致才有效。输出相同但输入变化（用户追加/重答/撤销——`input_snapshot_id` 变）→ 旧记录失效；声明变化（含检查器版本升级——spec_version 变）→ 旧记录失效。
- `TaskView.acceptance` 折叠最新结论 + 绑定三元组 + 历史计数；快照序列化随之补字段（`test_view_serialization_coverage` 守卫强制覆盖）。

### D4 修正机制：额度预占先行 + 内联续跑兼容 + 双通道缺口

**跨崩溃额度保证**（评审 P1-3 的处置）——落盘顺序为定案：

```
① TASK_ACCEPTANCE_CHECKED（verdict=failed + 缺口 + 候选引用）
② TASK_ACCEPTANCE_RETRY_RESERVED（repairs_used 0→1 预占，任务回 PENDING）
③ 发起修正 run（下一轮 act 注入缺口）
```

预占事件**先于**修正生成落盘。崩溃语义：①②已落、③未起 → 恢复见 PENDING + 额度已用 → 该任务的续跑视为**完成本次已预占的修正**（attempt+1，repairs_used 不再增加），续跑候选仍失败则 FAILED；③已起即崩 → 恢复走 task-handoff 的 run 恢复链，额度已占、不重置。任何路径下重启都不得使修正次数超过 1。测试覆盖"预占前崩溃 / 预占后生成前崩溃 / 生成中崩溃"三点。

**修正轮**：必需失败 + 预算允许 + 缺口与上轮 findings 不完全相同 → 走 ②；下一轮 act 注入缺口（`next_step_hint` 投递一次性提示，保留瞬态语义；持久化事实在事件里，恢复经投影重建注入——通道实现期定）。下一轮 finalize 对新候选**重跑全部必需检查**。终止条件（缺口重复 / 预算耗尽 / 检查器 degraded/故障 / 额度已用）→ `FAILED + error_code=TASK_ACCEPTANCE_FAILED`，验收记录保留候选与剩余缺口。选 FAILED 而非 FINISHED+failed：天然复用 task-handoff 依赖条件（on_success 下游不放行）与 review 面 note 通道。

**compat 入口的内联续跑**（评审 P1-2 的处置）：`run_single_task` 无队列无 drain——处置为 ACCEPTANCE_RETRY 时由该入口**内联再执行一轮**（同一执行器循环，上限 1 次，等价于把 drain 折进调用方），返回最终结果。集成测试断言"首次检查失败 → 实际发生一次修正生成 → 返回修正后结果"，仅断言检查器被调用不够。

**影子模式**：`acceptance_mode: off | shadow | required`。行为矩阵（评审 P2-7 的处置，"零变化"限定为无任何检查声明或 mode=off）：

| mode \ 声明 | 无检查 | 仅提示性 | 含必需 |
|---|---|---|---|
| off | 零变化 | 不执行、无事件、零变化 | 不执行、无事件、零变化 |
| shadow | 零变化 | 执行+记录，不改交付 | 执行+记录，**不改交付**（失败也不重排） |
| required | 零变化 | 执行+记录，失败不阻塞 | 执行+记录+门禁（失败→缺口/修正/FAILED） |

恢复时检查器版本缺失（同 id 升版、旧版本未保留）的分档处置与上表同构：off 不执行验收、不影响原交付流程；shadow 记录 unverified、不改变交付决策；required 必需检查判 unverified、阻止成功交付。

### D5 成本口径与预算

修正轮的模型调用经现有 LLM 请求入口进入 usage 归集；TM 侧把 ACCEPTANCE_RETRY 计入任务成本累计，预算耗尽时直接走失败处置。检查器执行为本地 CPU，不进 token 口径，验收矩阵单列其时延与 degraded 次数。

## Risks / Trade-offs

- [FinalizeStep 边界遗漏某条成功路径] → D1 路径映射逐条测试 + "无验收记录落成功终态"断言守卫；compat 内联续跑单独集成测试。
- [中间交流被误当交付候选] → D1 显式区分 + park 回合不产生验收记录的负向测试。
- [修正环节放大模型弱点（生命周期轮 A3 教训）] → 缺口结构化带数据引用（实验 14/14 修正成功的依赖）；一次上限 + 重复缺口即停 + 跨崩溃预占。
- [预占与生成之间的窗口竞态] → 落盘顺序定案（①②先于③）；三点崩溃测试。
- [检查器拖垮事件循环] → 单线程池隔离 + 超时判 unverified + degraded 熔断 + 迟到丢弃；阻塞检查器并发测试。
- [候选大对象撑爆事件] → candidate_ref 上限直存/超限 spill，与 inputs 截断同纪律。
- [影子→必需切换回归] → 冻结 18 答复重放固定回归资产（10 不动 / 8 全拦 / 0 误报）+ off/shadow/required × 三种声明的行为矩阵测试。

## Migration Plan

- 存量事件：无 acceptance 字段 → None/缺省，旧语义完全保留；回退 = 旧代码忽略新字段（task-handoff 同一先例）。
- 默认 `acceptance_mode=off`；启用路径 off → shadow（观察）→ required（放量），与方案 §6 放量规则一致。

## Open Questions

- 检查器 version 的语义粒度（注册时静态字符串 vs 随参数哈希）——实现期定，不影响协议形状。
- 缺口注入通道（guidance 尾注 vs task_spec 小节）——两案接口相同，实现期按装配预算实测取舍。
