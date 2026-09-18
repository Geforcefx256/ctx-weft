# 未提交窗口 + memory 暂存 —— 机制审查（2026-09-17）

## 背景

分支 `feat/multimodal` 上对 HITL 两阶段终局（spec 2026-09-09）做了一轮扩展，目前**未提交**：

- **热应答也推迟终局**：`HitlService._commit(defer=True)` 冷热都只登记 `pending_decision`；被叫醒的
  协程由 gateway 重新武装提交点（`_rearm_commit_point_after_hot_reply`）。LLM 开口前暂停 → 热撤销
  （`TaskManager.abandon_round` + `HitlPark`）。
- **memory 暂存**：窗口开着时，task 层 memory 写入经 `ingest_or_stage`（`core/loop/driver.py`）暂存进
  `RoundSnapshot.staged_memory`；`commit_round` 钩子 `_commit_round_writes`（`core/runtime.py`）先终局
  HITL（`HitlResolved`）再按序落盘；丢弃 / 热撤销时随快照扔掉。装配（PrepareStep → AgentRecallSource）
  与 token 估算叠加暂存区。
- **额外提交点**：PrepareStep 压缩前、ActStep 离开循环前、gateway 新开 HITL 前；`control__ask_user`
  调用前（step 5）**不**提交。
- **提交钩子**：所有提交路径（含 `_run_task` 兜底）统一经 `commit_round` 钩子终局待终局 HITL。
- **恢复核对**：`_verify_task_prompts_in_memory` 核对「已开始的 task 提问在 memory 里」这一推断。

本文是对上述机制的只读审查结果。三路并行核查：窗口泄漏、绕过暂存的读写、崩溃一致性。
**新引入** = 暂存 / 热推迟改动带来的；**原有** = HEAD 上已存在（可能被放大）。
CONFIRMED = 代码路径完整追通；PLAUSIBLE = 需运行时条件 / 竞态确认。
行号以审查时的工作区为准，会漂移，以函数名为准。

## 修复状态（2026-09-17，未提交）

| # | 状态 | 修法 | 回归测试（`tests/integration/test_round_staging_fixes_e2e.py`） |
|---|---|---|---|
| H3 / H4 / M3 | 已修 | `TaskManager.commit_round` 全部成功后才弹快照，`events_flushed` 标记让重试跳过补投；`HitlService.commit` 发射失败回滚为待终局（`HitlRegistry.uncommit_claim`）；提交钩子不再吞异常；act 成功后才置 `ROUND_COMMITTED_KEY`；`_run_task` 兜底提交失败按停机处理（`_commit_round_or_halt`）；`stage_memory` 同 id 替换 | `test_commit_retries_after_hitl_resolved_fails_once` / `test_commit_failure_halts_without_losing_the_reply` |
| H2 | 已修 | 结果就是人的答复（工具阶段 `reply_as_result` 请求，`HitlRegistry.result_is_human_reply`）时，reconcile 的事件流完成判定与 gateway 重入重放都不采信事件里那份结果，恒按 HITL 决定重新生成。顺带解决了 M7 里 ask_user 那部分（答复带图不再被补写成文本标记） | `test_restart_after_crash_mid_commit_uses_the_new_answer` |
| H1 | 已修 | 新事件 `TaskMessageAppended{memory_id, agent_id, content, source, timestamp}`，`_inject_user_turn` 写入 / 暂存消息后发出；恢复期 `_restore_appended_messages` 把 memory 里缺的按原 id、原时间戳补回 | `test_recovery_restores_an_appended_message_from_the_event_log` / `test_injected_message_is_recorded_in_the_event_log` |
| H5 | 已修 | act `_execute_tool_calls` 两个暂停检查点先走 `_discard_round_if_uncommitted`；热撤销时剩下的工具调用不补结果，人重答后由 reconcile 当作首执执行 | `test_pause_between_tools_after_a_hot_reply_retracts_it` |
| M1 | 已修 | `reply_to_hitl` 入口遇 `claim_pending` 直接幂等返回 None；记录窗口是否本次所开，`resolve` 抛出或返回 None 时用 `TaskManager.drop_round` 撤掉（不调撤销钩子） | `test_rejected_reply_does_not_leave_a_window_open` / `test_duplicate_reply_keeps_the_original_window` |
| M2 | 已修 | `_inject_user_turn` 重排没排上时只提交本次开的窗；别人的窗里消息随那一轮算数或撤回。注：agent 在跑时 `send_message` 先报忙，此形状从公开入口只在「窗开着、task 已排队未开跑」时可达 | `test_injecting_into_an_open_hot_round_does_not_commit_it` |
| M5 | 已修 | run 收尾（`_settle` 释放槽位）之后自检：这次 park 所等的那个 `hitl_id` 若已有决定，说明应答落在收尾窗口里、它的重排被「还在跑」挡掉了，就地补一次 `resume_task` + `drain`。判据经新钩子 `human_answer_ready` 由 runtime 读 registry 给出（`_hitl_answer_ready`），只认这一次 park 的问题，故不会退化成无条件重排 | `tests/integration/test_cold_reply_settle_race_e2e.py`（两种落点 + 无应答不得重排的反例）|
| M6 | 已修 | 提问记录改用确定性 id（`task_prompt_record_id`：task + 提问文本/部件指纹）；`_verify_task_prompts_in_memory` 改成**双向**核对（猜「已写」却没有 → 补；猜「没写」其实有 → 挡住重复写），判据 = 确定性 id 命中，或存量自动 id 的提问回合文本前缀匹配（reopen 改写过则哈希不同、照常写入）；`suspend.py` 的重复落库改走同一入口 | `tests/integration/test_prompt_and_window_guards_e2e.py` |
| L6 | 已修（其中两类） | 只有真会续跑的应答才开窗：`NoResumeDelivery` 与打到已终态 / 不在 TM 名下的 task 的应答不开窗，走一步终局。停机路径上的外部取消仍会留下开着的窗，与 L5 一起处理 | 同上 |
| M7（截断） | 已修 | reconcile 的事件流完成判定加上 `not f.truncated`，与 gateway 的重放短路同口径 | 同上 |
| L10 | 已修 | 三处过时注释改写（driver 的提问落库说明、runtime 与 registry 的 `reply_memory_id` 段）；并注明「第几次应答」那一维已成历史包袱：撤销的答复不再进 memory，`reply_attempt` 与 `HitlReplyRetracted` 的折叠只剩这一个用途 | — |
| M8 | 已修 | 补写不再借用「有未决 HITL 就跳过」（那是重排判据），只避开**正在跑**的 task；补写的时间戳改用那条答复真正终局的时刻（`resolved_at`），对话顺序不再颠倒；活 TaskManager 那条续跑路径也补跑一次 | `tests/integration/test_recovery_backfill_and_cancel_e2e.py` |
| L5 | 已修 | `cancel_session` 先丢弃开着的窗（待终局答复随之退回待答）再收口气泡，事实照常落盘；「会话安静」判据补上「有待终局答复」与「有开着的窗」两条（`claim_pending_for_session`）；逐出会话前用 `drop_round_buffer` 兜底清掉总线缓冲 | 同上 |
| M7（图片） | 已修 | 病因是**两套图片标记并存**：事件用的是自制短标记（只留 ref 前 12 字符，取不回），而仓库里 L0.5 占位本就是「纯文本 + 完整 ref + `media:get_image` 可取回」。`redact_content_for_event` 改用后者；从事件补写的记录用 `placeholder_refs` 声明 ref（GC 的 mark 只看结构化字段）。内联 base64 无 ref 可写，保留短标记 | `tests/integration/test_result_image_survives_backfill_e2e.py` |
| L1–L4 / L7 / L8 | 已修 | L4 审计记录、L3 继承复制各自改用确定性 id（重入 / 撤销后重跑不再叠加）；L1 assistant 回合也走暂存分流（零 chunk 收场时不再直写）；L2 压缩的忙判据加上「有开着的窗 / 有待终局答复」；L7 暂存块的排序键取视图最大序号往后排（且不写进将要落盘的 metadata）；L8 消息注入只让**同一个 task** 的气泡跟着这一轮 | `tests/integration/test_deterministic_ids_and_guards_e2e.py` |
| 其余 | 未修/接受 | M4 事件面残留（重入多一条 `CapabilityInvoked`，是事实记录）、L9（`RoundCommitted` 必须排在补投之前，改不动帧序契约）、M7 里人备注的图（未外部化、无 ref；由「按决定重新生成」覆盖） | — |

---

## 高危

### H1 注入消息的正文不在任何事件里，崩溃即丢 —— 新引入，CONFIRMED

- `_inject_user_turn`（runtime.py）算出 `_event_jsonable` 后丢弃；`requeue_for_message` 发的
  `TaskRequeued` payload 只有 `reason`；全仓无承载注入消息正文的事件。
- 提交顺序是「事件先落盘、暂存后落盘」。崩在两者之间：日志里这一轮已发生（`RoundCommitted` /
  `TaskRequeued` / `TaskStarted`），memory 里没有这条消息；host 已把用户消息帧发出去，界面上消息在，
  模型永远看不见。
- HEAD：消息即时写 memory，这几个崩溃点不丢。

### H2 ask_user 崩溃后重答，模型拿到旧答案 —— 新引入，CONFIRMED

- ask_user 重入不关窗（gateway step 5 例外）+ 热答复推迟终局 → `CapabilityFinished` 进窗口缓冲，
  提交时随 `commit_provisional` **先于** `HitlResolved` 落盘。
- 崩在两者之间 → 重启后气泡 pending、task 挂起，人重答。ReconcileStep 先判 `f.finished`
  （事件流完成判定）再判 `_waiting_human`，直接用事件里的旧结果补写（`via=event-finished`）；
  gateway 重入的「已完成即重放」短路同理。
- 后果：新 `HitlResolved` 与 memory 内容不一致，人的第二次回答被静默丢弃——又是「答了没用」。
- HEAD：step 5 对 ask_user 也提交、热投递就地终局，`HitlResolved` 先于 Finished，无此问题。

### H3 `HitlResolved` 发射失败，答复照样进 memory —— 新引入，CONFIRMED

- `HitlService.commit` 先 `registry.commit_claim`（内存判终局）再 `_emit_resolved`；发射抛
  `PersistenceUnavailableError` 时，`_commit_round_writes` 吞掉异常后照样 ingest 暂存区。
- 结果正是本机制要避免的反向不一致：memory 有答复、日志里气泡仍 pending。重启后重答：
  - UserTurn：`reply_attempt` 仍为 0，记录 id 相同 → memory 幂等 no-op，新答案被吞；
  - ask_user：结果 id 已在 memory → 不算 dangling，新答案被忽略。

### H4 提交时补投失败：暂存丢失、HITL 卡死、窗口成孤儿 —— 先弹快照原有，后果新引入，CONFIRMED

- `TaskManager.commit_round` 在任何 await 之前 `self._rounds.pop(task_id)`；随后 `RoundCommitted`
  发射或 `bus.commit_provisional` 抛 `PersistenceUnavailableError`：
  - 暂存列表随快照丢失；
  - 待终局 HITL 既未 commit 也未 release：不在 `list_pending`，`resolve` 因 `claim_pending` 返回
    None —— 进程内无法重答；
  - 总线把窗口原样放回（`commit_provisional` 失败语义），但 TM 已无这一轮 → 此后该 task 的事件全进
    孤儿缓冲，而 memory 写入因 `is_round_open=False` 直接落盘，形成「memory 有、日志无」。
- act `_commit_round` 在提交**之前**置位 `ROUND_COMMITTED_KEY`，失败后不会重试。
- `_run_task` 在 `except Exception` 块内部调用的兜底 `commit_round` 若抛
  `PersistenceUnavailableError`，不被同级 `except PersistenceUnavailableError` 捕获，逃出
  `_run_task`，`_running_tasks` 不清理，agent 在进程内一直显示忙。

### H5 热答复后在工具之间暂停，不撤回 —— 新引入，CONFIRMED

- act `_execute_tool_calls` 的两个暂停检查点（工具间 `_interrupt_pending`、`_await_tool_or_stop`
  返回 False）直接补「已取消 / 被打断」合成结果 + `launch_background_observe` + `_park_for_interrupt`，
  **不经** `_discard_round_if_uncommitted`。
- 热应答醒来后到下一个提交点之前命中这两处：答复不被撤回（违背热撤销设计），合成结果被暂存，
  `_cold_park` 的 `HitlOpened` 在窗口里缓冲，最终由 `_run_task` 兜底提交。background_observe 装配
  不叠加暂存区，复盘看不到人的答复。
- 现有测试只覆盖「LLM 开口前暂停」。

---

## 中等

### M1 `reply_to_hitl` 开窗后失败不关窗；入口不认 `claim_pending` —— 原有，被放大，CONFIRMED

- 开窗（`begin_round`）之后 `hitl.resolve` 里 `ReplyIntake.normalize` 校验失败抛异常 → 异常直接抛给
  host，窗口不关。
- 入口守卫只判 `not pending.resolved`：同一 HITL 已待终局、原窗口已被弹出时（例如前一次提交卡在 I/O
  await 中），双击 / 陈旧页签重放开出**新**窗口，`registry.claim` 返回 None → `return None`，窗口泄漏。
- 泄漏窗口 + 该 task 正由已提交过的 run 在跑 → 其提交点全是 no-op，下一个问题的 `HitlOpened` 被缓冲；
  热重武装因 `round_hitl_id` 对不上而不触发。现在工具结果 / TOOL_AUDIT 也被暂存，压缩与 observe 看不到。

### M2 消息注入时提交了**别人**的窗口 —— 新引入，PLAUSIBLE

- `_inject_user_turn` 在 `requeue_for_message` 返回 False 时就地 `tm.commit_round`，但 `begin_round`
  对已开的窗是 no-op。注入到一个首 chunk 前的新 task、或开着热应答窗 / 冷续跑窗的 task 时，消息被塞进
  那一轮并把它提前提交：那一轮失去撤销，其中 HITL 在 LLM 开口前被终局、暂存提前落盘。

### M3 暂存落盘失败被吞 —— 新引入，CONFIRMED

- `_commit_round_writes` 逐条 ingest，异常只记日志。快照已弹出 → 进程内装配看不到该记录，
  `user_prompt_in_memory` 仍为 True → task 缺原始提问；task 若已 FINISHED，恢复时跳过终态，永久缺失。
  ACTIVE/SUSPENDED 重启后补写的提问时间戳是恢复时刻，排在已有回合之后。

### M4 存储故障停机后窗口跨 run 保留，暂存重复 —— 新引入，PLAUSIBLE

- `PersistenceUnavailableError` 分支不提交也不丢弃 → 窗口与暂存保留；下一次 run `begin_round` 幂等
  复用旧快照。`_reconcile_or` / `_dangling_tool_calls` 看不到暂存结果 → 重新 invoke dangling ask_user：
  同 `res_` id 的结果再暂存一份、TOOL_AUDIT 以新随机 id 再暂存、重复发 `CapabilityInvoked`。
  AgentRecallSource 的叠加不按 id 去重 → prompt 里同一 tool_call 两份结果。

### M5 冷应答与 run 的 park 收尾竞态，task 不重排 —— 原有，PLAUSIBLE

- run 抛 `HitlPark` 后，在 `_run_task` 的兜底提交 → `apply_run_outcome` → `_settle` 这几个 await 之间
  冷应答到达：`resume_task` 因 task 仍在 `_running_tasks` 直接返回、不 drain。另一条来路：
  `_resolve_human` 在 `hitl.open` 发事件的 await 中被应答抢先，waiter 见 `claim_pending` 判驱逐 → park。
- 应答早于兜底提交：HITL 终局、答复落盘，但 task 停在 AWAITING_HUMAN 永不重跑，`list_pending` 为空，
  会话「答了没反应」，只有 `/resume` 能救。晚于兜底提交：窗口一直开着，答复留在暂存区。

### M6 恢复期提问核对的两个方向误判 —— 原有 + 新引入，PLAUSIBLE

- `task_from_projection` 只把 ACTIVE/SUSPENDED 推断为「提问已在 memory」；AWAITING_HUMAN / INTERRUPTED /
  恢复后的 PENDING 推断为 False → driver 会**重复写入**原始提问（时间戳为当前时刻，排在对话末尾）。
  `_verify_task_prompts_in_memory` 只核对 True 的那部分，不反向修正。（原有，未确认 composer 是否去重）
- 反向误判（新引入的核对谓词「任意一条无 source 的 user 回合」）：reopen 后旧提问仍在 memory，
  `TaskStarted` 落盘后崩在修订版提问写入之前 → 旧提问让核对通过，修订版永不补写；老版本注入的答复
  （无 `source`）也会被当成提问。

### M7 从 `CapabilityFinished` 补写的结果是脱敏 + 截断版 —— 原有路径，被放大，CONFIRMED

- 事件 payload 是 `redact_content_for_event(content)[:8000]`，图片被渲染成
  `[image <mt> <src>:<12 字符>…]` 文本；reconcile 补写时 `content = str(result)` 且 `converge=False`
  → 人答复里的图变成文本标记，`get_image` 取不回。`_conclude_without_rerun` 带图的 message 同理。
- reconcile 补写不检查 `facts.truncated`（gateway 重放检查了）：`spillable=False` 且超 8000 字符的
  工具崩在 Finished 与 ingest 之间，截断内容被当完整结果写入。
- 暂存让「Finished 已落盘、memory 未写」这个窗口从极窄变成整个提交过程。

### M8 同一 task 多条答复、终局到一半崩溃，答复永久缺失 —— 原有，PLAUSIBLE

- 同一 task 还挂着别的 pending 时，`_inject_resolved_user_turns` 按 `parked_or_inflight_task_ids`
  跳过它；之后人答另一条走活 TM 路径（`_recover_in_existing_tm` / `_resume_in_existing_tm`），那条路径
  根本不调补写。冷路径下补写排在 `user_reply` 之后，时间戳顺序颠倒。

---

## 低危 / 边角

| # | 问题 | 性质 |
|---|---|---|
| L1 | LLM 流零 chunk 正常结束：`_ingest_assistant_turn` 在窗口里直接写 memory（新 task 时提问还在暂存区），plain-text 的 background_observe / `_cold_park` 也跑在窗口里 | 新引入，PLAUSIBLE |
| L2 | `compact_agent` 判忙只看 run 令牌，与 `_inject_user_turn` / `reply_to_hitl` 开窗之间有竞态，压缩看不到暂存 | PLAUSIBLE |
| L3 | 丢弃一轮会还原 `user_prompt_in_memory`，但 `inherit_memory` 的 AGENT 层复制不撤销，下次 run 重复复制 | 新引入，PLAUSIBLE |
| L4 | TOOL_AUDIT 暂存时拿随机 id：走事件补写路径时永不补；多次冷重入各写一条 | 新引入 |
| L5 | 窗口开着时 cancel / purge：`_cancel_session_hitl` 按 `list_pending` 收口、漏掉待终局气泡；`TASK_CANCELED` 进缓冲；TM 被逐出后 bus 的 `_provisional` 泄漏；purge 摘掉未终局 HITL，重启复活 | 原有，暂存数据一并丢 |
| L6 | 其他泄漏窗口的边角：`NoResumeDelivery` 应答（开窗后无续跑）、应答打到已终态 task、`_run_task` 遇外部 `CancelledError`（进程关闭） | 原有；暂存数据丢失为新 |
| L7 | 提交前暂存块在 prompt 里用 `seq_no = idx` 兜底，与 provider 的 per-scope seq 不可比；时间戳相同时提交前后排序可能不一致（Windows 时钟分辨率 15.6ms） | 新引入，PLAUSIBLE |
| L8 | 被收口的旧气泡属于其他 task 时：`_cancel_pending_hitl_of(defer=True)` 按 agent 收口，提交 / 撤销钩子按 task 查 → 卡在待终局 | 原有，PLAUSIBLE |
| L9 | 新 task 崩在 `RoundCommitted` 之后、`commit_provisional` 之前：host 已发用户消息帧，core 无此 task | 原有 |
| L10 | 注释过时：`act._commit_round` 附近 driver 相关描述、runtime `_write_hitl_reply_turn` 里 `reply_memory_id` 那段仍描述 fold 撤销 | 新引入 |

## 查过、确认安全的点（节选）

- 冷应答、消息注入、新 task 三条主路径上，常规写入要么经暂存、要么只在提交点之后发生。
- `_last_user_prompt`（打断续接前缀）只在提交后可达；热撤销走非 edit 的 park。
- token 基线：`_estimate_tokens` 叠加暂存；每个 run 新建 `LoopGuard`，基线只在 act 提交后才存在。
- 暂存记录落盘时保留原时间戳；`ingest_tool_result` / `ingest_or_stage` 的返回值无人使用。
- born-cancel：`_run_loop` 把 `CancelledError` 转成 CANCELED 结局，兜底提交关窗；born-pause 走
  `discard_round` 正常关窗。
- 「owner TM 不 alive 于是新建 TM、窗口留在旧 TM」实际不可达（`is_current` 与取 TM 用同一张 map）。
- D 点（`HitlResolved` 全发完、暂存落盘到一半）崩溃：UserTurn 由 `_inject_resolved_user_turns` 补写，
  「Blocked by human」/ 执行前错误 / `_conclude_no_record` 由 reconcile 按缓存决定重入得到同内容。
