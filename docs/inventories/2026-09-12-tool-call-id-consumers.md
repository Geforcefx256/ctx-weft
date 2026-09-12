# tool_call_id 消费点盘点（conversation-pairing · 任务 1.1）

> 目的：核对「单一供值源」覆盖完整性。铸造点 = assistant 回合摄入（act `_run_llm_turn`
> 事件发射前 + observe `run_observe_react` 事件发射前）；锚 = 预铸 assistant 记录 id
> （`MemoryEvent.id` 采纳）。盘点基准：commit 70d0cb6 之后、本 change 实施时。

## 生产（铸造，改动点）

| 位置 | 说明 |
|---|---|
| `core/loop/steps/act.py::_run_llm_turn` | act 回合唯一铸造点：流结束→打断处理之后、`LLM_RESPONSE_FINISHED` 之前；结果随 `_LLMTurnOutput.tool_calls/minted/anchor` 下发 |
| `core/loop/steps/observe.py::run_observe_react` | observer ReAct 回合铸造（live 平面，工具为 SILENT/控制面不入 task memory；锚取唯一值不持久） |
| `core/utils/ids.py::mint_turn_call_ids` | 铸造原语：`tc_{seq36}_{ord36}_{sha256(anchor\|ordinal\|raw)[:12]}`，回合内唯一性自检 |

## 值的中转（消费铸造值，逻辑零改动——透传 tc.id / tool_call_id 参数）

| 位置 | 链路 |
|---|---|
| `act.py::execute` | current_messages 追加 assistant/tool 回合（`persisted_turn.tool_calls`、`tr["tool_call_id"]`） |
| `act.py::_execute_tool_calls` / `_invoke_tool` | `gateway.invoke(tool_call_id=tc.id)` |
| `act.py::_ingest_synthetic_tool_result` | 打断/取消补写的 TOOL_RESULT metadata |
| `capability_gateway.py` invoke 全链 | 事件 payload（CAPABILITY_INVOKED/FINISHED）、TOOL_AUDIT、TOOL_RESULT、plan 派发框 + ack 对、错误补写（`_error_and_record`）、HITL 登记键（`hitl.open`/`registry.decision_for`）、`origin_tool_call_id` 透传给子 task |
| `reconcile.py` | 重入 `invoke(tool_call_id=tc["id"])`（读自 memory 记录 = 铸造值）；补写 metadata 同值 |
| `finalize.py::_ensure_dispatch_frame` | 优先 `task.origin_tool_call_id`（铸造值）；兜底 `generate_id("tcall")`（ULID，charset 安全） |
| `hitl/*`、`control/*`（reducers/converters/types）、`orchestrator/task/*`、`models/task.py` | 纯值存储/投影，无格式校验 |
| `protocols/_legacy_dispatch.py` | load_view 内按 `tool_call_id` 配对派发框——值不透明，铸造值照常配对 |

## 配对/校验算法（不改算法，输入唯一后自然正确）

- `assembler/budget.py::_coalesce_tool_pairs`（裁剪同生共死单元）
- `loop/llm_gateway.py`：`drop_dangling_tool_calls` / `reorder_tool_results_after_calls` / `drop_orphan_tool_results`
- `assembler/sources/_history.py` / `agent_recall.py` / `composer.py`（重建 metadata→LLMMessage）

## 唯一的存量适配（legalize 层留痕）

- `llm_gateway.py::reorder_tool_results_after_calls`：命中「同一 id 被多个 assistant 携带」
  （仅可能来自改造前裸 wire id）→ ERROR 日志（含 id 清单），行为不变。

## adapter（原样透传，验收对象）

- `providers/llm/openai.py`：`tool_calls[].id` 直通 payload。
- `providers/llm/anthropic.py`：`tool_use.id` 直通（`_serialize_messages`）。
- 内部标识形态 `tc_` 前缀、`[a-z0-9_]`、≤64 字符，落在两家 provider 工具 id 约束内。

## 值域变化（对外可见）

- 事件（CAPABILITY_* / LLM_RESPONSE_FINISHED.tool_calls）与 HITL 请求中的 `tool_call_id`
  从裸 wire id 变为内部标识；raw id 保留于 assistant 记录 metadata（`tool_calls[].raw_id`），
  `op_id`（operation_id）随伴随写入。
