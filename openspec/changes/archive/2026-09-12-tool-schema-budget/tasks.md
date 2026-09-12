# Tasks: tool-schema-budget

## 1. 费率与装配预留

- [x] 1.1 `core/utils/estimate.py` 新增 `estimate_tools_tokens(tools, count)`（name+description+schema JSON 单一真源），prepare 与 gateway 两处重复公式收敛改为调用——单测：与既有两处公式逐值一致；`pytest` 全绿
- [x] 1.2 assemble 扣减：budget.apply 前按 cache 快照 + purpose 现算预留（compact 空面零预留、cache 缺失零预留回退旧行为），`AssembledPrompt.metadata` 写 `tools_reserved_tokens` 与 `tools_signature`；`ContextOverflowError` 报错仍用真有效窗口——单测：注册大 schema 工具后预留增长、内容预算缩小、超限裁内容不裁工具声明
- [x] 1.3 装配级回归：动态 pin 新工具后再次装配，预留与指纹更新——集成用例（capability cache pin → 重新 assemble 断言 metadata 变化）

## 2. 循环增量追踪

- [x] 2.1 `loop_guard` 增瞬态 `last_tools_signature` / `last_tools_est`（不进持久快照）；指纹 = 规范化工具定义（name+description+schema，key 排序）整体哈希，与估算共用同一次序列化转储；`request_prompt_estimate` 增量路径在指纹变化时 `delta += estimate_tools_tokens(request.tools) − last_tools_est`，不变时行为与现状逐字节一致——单测：两分支对照 + **名称不变 schema 显著增长 → 指纹变化、估算增长**
- [x] 2.2 pin 大 schema 回归：act 循环中段 pin 大 schema 工具，下一请求估算明显增长、`apply_dynamic_max_tokens` 按新 used 收紧；工具面不变多轮估算与改造前一致（对照断言）
- [x] 2.3 usage 回喂时更新 last_tools_*（真实基线吸收差值）——单测：变化后下一轮恢复纯消息增量口径

## 3. 发送前策略与观测

- [x] 3.1 超限不硬拒：估算超有效窗口时请求照发 + WARNING 留痕（used / eff / tools_signature）+ 既有 compact 比例机制下轮触发——单测断言不抛、日志可检索、max_tokens 收紧
- [x] 3.2 偏差留痕（SHOULD）：tokenizer.observe 回喂处附 tools 分项字段（预留值/指纹/实测 prompt tokens）——单测 caplog 断言三字段同现

## 4. 收尾

- [x] 4.1 与 context-evidence-delivery 组合用例（若彼 change 已落地）：schema 预留先扣减、证据地板在剩余内生效；未落地跳过注明——`ruff` / `mypy` 通过，README 预算章节补工具面口径与决策（不硬拒）说明
