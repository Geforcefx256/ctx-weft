# Tasks: tool-result-recovery

## 1. 结果存储与收敛

- [x] 1.1 定义 `ToolResultStore` 协议（put 全文 / get 按 offset+limit 或 tail 窗口读取）+ 内存默认实现（LRU 双上限逐出，条数/字节可配）——单测：分页拼接等于原文、tail 直读末尾、逐出后 get 返回显式未命中
- [x] 1.2 gateway 收敛重构：`_maybe_spill` 改为 `converge(text, invocation_id)`（引用+全长+头预览+尾预览+失败标记），invoke 顺序调整为 全文 put store → 账本 completed 持收敛前全文 → 收敛 → 对话/memory/事件——单测：超阈值时账本 result 为全文、对话 TOOL_RESULT 与 memory 记录均为收敛版
- [x] 1.3 runtime 接线：ToolResultStore 显式注册 > 内存默认（与 OperationStore 同口径）；`spill_tail_chars` 配置项——构造测试：未注册时回取仍可用（内存默认）

## 2. 回读工具

- [x] 2.1 实现 `read_tool_output(invocation_id, offset|tail, limit)`（builtin provider 注册，purposes 含 act，spillable=False，limit 硬上限，无窗口参数时默认首页）——单测：分页/尾部/未命中三形态与有界性
- [x] 2.2 端到端证据贯通：大输出（尾部放关键标记串）→ 收敛版尾部预览含标记 → 模型经 read_tool_output tail 模式读回全文片段——集成用例断言标记可见且可回取

## 3. 重放与恢复收敛

- [x] 3.1 completed 短路重放路径改经 converge（禁止全文直灌对话），收敛引用沿用账本记录的**原执行 invocation_id**（OperationRecord 附带该字段；MUST NOT 用重放路径新铸的 invocation_id）——单测：重放入对话形态与首次一致、引用 id 可在 store 解析
- [x] 3.2 补齐 reconcile「账本 completed 而 task 视图缺 result 记录」分支（现状为直接 continue）：以账本全文经 converge 补写 memory；queryable 查询重放与宿主 `supply_result` 补结果同样统一过 converge——单测覆盖四个入口各自收敛形态；无 store 旧数据（改造前账本截断文本）补写时原样通过、不二次收敛
- [x] 3.3 重放重新入库与回读端到端：清空内存 store 后，经持久账本（SQL OperationStore）completed 短路重放 → 以账本全文重新入库（键 = 原执行 invocation_id）→ `read_tool_output` 实际取回内容——集成用例断言闭环；re-put 失败分支断言显式不可用标记

## 4. 失败显式与收尾

- [x] 4.1 store 写失败/SpillSink 双失败场景：收敛版含显式「全文不可用」标记 + ERROR 日志；SpillSink 成功时路径提示与 read_tool_output 引用并存——单测覆盖三分支（store ok+spill ok / store ok+spill fail / store fail）
- [x] 4.2 既有 spill 截断相关测试更新为新收敛形态；README 工具输出章节更新（回取指引、配置项、内存默认的易失性说明）——`pytest` 全绿 + `ruff` / `mypy` 通过
