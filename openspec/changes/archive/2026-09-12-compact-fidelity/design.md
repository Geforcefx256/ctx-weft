# Design: compact-fidelity

## Context

压缩摘要的生成链：`summarize_for_compact`（purpose=compact 装配 + 单次 LLM 调用，cue 为 `_COMPACTION_INSTRUCTION` / `_AGENT_COMPACTION_INSTRUCTION` 两段散文）→ 产出自由文本 → 直接落库（L3 坍缩为 USER_PROMPT 两节结构；L1 折叠为 AGENT 层 SUMMARY）。cue 散文里「Preserve: goal, facts, decisions, unfinished threads」是无结构约束；fold 一次 supersede 原文不可逆。段摘要（act_recap / Progress So Far）由 observe/background_observe 生成，契约在 ROLE facet，量多且逐段——结构化它会显著放大改动面。

## Goals / Non-Goals

**Goals:**

- 两域 digest（L3 task / L1 agent）结构化 + 宽松降级 + 保真夹具。
- Evidence 引用与 tool-result-recovery 的回读通路联通。

**Non-Goals:**

- 段摘要（act_recap / Progress So Far / retry 进度）**不结构化**——它逐段生成、消费方是下一轮 observe 的 recap 锚点，改格式牵动观察者 cue 与锚点标题契约；其保真由既有「honestly recapping」措辞 + 本 change 夹具间接覆盖（夹具可断言段摘要不丢未完成事项）。
- 不改 fold/supersede 语义（原文不可逆是既定设计，契约化明示而非推翻）。
- 不做摘要质量打分/自动重生成（夹具失败先人工归因）。

## Decisions

### D1 小节集与 cue 改造

- 小节：`Goal / Constraints / Done / Remaining / Evidence` 五节，markdown H2，与仓库「标题即锚点」惯例（`## Progress So Far`）一致。
- cue 在两域现有 Preserve 语义上重写为逐节模板 + 「每节一行起步、有则多写、无则写 none」的指示——显式允许空节防模型硬编。
- **备选（弃）**：JSON 结构化输出——compact 走纯文本流（tools=[]），改 JSON 引入解析脆性与转义地狱；markdown 小节宽容得多。

### D2 宽松解析与降级

- 正则抓 `## <name>` 节；五节中 Goal/Constraints/Remaining/Done 任一缺失 → 整文作 digest、metadata 标 `degraded: true`（Evidence 允许缺）。
- 解析器纯函数、独立单测；降级不重试（重试消耗一次 LLM 调用换不回格式保证，散文 digest 仍可用）。

### D3 Evidence 引用格式与联通

- 引用行模板：`- read_tool_output(<invocation_id>): 一句话说明`。invocation_id 在装配给 compact 的历史里可见（工具结果收敛版/记录 metadata 均含，tool-result-recovery 落地后收敛版自带）。
- **硬依赖声明**：回取通路 = tool-result-recovery 的 read_tool_output；该 change 未落地时，本 change 任务组 2 阻塞、其余组不受影响（Evidence 节允许为空，digest 不因无引用而 degraded）。
- 旧数据（结构化前 digest）无小节：解析按降级路径自然兼容，无需迁移。

### D4 保真验收分三层（证明力与守卫对象匹配）

固定 mock 的输出不随 cue 变化——用「预排摘要 + 存活断言」证明不了 cue 的语义作用（审核探针证实）。验收拆三层，各守各的：

- **管道层（CI · 脚本化 mock）**：预排多轮响应（含埋唯一标记的约束/待办/失败原因/工具产出 + 预排 digest），驱动 summarize + collapse/fold 循环；断言：输入材料进入压缩请求的历史、预排 digest 的标记经解析落库并跨轮存活、降级路径正确、digest 长度有界（≤ 旧版 1.5×）。**只证管道，不证 cue。**
- **请求层（CI · 请求断言）**：捕获 compact 的 LLMRequest，断言 cue 文本含五节契约的完整指示（节名 + 逐节语义句）；cue 被删改（如去 Constraints）→ 此层失败。**cue 契约的唯一 CI 守卫。**
- **语义质量层（非 CI · 真实模型评测）**：真模型是否按 cue 产出含约束的 digest——发布前人工/脚本评测（可用既有 benchmarks 目录承载），MUST NOT 进 CI（不可重复、防不可靠）。
- 夹具入库 `tests/`（unit 级：直接驱动 summarize_for_compact + collapse/fold 循环；不引真 LLM）。
- 「重做检测」在管道层的实现：预排 digest 自带 Done/Remaining 划分，断言落库后归属不变（标记串匹配，不做语义判断）。

## Risks / Trade-offs

- [结构化 cue 推高摘要长度（五节骨架开销）] → 每节一行起步的指示 + 夹具长度断言（digest 不超旧版 1.5×）；L3 坍缩的两节结构（原文 + 摘要）本就有界。
- [模型不守格式 → 大量 degraded] → degraded 有 metadata 标记可观测；夹具量化降级率；必要时后续加一次重试（Open Question）。
- [五维断言的夹具可能脆（依赖措辞）] → 断言用夹具自控的标记串（如约束文本里埋唯一 token），不依赖语义判断。
- [固定 mock 冒充语义证明] → 分层验收（D4）：mock 只证管道、请求断言证 cue 契约、语义质量归真实模型评测——CI 内不假装证明语义。
- [Evidence 引用指向已逐出的存储条目] → 回读工具返回显式未命中（tool-result-recovery D5 语义），digest 不受影响。

## Migration Plan

默认生效；旧格式 digest 走降级路径自然兼容，无迁移。回滚 = 还原代码（degraded 标记与夹具随之失效，无状态残留）。

## Open Questions

- degraded 率偏高时是否加一次「格式修复重试」——待夹具实测降级率后再议（不影响接口与任务结构）。
