# Design: context-evidence-delivery

## Context

装配流水线：sources 并发产块 → `PriorityBudgetStrategy.apply`（按 `slot_priority` 静态档 + 两个动态覆盖裁剪）→ `DefaultComposer.compose`（按 kind 渲染）。现状三个断点：composer 渲染的 kind 集合（identity/background/history/task_spec/directive/capabilities/guidance）不含 `reference`/`summary`/`blackboard`；`slot_priority` 把 reference/summary 固定为 7（最先丢，有测试钉死）；budget 的丢序键 `(-p, ts, -tok)` 无相关性维度，而两个 source 的块 metadata 已带 `score`（knowledge.py / long_memory.py）。guidance 的「尾部动态区」模式（GuidanceSource → composer 末条 user 收尾）是现成的渲染先例。

## Goals / Non-Goals

**Goals:**

- 证据有渲染归宿，且落位不破坏 prompt cache 前缀稳定性。
- 相关性参与裁剪决策，但静态阶梯与既有地板语义不变。
- 静默盲区清零：丢弃留痕 + 未识别 kind 警告。

**Non-Goals:**

- 不复活 blackboard 渲染（Phase 3 裁定：predecessor 结果经 memory recall 浮现；本 change 仅让其享受警告）。
- 不引入向量重排/二次相关性模型——只用 source 已产出的 score。
- 不改 compact/observe 等 facet purpose 的证据渲染（列为可选后续）。
- 不动 `slot_priority` 的静态档数值。

## Decisions

### D1 渲染落点：尾部动态区，guidance 之前

- 证据随每次装配重新查询（query = 当前 user_prompt），内容逐轮变——注入 task 锚回合会每轮打穿 cache 前缀，恰好违反该回合存在的理由；独立 user 消息破坏「末条 = 当前消息」的结构约定。
- 形态：`## Retrieved Evidence` 单节（两小节：References / Recalled Memories），每条 `- [score] 来源: 内容`；置于 guidance 之前（guidance 恒收口末尾，act 态势提示优先级高于证据罗列）。
- 空节不渲染（无命中零开销）。

### D2 相关性轴：动态提级（rank∧floor，硬上界 K）+ 档内 score 丢序

- **提级判据（两条件同时满足，per-source）**：块 `source` 内 score 排名 ≤ K（默认 3，`evidence_top_k` 可配，K=0 关闭）**且** `score >= evidence_score_floor`（默认 0，即仅按排名；可配置提高门槛）→ 档位 7 → 4。**AND 语义保证硬上界**：无论多少条目高分达标，每来源提级条数 ≤ K——审核探针证实 OR 语义下（K=3、十条均 0.9）会十条全提级，上界失效。与现有「当前 task 内容 → 4」同一档（不高于它，证据不该压过当前任务本体）。
- **档内丢序键：`(-p, score, ts, -tok)`，score 升序（低分先丢）**。注意符号方向：丢弃循环按 sorted 升序从前遍历，`-p` 使高档先丢；score 必须**取正号**才能让低分排前——写成 `-score` 会先丢高分（审核探针证实丢序 0.9 → 0.6，方向反了）。
- **缺失 score = +inf（同档内最后丢）**：当前任务历史等无 score 块与提级证据同处 4 档时，有 score 者先丢、无 score 者殿后——防修改符号后反噬当前任务历史。既有 5/6 档全为无 score 块（彼此同 +inf），退化回 `(ts, -tok)` 原序，行为不变。
- **为什么提级到 4 而非保留 7 改丢序**：丢序只决定同档内先后，7 档本身先于 6/5 整档消失——不提级则证据永远先于胶囊死，score 排序无意义。
- **备选（弃）**：预留固定预算份额（evidence reserve）——份额与实际命中量脱钩，无命中时浪费、多命中时不足；top-K 提级天然随命中量伸缩且有硬上界。
- 上界：per-source ≤ K（默认两来源合计 ≤ 3+3 条提级），多 provider 各自计 K。

### D3 丢弃留痕：结构化日志先行，事件可选

- budget 丢弃循环内对每个被丢单元 `logger.info`（kind/source/token/eff_prio）；证据类丢弃单列 warning。
- **备选（弃）**：新增 EventType（如 ContextPruned）——事件枚举与 host 契约面广，先以日志满足验收，事件列为可选任务。
- composer 未识别 kind：compose 入口处对渲染后剩余 kind 集合 warning（一次一条，避免刷屏）。

### D4 与 tool-schema-budget 的口径组合

- 若两 change 皆落地：有效预算 = `effective_limit − tools_reserved`（D 的扣减先行），证据地板在其剩余内生效。两 change 独立可单独回滚；组合语义在两侧测试中各加一条组合用例。

### D5 既有断言迁移

- `test_source_priorities` 的 `reference == 7` 改为断言「静态基线 7 + 提级路径 4」两个口径；`slot_priority` 函数签名不变（提级逻辑在 budget 层，函数保持纯静态）。D2 与 D4（tool-schema-budget）的组合口径见彼 design——schema 预留先扣减、证据地板在其剩余内生效。

## Risks / Trade-offs

- [score 语义异构（knowledge 的排序分 vs memory 的相似度）跨 source 不可比] → top-K 按 source 内排名（k 位次而非绝对分），阈值仅作可选加严；文档注明 score 不跨源比较。
- [证据段增大尾部、加剧 cache 尾部重付] → 证据位于末条 user 尾部动态区本就是每轮重付区，增量与命中量成正比且有 top-K 上界；预算估算按内容自然计入。
- [低分证据仍在 7 档、窗口极紧时照样全灭] → 有意为之（有界保护）；留痕日志让「证据被裁」可见，运营可调 top-K/阈值。
- [警告刷屏（高频装配 × 未识别 kind）] → compose 级去重（同 kind 每次装配至多一条）；blackboard **同样警告**（与 spec 一致，不设静默名单——它正是需要被看见的常驻盲区）。

## Migration Plan

默认行为直接切换（渲染 + 提级 + 留痕同时生效）；配置项 `evidence_top_k` / `evidence_score_floor` 提供回退调节（top_k=0 等效关闭提级）。无数据迁移。

## Open Questions

（无——落点、提级档位、tiebreak、留痕形式均已定；facet purpose 证据渲染与 ContextPruned 事件已明确列为可选后续。）
