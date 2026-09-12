# Proposal: context-evidence-delivery

## Why

知识检索（`reference`）与语义召回（`summary`）两个装配源正常产出内容块，但默认 Composer 没有任何渲染路径——块先占用预算配额、通过裁剪，然后在渲染阶段静默消失，最终 LLM 请求不含任何命中证据，也无报错与日志（`_render_bb` 为死代码，`blackboard` 同样静默）。同时裁剪只有来源等级一根轴：reference/summary 恒为最低保护档（7，最先被丢，低于已完成任务的胶囊与摘要），块上携带的相关性 `score` 从未被读取——窗口紧张时直接回答当前问题的证据最先死。

## What Changes

- **渲染路径**：DefaultComposer 新增证据段（检索参考 + 语义召回，含来源与 score 标识），渲染于**尾部动态区**（末条 user、guidance 之前）——贴近生成点、不打穿 prompt cache 前缀（当前 task 锚回合保持字节稳定）。purpose=act 为 MUST，其余 facet purpose 不强制。
- **相关性轴**：预算裁剪保持静态阶梯，新增动态提级——reference/summary 中**位列来源 top-K 且 score 达标**（两个条件同时满足；per-source 硬数量上界，默认 K=3，`top_k=0` 关闭）的条目提级到与当前任务内容同档（4），窗口紧张时**先丢低分证据与陈旧已完成胶囊的顺序反转**；同档内 tiebreak 引入 score——**低分先丢**，无 score 的块（如当前任务历史）在同档内视为最受保护、最后丢。提级以 per-source top-K 为硬上界，防大量高分条目整体挤爆预算。
- **丢弃可观测**：预算裁剪丢弃的块 MUST 留痕（至少结构化日志：kind / source / token / 原因档位）；装配层遇到无渲染路径的 kind MUST 显式 warning（今日 `reference`/`summary`/`blackboard` 三种静默）。
- **贯通验收**：注册真实 provider → 召回命中 → 断言最终 LLM 请求 messages 含证据与来源；预算不足裁掉时断言不出现幽灵内容。
- **blackboard 维持不渲染**（Phase 3 设计裁定不变），仅受益于未识别 kind 警告。

## Capabilities

### New Capabilities

- `context-evidence`: 证据供给契约——检索/召回内容块必有渲染归宿（或显式留痕丢弃）、渲染位置不打穿缓存前缀、裁剪含相关性轴且证据保护有界。

### Modified Capabilities

（无——装配/预算/渲染此前无 spec 覆盖，本 change 首立契约。）

## Impact

- **代码**：composer.py（证据段渲染）、budget.py（动态提级 + 档内 score tiebreak + 丢弃留痕）、priority.py（提级判据入口，静态档位不变）、sources（metadata 透传 score 已在场）、runtime 无需改动（默认组件直改）。
- **测试**：`test_source_priorities` 中 `reference == 7` 的钉死断言需按「静态基线 + 动态提级」更新语义；新增贯通用例（provider → 召回 → 最终请求）。
- **协同**：与 tool-schema-budget 独立；若两者皆落地，预算口径组合为 schema 预留先扣减、证据地板在其剩余内生效（两个 design 互引）。
- **顺序**：独立 change；建议在 conversation-pairing 之后（证据正确供给建立在配对正确之上）。
