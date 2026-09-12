# Tasks: context-evidence-delivery

## 1. 渲染路径

- [x] 1.1 composer 证据段：`## Retrieved Evidence`（References / Recalled Memories 两小节，条目含 score 与来源标识），落末条 user 尾部动态区、guidance 之前；空节零渲染；未识别 kind 显式 warning（compose 级去重：同 kind 每次装配至多一条；blackboard 同样警告、不设静默名单）——单测：命中两类证据的最终 prompt 含内容与来源；未识别 kind（含 blackboard）触发 warning
- [x] 1.2 cache 前缀稳定回归：interactive 两轮装配、第二轮检索内容变化，断言当前 task 锚回合（Current Task/Capabilities 落点）两轮字节一致，证据仅随尾部变化——新增 `tests/unit/test_evidence_cache_prefix.py`
- [x] 1.3 贯通验收：注册 in-memory Knowledge/Memory provider（真实召回路径）→ 命中 → 断言最终 LLMRequest.messages 含证据与来源；预算压到裁掉证据时断言请求不含幽灵内容——集成用例

## 2. 相关性裁剪

- [x] 2.1 budget 动态提级：`reference`/`summary` 满足 per-source 排名 ≤ K **且** `score >= floor`（配置 `evidence_top_k` 默认 3 / `evidence_score_floor` 默认 0；K=0 关闭）；丢序键扩为 `(-p, score, ts, -tok)`——score 取正号（低分先丢）、缺失 score 记 +inf（同档内最后丢，当前任务历史受保护）——单测边界：单源十条均 0.9 仅前 3 条提级、K=0 全不提级、多 provider 各自计 K、无 score 块在同档内最后丢、既有 5/6 档丢序不变
- [x] 2.2 证据保护对照回归：同预算下「低分证据 → 陈旧胶囊 → 达标证据」的丢弃顺序断言；对照改造前证据存活率提升（同数据两版本行为对照用例）；地板不受侵蚀（identity/当前 user_prompt 恒在）
- [x] 2.3 既有断言迁移：`test_source_priorities` 的 `reference == 7` 更新为「静态基线 7 + 提级 4」双口径——`pytest` 全绿

## 3. 可观测与收尾

- [x] 3.1 budget 丢弃留痕：被丢单元结构化日志（kind/source/token/eff_prio），证据类丢弃单列 warning——单测以 caplog 断言可还原丢弃清单
- [x] 3.2 与 tool-schema-budget 组合用例（若彼 change 已落地）：schema 预留先扣减、证据地板在其剩余内生效；未落地则跳过并注明——`ruff` / `mypy` 通过，README 装配章节补证据供给与配置说明
