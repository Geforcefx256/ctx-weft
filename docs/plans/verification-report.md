# 可靠性方案核验表（2026-09-11，reliability-wp8 产出）

> 方案 §10.3 的逐项填写。空项写「未验证」——不写「预计通过」。
>
> ⚠️ **本表是 2026-09-11（reliability-wp8 收口时）的快照。** 其后 wp7/wp8 的执行限制被
> 整体撤销、工具操作账本的身份与恢复策略被重做（`operation_id` 合并进摄入点铸造的
> tool_call 标识、策略收成两值、「结果不确定」从控制流降级为工具结果）。H1–H4 的结论
> 不受影响（探针 `--expect fixed` 仍四项 True），H5 见下表已作废行。下方「测试基线」的
> 计数已按 2026-09-14 复测更新；性能基线仍是 2026-09-11 的原始数据，未重测。

| 项目 | 基线结果 | 候选结果 | 证据文件/命令 | 结论 |
|---|---|---|---|---|
| H1 存储失败后不伪报成功 | 探针 H1 baseline: emit_rejected=False, notified=53, stored=0 | 探针 `--expect fixed` H1=True；无桩 h1 fixed:true | `docs/plans/verification/verify_agent_architecture.py` + `verify_no_stubs_e2e.py h1` + `tests/integration/test_runtime_storage_failure.py` | ✅ 已修复（wp3） |
| H2 三种状态一致 | 探针 H2 baseline: snapshot=[b] vs full=[a,b] | 探针 H2=True；无桩 h2 fixed:true | 同上 + `tests/unit/test_snapshot_consistent_cut.py` + `tests/integration/test_snapshot_commit_interleaving.py` | ✅ 已修复（wp4） |
| H3 不盲目重复副作用 | 探针 H3 baseline: effect_count=2 | 探针 H3=True；无桩 h3 fixed:true（1→1） | 同上 + `tests/unit/test_operation_recovery_policy.py` + `tests/integration/test_operation_crash_matrix.py`（真退出） | ✅ 已修复（wp5+wp6） |
| H4 执行参数正确 | 探针 H4 baseline: Authorization=*** | 探针 H4=True；无桩 h4 fixed:true（REAL_TOKEN） | `verify_agent_architecture.py` + `verify_no_stubs_e2e.py h4` + `tests/unit/test_gateway_argument_channels.py` | ✅ 已修复（wp0-wp1） |
| ~~H5 限制与弃用可解释~~ | 静态零执行点 | — | — | ❌ **已撤销**——wp7 的 `ExecutionLimits` 连同 wp8 的 budget 接线整体移除（计时与打断是宿主策略，不是 SDK 职责）。`Task.timeout_ms` / `RuntimeConfig.default_task_timeout_ms` 两个死字段已删；本行引用的两个测试文件不在仓内 |
| H6 扩展无需改 core | 当前缺少此接缝 | **未实施** | — | ⬜ 可选（wp9，无收益证据可否决） |
| 性能预算 | 未测量 | 首轮绝对基线已留（见下表） | `scripts/benchmark_runtime_commit.py` + `benchmarks/2026-09-11-baseline.json` | 📊 基线就位（无改前对照——后续变更以此为准） |
| PostgreSQL 真库 | 未验证 | 未验证（本地无 PG 实例） | — | ⬜ 未验证 |

## 首轮绝对基线数据（benchmarks/2026-09-11-baseline.json）

| 场景 | p50 | p95 |
|---|---|---|
| single_session_100_tools | 4.0 ms | 4.5 ms |
| multi_session_10 | 3.1 ms | 5.9 ms |
| recovery_1000_nosnap | 2.2 ms | 3.2 ms |
| recovery_1000_snapshot | 0.1 ms | 0.1 ms |
| observer_backpressure | 0.2 ms | 0.3 ms |

> 快照后恢复 0.1ms vs 无快照 2.2ms——「日志增长但快照后 delta 固定时恢复读取不随总日志线性增长」的目标已兑现（1k 事件级别）。10k/100k 量级未测（事件构造成本高，留宿主生产数据）。

## 测试基线

| 套件 | 结果 |
|---|---|
| `pytest tests` | **3163 过 / 2 失败（既知预存）/ 2 skip**（3167 收集，2026-09-14 复测）|
| 预存失败 ① | `test_compact_flow_e2e.py::test_multiround_retry…`（compact L3 坍缩，非可靠性范畴） |
| 预存失败 ② | `test_observe_outcomes.py::test_default_role_prompt…`（ROLE.md 外部路径，环境性） |
| skip ① | `test_memory_conformance.py`（provider 不暴露 writable cursor——协议面注释） |
| skip ② | `test_grep_tool.py`（环境未装 ripgrep） |
| 探针 `--expect fixed` | H1=True, H2=True, H3=True, H4=True |
| 无桩 e2e | h1/h2/h3/h4 全 fixed:true |

## 验收矩阵覆盖

见 `docs/plans/acceptance-matrix.md`（E-T 14/14 有锚；O-T 14 条有锚、O-T11/O-T12 随 `resolve_operation` 作废、O-T16 未落地；L-T 整表随 wp7 撤销而作废；X-T 2 归 WP9 可选）。
