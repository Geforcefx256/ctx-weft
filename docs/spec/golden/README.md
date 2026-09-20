# 黄金用例（golden cases）

> 语言无关的 `事件序列 → 期望投影` 用例。三份实现各自读这些 JSON，跑同一断言。

## 格式

每个 `*.json`：

```jsonc
{
  "name": "用例名",
  "description": "意图",
  "runId": "传给 reduceEvents 的 runId",
  "events": [ /* Event[]，字段见 ../01-events.md */ ],
  "snapshotAt": 6,          // 可选：在第 N 条（1-based）处切分，验证 快照+增量 == 全量
  "expected": {             // reduceEvents(events) 后 RunStateView 的断言子集
    "sessionStatus": "...",
    "tasks": { "<id>": { "status": "..." } },
    "agents": ["<id>", ...]
  }
}
```

## 断言规则

1. **全量**：`reduceEvents(events, runId)` 的字段 ⊇ `expected`，逐字段相等。
2. **快照+增量**（当存在 `snapshotAt=k`）：
   `applyEvents(events[k:], reduceEvents(events[:k]))` 必须等于 `reduceEvents(events)`。
3. `expected` 只列需断言的字段，未列字段不检查（允许实现有额外内部字段）。

## 用例清单

| 文件 | 覆盖的 reducer 分支 |
|------|------|
| `01-session-two-tasks.json` | 基本生命周期 + TASK_STATUS 映射 + 快照/增量一致；**RunStarted / RunFinished 不写 sessionStatus** |
| `02-reopen-requeue.json` | TaskRequeued：回 PENDING、清 outputs、恢复改写后的 prompt（存量 reopen 事件流） |
| `03-suspend-resume.json` | TaskSuspended → TaskResumed 状态分支 |
| `04-subagent-spawn-depth.json` | `_rebuild_agents` 子 agent 推算（spawnDepth=1、parent=root） |
| `05-step-and-context-progress.json` | RunStarted / StepStarted / StepCompleted → currentStep；ReasonCompleted；ActTurnCompleted |
| `06-session-status-transitions.json` | 会话状态：SessionCreated → SessionFinished（中间三层 TaskQueueBlocked/SessionWaiting/SessionRunning 已随类型删除，2026-09-05） |
| `07-session-resumed.json` | RunInterrupted(no-op) → TaskInterrupted(task→INTERRUPTED)；SessionResumed 更新 userPrompt + 回 RUNNING |
| `08-metadata-filler-goal.json` | RecognizeIntentToolCall 回填 session.goal |
| `09-failure-threshold.json` | failureCounter 折叠：TaskFailed +1、熔断失败不计、FailureThresholdHit 本身 no-op（跨快照边界） |
| `10-reopen-chain-multi-step.json` | 三步 plan 的级联 TaskRequeued（存量 reopen 事件流） |
| `11-compact-task.json` | compact 子任务作为普通 task；compact 域事件 no-op |
| `12-inert-events-noop.json` | token budget / capability / HITL / LLM 事件对投影无副作用 |
| `13-task-canceled.json` | TaskCanceled → CANCELED；**RunCanceled 在 reducer 中 no-op**（会话取消看 SessionFinished{CANCELED}） |
| `14-task-failed.json` | TaskFailed → FAILED，error 由 `TaskFailed.error_message` 折入；TaskFinalized 只补 `finishedAt`——它带的 `outputs{output,summary}`/`error` 是给 host 的，reducer 不折 |
| `15-recognize-intent-metadata.json` | RecognizeIntentToolCall 更新 task title/description + session.goal（空值不覆盖） |

> 覆盖范围：`reducers._apply` 的全部分支 + `_rebuild_agents` 的 root/子 agent 推算 + 惰性事件 no-op。
> 每个用例的 `expected` 均由真实 `reduce_events` 验证（见 runner，31 项全过）。

参照实现：`tests/unit/test_golden_conformance.py` —— 读本目录所有 JSON，对真实
`reduce_events` 跑全量 + 快照/增量两类断言。**这是 Java / TS 移植可直接照搬的测试模板**。

> 扩充用例时优先把 Python `test_subtask_reviews.py`、`test_task_scheduling.py` 的断言抽成本目录的 JSON，
> 使三份实现共用同一套真相。
