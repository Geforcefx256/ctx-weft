"""调度内核：队列 → 派发 → 结局 → task 状态。

`manager.py` 是驱动器；其余都是它的协作者，且三个是**纯函数层**：
`disposition`（run 结局 + 重试预算 → task 处置）、`failure_threshold`（熔断清场
分类）、`hooks`（TaskManager 需要外界给它什么）。

同一条切法贯穿这三个：**纯函数回答「是什么 / 该做什么」，manager 负责「照办 +
发事件」。** 顺序与事件发射永远留在 manager，因为那才是契约。

`disposition.py` 同时是**对 `core.loop` 的契约面**：loop 跑完一次 run 用它的词表
（`RunOutcome` / `RunOutcomeKind`）回报「发生了什么」。它是纯 stdlib 叶子，正适合
当这个角色。

本包不做 re-export，请按子模块引。
"""
