"""HitlSnapshot：折叠事件得到的、可直接装填进 `HitlRegistry` 的内存态。

恢复是「喂进来」，不是「查回去」：core 的一切 HITL 查询只读内存，装填的完备性
由恢复路径承担（spec §3.1）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ctx_weft.core.hitl.registry import PendingHitl
from ctx_weft.protocols.hitl import HitlDecision


@dataclass
class HitlSnapshot:
    """`pending`：仍未终局的请求。

    `decisions_for`：`(session_id, tool_call_id, stage) → (决定, resume_state)`。键必须
    是这三维——只按 tool_call_id 会让 A 会话的批准替 B 会话同名 id 的调用开门，也会让
    同一 tool_call_id 下的授权决定与工具阶段决定相互覆盖（Task 4.5：跨会话 / 跨阶段两个
    洞）。**成对**是硬要求——冷路径重入调 `resume(ask_id, decision, resume_state, ctx)`，
    丢掉 resume_state 就要求 provider 重做让出前的工作（spec §7.2）。只收录**可用**的决定。

    **`decisions_for[*][0].message` 仍是 event 侧引用**：折叠只做到
    `content_from_jsonable`，得到的 `ContentPart` 里若含 blob 引用，那引用落在
    **事件** blob store 的命名空间下，不是记忆 blob store 能打开的。装填进
    `HitlRegistry`/喂给消费方之前，必须先 `hydrate_event_content` 再
    `normalize_content` 写入记忆 blob store，并对失败路径做 `downgrade_images_to_text`
    兜底（该兜底本身绝不可再抛）——现成实现见 `runtime.py:1943` 的
    `_cold_hitl_decision`（spec §12.3.3 第二段）。`fold_hitl_snapshot` 保持同步，
    结构上做不了这一步；这是调用方（下一阶段接 `load_snapshot` 到恢复路径时）的责任。
    """

    #: 折叠期的工作账：**开过的请求**（按 hitl_id）。`HitlClosed` 与 `outcome=cancelled`
    #: 会把条目摘掉，所以它的大小 = 「开了还没了结的」≈ 活跃数，不随会话长度增长。
    #:
    #: 为什么在这里而不是折叠的本地变量：`apply_hitl_event` 要能被**增量** apply，那就要求
    #: 累加器是**唯一**的状态。放在外面，一次性折叠和增量折叠各带一份，两份迟早分叉。
    opened: dict[str, PendingHitl] = field(default_factory=dict)
    #: 折叠期的工作账：`decisions_for` 的键 → 当前那条决定是谁的。`HitlClosed` 据此只销
    #: 自己那一条——同键会被「重问副本」的后一条决定覆盖，不记主人就直接 pop，一条**旧**
    #: 请求的了结会把**新**请求的决定连带删掉 → 同一个工具重新求批。
    decision_owner: dict[tuple[str, str, str], str] = field(default_factory=dict)

    #: 折叠期的工作账：**开过的请求**（按 hitl_id）。`HitlClosed` 与 `outcome=cancelled`
    #: 会把条目摘掉，所以它的大小 = 「开了还没了结的」≈ 活跃数，不随会话长度增长。
    #:
    #: 为什么在这里而不是折叠的本地变量：`apply_hitl_event` 要能被**增量** apply，那就要求
    #: 累加器是**唯一**的状态。放在外面，一次性折叠和增量折叠各带一份，两份迟早分叉。
    opened: dict[str, PendingHitl] = field(default_factory=dict)
    #: 折叠期的工作账：`decisions_for` 的键 → 当前那条决定是谁的。`HitlClosed` 据此只销
    #: 自己那一条——同键会被「重问副本」的后一条决定覆盖，不记主人就直接 pop，一条**旧**
    #: 请求的了结会把**新**请求的决定连带删掉 → 同一个工具重新求批。
    decision_owner: dict[tuple[str, str, str], str] = field(default_factory=dict)

    #: 折叠期的工作账：**开过的请求**（按 hitl_id）。`HitlClosed` 与 `outcome=cancelled`
    #: 会把条目摘掉，所以它的大小 = 「开了还没了结的」≈ 活跃数，不随会话长度增长。
    #:
    #: 为什么在这里而不是折叠的本地变量：`apply_hitl_event` 要能被**增量** apply，那就要求
    #: 累加器是**唯一**的状态。放在外面，一次性折叠和增量折叠各带一份，两份迟早分叉。
    opened: dict[str, PendingHitl] = field(default_factory=dict)
    #: 折叠期的工作账：`decisions_for` 的键 → 当前那条决定是谁的。`HitlClosed` 据此只销
    #: 自己那一条——同键会被「重问副本」的后一条决定覆盖，不记主人就直接 pop，一条**旧**
    #: 请求的了结会把**新**请求的决定连带删掉 → 同一个工具重新求批。
    decision_owner: dict[tuple[str, str, str], str] = field(default_factory=dict)

    pending: dict[str, PendingHitl] = field(default_factory=dict)
    decisions_for: dict[tuple[str, str, str], tuple[HitlDecision, dict[str, Any] | None]] = field(
        default_factory=dict)
    #: `hitl_id` → 那条**已终局请求本身**（`task_id` / `delivery` / `form` /
    #: `created_at` / `decision` 都在）。取消（cancelled）不收录——它不是可续跑的决定。
    #:
    #: **键是 `hitl_id`，不是 `decisions_for` 的三元组**：`decisions_for` 是决定缓存，
    #: 它的三维键（session / tool_call / stage）是一条安全边界，不能动；而本表要回答的是
    #: 「这个 session 里哪些请求已终局」，对**没有 tool_call_id 的请求同样成立**——
    #: `_cold_park`（`act.py`）开的 `UserTurn` park 就没有 tool_call_id，用三元组
    #: 做键会把整整一类请求排除在外（Task 9 复审 Finding 2）。
    #:
    #: 为什么单开一份而不只留 `decisions_for` 的 `(decision, resume_state)`：那个值里
    #: 没有 `task_id`、没有 `delivery`。少了这一份，`load_snapshot` 装出来的全是
    #: `task_id=""` 的占位，`resolved_for_session()` 既认不出该唤醒哪个 task、也认不出
    #: 哪些是 `UserTurn`——「人答过了，那句话却静默消失」这条故障就留在原地。
    #:
    #: 可选：手工构造的快照（既有单测、host 直接喂）不填它，`load_snapshot` 退回
    #: 只带决定的占位项，行为与本字段引入之前逐字节一致。
    resolved: dict[str, PendingHitl] = field(default_factory=dict)
