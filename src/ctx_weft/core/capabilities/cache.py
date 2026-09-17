"""CapabilityCache：per-session capability 快照。

**填充方是 PrepareStep，不是 ALM**：全仓唯一的 `put()` 调用点在
`core/loop/steps/_capabilities.py`——每次 run 的 prepare 阶段解析完 capability 后写入，
`core/loop/capability_gateway.py` 在每次工具调用时读。改造前这里写的是「实例化时填充
（AgentLifecycleManager.instantiate）」，那条线早已搬去 PrepareStep，文件与注释都没跟着走
（同批还查出 `TaskManager._session_registry` 的 docstring 也在描述一个已不存在的关系）。

对象本身由 `runtime` 在装配期构造，跨 run 存活；per-agent 快照可被 `evict` 逐出，
`register_global` 存的 session 全局控制工具不随之逐出。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ctx_weft.protocols.capability import Capability, qualify

logger = logging.getLogger(__name__)


class DuplicateCapabilityName(Exception):
    pass


@dataclass
class CapabilityCache:
    """Per-session in-memory capability 快照——**工具面的唯一真相源**。

    三个槽，按生命周期分：
      _store / _by_*     per-agent，每个 run 由 prepare 重写（put）、run 收尾逐出（evict）
      _global_by_*       session 全局控制工具，register_global 写入，不随 evict 逐出
      _pinned            per-task，运行期 pin 进来的能力，跨同 task 的多个 run 存活

    支持 by-id 和 by-name 查找。LLM 看到的工具数组由 AssembledPrompt 每次读 tools 时经
    `available()` 现算（sources/capability.py::build_llm_tools），不再是装配期的拷贝——
    所以运行期 pin 进来的能力当轮即可见。
    """

    _store: dict[str, list[Capability]] = field(default_factory=dict)
    _by_name: dict[tuple[str, str], Capability] = field(default_factory=dict)
    _by_qualified: dict[tuple[str, str], Capability] = field(default_factory=dict)
    # session 全局区：内置控制工具——对所有 agent 可见、生命周期为 session、**不随 evict 逐出**。
    # 与 per-agent（_store/_by_*）互补：put 存 per-run 解析的 skill/mcp/agent；register_global 存
    # 控制工具（report_task_outcome / collect_process_report / finish_task …）。
    _global_by_name: dict[str, Capability] = field(default_factory=dict)
    _global_by_qualified: dict[str, Capability] = field(default_factory=dict)
    # per-task pin 区：运行期由 provider 经 CapabilityEvent(kind="pin") 加进来的能力
    # （task_id → 能力列表，插入序即 LRU 序）。**独立于 _store 的第三个槽**，理由是生命周期
    # 不同：_store 按 agent 存、每个 run 重写重逐（put/evict），pin 按 task 存、要跨同一 task 的
    # 多次 run（retry）活着。混进 _store 就等于「每次 retry 丢光刚发现的工具」。
    _pinned: dict[str, list[Capability]] = field(default_factory=dict)
    # agent_id → 模板 forbidden 的 capability id。put() 时记下，pin() 就地据此挡回——
    # 模板的禁用是硬边界，运行期的 pin 不能绕过它。
    # **不随 evict 清**：evict 每个 run 都跑，而 fire-and-forget 的 background observe 可能在
    # evict 之后才真正走到 gateway；那一刻若 _forbidden 已空，一个 pin 就能绕过模板禁用。
    # 代价只是常驻一份 id 集合，量级与 runtime 持有的 agent 记录同阶，不是新的一类泄漏。
    _forbidden: dict[str, set[str]] = field(default_factory=dict)
    # 单个 task 的 pin 上限：越界按 LRU 挤掉最早的。工具面无限膨胀会把上下文吃光，
    # 而「挤掉最早的」正对应「早先 pin 的那些已经用过了」。
    max_pins: int = 8

    def register_global(self, capabilities: list[Capability]) -> None:
        """注册 session 全局能力（控制工具）。幂等覆盖；不进 per-agent，故 evict 不逐出。"""
        for cap in capabilities:
            self._global_by_name[cap.name] = cap
            self._global_by_qualified[qualify(cap.id)] = cap

    def put(
        self,
        agent_id: str,
        capabilities: list[Capability],
        forbidden_ids: set[str] | None = None,
    ) -> None:
        """存入 agent 的 capability 快照，按 cap.id 去重（qualified 名作 LLM 入口键）。

        `forbidden_ids` 是模板声明的禁用集合，记下来供 pin() 就地过滤。

        **绝不碰 _pinned**：put 是每次 run 的 prepare 都跑的，同一 task 的 retry 会再跑一遍；
        在这里清 pin 就等于「agent 每次重试都要重新发现同一批工具」。pin 的清理只有两个
        出口：task 落终态（TaskManagerHooks.on_task_terminal）与 context_limit 退出（ActStep）。
        """
        if forbidden_ids is not None:
            self._forbidden[agent_id] = set(forbidden_ids)
        seen_ids: set[str] = set()
        for cap in capabilities:
            if cap.id in seen_ids:
                raise DuplicateCapabilityName(
                    f"Duplicate capability id '{cap.id}' for agent {agent_id}"
                )
            seen_ids.add(cap.id)

        self._store[agent_id] = list(capabilities)
        for cap in capabilities:
            self._by_name[(agent_id, cap.name)] = cap          # bare: skill lookup / display
            self._by_qualified[(agent_id, qualify(cap.id))] = cap  # qualified: LLM tool calls

    def get(self, agent_id: str) -> list[Capability]:
        """获取 agent 的完整 capability 列表（per-agent 快照 + session 全局控制工具）。

        per-agent 快照缺失（未 put 过，或已被 evict）时不再 raise：退化为「只有全局控制工具」而
        非整体报错——调用方（如 fire-and-forget 的 background observe）常在 evict 之后才真正跑到
        这里，此时仍应能解析全局控制工具（register_global 不随 evict 逐出，见 get_by_qualified_name
        同一注释）。
        """
        caps = self._store.get(agent_id, [])
        if not self._global_by_qualified:
            return list(caps)
        seen = {c.id for c in caps}
        return list(caps) + [c for c in self._global_by_qualified.values() if c.id not in seen]

    def get_by_name(self, agent_id: str, name: str) -> Capability | None:
        """按裸名查找（skill 加载等内部路径）；per-agent miss 回退 session 全局区。"""
        return self._by_name.get((agent_id, name)) or self._global_by_name.get(name)

    def get_by_qualified_name(
        self, agent_id: str, qualified_name: str, task_id: str = "",
    ) -> Capability | None:
        """按 LLM 所见的 qualified 名查找（gateway 工具调用路径）：per-agent 快照 → session
        全局区 → 本 task 的 pin。

        per-agent miss 回退全局区——控制工具非 per-agent 可逐出资源，evict 后仍能解析（修
        background observe 竞态）。pin 区排在最后且**必须显式给 task_id**：不给的调用点
        （不在 task 语境里的内部路径）行为一字不变。
        """
        hit = (
            self._by_qualified.get((agent_id, qualified_name))
            or self._global_by_qualified.get(qualified_name)
        )
        if hit is not None or not task_id:
            return hit
        return next(
            (c for c in self._pinned.get(task_id, []) if qualify(c.id) == qualified_name),
            None,
        )

    # ── per-task pin 区 ───────────────────────────────────────────────────────

    def pin(self, agent_id: str, task_id: str, caps: list[Capability]) -> None:
        """运行期把能力加进当前 task 的可用面（provider 的 `pin` 事件经 gateway 落到这里）。

        就地过 forbidden（模板禁用是硬边界，运行期绕不过去）；按 id 去重（重复 pin 同一个
        工具是常态，不该占额度）；超过 max_pins 按 LRU 挤掉最早的——工具面无限膨胀会把
        上下文吃光，而最早 pin 的那批通常已经用过了。
        """
        if not task_id or not caps:
            return
        forbidden = self._forbidden.get(agent_id) or set()
        slot = self._pinned.setdefault(task_id, [])
        existing = {c.id for c in slot}
        for cap in caps:
            if cap.id in forbidden or cap.id in existing:
                continue
            slot.append(cap)
            existing.add(cap.id)
        if len(slot) > self.max_pins:
            del slot[: len(slot) - self.max_pins]

    def clear_pins(self, task_id: str) -> None:
        """清掉该 task 的 pin。幂等（重复清、清不存在的 task 都安全）。

        唯一出口是 task 落终态的 `on_task_finished`。**ActStep 的 context_limit 退出曾是
        第二个出口，已移除**：`_pinned` 按 task_id 分区，新 task 天然看不到旧 task 的 pin，
        所以「上下文爆了要清掉多占的工具描述」这个理由站不住——真正的膨胀上界由 `pin()` 的
        max_pins LRU 提供（最多 8 条）。而在 act 中途清掉它，等于让 agent 压缩续跑后从零重新
        发现工具，正是 `evict` 的注释明确拒绝在 run 边界做的那件事（续跑连 run 边界都不跨）。

        **已知边界：不是严格「每个 task 必清」。** 出口挂在正常路径上，绕开它的异常终止路径
        （如 cancel_all 清掉从未 start 过的排队任务）会留下残余。残余上限是单 task 的
        max_pins 条，且随 CapabilityCache 本身（per-session）一起回收，故按已知边界接受。
        """
        self._pinned.pop(task_id, None)

    def available(self, agent_id: str, task_id: str) -> list[Capability]:
        """该 agent 在本 task 下的完整可用面：**全局控制工具 → per-agent 快照 → 本 task 的 pin**。

        顺序是刻意的，不是随手排的：它复刻改造前 `resolve_and_bind` 的 `builtin_caps + resolved`
        ——控制工具在前，其余在后。活工具面就是拿这个列表去产 LLM 的 tools 数组，顺序一变，
        tools 的 KV cache 前缀就跟着变一次；只换来源不换内容，就不该付那笔无谓的失效代价。
        （所以**不能**直接复用 `get()`：它是 store-then-global 的相反顺序，且另有调用方
        ——background observe / observe 取绑定清单——它们的顺序语义不该被这次改造牵动。）

        `get()` 保持不含 pin，语义不变；pin 排在最末，是「运行期后加的」这一事实的自然体现。
        """
        store = self._store.get(agent_id, [])
        seen: set[str] = set()
        caps: list[Capability] = []
        for cap in list(self._global_by_qualified.values()) + list(store):
            if cap.id in seen:
                continue
            seen.add(cap.id)
            caps.append(cap)
        if not task_id:
            return caps
        return caps + [c for c in self._pinned.get(task_id, []) if c.id not in seen]

    def get_by_id(self, agent_id: str, capability_id: str) -> Capability | None:
        """按 id 查找。"""
        for cap in self._store.get(agent_id, []):
            if cap.id == capability_id:
                return cap
        return None

    def evict(self, agent_id: str) -> None:
        """per-agent 快照清理（runtime `_run_loop` 的 finally，**每个 run 一次**）。

        **绝不碰 _pinned**：它按 task 存活，而同一个 task 可以跑多个 run（retry / resume）；
        在这里连坐清 pin，agent 每次重试都得从零重新发现工具。清理出口见 clear_pins。
        """
        caps = self._store.pop(agent_id, [])
        for cap in caps:
            self._by_name.pop((agent_id, cap.name), None)
            self._by_qualified.pop((agent_id, qualify(cap.id)), None)

    def get_by_kind(self, agent_id: str, kind: str) -> list[Capability]:
        return [c for c in self._store.get(agent_id, []) if c.kind == kind]

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._store
