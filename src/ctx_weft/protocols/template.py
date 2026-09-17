"""AgentTemplate / IdentityFacet。

Identity 是 AgentTemplate 的内禀字段（一等公民），不嵌套在 Capability 里。
Capability 通过 capability_refs 引用外部能力。

模板进入 core 的唯一通道是 AgentCapabilityProvider（spec 2026-07-22 方案 B）；
目录格式 loader 见 providers/agent_template_local。

详见设计文档 §4.6。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ctx_weft.protocols.capability import Purpose

CapabilityMode = Literal["optional", "required", "forbidden"]


# ── Identity ──────────────────────────────────────────────────────────────────


@dataclass
class IdentityFacet:
    """Agent 在某个 purpose 下的身份呈现。

    例：
      facets["act"] → Actor 阶段的 SOUL（人格、价值观、行为风格）
      facets["observe"] → Observer 阶段的 ROLE（职责定义、评估准则）
    """

    text: str  # 该 purpose 下的身份文本（SOUL 或 ROLE 内容）
    style: str | None = None  # 该 purpose 的输出风格偏好（可选）


# ── Capability Reference ──────────────────────────────────────────────────────


@dataclass
class CapabilityRef:
    """Template 中对外部 capability 的声明引用。"""

    capability_id: str  # 匹配某 provider 返回的 Capability.id
    mode: CapabilityMode = "optional"
    # optional  → 由 retrieve(ctx) 决定是否出现
    # required  → 强制加载，走 list() 精确查找
    # forbidden → 强制排除，即使 retrieve() 返回也过滤掉
    # 注：purposes 不在 ref 中——以 Capability 自身声明为准（v0.3 决议）


# ── Config 子结构 ──────────────────────────────────────────────────────────────


@dataclass
class MemoryConfig:
    """Memory 配置（从 template 拷贝到 Agent 实例化时快照）。"""

    short_window_size: int = 20
    summary_threshold: int = 20
    use_long_term: bool = True
    subscribed_blackboard_topics: list[str] = field(default_factory=list)


@dataclass
class LoopConfig:
    """Loop 配置（从 template 拷贝到 Agent 实例化时快照）。

    完整字段见设计文档 §6.8.2（含 compact 阈值等）。
    """

    max_turns_per_act: int = 50
    max_turns_per_observe: int = 5       # ObserveStep ReAct 循环上限
    failure_threshold: int = 3
    max_spawn_depth: int = 4
    compact_token_ratio: float = 0.8
    # 压缩「压到」目标比率（滞后区下沿）：触发后一路升级直到 token 估算 < 此比率 * context_limit。
    # 0 = 无滞后，回退等于 compact_token_ratio（压到刚低于触发比率即停）。应设得比 compact_token_ratio 低。
    #
    # 默认 0.4 而非 0：滞后区是 mid-act 恢复（见 context_limit_stop_ratio）能不能成立的前提。
    # 无滞后时 compact 把估算压到「刚低于 0.8」就停，act 续跑第一轮加上本轮输出立刻又越线，
    # 一次恢复配额当场蒸发；0.4 给出足够的续跑空间，代价是每次恢复折得更狠（单个长任务的
    # 典型路径是 L0.5 → L1/L2 无可折对象跳过 → L3 坍缩当前 task 层，一次摘要 LLM 调用）。
    compact_target_ratio: float = 0.4
    compact_message_delta: int = 20      # DEPRECATED（2026-07-01）：compact 改纯预算驱动，本字段不再被读
    compact_keep_last: int = 6           # 保留底线（非触发门）：agent 层折叠保留的胶囊数；更老的折成摘要
    collapse_keep_last: int = 3          # 保留底线（非触发门）：task 坍缩保留的最近段摘要条数
    # L0.5 图片降级（image fold/replay 子设计 §6/§12）：escalating_compact 在 L1 之前把
    # memory 里的真图换成含 ref 的文本占位，保留**最近这么多张**不降（按图片张数数，一条
    # 记录可以只降一部分）。模型对最近几张图的依赖最强，且降级可逆（media:get_image 取回）。
    # §12 记为未决参数，暂定 2，待实测校准——故做成配置项而非常量。
    compact_keep_recent_images: int = 2
    # 派发前压缩阈值：派发工具调用执行前，若本轮 prompt token / context_limit >= 此值，
    # 先对父自身 task 层对话压缩一次（fold_task），使父 resume 更精简、inherit 快照为
    # 压缩后版本。0 = 关闭（默认）。通常设得比 compact_token_ratio 更早触发。
    predispatch_compact_token_ratio: float = 0.0
    # act 的上下文停机线：单轮**真实** prompt_tokens / effective_limit ≥ 此值 → 结束本段 act。
    # 曾硬编码 0.8，与 compact_token_ratio 的默认值重合——于是 prepare 刚把估算压到「刚低于
    # 0.8」，act 第一轮就又越线。两条线必须分工：compact_token_ratio 管 run 开头的常规压缩
    # （估算口径），本值管 mid-act 兜底（真实口径），故设得更高。
    context_limit_stop_ratio: float = 0.9
    # 一次 run 内最多几次「上下文恢复」：act 越过 context_limit_stop_ratio 时不再直接退出吃
    # 一次 retry，而是回 PrepareStep 压缩后原地续跑。配额耗尽（或 compact 已压不动，见
    # `COMPACT_NOOP_KEY`）才退回 observe → retry 的老路。0 = 关闭恢复，行为回到改造前。
    max_context_recoveries: int = 2
    # 短任务（叶子）保留阈值（spec 2026-06-23）：finish 时 task 层对话 token ≤ threshold
    # 且 LLM_RESPONSE 轮次 ≤ turn_cap → 不 close（保留完整对话）；否则 close 成残留。
    short_task_token_threshold: int = 1000
    short_task_turn_cap: int = 2
    # 段边界免折阈值（background_observe）：interactive/interrupt 段（末条 UP 之后）的
    # active raw ≤ 此值时免折、该段永久保 raw（不跑后台 LLM；折叠段作用域化后前段残留
    # 不跨段合折，2026-07-21）。按「一条 recap 约 150-300 token，raw 明显大于 recap 才
    # 值得一次后台 LLM」定档，比 short_task_token_threshold 紧。
    short_segment_token_threshold: int = 400


# ── AgentTemplate ─────────────────────────────────────────────────────────────


@dataclass
class AgentTemplate:
    """Agent 实例化的蓝图。core 通过 AgentCapabilityProvider 读取。"""

    id: str
    name: str
    version: str  # semver；既是 template 版本也是 identity 版本

    # 身份定义（inline，一等字段）
    identity: dict[Purpose, IdentityFacet]
    # 外部能力引用：tool / skill / sub-agent
    capability_refs: list[CapabilityRef]

    # 子配置
    memory_config: MemoryConfig
    loop_config: LoopConfig
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
