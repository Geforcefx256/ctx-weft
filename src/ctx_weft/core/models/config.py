"""RuntimeConfig — core 运行期可调旋钮，由 host 在实例化时注入。

默认值 = 历史行为，使直连 core / 测试零改动。host 始终注入实际值。
core 不再自读 os.environ；LLM 凭证/模型的 env 自举（bootstrap_from_env）由 host 子类实现。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    hitl_timeout_sec: int | None = None
    hitl_max_resolved: int = 1000
    # 事件提交策略（spec: event-commit）：required（默认）= emit 先经 CommitGate 确认
    # 存储提交再对外通知，存储失败显式抛 PersistenceUnavailableError 并隔离会话；
    # best_effort = 旧观察者路径（吞存储错误），启动告警、不可靠恢复。
    event_commit_policy: str = "required"
    task_max_concurrent: int = 4
    task_max_retries: int = 3
    default_token_budget: int = 200_000
    # 工具输出落盘（spill）阈值：CapabilityGateway 读取。host 可覆盖。
    spill_threshold: int = 4000
    spill_preview_chars: int = 1000
    # spec: tool-result-recovery——收敛版尾部预览字符数（错误/结论高发区立即止血）。
    spill_tail_chars: int = 1000
    # LLM 瞬时故障自愈预算（stream_llm_resilient 读取；默认=历史安全值）
    llm_self_heal_max_attempts: int = 8
    llm_self_heal_max_duration_sec: float = 300.0
    llm_self_heal_base_delay_sec: float = 2.0
    llm_self_heal_max_interval_sec: float = 60.0
    # 动态 max_tokens（apply_dynamic_max_tokens 读取；默认=安全值）
    # margin=8192：覆盖本轮新增尾段估算的残余 CJK 低估（len//4 对汉字系统性偏低），
    # 并与输入侧 reserved_output_tokens=8192 对称。仅在贴近满窗时才实质压小输出（届时已近
    # compact），常态下几万量级的输出预算无感。
    dynamic_max_tokens_margin: int = 8192
    # margin 比例制：实际 margin = max(margin, ratio*estimate)。固定 8192 只兜得住小 prompt 的
    # 估算残差；估算误差按比例放大（10 万 token 的 5% 是 5000），margin 也须随体量放大。
    dynamic_max_tokens_margin_ratio: float = 0.05
    dynamic_max_tokens_floor: int = 1024
    # 输出软顶：常态下不把整个剩余窗口都放给输出，按 context_limit 的比例封顶（但不低于
    # output_min）。ceiling = min(output_ceiling or context_limit, max(ratio*L, output_min))。
    # 好处：没人需要单轮几万 token 输出；且直接降低"max_tokens 超模型真实输出上限"的 400。
    # used 越过 ~(1-ratio) 后软顶不再绑定，回落到 L-used-margin 的紧缩段。
    dynamic_max_tokens_output_ratio: float = 0.2
    dynamic_max_tokens_output_min: int = 4096
