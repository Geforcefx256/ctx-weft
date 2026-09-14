"""工具操作账本协议（spec: tool-operations；change reliability-wp5，方案 §5.3）。

区分三个身份（H3 的根因修复面）：

- ``tool_call_id`` —— LLM wire 配对字段。模型复用 ``call_1`` 是常态，MUST NOT 用作
  恢复匹配依据。
- ``invocation_id`` —— 单次**执行尝试**身份（gateway 每次分配，provider 据此登记
  在途句柄供 cancel）。
- ``operation_id`` —— 跨重启稳定的**逻辑调用**身份（本模块），由
  ``(tenant, session, agent, assistant_record_id, tool_ordinal)`` 确定性派生。
  两条内容相同的合法调用得到不同 id（不误去重）；同一逻辑调用经普通执行 / 热 HITL
  resume / 冷恢复重入，读到同一个 id。

账本与工具副作用**不组成分布式事务**（方案明令不据此声称 exactly-once）；它把
「副作用是否已发生、结果是什么」变成可判定事实，供 WP6 的恢复策略表消费。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from ctx_weft.protocols.context import ProviderContext

__all__ = [
    "OperationStatus",
    "OperationRecord",
    "OperationUpdate",
    "OperationStore",
    "operation_id_for",
    "operation_memory_result_id",
    "RecoveryPolicy",
    "normalize_recovery_policy",
    "AdjudicationOutcome",
    "Adjudication",
    "OperationAdjudicator",
]


class OperationStatus(StrEnum):
    """账本状态机（spec: tool-operations）。合法转移见 OperationRecord docstring。"""

    PREPARED = "prepared"            # 身份已持久、尚未执行（prepared→started 之前无外部副作用）
    STARTED = "started"              # provider 调用进行中（副作用可能已发生）
    COMPLETED = "completed"          # 结局已确认（含 outcome=error 的确认失败）
    WAITING_HUMAN = "waiting_human"  # 停在人工节点（HITL park）
    UNKNOWN = "unknown"              # 无法判定业务结果（WP6 消费：停住等宿主处置）


# ── 恢复策略与裁决（spec: tool-operations，wp6）───────────────────────────────


class RecoveryPolicy(StrEnum):
    """崩溃后 started 的操作能不能自动重跑——**只有两类**。

    分类判据是「core 要不要做决定」，不是「副作用长什么样」：

    - ``IDEMPOTENT``：重跑安全，core 直接同 operation_id 重跑。至于安全的**理由**是
      「重跑本就无害」还是「Provider 以 operation_id 去重」，core 不关心——两者的
      行为逐字相同，分成两个值只会让调用方以为存在不存在的差别。
    - ``REVIEWED``（默认）：core **绝不自行重跑**，把这次不确定交给裁决者。

    「谁来裁决」不是策略值，而是一条链——把它压进枚举会让「core 判不了」这一件事
    裂成若干个看似并列的值：

        Provider 实现 OperationAdjudicator？
          ├─ 是 → 调用它：COMPLETED → 回填不执行；DEFINITELY_NOT_STARTED → 执行；
          │        UNKNOWN → 落到人
          └─ 否 → 落到人：账本 unknown + OperationUncertain + runtime.resolve_operation

    裁决能力**靠发现不靠声明**：``OperationAdjudicator`` 是 Provider 级接口，用
    capability 级字段去声明它是错配，还得额外拿启动期校验去堵「声明了却没实现」。
    isinstance 发现则没有这个失败模式；而且 Provider 可以逐次决定——判得了的给权威
    结论，判不了的返回 UNKNOWN 自动落到人。
    """

    IDEMPOTENT = "idempotent"
    REVIEWED = "reviewed"


def normalize_recovery_policy(value: object) -> RecoveryPolicy:
    """把声明值归一为 RecoveryPolicy；无法识别则抛 ValueError。

    **不静默兜底**：拼错的取值必须响亮失败，由 resolver 在启动期一次性拦下——静默降级
    成保守值会让「我标了 idempotent 为什么崩溃后还在等人」变成一个读源码才能查的问题。
    ``None`` / 缺省 → REVIEWED（Provider 没表态时按需审核处理）。
    """
    if value is None or value == "":
        return RecoveryPolicy.REVIEWED
    if isinstance(value, RecoveryPolicy):
        return value
    try:
        return RecoveryPolicy(str(value))
    except ValueError:
        raise ValueError(
            f"unknown recovery_policy {value!r}; expected one of "
            f"{[p.value for p in RecoveryPolicy]}") from None


@dataclass
class OperationRecord:
    """一条逻辑调用的账本行。

    合法转移：prepared→started→completed；started/waiting_human→unknown（WP6）。
    ``revision`` 乐观锁：每次 CAS +1，compare_and_set 期望值不匹配即拒绝。
    ``result``：完整规范化结果（str | ContentParts 列表 | blob ref 字符串）——非审计
    事件的截断文本（两条通道目的不同，不合并）。``args_hash``：授权后参数指纹
    （复用 gateway 的 invocation_key 实现）。
    """

    operation_id: str
    tenant_id: str
    session_id: str
    agent_id: str
    assistant_record_id: str
    tool_ordinal: int
    tool_name: str
    # 派生 task 维度（wp6 resolve_operation 补写 memory 用；空=未携带，宿主可经
    # OperationUncertain payload 自行定位 task）
    task_id: str = ""
    status: OperationStatus = OperationStatus.PREPARED
    revision: int = 1
    args_hash: str = ""
    recovery_policy: str = RecoveryPolicy.REVIEWED   # WP5 只落库；WP6 消费
    attempts: list[str] = field(default_factory=list)   # invocation_id 列表（每次执行尝试）
    result: Any = None                        # 完整结果 / ContentParts / blob ref
    error: str | None = None
    memory_result_id: str = ""                # TOOL_RESULT 记录 id（确定性派生）
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class OperationUpdate:
    """CAS 携带的增量：status/result/error/attempt 追加。None 字段不更新。"""

    status: OperationStatus | None = None
    result: Any = None
    result_set: bool = False                  # 显式区分「未更新」与「更新为 None」
    error: str | None = None
    append_attempt: str | None = None


@runtime_checkable
class OperationStore(Protocol):
    """操作账本抽象。host 提供持久实现（SQL）；runtime 缺省注册内存版。

    写失败按存储不可用处理（复用 event-commit 的隔离链路）——账本是 H3 恢复的
    依据，静默降级会重新制造「伪装成功」。
    """

    async def get(self, operation_id: str, ctx: ProviderContext) -> OperationRecord | None:
        raise NotImplementedError

    async def prepare(self, record: OperationRecord, ctx: ProviderContext) -> OperationRecord:
        """登记 PREPARED 行。幂等：同 id 同内容 no-op 返回既有行。"""
        raise NotImplementedError

    async def compare_and_set(
        self,
        operation_id: str,
        expected_revision: int,
        update: OperationUpdate,
        ctx: ProviderContext,
    ) -> OperationRecord:
        """乐观锁推进状态机；revision 不匹配抛 RevisionConflict。"""
        raise NotImplementedError


class RevisionConflict(Exception):
    """CAS 期望 revision 不匹配——后到者被拒，状态机不倒退不跳跃。"""


def operation_id_for(
    tenant_id: str, session_id: str, agent_id: str,
    assistant_record_id: str, tool_ordinal: int,
) -> str:
    """确定性派生：同一逻辑调用跨进程/跨重启必然同值（ULID 做不到）。

    截断 24 hex（96 bit）：同会话内 record_id 已是 ULID，碰撞概率可忽略；``op_`` 前缀
    保可读。同参两次合法调用因 record_id/ordinal 不同必然异 id（O-T09）。
    """
    digest = hashlib.sha1(
        f"{tenant_id}|{session_id}|{agent_id}|{assistant_record_id}|{tool_ordinal}".encode()
    ).hexdigest()[:24]
    return f"op_{digest}"


def operation_memory_result_id(operation_id: str) -> str:
    """TOOL_RESULT 的 memory 记录 id 由 operation_id 确定性派生（spec §5.3）：
    completed 后 memory 写失败时，恢复路径按同一 id 幂等补写。"""
    return f"res_{operation_id[3:]}"


# ── 裁决接口（spec: tool-operations，wp6）─────────────────────────────────────


class AdjudicationOutcome(StrEnum):
    """裁决一次不确定操作的三态结论（方案 §5.4）。

    ``DEFINITELY_NOT_STARTED`` 是**权威否定**——查得到完整执行记录才算；
    「暂时查不到」必须归 UNKNOWN，不得用否定冒充（否则会重跑已发生的副作用）。
    """

    COMPLETED = "completed"                    # 外部已完成（携带结果）
    DEFINITELY_NOT_STARTED = "definitely_not_started"
    UNKNOWN = "unknown"


@dataclass
class Adjudication:
    """裁决返回：结论 + COMPLETED 时的外部结果（Provider 结果结构）。"""

    outcome: AdjudicationOutcome
    result: Any = None


@runtime_checkable
class OperationAdjudicator(Protocol):
    """Provider 可选实现：替 core 裁决一次结果不确定的操作。

    实现即生效——**无需在 capability 上声明**。判不了的操作返回
    ``AdjudicationOutcome.UNKNOWN``，core 自动落到人工处置。
    """

    async def adjudicate(
        self, operation_id: str, ctx: ProviderContext,
    ) -> Adjudication:
        raise NotImplementedError
