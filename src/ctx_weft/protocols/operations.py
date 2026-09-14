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
    "Adjudication",
    "OperationAdjudicator",
]


class OperationStatus(StrEnum):
    """账本状态机（spec: tool-operations）。合法转移见 OperationRecord docstring。"""

    PREPARED = "prepared"            # 身份已持久、尚未执行（prepared→started 之前无外部副作用）
    STARTED = "started"              # provider 调用进行中（副作用可能已发生）
    # 这次逻辑调用已有最终结果——**含「结果无法确定」这一种结果**。曾经另有一个
    # UNKNOWN 状态承载后者，但它的两个消费方（停机闸门、宿主 resolve_operation）都
    # 已删除；结论本身住在 result 文本里，状态再分一档没有消费者。
    COMPLETED = "completed"
    WAITING_HUMAN = "waiting_human"  # 停在人工节点（HITL park）


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

    合法转移：prepared→started→completed；waiting_human 见状态机表。
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


@dataclass
class Adjudication:
    """裁决结论。core 只需要两件事：**该不该重跑**，以及不重跑时这次调用的结果写什么。

    这里刻意**不是**「发生了什么」的枚举（曾经是 completed / definitely_not_started /
    unknown 三态）。那问的是事实，而事实在不确定时无法断言，于是不得不留一个
    ``unknown`` 兜住「这个问题我答不了」——core 再拿着那个 unknown 去发明一整套停机
    等人的控制面。

    换成「该不该重跑」之后不需要第三态：那是**建议**不是断言，不知道的时候答
    ``rerun=False`` 既不用猜、也正是想要的答案。而「为什么不重跑」——查到了真结果、
    还是压根查不到——是 ``result`` 的**内容**，由知情的裁决者自己写，core 不替它组织措辞。
    """

    rerun: bool
    result: Any = None        # rerun=False 时写入 capability result

    @classmethod
    def rerun_safe(cls) -> "Adjudication":
        """权威判定「这次没跑成」——同 operation_id 重跑是安全的。"""
        return cls(rerun=True)

    @classmethod
    def conclude(cls, result: Any) -> "Adjudication":
        """不重跑，把 ``result`` 作为这次调用的结果。

        无论是查到了外部真实结果，还是只能说明「查不到、副作用可能已发生」——
        对 core 都是同一件事，区别全在文本里。
        """
        return cls(rerun=False, result=result)


@runtime_checkable
class OperationAdjudicator(Protocol):
    """Provider 可选实现：替 core 裁决一次结果不确定的操作。

    实现即生效——**无需在 capability 上声明**。未实现时 core 按「无从查证」作结，
    同样不重跑。

    ``adjudicate`` 判不了时返回 ``Adjudication.conclude(<说明文本>)``：说清查到了什么、
    查不到什么，那段文本会成为这次工具调用的结果，交给 agent 在任务上下文里处理。
    core **不会**因此停机等人——「不确定」是一种工具结果，不是一种控制流。

    重复调用的防护归 Provider 自理：core 的承诺止于「不自行重跑」；agent 下一轮主动
    再调一次是一次**新的逻辑调用**（新 operation_id），与崩溃恢复无关。
    """

    async def adjudicate(
        self, operation_id: str, ctx: ProviderContext,
    ) -> Adjudication:
        raise NotImplementedError
