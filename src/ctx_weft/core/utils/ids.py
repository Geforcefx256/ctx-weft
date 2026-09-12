"""ULID 前缀 id 生成 + 工具调用内部标识铸造。

`evt_` / `tsk_` / `ses_` / `agt_` 等前缀是**对外可见的 id 形态**——host 与事件流都按
它认类型，改前缀等于改契约。

内部调用标识（spec: conversation-integrity）：LLM 分配的 tool_call wire id 可被模型
复用（`call_1` 这类短值是常态），跨轮次/跨任务召回重建后按裸 id 配对会错配。每个
assistant 回合摄入时把 wire id 替换为 `tc_{seq36}_{ord36}_{hash12}`——字符集
`[a-z0-9_]`、长度 ≤64（主流 provider 工具 id 约束的交集内，adapter 原样透传即合法），
由回合锚 + 调用序号 + 原始 id 确定性派生。raw id 保留在记录 metadata（`raw_id`）。
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from ulid import ULID

from ctx_weft.protocols import ToolCall

__all__ = ["generate_id", "mint_call_id", "mint_turn_call_ids", "MintedCall"]

# 内部调用标识的形态契约（adapter 验收 / 单测共用同一判据）。
INTERNAL_CALL_ID_RE = re.compile(r"^tc_[0-9a-z]+_[0-9a-z]+_[0-9a-f]{12}$")
INTERNAL_CALL_ID_MAX_LEN = 64

# tc_ + seq36 + _ + ord36 + _ + 12 hex：两位 base36 段封顶长度，防极端入参撑爆 64。
_MAX_SEQ36_LEN = 13
_MAX_ORD36_LEN = 4


def generate_id(prefix: str) -> str:
    """Generate a ULID-based primary key (time-sortable + globally unique).

    Format: {prefix}_{ulid}, e.g. ses_01H8K9XPYJ7DRT2RY3JFXSF7M2
    """
    return f"{prefix}_{ULID()}"


def _base36(n: int) -> str:
    if n < 0:
        raise ValueError("base36 expects non-negative int")
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


def mint_call_id(*, anchor: str, ordinal: int, raw_id: str, turn_seq: int) -> str:
    """铸单个内部调用标识：`tc_{seq36}_{ord36}_{sha256(anchor|ordinal|raw)[:12]}`。

    确定性：同 (anchor, ordinal, raw_id) 恒同值——同一回合的两个供值平面（memory 记录 /
    活动消息列表）来自同一次铸造，天然一致；跨会话稳定（不掺随机量）。
    """
    seq36 = _base36(turn_seq)
    ord36 = _base36(ordinal)
    if len(seq36) > _MAX_SEQ36_LEN or len(ord36) > _MAX_ORD36_LEN:
        raise ValueError(f"turn_seq/ordinal out of representable range: {turn_seq}/{ordinal}")
    digest = hashlib.sha256(f"{anchor}|{ordinal}|{raw_id}".encode()).hexdigest()[:12]
    minted = f"tc_{seq36}_{ord36}_{digest}"
    if len(minted) > INTERNAL_CALL_ID_MAX_LEN:  # pragma: no cover — 上面的长度封顶已拦
        raise ValueError(f"minted id exceeds {INTERNAL_CALL_ID_MAX_LEN} chars: {minted!r}")
    return minted


@dataclass(frozen=True)
class MintedCall:
    """一次铸造的完整产物：替换后的 ToolCall + 审计伴随字段。

    raw_id 是 LLM 的原始 wire id（落 metadata 供追溯）；ordinal 是该调用在回合内的
    序号（与 operation_id 的派生输入同口径，见 act._execute_tool_calls）。
    """

    call: ToolCall
    raw_id: str
    ordinal: int


def _with_id(tc: ToolCall, minted: str) -> ToolCall:
    """返回 id 已替换的调用副本，原对象不动。

    chunk.tool_call 的契约是 ToolCall，但实际供给里混有鸭子类型（测试的
    SimpleNamespace 等）——浅拷贝 + setattr 对两者都成立且保留原类型与额外字段；
    冻结 dataclass 兜底退回构造标准 ToolCall。
    """
    dup = copy.copy(tc)
    try:
        dup.id = minted
        return dup
    except dataclasses.FrozenInstanceError:  # pragma: no cover — 今日无冻结供给
        return ToolCall(id=minted, name=tc.name, arguments=tc.arguments)


def mint_turn_call_ids(
    tool_calls: Iterable[ToolCall],
    *,
    anchor: str,
    turn_seq: int,
) -> list[MintedCall]:
    """对一个回合的全部 tool_calls 铸内部标识（唯一供值点，两个平面共享其结果）。

    回合内唯一性自检：同回合撞哈希即抛（12 hex 撞率工程上为零，自检防实现错误）。
    """
    out: list[MintedCall] = []
    seen: set[str] = set()
    for i, tc in enumerate(tool_calls):
        minted = mint_call_id(anchor=anchor, ordinal=i, raw_id=tc.id, turn_seq=turn_seq)
        if minted in seen:
            raise ValueError(f"internal call id collision within turn: {minted}")
        seen.add(minted)
        out.append(MintedCall(call=_with_id(tc, minted), raw_id=tc.id, ordinal=i))
    return out


# CJK 表意字 / 假名 / 谚文 / 全角标点等：这些脚本 len//4 会严重低估（真实 0.6~1 token/字），
# 单列出来按更保守的每 1.5 token 估。ASCII 起始都 < 0x3000，findall 走 C 级、对大文本仍快。
