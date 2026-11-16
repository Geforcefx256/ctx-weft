"""L0.5 降级的**选取策略**——纯函数、无 IO（子设计 §7）。

回答两个问题，都不碰 memory：

1. **哪些图该降**（`keep_recent` 口径 + 「只降 ref 形态」判据）；
2. **哪些记录必须同批重写**——同 timestamp 记录的相对顺序只由 `seq_no` 兜住，
   而补偿记录必然拿到新的（最大的）`seq_no`，故部分重写同刻组会静默乱序。
   见下方「同刻 tie 组」。

执行（读记录 / 改 content / `memory.fold()`）在 `fold.py`。本模块只输出 `DemotionPlan`。

════════════════════════════════════════════════════════════════════════════
判断题 1 —— `keep_recent` 按**图片张数**数，且允许**一条记录内部分降级**
════════════════════════════════════════════════════════════════════════════

子设计 §8 的措辞是「除最近 `keep_recent` **张**之外的所有图片」——单位是张不是条。
一条记录可含多张图，于是「最近 N 张」可能落在一条记录的中间。两种口径：

- **整条保护**（记录里只要有一张被保护就整条不动）：一条含 20 张图的消息在
  `keep_recent=2` 下会把 20 张全部扣住，L0.5 一个 token 都释放不出来——而
  「一次贴一叠图」恰恰是最需要 L0.5 的场景。口径与文档也对不上（说好数张，实际数条）。
- **部分降级**（本模块采用）：按视图顺序把所有图排成一列，末尾 `keep_recent` 张原样
  保留，其余逐张换占位。同一条记录里被保护的图仍是真图，被降的图就地变成占位——
  占位留在原下标上，故「第几张图」的相对位置不变（`get_image` 的位置信息要用）。

计数覆盖**所有**图片 part（判据 `not hasattr(p, "text")`，与 `utils.image_part_count`
同源、spec §13 冻结），而不只是可降的那些：`keep_recent` 要保的是「模型还能看见的最近
N 张」，一张不可降的 base64 图同样占着模型的视线，故它照样占一个保护名额。

════════════════════════════════════════════════════════════════════════════
判断题 2 —— 只降 `source_type == "ref"` 的图
════════════════════════════════════════════════════════════════════════════

占位是**落库、永久**的，写下去就把原图从记录里换走了。inline base64 的图一旦换成占位，
ref 无从取回 = 图**永久丢失**——那不是降级，是删除。故判据是「这张图能不能被
`media:get_image` 取回来」，即 `core.utils.content.extract_blob_refs` 认得的 `blob:<sha>`
（复用归一层这个**唯一真源**，不在本模块另写一遍 isinstance，见该函数 docstring）。

这与 §10「`MemoryBlobStore` 未注册 → 返回 0、不降级」**不冲突，且严格更细**：未注册时
`normalize_content` 原样返回、没有任何 part 会变成 ref 形态，于是本判据自然选中空集、
`fold.py` 一次 `fold()` 都不发——「行为与改造前逐字节一致」由同一条判据兜住，不需要
第二处 registry 探询（多一处判据就多一处会分叉的真源）。更细则体现在：宿主**已**注册
MemoryBlobStore 但记录里仍有存量 inline base64（宿主直接 ingest 的、Phase 3b 之前落的）时，
registry 探询会放行并把它们弄丢，本判据不会。

ref 形态但 ref 本身不是可解析 token（空 / 含空白 / 含 ``]``）的，同样**不选**：
`refs.encode_image_placeholder` 对这类输入刻意抛 ValueError（占位落库即永久，静默产坏
占位等于丢图），本模块自己先判、直接跳过，不指望 refs 兜底（子设计 §10 第三行）。

════════════════════════════════════════════════════════════════════════════
判断题 3 —— 同刻 tie 组：整组同批重写（子设计的「位置不变」论证有误）
════════════════════════════════════════════════════════════════════════════

子设计 §4.1 说「新事件的 timestamp / role / metadata（含 `seq_no`）原样带过去 → 位置
不变」。**`seq_no` 带不过去**：它由 provider 在 ingest 时自己分配
（`in_memory.py:113`），补偿记录必然拿到一个新的、最大的 seq_no。真正保住位置的是
timestamp（`load_view` 按 `(timestamp, seq_no)` 升序），`seq_no` 只在同刻时做 tie-break。

于是同刻组里**只降一部分**会乱序：

```
折叠前:      ['msg0', 'msg1', 'msg2']            # 三条完全同 timestamp
只降 msg0:   ['msg1', 'msg2', 'DEMOTED-msg0']    # msg0 跑到组尾
```

**同刻不是只有测试才构造得出来——生产里是显式不变量**：`finalize.py:157/213` 的派发框
（assistant）与其 result（tool）**共用同一个锚点 ts**，并明写「框与 result 同锚、严格
相邻」是不变量。真降到这一对上，框会掉到 result 后面。故「接受并记录」这条路不可取。

**采用：tie 组内从第一条被降的记录起，到该组结尾为止，整段进同一次 `fold()`**，
replacements 按原顺序给出。于是它们拿到的新 seq_no 依原序递增，组内相对顺序保持；
组前未被触碰的记录 seq_no 更小、仍排在前面；组整体的位置由 timestamp 钉住，**不后移**
（timestamp 是主序键，同刻组之外的记录一律按 timestamp 分开）。代价是组内那些**本不需要
降级**的记录也被原样重写一遍（内容逐字段照抄，只换 record id）——这是为顺序正确性付的
账，且只在同刻组里发生。

不选「跳过会乱序的记录」：那会让 L0.5 在任何 timestamp 精度粗糙的 provider 上**静默
失效**（全是同刻组 → 全部跳过 → 一张不降），失效方式还不可观测。
不选协议级修法（`fold` 的 replacement 在 1:1 时继承被 supersede 记录的 `seq_no`）：
`supersede_ids` 与 `replacements` 是两个无配对关系的列表，属协议改动，本任务只提出、
不实施（见 task-2 简报第 (c) 条）。

**已知边界**（钉在测试里）：本模块只看得见传进来的那一份视图。被 `kinds` 过滤掉的
同刻记录、以及别的 scope 分区里的同刻记录，其与补偿记录的相对顺序不受保护——前者
本就不进装配视图，后者按 scope key 各自排序，都不影响对话历史的呈现顺序。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ctx_weft.protocols.content import extract_blob_refs

__all__ = [
    "DemotionPlan",
    "demotable_ref",
    "is_image_part",
    "plan_demotions",
]


def is_image_part(part: Any) -> bool:
    """该 part 是不是图片。判据 ``not hasattr(p, "text")``——**spec §13 冻结**（裁定 D1）。

    与 `utils.image_part_count` / `utils.image_tokens` 同一判据：本模块数出来的「图片
    张数」必须与 `freed_tokens` 的口径同源，否则 L0.5 会报出「降了 3 张但 token 没动」。
    """
    return not hasattr(part, "text")


def demotable_ref(part: Any) -> str | None:
    """这张图能被降级吗？能则返回可写进占位的 ref，否则 ``None``（含「不是图」）。

    判据见模块 docstring 判断题 2：必须是 `extract_blob_refs` 认得的 `blob:<sha>`
    （归一层唯一真源），且 ref 本身是可解析 token（非空、无空白、无 ``]``）。
    """
    if not is_image_part(part):
        return None
    refs = extract_blob_refs([part])
    if not refs:
        return None
    ref = refs[0]
    # refs.encode_image_placeholder 的准入条件，此处先判、不让它抛（子设计 §10）。
    if not ref or any(ch.isspace() for ch in ref) or "]" in ref:
        return None
    return ref


@dataclass(frozen=True)
class DemotionPlan:
    """一次降级的完整计划。纯数据，无 IO。

    - ``batches``：每个元素是**一次 `fold()`** 的 record id 序列，按视图顺序。
      长度 > 1 只在同刻 tie 组时出现（见模块 docstring 判断题 3）；其中可能含
      ``demote_indices`` 为空的记录——它们是为保住顺序而**原样重写**的。
    - ``demote_indices``：record id → 该记录 content 里要换成占位的 part 下标（升序）。
    - ``image_count``：计划降级的**图片张数**（`demote_indices` 的下标总数）。
      它是上界：`fold()` 失败的批次不计入 `fold.py` 的返回值。
    """

    batches: tuple[tuple[str, ...], ...] = ()
    demote_indices: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    image_count: int = 0


def plan_demotions(
    records: Sequence[Any],
    *,
    keep_recent: int,
    only_ids: Iterable[str] | None = None,
) -> DemotionPlan:
    """选出该降的图 + 该同批重写的记录。**纯函数**，不碰 memory、不改 records。

    ``records`` 必须是 `load_view` 的原序（``(timestamp, seq_no)`` 升序）——「最近
    N 张」与同刻分组都建立在这个顺序上。

    ``keep_recent``：末尾保留的**图片张数**（负数按 0 处理；见判断题 1）。
    ``only_ids``：限定只降这些记录里的图（`demote_all` 用；``None`` = 全视图）。
      注意保护名额仍按**整个视图**数——`demote_all` 的调用方给的是「即将被折走的那
      一段」，若只在这一段里数「最近 N 张」，被保护的会是这一段的末尾而不是真正最近的
      几张。不过 `demote_all` 本就不受 `keep_recent` 保护（传 ``keep_recent=0``）。
    """
    keep = max(0, int(keep_recent))
    limit = None if only_ids is None else set(only_ids)

    # 1) 视图顺序上的全部图片槽位（含不可降的），末尾 keep 张受保护。
    slots: list[tuple[int, int]] = []          # (记录下标, part 下标)
    for ri, rec in enumerate(records):
        content = getattr(rec, "content", None)
        if not content or isinstance(content, str):
            continue
        for pi, part in enumerate(content):
            if is_image_part(part):
                slots.append((ri, pi))
    # max(0, ...)：keep 大于总张数时必须保护**全部**——写成 slots[len-keep:] 会因负数
    # 起点变成「只保护最后 keep-len 张」，越保护越少，恰好反了。
    protected = set(slots[max(0, len(slots) - keep):]) if keep else set()

    # 2) 逐记录选出要换占位的 part 下标。
    selected: dict[str, tuple[int, ...]] = {}
    per_index: list[tuple[int, ...]] = []
    for ri, rec in enumerate(records):
        rid = str(getattr(rec, "id", "") or "")
        content = getattr(rec, "content", None)
        picks: list[int] = []
        if rid and content and not isinstance(content, str) \
                and (limit is None or rid in limit):
            for pi, part in enumerate(content):
                if (ri, pi) in protected:
                    continue
                if demotable_ref(part) is not None:
                    picks.append(pi)
        per_index.append(tuple(picks))
        if picks:
            selected[rid] = tuple(picks)

    if not selected:
        return DemotionPlan()

    # 3) 同刻 tie 组：连续同 timestamp 为一组；组内自第一条被选中的记录起整段同批重写。
    batches: list[tuple[str, ...]] = []
    n = len(records)
    start = 0
    while start < n:
        end = start + 1
        ts = getattr(records[start], "timestamp", None)
        while end < n and getattr(records[end], "timestamp", None) == ts:
            end += 1
        first = next((i for i in range(start, end) if per_index[i]), None)
        if first is not None:
            ids = tuple(str(getattr(records[i], "id", "") or "")
                        for i in range(first, end))
            if all(ids):
                batches.append(ids)
        start = end

    counted = sum(len(selected.get(rid, ())) for b in batches for rid in b)
    return DemotionPlan(
        batches=tuple(batches),
        demote_indices={rid: selected[rid] for b in batches for rid in b if rid in selected},
        image_count=counted,
    )
