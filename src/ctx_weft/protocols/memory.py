"""MemoryProvider 协议（统一）。

定位：ctx-weft 中唯一的 memory 抽象。承载所有"短期 / 长期 / Blackboard"职能——
它们在 ctx-weft 中是**同一概念**。

核心思想：Provider 是一个**事件摄取 + 多模召回**的黑盒：
- core 把所有重要事件通过 ingest() 喂给 provider
- Provider 自由决定如何内化
- core 通过三种 recall 接口取回：recall_recent / recall_topic / recall_semantic

详见设计文档 §4.3。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from ctx_weft.protocols.context import (
    BLOB_REF_PREFIX,  # noqa: F401  # 刻意的 re-export：既有调用点仍从这里取
    ContentPart,
    ProviderContext,
    normalize_content_parts,
)


# ── Event types ───────────────────────────────────────────────────────────────


class MemoryEventType(StrEnum):
    """Memory 事件类型——core 通过 ingest() 喂给 provider 的事件分类。

    继承 StrEnum 以保持 JSON / 字符串语义。

    分层见 spec/06 §3：每个类型由 EVENT_LAYER 唯一映射到一层（task / agent / session）。

    【目标形态·待重构】完整设计见 docs/superpowers/specs/2026-07-06-memory-protocol-v2-design.md。
    一句话：**记忆按 scope 归档、按 address 定位**。要点：现状 type 承担双重职责（归属路由 +
    内容筛选）且两套回合编码并存（task 层 type 区分 / agent 层 role+metadata 区分），框架机制
    演进屡次穿透协议铸新类型、事后僵尸化。v2 沿「内容种类」慢轴重画，方法面 11 → 8：
      kind:  CONVERSATION_TURN | SUMMARY | TOOL_AUDIT | PUBLICATION（封死，永不为新机制扩）
      scope: TASK | AGENT | SESSION —— 归属范围（原 MemoryScope 更名；"层"误导纵向堆叠，实为
        横向归属分区），MemoryEvent 显式字段，EVENT_LAYER 退役为 legacy 兜底；可随执行模型
        缓慢生长（如将来的 USER/TENANT）
      address: MemoryAddress（原 MemoryAddress 数据类更名——它是坐标不是范围）：全址 = ingest
        归档地址，半址 = 过滤模式（None 字段 = 通配，非法字段抛 ValueError）
      role:  user | assistant | tool —— 回合按 LLM message 模型编码，机制住 metadata（约定注册表）
      读 = 三种记忆动作：load_view(address, scope, kinds=None) 全量幸存视图、时间正序、无
        limit/count（22 调用点普查：limit 是伪能力兼丢最老端的潜伏 bug）/ recall_topic /
        recall_semantic；recall_recent_by_agent 消解为半址 MemoryAddress(agent_id=A)。
      写 = ingest + fold(supersede_ids, replacements)：遗忘+补偿原子完成（修徒手 supersede+
        ingest 的崩溃丢摘要窗口）；keep_last/protect_types 策展政策上移框架侧，apply_compact 消亡。
    演进规则：新框架机制 = 新 metadata 约定，永不铸新 kind；存量旧类型零数据迁移、读侧统一归一
    （legacy_dispatch shim 届时并入唯一归一化模块；postgres 列名 layer 不改，provider 内部映射）。
    """

    # ── task 层（一次 task 的私有执行对话）──
    USER_PROMPT = "user_prompt"            # 用户任务输入；retry 时也追加于此
    LLM_RESPONSE = "llm_response"          # actor LLM 一轮的完整输出（metadata.tool_calls 供无损重建）
    TOOL_INVOCATION = "tool_invocation"    # actor 一次「真实能力」工具调用（仅审计，不进装配）
    TOOL_RESULT = "tool_result"            # 「真实能力」工具返回值
    TASK_COMPACT_SUMMARY = "task_compact_summary"    # task compact（observe active）产出的 [Context so far]

    # ── agent 层（该 agent 的对话日志，跨 task 持久）──
    AGENT_COMPACT_SUMMARY = "agent_compact_summary"  # agent compact 折叠超龄单元产出的滚动经验摘要
    # agent 层统一对话回合：dispatch 框/result（gateway plan 框 + finalize 铸框/回填）、finish 对
    # （finalize 合成、bg observe 替换）、inherit 快照（spawn 镜像父视图）。与 task 层同构——
    # 靠 role（assistant/tool）+ metadata（tool_calls/tool_call_id/origin_task_id/parent_task_id）
    # 区分回合并无损重建，不再按事件类型区分。
    AGENT_CONVERSATION_TURN = "agent_conversation_turn"

    # ── session / topic ──
    BLACKBOARD_PUBLISH = "blackboard_publish"  # 显式 topic 发布（见 spec/04）

    # ── legacy（§5.0 枚举永不物理删除；写侧已死，仅读侧兼容存量数据）──
    TASK_DISPATCH = "task_dispatch"               # legacy：新数据不再写；读侧 normalize_legacy_dispatch 归一为 AGENT_CONVERSATION_TURN（assistant 回合）
    TASK_DISPATCH_RESULT = "task_dispatch_result"  # legacy：同上（tool 回合）；线上无存量后删 legacy_dispatch 模块即可日落
    OBSERVER_SUMMARY = "observer_summary"  # 半僵尸：尚存两个写点（runtime tracking flush / suspend 挂起摘要）但不进装配，仅影响计数/估算口径
    COMPACT_SUMMARY = "compact_summary"    # 死类型：已无写点（apply_compact 按层写 TASK/AGENT_COMPACT_SUMMARY）；读侧仅 prepare 估算仍带到


class MemoryScope(StrEnum):
    """归属范围（v2 §2 · P4c 终名，原 MemoryLayer）：这条记忆归谁。

    task-scoped 私有执行转录 / agent-scoped 跨 task 经验 / session-scoped 共享黑板。
    横向归属分区（非纵向抽象堆叠）；成员可随执行模型缓慢生长（如将来的 USER/TENANT）。
    scope key 与 seq 计数按分区隔离。"""

    TASK = "task"        # tenant|session|task|<task_id>
    AGENT = "agent"      # tenant|session|agent|<agent_id>
    SESSION = "session"  # tenant|session


# host 兼容别名（P4c）：host postgres provider 仍 import MemoryLayer；host 迁移
# 完成后独立 PR 删除。新代码一律 MemoryScope。
MemoryLayer = MemoryScope


class MemoryKind(StrEnum):
    """v2 内容种类（设计 §2）。判据：不同 kind = provider 可施加不同存储/索引/保留策略。

    与 MemoryScope 正交：kind 答「这是什么内容」，scope 答「这条记忆归谁」。
    封死——永不为新机制扩，新机制 = 新 metadata 约定（见 MemoryEventType docstring）。
    """

    CONVERSATION_TURN = "conversation_turn"  # 对话回合（user/assistant/tool），各 scope 通用
    SUMMARY = "summary"                      # 遗忘补偿：段摘要 / 经验摘要（fold 的 replacement）
    TOOL_AUDIT = "tool_audit"                # 真实能力调用审计；默认不进视图装配
    PUBLICATION = "publication"              # topic 发布；按流读取（recall_topic）


# 事件类型 → 层（spec/06 §3）。唯一映射，ingest/recall 据此选 scope key。
# 注：「type 唯一决定层」已在松动——apply_compact 显式传 layer、provider recall 宽容混层；
# 目标形态下 layer 是 MemoryEvent 显式字段，本映射仅为 legacy 类型兜底（见 MemoryEventType docstring）。
EVENT_LAYER: dict[MemoryEventType, MemoryScope] = {
    MemoryEventType.USER_PROMPT: MemoryScope.TASK,
    MemoryEventType.LLM_RESPONSE: MemoryScope.TASK,
    MemoryEventType.TOOL_INVOCATION: MemoryScope.TASK,
    MemoryEventType.TOOL_RESULT: MemoryScope.TASK,
    MemoryEventType.TASK_COMPACT_SUMMARY: MemoryScope.TASK,
    MemoryEventType.AGENT_COMPACT_SUMMARY: MemoryScope.AGENT,
    MemoryEventType.AGENT_CONVERSATION_TURN: MemoryScope.AGENT,
    MemoryEventType.BLACKBOARD_PUBLISH: MemoryScope.SESSION,
    # legacy 类型（写侧已死，读侧兼容存量）
    MemoryEventType.TASK_DISPATCH: MemoryScope.AGENT,
    MemoryEventType.TASK_DISPATCH_RESULT: MemoryScope.AGENT,
    MemoryEventType.OBSERVER_SUMMARY: MemoryScope.AGENT,
    MemoryEventType.COMPACT_SUMMARY: MemoryScope.AGENT,
}


# v2 P4b-2：layer_for_types 随类型清单召回日落删除（读侧统一 kind 视图，无调用点）。


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class MemoryAddress:
    """记忆范围限定。

    【目标形态更名 MemoryAddress】它是坐标不是范围：全址 = ingest 归档地址，
    半址 = load_view 过滤模式（None 字段 = 通配）；"scope" 一名让位给归属范围枚举
    （原 MemoryScope）。见 v2 设计 §3。"""

    session_id: str
    task_id: str | None = None
    agent_id: str | None = None


# v2 正名（P4b-1 完成实体互换）：类本体即 MemoryAddress；旧名 MemoryScope 进入
# 名字真空（P4c 由归属范围枚举 MemoryScope 接名）——漏网引用是 loud NameError。


@dataclass
class MemoryEvent:
    """ingest 的输入：一条要写入的 memory 事件。

    过渡形态（v2 P2b · 2026-07-27）：type（legacy 词汇）与 kind+layer（v2 词汇）二选一——
    - legacy 构造：type 给定，行为逐字节不变（不做 kind 补全，归一化在读侧）；
    - v2-native 构造：kind+layer 显式、type=None；须满足 §4 ingest 全址不变量
      （TASK → scope 必携 task_id **且** agent_id；AGENT → agent_id；SESSION → 仅 session_id）。
    全字段带默认值（既有调用点均为关键字构造）；scope/content/timestamp 缺失 → ValueError。
    """

    type: MemoryEventType | None = None
    address: MemoryAddress | None = None  # 归档坐标（v2 §3 终名，原 scope 字段）
    content: str | list[ContentPart] | None = None  # 必给；显式空串合法（占位回合）。可能是 list[ContentPart]（多模态）；宿主 provider 必须能持久化并原样返回，见 MemoryProvider 多模态契约
    timestamp: datetime | None = None
    # 调用方预生成 record id（v2 设计 §4 · 2026-07-27 增补，投影化前置）。
    # 给定 → provider 必须采用并按 id 幂等（重复 ingest = no-op）；None → provider 生成。
    id: str | None = None
    # v2 词汇（设计 §2）：kind = 内容种类，scope = 归属范围（原 layer 字段）。
    kind: MemoryKind | None = None
    scope: MemoryScope | None = None
    role: Literal["user", "assistant", "system", "tool"] | None = None
    topic: str | None = None  # 用于 topic-style 事件（含父子 task 通信）
    causation_id: str | None = None  # 关联上游 event
    metadata: dict = field(default_factory=dict)
    # GC 的 mark 输入（缺陷 2026-08-27）：本事件引用了哪些 blob，但**没有**以结构化
    # ImagePart(source_type="ref") 形式出现在 content 里。
    #
    # 唯一的填写方——L0.5 降级（core/media/fold.py）：它把 ImagePart(ref) 换成文本
    # 占位，ref 就此掉进自由文本，provider 再也扫不出来 → 引用归零 → 字节被回收 →
    # media:get_image 取不回，L0.5 承诺的「可逆」失效。
    #
    # 普通写侧**不必填**：content 里结构化的 ref part 由 collect_blob_refs 自动采集。
    # provider 建引用边时一律走 core.content.collect_blob_refs，不得自行判断。
    blob_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # 误型 loud（字段终名切换护栏）：旧习惯 scope=<坐标> / address=<枚举> 静默换义
        # 是最危险的错，必须 TypeError 指路。
        if isinstance(self.scope, MemoryAddress):
            raise TypeError(
                "MemoryEvent.scope now takes the MemoryScope enum (归属范围); "
                "pass the coordinate via address= (原 scope 字段已改名 address)")
        if isinstance(self.address, MemoryScope):
            raise TypeError(
                "MemoryEvent.address takes a MemoryAddress coordinate; "
                "pass the MemoryScope enum via scope= (原 layer 字段已改名 scope)")
        if self.type is None and self.kind is None:
            raise ValueError("MemoryEvent requires type (legacy) or kind (v2)")
        if self.address is None:
            raise ValueError("MemoryEvent.address is required")
        if self.content is None:
            raise ValueError("MemoryEvent.content is required")  # 空串合法（占位回合）
        if self.timestamp is None:
            raise ValueError("MemoryEvent.timestamp is required")
        if self.kind is not None:
            # v2-native：scope 必须显式 + §4 全址不变量（legacy 构造不强制，迁移期宽松）
            if self.scope is None:
                raise ValueError("v2 MemoryEvent (kind given) requires explicit scope")
            if self.scope is MemoryScope.TASK:
                if not (self.address.task_id and self.address.agent_id):
                    raise ValueError(
                        "TASK-scoped v2 event requires full address (task_id AND agent_id); "
                        f"got {self.address!r}")
            elif self.scope is MemoryScope.AGENT:
                if not self.address.agent_id:
                    raise ValueError(
                        f"AGENT-scoped v2 event requires agent_id; got {self.address!r}")

        # 写侧归一（Phase 3c Task E2），与读侧 MemoryRecord 对称：dict 形态 part 是
        # 协议违规（本字段类型声明即 str | list[ContentPart]）。今日写侧 content 全由
        # core 构造（都是 dataclass），但宿主直接 ingest(MemoryEvent(content=[dict]))
        # 时 dict 会原样落库、读侧再归一回来——结果正确，只是往返多绕一圈，且写读两侧
        # 语义不对称。放在全部校验**之后**：content=None 等既有报错路径不得被归一抢先。
        # 共用实现见 core.content.normalize_content_parts（经 protocols.context 的惰性
        # 绑定转调，层序 + 热路径理由同 MemoryRecord.__post_init__）。
        self.content = normalize_content_parts(self.content)


@dataclass
class MemoryRecord:
    """recall 返回的统一表示。

    过渡形态（v2 P2b）：legacy 行 type 非 None；v2 行 type=None、kind/layer 给定。
    load_view 返回前经 normalize_view 统一重打 kind/layer/address（来源回显，
    取代 metadata["task_id"] 打标——打标过渡期保留，读侧优先 address）。
    """

    id: str
    type: MemoryEventType | None
    content: str | list[ContentPart]
    timestamp: datetime
    role: Literal["user", "assistant", "system", "tool"] | None = None
    topic: str | None = None
    score: float | None = None  # 仅 recall_semantic 时填
    kind: MemoryKind | None = None       # v2：内容种类
    scope: MemoryScope | None = None     # v2：归属范围（终名，原 layer 字段）
    address: "MemoryAddress | None" = None  # v2：来源回显（归档坐标）
    metadata: dict = field(default_factory=dict)
    # 与 MemoryEvent.blob_refs 对称的读侧回显（缺陷 2026-08-27）。
    # provider 必须还原它——否则一条被降级过两次的记录，第一次降的 ref 在第二次
    # 重建补偿事件时就没人认领了（core/media/fold.py::_rebuild 要累积它）。
    blob_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """把 dict 形态的 content 归一回 ContentPart dataclass（多模态 Phase 3c）。

        ``content`` 的类型声明就是 ``str | list[ContentPart]``，多模态契约第 2 条要求
        「召回时原样返回——形态必与入库时相同」。JSON 往返的 provider 若回吐
        ``[{"type":"text","text":"hi"}]`` 就是**协议违规**，后果实测过两条且互相掩护：
        ``image_part_count`` 把文本计成图（多算 1600 token），``content_to_text``
        返回空串（摘要器完全看不见内容）——token 涨了看着"内容很多"，实际内容全丢。

        修法落在**类型自己的边界**而不是每个读取方：core 读 memory 有 11 个调用点
        （act/compact/finalize/prepare/reconcile/segment_fold/background_observe/
        runtime/agent_recall/blackboard/long_memory），逐个兜底既漏又会随新调用点
        腐化；而任何 provider——含未来的第三方——都必须构造 MemoryRecord。判据
        ``not hasattr(p,"text")`` 因此保持冻结（用户裁定 D1），不为违规形态解冻。

        归一实现是三处边界共用的 ``core.content.normalize_content_parts``
        （Phase 3c Task E2 抽取；另两处是 ``MemoryEvent`` 与 ``LLMMessage``）——spec §3①
        要求形态转换收在归一层，三处各抄一份正是它要防的散点。快路径与「已合规不重建」
        的性能性质由该函数保证。经 ``protocols.context.normalize_content_parts`` 这个
        **惰性绑定**转调（层序 + 热路径理由见该绑定的 docstring）。
        """
        self.content = normalize_content_parts(self.content)


@dataclass
class Subscription:
    """task 对 topic 的订阅（持久化）。

    task_id = 订阅方任务（读取该 topic 的 task）；"" 表示 session 级订阅（如 long_term_*）。
    """

    session_id: str
    topic: str
    cursor: int  # 已读到的 seq_no
    # subtask=自己派生的子任务结果（可 review）；predecessor=同 plan 前序结果（只读）；
    # long_term_*=跨 session 长期上下文。
    intent: Literal["subtask", "predecessor", "long_term_background", "long_term_project_log"]
    task_id: str = ""
    priority: int = 5


# v2 P4b-2：CompactResult 随 apply_compact 消亡（框架侧对应物 SegmentFoldResult）。


@dataclass
class MemoryProviderInfo:
    """Provider 能力声明。"""

    name: str
    supports_semantic: bool = False  # 是否支持 recall_semantic
    supports_topic: bool = True  # 是否支持 topic / subscription
    archives_superseded: bool = True  # fold 标记的 superseded 行是否物理归档（原 supports_compact_archival）
    max_event_size_bytes: int | None = None


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class MemoryProvider(Protocol):
    """统一 memory：摄取所有事件 + 多模召回。单实例，必需。

    【多租户隔离契约】
    ``ProviderContext.tenant_id`` 是**硬隔离边界**，不是标签。宿主可自带 session_id
    （见 ``start_session``），故「同 session_id、不同 tenant」是可达状态——provider
    **必须**按租户分区，否则就是跨租户数据泄漏（实测发生过）。要求：
    1. ingest 时记录 ``ctx.tenant_id``（归一后）随行落库；
    2. 所有读接口（``load_view`` / ``recall_topic`` / ``recall_semantic`` /
       ``list_subscriptions``）只返回**本租户**的行——即便 session_id / task_id /
       agent_id / topic 全部撞车；
    3. 归一：``None`` 与空串一律视为 ``"default"``（与 ``ProviderContext.tenant_id``
       的字段默认值一致），**写读两侧必须用同一个归一**。这条不是洁癖：串接不一致
       时若各自比原值，读侧会静默返空（「会话突然失忆」），比泄漏更难诊断；
    4. 隐式的**跨行扫描**同样按租户分区——典型是 topic 发布的覆盖语义（下方
       ``recall_topic``）：一个租户的发布绝不能把另一个租户同 topic 的行标为
       superseded（它甚至读不到那些行）；订阅表同理，唯一键须含 tenant。
    5. 尚未收口的一处（记录在案，不属本契约要求）：**记录 id 的命名空间仍是全局的**
       ——``ingest`` 的按 id 幂等与 ``fold`` 的按 id 遗忘都不看 tenant。它们是调用方
       显式给出 id 的操作，与「隐式扫描」不同类；要改需先定 id 是全局唯一还是租户内
       唯一。宿主若跨租户复用记录 id，须自行保证。

    【多模态无损存取契约】
    MemoryEvent.content 可能是 str 或 list[ContentPart]（后者为多模态，含文本与图片）。宿主 provider 必须：
    1. 持久化时保持原样——若 ingest 接收 list[ContentPart]，则无损保存整体结构；
    2. 召回时原样返回——load_view / recall_topic / recall_semantic 返回的 MemoryRecord.content
       形态必与入库时相同（str 返 str，list 返 list，无转换）；
    3. 落库推荐形态：``ctx_weft.core.utils.content.content_to_jsonable`` 用于持久化前的 JSON 序列化
      （ContentPart 是普通 dataclass，需过此转换才能 json.dumps；取回时用 ``content_from_jsonable`` 还原）；
    4. 禁止在持久化层拍扁成纯文本——装配期是否拍扁为 str 由框架决定（见 composer.py），
       provider 的职责仅是无损存取。
    """

    name: str

    # ── 摄取（write）──

    @abstractmethod
    async def ingest(
        self,
        event: MemoryEvent,
        ctx: ProviderContext,
    ) -> str:
        """写入一个 memory event；返回 event id。

        core 调用方必须为每个重要事件调用 ingest；provider 自由决定是否持久化、
        如何索引。

        id 契约（v2 设计 §4 · 2026-07-27 增补）：event.id 给定时必须采用并原样回显，
        且**按 id 幂等**——已存在的 id（含已 superseded）= no-op，返回该 id，不比对
        内容、不重复写入、不推进任何计数器。None → provider 自行生成。
        """
        ...

    @abstractmethod
    async def fold(
        self,
        supersede_ids: list[str],
        replacements: list[MemoryEvent],
        ctx: ProviderContext,
    ) -> list[str]:
        """原子"遗忘 + 补偿"（v2 设计 §4）：标 superseded 并写入 replacements，一个事务内完成。

        - replacements 可空（纯遗忘）、可多条（finish 对替换这类成对写入）；返回新事件 id 列表。
        - 已 superseded / 不存在的 id 跳过（幂等）。
        - replacement 带 ``event.id`` 时按 record-id 契约采用且按 id 幂等（重放安全）。
        - 取代 v1 的 supersede + apply_compact：策展政策（keep_last/protect/段界/锚点）
          上移框架侧（segment_fold 等），provider 只按显式 id 集与显式 replacement 执行。
          修徒手 supersede+ingest 的崩溃丢摘要窗口。
        """
        ...

    # ── 召回（read，三种模式）──

    @abstractmethod
    async def load_view(
        self,
        address: "MemoryAddress",
        scope: MemoryScope,
        ctx: ProviderContext,
        kinds: list[MemoryKind] | None = None,
    ) -> list[MemoryRecord]:
        """工作记忆回放（v2 设计 §4）：返回该归属分区**全量幸存**记录，**时间正序**。

        - 排序键 (timestamp, seq_no) 升序；"最近一条"取 ``[-1]``。无 limit / 无 count——
          视图天然有界（≈一个 LLM context，超了 compact 触发）。
        - kinds=None 默认 = [CONVERSATION_TURN, SUMMARY]（工作记忆视图的定义）；
          需要 TOOL_AUDIT（如计算折叠 id 集）时显式传。
        - 半址过滤（非 None 字段皆为条件，非法非 None 字段抛 ValueError）：
          TASK → task_id 给定=单 task 视图 / 仅 agent_id=跨 task 聚合 / 二者皆 None=ValueError；
          AGENT → agent_id 必给、task_id 非 None=ValueError；
          SESSION → task_id/agent_id 非 None=ValueError。
        - 返回前经 memory_compat.normalize_view（legacy dispatch 配对 + kind/layer 重打）；
          record.address 回显来源归档地址。
        - **租户隔离**（见类 docstring）：只返回 ``ctx.tenant_id``（归一后）分区内的行。
          同 session_id 的另一个租户的记录必须不可见；反之本租户的行一条都不能少
          （修过头 → 返空 = 会话失忆）。
        """
        ...

    # v2 P4b-2：recall_recent / recall_recent_by_agent / count_recent 自协议删除——
    # 读取面收敛为三种记忆动作（load_view / recall_topic / recall_semantic）。
    # in-memory provider 保留同名实例方法仅为存量测试兼容（非协议，见 P4a 范围决策）。

    @abstractmethod
    async def recall_topic(
        self,
        topic: str,
        since: int,
        ctx: ProviderContext,
    ) -> tuple[list[MemoryRecord], int]:
        """按 topic 拉取（自 since seq_no 之后），返回 (events, new_cursor)。

        必需实现。父子 task 通信 + 跨 session 订阅式上下文用。

        - 覆盖语义（黑板）：同 topic 的 PUBLICATION 只保留最新一条，旧的标 superseded；
          其它 kind 的带 topic 记录累积保留。``BlackboardSource`` 依赖这条。
        - **租户隔离**（见类 docstring）：topic 名跨租户撞车同样可达（宿主自定的
          long_term_* topic 是常态），故按 ``ctx.tenant_id``（归一后）过滤；上面的
          覆盖扫描也**只在本租户分区内**进行。
        """
        ...

    @abstractmethod
    async def recall_semantic(
        self,
        query: str,
        scope: MemoryAddress,
        top_k: int,
        ctx: ProviderContext,
    ) -> list[MemoryRecord]:
        """语义相似度召回。可选——core 默认实现返空；外部实现核心能力。

        Provider 通过 describe() 声明是否支持。
        **租户隔离**（见类 docstring）同样适用：向量检索也必须按 ``ctx.tenant_id``
        预过滤，否则相似度会直接把别的租户的内容捞出来。
        """
        ...

    # ── 订阅（cross-session 长期上下文）──

    @abstractmethod
    async def subscribe_topic(
        self,
        session_id: str,
        topic: str,
        intent: Literal["subtask", "predecessor", "long_term_background", "long_term_project_log"],
        ctx: ProviderContext,
        task_id: str = "",
    ) -> str:
        """task 订阅 topic；返回 subscription id。幂等：同 (session, task, topic) 重复订阅保留游标。

        幂等的唯一键**须含 tenant**（见类 docstring 第 4 条）：否则同 session_id 的
        另一个租户会撞上幂等分支、拿到别人的订阅与游标。
        """
        ...

    @abstractmethod
    async def list_subscriptions(
        self,
        session_id: str,
        ctx: ProviderContext,
        task_id: str | None = None,
    ) -> list[Subscription]:
        """列出订阅。task_id 给定时只返回该 task 的订阅 + session 级订阅（task_id=""）；None 返回全部。

        "全部" 仍以租户为界——只返回 ``ctx.tenant_id``（归一后）分区内的订阅
        （topic 名与 intent 同样是租户数据）。
        """
        ...

    # v2 P4b-2：apply_compact / supersede 自协议删除——写面收敛为 ingest + fold。
    # 策展政策（keep_last / protect / 段界 / 锚点）上移框架侧 segment_fold 等；
    # 排序契约（渲染序 (timestamp, seq_no)）与锚点语义随之移交（见 segment_fold docstring）。

    # ── 能力 ──

    @abstractmethod
    async def describe(self, ctx: ProviderContext) -> MemoryProviderInfo:
        """返回 provider 能力声明。"""
        ...


# ── Blob 存储（多模态字节侧）─────────────────────────────────────────────────


class MemoryBlobStore(ABC):
    """core 的「二进制 sink」契约：存取图片等二进制内容，core 只见 ref。

    与 ``protocols.filesystem.SpillSink`` 同形——core 不直接碰存储，只知道
    「有个 sink 能存能取」。

    **本协议与 MemoryProvider 同处一个模块，但刻意不是它的方法**（裁定 D4）：
    D4 论证的是**引用边**（哪个 part 指向哪个 blob ref）该与 ingest 同事务，
    不是**字节**该进 RDBMS——字节侧可以是文件系统 / 对象存储等任意独立部署，
    memory 只需持有、维护引用边（见 ``MemoryProvider.live_blob_refs()``），
    不必是字节的持有者。语义上这仍是 memory 多模态支持的字节侧，与上面
    ``MemoryProvider`` 的【多模态无损存取契约】（part 结构侧）是同一件事的两面，
    宿主实现多模态 memory 时应在本模块一次读全。
    保持独立 ABC 而不并入 ``MemoryProvider``，是因为两者的能力**正交**：字节放在哪
    （文件系统 / 对象存储 / 数据库）与记忆怎么存是两个独立的部署决策，宿主换其中一个
    不应被迫换另一个。仓内实现见 ``ctx_weft.providers.blob.fs.FsBlobStore``
    （它同时也满足 ``EventBlobStore``，那是实现层的方便，不是协议层的关系）。

    ⚠️ 历史：Phase 3b 曾把本协议挂在 ``FilesystemToolsProvider`` 上（裁定 D5 移除），
    随后 ``SqlMemoryProvider`` 曾同时实现它、由 ``ProviderRegistry`` 自动解析
    （spec 2026-08-29 §5 移除——D4 论证的是引用边该与 ingest 同事务，不是字节该进 RDBMS）。
    现在字节侧只由显式注册决定。

    put 必须**内容寻址且幂等**：同样的 data 返回同样的 ref，重复调用不重复存。
    这同时给到三件事：写入端去重、重放安全、以及 rehydrate 字节稳定——同一 ref
    每次还原出的 base64 完全一致，Anthropic 的 prompt cache 前缀不会被打碎。

    get 对不存在 / 已回收的 ref 返回 None，**不得 raise**：blob 过期、宿主换机、
    GC 误删都会发生，调用方据此降级为文本占位，绝不因取图失败中断 loop。
    """

    @property
    def can_externalize(self) -> bool:
        """本 store 是否真的能存——``NullMemoryBlobStore`` 返回 False。

        调用方据此**先探询、再决定**，而不是调用 put 并捕获 NotImplementedError：
        后者会把「响亮失败」降级成控制流，让真正的接线错误也被静默吞掉
        （Phase 1 终审契约）。基类默认 True，既有实现无需改动。
        """
        return True

    @abstractmethod
    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        ...

    @abstractmethod
    async def get(
        self, ref: str, ctx: ProviderContext
    ) -> "tuple[bytes, str] | None":
        ...


class NullMemoryBlobStore(MemoryBlobStore):
    """未注册 MemoryBlobStore 时的默认实现——保证不接 blob 的宿主行为完全不变。

    put 刻意抛错：Phase 1 内没有任何调用方（外部化在 Phase 3），抛错可在
    Phase 3 接线错误时立刻暴露，而不是静默产出一个假 ref。调用方（
    ``core.content.normalize_content``）先探询 can_externalize 决定是否外部化，
    **不**捕获这里的 NotImplementedError——它仍是接线错误的响亮信号。
    """

    @property
    def can_externalize(self) -> bool:
        return False

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        raise NotImplementedError(
            "No MemoryBlobStore registered; register one via "
            "ProviderRegistry.register_memory_blob_store() before externalizing content."
        )

    async def get(
        self, ref: str, ctx: ProviderContext
    ) -> "tuple[bytes, str] | None":
        return None
