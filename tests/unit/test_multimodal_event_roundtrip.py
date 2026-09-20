import hashlib
import json
from datetime import datetime, timezone

from ctx_weft.core.utils.content import content_to_event_jsonable, content_to_jsonable
from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events, serialize_view
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.core.orchestrator.task.manager import task_payload
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import ImagePart, ProviderContext, TextPart
from ctx_weft.protocols.events import EventBlobStore


class _StubEventBlobStore(EventBlobStore):
    """最小 EventBlobStore 桩，仅供本文件把 base64 算成 event-jsonable 用。"""

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        ref = f"blob:{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _content():
    return [TextPart(text="看这张图"), ImagePart(data="ZGF0YQ==", media_type="image/png")]


def _ts():
    return datetime(2026, 8, 23, tzinfo=timezone.utc)


def _ev(seq: int, type_: str, **payload) -> Event:
    """与 tests/unit/test_snapshot_recovery.py 同一构造惯例。"""
    task_id = payload.pop("task_id", None)
    return Event(
        id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id="s1",
        type=type_, timestamp=_ts(), task_id=task_id, payload=payload,
    )


# ── 写侧：payload 必须是可 json 化的形态 ──────────────────────────────────

async def testtask_payload_is_json_serializable():
    """ContentPart 是普通 dataclass，直接进 payload 会让宿主的 json 持久化炸掉。

    ``user_prompt_jsonable`` 现在是必传参数（I2）：调用方须先经
    ``content_to_event_jsonable`` 把 base64 外部化成 ref，task_payload 不再有
    「退回同步 content_to_jsonable、原样塞回 base64」的隐藏分支——这里传入的正是
    event-jsonable 结果，而不是含字节的 ``content_to_jsonable(_content())``。
    """
    task = Task(
        id="tsk_1", session_id="s1", status="ACTIVE", tenant_id="default",
        assigned_agent_id="a1", creator_agent_id="a1",
        title="T", description="d", user_prompt=_content(), created_at=now_utc(),
    )
    evt = _StubEventBlobStore()
    user_prompt_jsonable = await content_to_event_jsonable(
        task.user_prompt, event_blob_store=evt, ctx=_ctx())
    payload = task_payload(task, user_prompt_jsonable=user_prompt_jsonable)
    json.dumps(payload)  # 不抛即通过
    assert payload["task"]["user_prompt"] == user_prompt_jsonable
    # 事件 payload 里绝不能出现原始字节：base64 已被替换成 ref
    assert "ZGF0YQ==" not in json.dumps(payload)


def testtask_payload_plain_text_unchanged():
    task = Task(
        id="tsk_1", session_id="s1", status="ACTIVE", tenant_id="default",
        assigned_agent_id="a1", creator_agent_id="a1",
        title="T", description="d", user_prompt="纯文本", created_at=now_utc(),
    )
    payload = task_payload(task, user_prompt_jsonable="纯文本")
    assert payload["task"]["user_prompt"] == "纯文本"


# ── 读侧：事件回放 → 投影 → 运行时模型，完整还原 ─────────────────────────

def test_task_user_prompt_survives_event_replay():
    """跨重启的核心保证。"""
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="看这张图",
            template_id="tmpl_a", root_agent_id="agt_root"),
        _ev(2, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "ACTIVE", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
            "user_prompt": content_to_jsonable(_content()),
        }),
    ]
    view = reduce_events(events, run_id="s1")
    task = task_from_projection(view.tasks["tsk_1"])
    assert task.user_prompt == _content(), "多模态 prompt 必须经事件回放完整还原"


def test_task_requeued_replay_restores_multimodal_prompt():
    """TASK_REQUEUED 回放分支须把携带的 prompt 完整还原为 part 列表。

    今天没有发射点往这个 payload 里放 `user_prompt`（已删的 reopen 是唯一一个），
    但**存量日志里有**——这条覆盖的正是那条存量兼容读取：不还原它，被 reopen 过的
    老会话重放后 prompt 会退回 TaskCreated 的原始值，即重启前后不一致。
    """
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="看这张图",
            template_id="tmpl_a", root_agent_id="agt_root"),
        _ev(2, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "ACTIVE", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
            "user_prompt": "纯文本",
        }),
        _ev(3, EventType.TASK_REQUEUED, task_id="tsk_1",
            user_prompt=content_to_jsonable(_content())),
    ]
    view = reduce_events(events, run_id="s1")
    task = task_from_projection(view.tasks["tsk_1"])
    assert task.user_prompt == _content(), "requeue 后的 user_prompt 必须还原为 part 列表"


def test_snapshot_roundtrip_preserves_parts():
    """快照路径（serialize_view）与事件回放路径必须同样无损。"""
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="看这张图",
            template_id="tmpl_a", root_agent_id="agt_root"),
        _ev(2, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "ACTIVE", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
            "user_prompt": content_to_jsonable(_content()),
        }),
    ]
    view = reduce_events(events, run_id="s1")
    blob = serialize_view(view)
    json.dumps(blob)  # 快照必须可 json 化
    from ctx_weft.core.control.reducers import deserialize_view
    restored = deserialize_view(blob)
    assert task_from_projection(restored.tasks["tsk_1"]).user_prompt == _content()
