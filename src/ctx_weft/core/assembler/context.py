"""Assembled prompt inputs, also persisted directly as the prompt event payload.

The private record cache reuses immutable records returned by Sources. It is
optional to compile and never serialized; no live task/session objects are kept.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from enum import Enum

from ctx_weft.protocols.content import content_to_jsonable
from ctx_weft.protocols import MemoryAddress, MemoryRecord
from ctx_weft.protocols.memory import MemoryEventType, MemoryKind, MemoryScope


def record_from_json(data):
    """Decode only an inline memory value, without reconstructing runtime state."""
    data = dict(data)
    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
    for name, klass in (("type", MemoryEventType), ("kind", MemoryKind), ("scope", MemoryScope)):
        if data.get(name) is not None:
            data[name] = klass(data[name])
    if data.get("address") is not None:
        data["address"] = MemoryAddress(**data["address"])
    return MemoryRecord(**data)


def json_value(value):
    """Detach source data while retaining its original JSON types."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported prompt input: {type(value).__name__}")


@dataclass
class PromptContext:
    purpose: str = "act"
    template_id: str = ""
    scope: dict = field(default_factory=dict)
    task: dict = field(default_factory=dict)
    skill_name: str = ""
    context_limit: int = 100_000
    reserved_output_tokens: int = 0
    judge_task: bool = True
    terminal_only: bool = False
    compact_scope: str = "task"
    observe_boundary: str = "normal"
    observe_act_turns: int | None = None
    subtasks: list[dict] = field(default_factory=list)
    memory_blocks: list[dict] = field(default_factory=list)
    capabilities: list[dict] = field(default_factory=list)
    guidance: list[dict] = field(default_factory=list)
    knowledge: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)
    token_counts: dict[str, int] = field(default_factory=dict)
    appended_messages: list[dict] = field(default_factory=list)
    tools: list[dict] | None = None
    def __post_init__(self):
        if self.purpose not in {"act", "observe", "compact", "recognize_intent"}:
            raise ValueError(f"unsupported prompt purpose: {self.purpose!r}")
        if self.terminal_only and not self.judge_task:
            raise ValueError("terminal_only requires judge_task")
        if self.compact_scope not in {"task", "agent"}:
            raise ValueError(f"unsupported compact scope: {self.compact_scope!r}")
        # Not a dataclass field: even a generic asdict() cannot persist bodies.
        self._records: dict[str, MemoryRecord] = {}
        self._record_context: tuple[str, str] | None = None
        # Explicitly opt-in, per-input runtime cache. Never part of event data.
        self._compiled_base = None
        # Raw Source values support standalone builders; never serialized.
        self._capability_blocks: list[dict] = []
        self._skill_definition: dict | None = None

    @classmethod
    def from_request(cls, request):
        """Start assembly with detached task/options, not live runtime objects."""
        extra, intent = request.extra or {}, request.intent
        return cls(
            purpose=request.purpose,
            template_id=getattr(request.template, "id", "") or "",
            scope={name: getattr(getattr(request, "scope", None), name, None)
                   for name in ("session_id", "task_id", "agent_id")},
            task=cls.task_fields(request.task),
            skill_name=getattr(getattr(request.task, "settings", None), "skill_name", ""),
            context_limit=request.resolved_model.context_limit,
            reserved_output_tokens=request.resolved_model.reserved_output_tokens,
            judge_task=getattr(intent, "judge_task", extra.get("judge_task", True)),
            terminal_only=bool(getattr(intent, "terminal_only", False)),
            compact_scope=getattr(intent, "scope", extra.get("compact_scope", "task")),
            observe_boundary=extra.get("observe_boundary", "normal"),
            observe_act_turns=extra.get("observe_act_turns"),
            subtasks=json_value(extra.get("subtasks", [])),
        )

    @staticmethod
    def task_fields(task):
        return {name: json_value(getattr(task, name, default)) for name, default in (
            ("id", ""), ("title", ""), ("description", ""), ("user_prompt", None),
            ("user_prompt_in_memory", False), ("outputs", None), ("switch_reason", ""),
        )}

    def add(self, block, *, staged_ids=()):
        """Append a Source result directly to the event's input representation."""
        value = block.value
        if block.kind == "identity":
            self.template_id = getattr(value, "id", "") or self.template_id
            return
        if block.kind == "task_spec":
            self.task = self.task_fields(value)
            self.skill_name = getattr(getattr(value, "settings", None), "skill_name", "")
            return
        position = sum(len(getattr(self, name)) for name in (
            "memory_blocks", "capabilities", "guidance", "knowledge", "references"))
        item = dict(position=position, id=block.id, source=block.source,
                    kind=block.kind, metadata=json_value(block.metadata))
        memory_id = getattr(block, "memory_id", None)
        if isinstance(value, MemoryRecord) or memory_id is not None:
            item["memory_id"] = memory_id if memory_id is not None else value.id
            if isinstance(value, MemoryRecord):
                self._records[value.id] = deepcopy(value)
                if value.id in staged_ids or not value.id:
                    item["record"] = json_value(value)
            self.memory_blocks.append(item)
        elif block.kind == "capabilities":
            from ctx_weft.core.assembler.sources.capability import capability_ref

            self.capabilities.append(capability_ref(value, block.metadata.get("provider_name")))
            item["definition"] = json_value(value)
            self._capability_blocks.append(item)
        elif block.kind == "directive":
            self._skill_definition = {name: getattr(value, name, "") for name in (
                "skill_id", "skill_name", "instructions")}
        elif block.kind == "guidance":
            item["state"] = json_value(value)
            self.guidance.append(item)
        elif block.kind == "reference" and is_dataclass(value):
            item["document"] = json_value(value)
            self.knowledge.append(item)
        elif block.kind in {"background", "blackboard", "summary", "reference"}:
            item.update(content=json_value(getattr(value, "content", value)),
                        value_metadata=json_value(getattr(value, "metadata", {})),
                        timestamp=json_value(getattr(value, "timestamp", None)))
            self.references.append(item)
        else:
            raise TypeError(f"unsupported prompt source {block.source!r} ({block.kind})")

    def to_payload(self):
        return {"version": 1, **{f.name: json_value(getattr(self, f.name))
                                for f in fields(self) if not f.name.startswith("_")}}

    @classmethod
    def from_payload(cls, payload):
        if payload.get("version") != 1:
            raise ValueError("unsupported prompt input version")
        names = {f.name for f in fields(cls) if not f.name.startswith("_")}
        return cls(**{name: deepcopy(value) for name, value in payload.items() if name in names})

    def for_request(self, appended_messages, tool_refs):
        if any(not isinstance(ref, dict) or set(ref) != {"provider", "id"}
               for ref in tool_refs):
            raise ValueError("prompt tools must contain only provider/id references")
        messages = []
        for message in appended_messages:
            item = dict(role=message.role, tool_calls=json_value(message.tool_calls),
                        tool_call_id=message.tool_call_id)
            if message.memory_id:
                item["memory_id"] = message.memory_id
            else:
                item["content"] = content_to_jsonable(message.content)
            messages.append(item)
        return replace(self, appended_messages=messages, tools=json_value(tool_refs))

    def memory_ids(self):
        ids = [item["memory_id"] for item in self.memory_blocks if item.get("memory_id")]
        ids += [item["memory_id"] for item in self.appended_messages if item.get("memory_id")]
        ids += [self.task[name] for name in ("user_prompt_memory_id", "outputs_memory_id") if name in self.task]
        return list(dict.fromkeys(ids))
