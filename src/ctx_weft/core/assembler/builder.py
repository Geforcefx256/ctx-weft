"""Prompt construction and budget enforcement used by Jinja templates.

Templates declare content, order, placement and discard priority through PromptBuilder.
The builder owns the mechanical work: exact token accounting, atomic-group trimming,
message decoration and protocol-preserving emission.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from ctx_weft.core.models.errors import ContextOverflowError
from ctx_weft.core.utils.content import image_part_count
from ctx_weft.protocols.content import content_to_text, content_with_prefix, content_with_suffix, downgrade_images_to_text
from ctx_weft.core.utils.estimate import (
    estimate_message_tokens,
    estimate_tools_tokens,
)
from ctx_weft.protocols import LLMMessage, LLMTool

OperationKind = Literal["system", "message", "before", "after", "user"]


@dataclass
class PromptOperation:
    kind: OperationKind
    content: str | LLMMessage
    priority: int
    required: bool
    sequence: int
    ref: str | None = None
    group: str | None = None
    timestamp: datetime | str | None = None
    token_count: int = 0
    merge: bool = False


def _intent_values(context) -> tuple[str, bool, str]:
    """Return template name, judge flag and compact scope after validation."""
    if context.purpose == "observe":
        return "observe", context.judge_task, "task"
    if context.purpose == "compact":
        return "compact", False, context.compact_scope
    if context.purpose == "recognize_intent":
        return "metadata", False, "task"
    return "act", False, "task"


def _identity_facet(template, name: str):
    if template is None:
        return None
    key = "recognize_intent" if name == "metadata" else name
    facet = template.identity.get(key)
    if facet is None and name == "metadata":
        facet = template.identity.get("act")
    return facet


def _is_blank(content: object) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content
    for part in content:
        if not hasattr(part, "text") or part.text:
            return False
    return True


def _memory_type(record) -> str:
    from ctx_weft.protocols.memory_compat import legacy_type_of

    return str(record.type or legacy_type_of(record.kind, record.scope, record.role))


def _history_facts(context, records=None):
    from ctx_weft.core.assembler.sources._history import history_facts
    from ctx_weft.protocols import MemoryScope

    records = context._records if records is None else records
    blocks = [b for b in context.memory_blocks if b["kind"] == "history"]
    base = max((r.metadata.get("seq_no", 0) for r in records.values()
                if r.scope == MemoryScope.TASK), default=0) + 1
    sequences = {}
    for block in blocks:
        record = records[block["memory_id"]]
        metadata = block["metadata"]
        sequences[record.id] = (base + metadata["staged_order"]
            if "staged_order" in metadata and "seq_no" not in record.metadata else
            metadata.get("sequence", record.metadata.get("seq_no", 0)))
    return history_facts([records[b["memory_id"]] for b in blocks],
                         task_id=context.task.get("id", ""), sequences=sequences)


_JSON_TYPE_ABBR = {
    "string": "str", "integer": "int", "number": "num", "boolean": "bool",
    "array": "list", "object": "obj", "null": "null",
}


def _compact_signature(schema: object) -> str:
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        return ""
    props = schema["properties"]
    required = set(schema.get("required") or ())
    result = []
    for name, spec in list(props.items())[:8]:
        raw_type = spec.get("type") if isinstance(spec, dict) else None
        if isinstance(raw_type, list):
            raw_type = raw_type[0] if raw_type else None
        result.append(
            f"{name}{'' if name in required else '?'}: "
            f"{_JSON_TYPE_ABBR.get(raw_type, raw_type or 'any')}")
    if len(props) > 8:
        result.append("…")
    return ", ".join(result)


def _shift_headings(markdown: str, base_level: int = 2) -> str:
    import re

    pattern = re.compile(r"^(#{1,6})\s")
    lines = markdown.split("\n")
    levels, fenced = [], False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith((chr(96) * 3, "~~~")):
            fenced = not fenced
        elif not fenced and (match := pattern.match(line)):
            levels.append(len(match.group(1)))
    shift = (base_level + 1 - min(levels)) if levels else 0
    if shift <= 0:
        return markdown
    output, fenced = [], False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith((chr(96) * 3, "~~~")):
            fenced = not fenced
        elif not fenced and (match := pattern.match(line)):
            line = "#" * min(6, len(match.group(1)) + shift) + line[len(match.group(1)):]
        output.append(line)
    return "\n".join(output)


def _capability_facts(context, purpose: str, capabilities=None) -> list[dict[str, Any]]:
    from ctx_weft.protocols.capability import qualify

    current_skill = context.skill_name
    blocks = [
        block for block in (context._capability_blocks if capabilities is None else capabilities)
        if purpose in block["definition"].get("purposes", ())
    ]
    sections = []
    for kind in ("skill", "tool", "agent"):
        candidates = [block for block in blocks if block["definition"]["kind"] == kind]
        if kind == "skill" and current_skill:
            candidates = [
                block for block in candidates
                if qualify(block["definition"]["id"]) != current_skill
            ]
        if not candidates:
            continue
        groups: dict[str, dict[str, Any]] = {}
        ungrouped = []
        for block in candidates:
            cap = block["definition"]
            item = dict(
                name=qualify(cap["id"]),
                signature=_compact_signature(cap.get("input_schema")),
                description=cap["description"],
            )
            provider = block["metadata"].get("provider_name", "")
            if not provider:
                ungrouped.append(item)
                continue
            group = groups.setdefault(provider, dict(
                name=provider,
                prefix=qualify(provider) + "__",
                description=(block["metadata"].get("provider_description") or "").strip(),
                entries=[],
            ))
            group["entries"].append(item)
        sections.append(dict(kind=kind, ungrouped=ungrouped, groups=list(groups.values())))
    return sections


def _prompt_facts(context, records=None, template=None, *, capabilities=None,
                  skill_definition=None) -> dict[str, Any]:
    from ctx_weft.core.utils.task_ref import task_label_parts, task_ref_parts
    from ctx_weft.protocols.content import normalize_content_parts

    records = context._records if records is None else records
    template_name, judge_task, scope = _intent_values(context)
    task = context.task
    title = task.get("title") or ""
    description = task.get("description") or ""

    def task_content(name):
        record_id = task.get(f"{name}_memory_id")
        return records[record_id].content if record_id else normalize_content_parts(task.get(name))

    raw_prompt = task_content("user_prompt")
    user_prompt = raw_prompt if isinstance(raw_prompt, str) else (
        content_to_text(raw_prompt) if raw_prompt else "")

    identity = template
    actor = _identity_facet(identity, "act")
    selected_name = (
        "observe" if template_name == "observe" and judge_task
        else "compact" if template_name in {"observe", "compact"}
        else template_name
    )
    selected = _identity_facet(identity, selected_name)
    compact = _identity_facet(identity, "compact")
    soul = getattr(selected, "text", "") if template_name == "act" else getattr(actor, "text", "")
    role = getattr(selected, "text", "")

    background_blocks = [item for item in context.memory_blocks + context.references if item["kind"] == "background"]
    background_blocks.sort(key=lambda item: item["position"])
    background = ""
    if background_blocks:
        item = background_blocks[0]
        background = content_to_text(records[item["memory_id"]].content if "memory_id" in item
                                     else normalize_content_parts(item["content"]))
    directive = None
    definition = skill_definition if skill_definition is not None else context._skill_definition
    if definition:
        directive = dict(
            skill_name=definition.get("skill_name", ""),
            body=_shift_headings(definition.get("instructions", "")),
        )

    capability_sections = _capability_facts(context, context.purpose, capabilities)
    guidance = None
    resume_snapshot = None
    if context.guidance:
        guidance = dict(context.guidance[0]["state"]["guidance"])
        resume_snapshot = context.guidance[0]["state"].get("resume")
    ctx = dict(
        title=title,
        description=description,
        user_prompt=user_prompt,
        task_ref=task_ref_parts(task.get("id", "") or "", title),
        soul=soul,
        project_background=background,
        directive=directive,
        capability_sections=capability_sections,
        guidance=guidance,
        resume_snapshot=resume_snapshot,
        needs_pointer=False,
        scope=scope,
        judge_task=judge_task,
    )
    if template_name == "observe":
        ctx.update(
            switch_reason=task.get("switch_reason", ""),
            terminal_only=context.terminal_only,
            evaluation_role=role if judge_task else "",
            summary_principles=getattr(compact, "text", "") if compact else "",
        )
        boundary = context.observe_boundary or "normal"
        supported_boundaries = {
            "actor_done", "finish", "finish_park", "normal", "plain_text",
            "max_turns", "context_limit", "interrupt", "dispatch", "mechanical", "task_switch",
        }
        ref = task_label_parts(task.get("id", ""), title, raw_prompt) if task.get("id") else ""
        outputs = task_content("outputs")
        output_text = outputs if isinstance(outputs, str) else (
            content_to_text(outputs) if outputs else "")
        inject = boundary in {"actor_done", "finish", "finish_park"}
        ctx.update(
            task_label=ref,
            boundary_kind=(boundary if boundary in supported_boundaries else "normal"),
            act_turns=context.observe_act_turns,
            observe_index=context.observe_index,
            final_output=output_text.strip() if inject else "",
        )
        if judge_task:
            subtasks = [
                dict(
                    ref=task_ref_parts(row.get("task_id", "") or "", row.get("title", "") or ""),
                    outcome=row.get("outcome", ""),
                    note=row.get("note"),
                )
                for row in context.subtasks
            ]
            ctx.update(
                plain_text_yielded=boundary == "plain_text",
                finish_park_yielded=boundary == "finish_park",
                subtasks=subtasks,
            )
        else:
            ctx["closed"] = boundary in {"finish", "normal"}
    else:
        ctx["role"] = role

    history, refs = _history_facts(context, records)
    sources = []
    for block in sorted(context.memory_blocks + context.knowledge + context.references,
                        key=lambda item: item["position"]):
        if block["kind"] not in {"blackboard", "summary", "reference"}:
            continue
        metadata = dict(block["metadata"])
        if "memory_id" in block:
            value = records[block["memory_id"]]
            content = value.content
            metadata = {**value.metadata, **metadata}
            timestamp = value.timestamp.isoformat() if value.timestamp else metadata.get("timestamp")
        else:
            value = block.get("document", block)
            content = normalize_content_parts(value["content"])
            metadata = {**value.get("metadata", value.get("value_metadata", {})), **metadata}
            timestamp = value.get("timestamp", metadata.get("timestamp"))
        sources.append(dict(
            kind=block["kind"],
            content=content_to_text(content),
            metadata=metadata,
            timestamp=timestamp,
            group=f"source:{block['id']}",
        ))
    return dict(
        template_name=template_name,
        ctx=ctx,
        history=history,
        source_blocks=sources,
        user_prompt_in_memory=task.get("user_prompt_in_memory", False),
        capabilities_priority=1,
        directive_priority=1,
        **refs,
    )


@dataclass
class PromptBuilder:
    """Mutable output surface exposed to a purpose Jinja template.

    Priority follows the existing convention: zero is pinned; larger numbers are
    discarded first. A group is atomic. If any member is required, the whole group is
    protected.
    """

    token_counter: Callable[[str], int]
    message_transform: Callable[[LLMMessage], LLMMessage] | None = None
    _operations: list[PromptOperation] = field(default_factory=list, init=False)
    _tools: list[LLMTool] = field(default_factory=list, init=False)
    _serial: int = field(default=0, init=False)
    _tool_groups: dict[str, str] = field(default_factory=dict, init=False)
    _last_refs: tuple[str, ...] = field(default=(), init=False)
    _last_trace: list[dict] = field(default_factory=list, init=False)
    _group_alternatives: dict[str, list[PromptOperation]] = field(default_factory=dict, init=False)

    def compile(self, context, *, tools: list[LLMTool] | None = None, records=None,
                template=None, trim: bool = True, capabilities=None, skill_definition=None):
        """Compile one PromptContext, optionally enforcing its input budget."""
        from ctx_weft.core.assembler.prompts import render_purpose
        from ctx_weft.core.utils.estimate import effective_limit, tools_signature

        token_limit = effective_limit(context.context_limit, context.reserved_output_tokens) if trim else None
        self._operations.clear()
        self._serial = 0
        self._tool_groups.clear()
        self._group_alternatives.clear()
        self.tools(tools or [])
        facts = _prompt_facts(context, records, template, capabilities=capabilities,
                              skill_definition=skill_definition)
        purpose = facts.pop("template_name")
        self.message_transform = None
        if purpose != "act":
            self.message_transform = lambda message: replace(
                message, content=downgrade_images_to_text(message.content))
        render_purpose(
            purpose,
            prompt=self,
            purpose=purpose,
            context=context,
            **facts,
        )
        assembled = self.build(
            token_limit,
            context_limit=context.context_limit,
            reserved_output_tokens=context.reserved_output_tokens,
        )
        assembled.metadata["tools_signature"] = tools_signature(assembled.tools)
        assembled.metadata["purpose"] = context.purpose
        return assembled

    def _add(
        self,
        kind: OperationKind,
        content: str | LLMMessage,
        *,
        priority: int,
        required: bool = False,
        ref: str | None = None,
        group: str | None = None,
        timestamp: datetime | str | None = None,
    ) -> str | None:
        if priority < 0:
            raise ValueError("prompt operation priority must be non-negative")
        if kind in {"before", "after"} and not ref:
            raise ValueError(f"{kind} operation requires a message ref")
        if kind == "message" and not ref:
            ref = f"message:{self._serial}"
        if kind == "user" and ref is None:
            ref = f"generated:{self._serial}"

        if isinstance(content, LLMMessage):
            counted = self.message_transform(content) if self.message_transform else content
            tokens = estimate_message_tokens(counted, self.token_counter)
        else:
            tokens = self.token_counter(content)

        operation = PromptOperation(
            kind=kind,
            content=deepcopy(content),
            priority=priority,
            required=required,
            sequence=self._serial,
            ref=ref,
            group=group,
            timestamp=timestamp,
            token_count=tokens,
        )
        self._serial += 1
        self._operations.append(operation)
        return ref

    def system(
        self, content: str, *, priority: int, required: bool = False,
        group: str | None = None,
    ) -> None:
        if content:
            self._add("system", content, priority=priority, required=required, group=group)

    def message(
        self,
        message: LLMMessage,
        *,
        ref: str | None = None,
        priority: int,
        required: bool = False,
        group: str | None = None,
        timestamp: datetime | str | None = None,
        content: Any = ...,
        role: str | None = None,
    ) -> str:
        if content is not ...:
            message = replace(message, content=content)
        if role is not None:
            message = replace(message, role=role)
        # Derive an atomic group for tool exchanges when the template does not provide one.
        if group is None and message.role == "assistant" and message.tool_calls:
            group = f"tool-exchange:{ref or self._serial}"
            for call in message.tool_calls:
                call_id = str(call.get("id", ""))
                if call_id:
                    self._tool_groups[call_id] = group
        elif group is None and message.role == "tool" and message.tool_call_id:
            group = self._tool_groups.get(message.tool_call_id)

        result = self._add(
            "message", message, ref=ref, priority=priority, required=required,
            group=group, timestamp=timestamp,
        )
        assert result is not None
        return result

    def before(
        self, ref: str, content: str, *, priority: int, required: bool = False,
        group: str | None = None, timestamp: datetime | str | None = None,
    ) -> None:
        if content:
            self._add(
                "before", content, ref=ref, priority=priority,
                required=required, group=group, timestamp=timestamp,
            )

    def after(
        self, ref: str, content: str, *, priority: int, required: bool = False,
        group: str | None = None, timestamp: datetime | str | None = None,
    ) -> None:
        if content:
            self._add(
                "after", content, ref=ref, priority=priority,
                required=required, group=group, timestamp=timestamp,
            )

    def user(
        self,
        content: str,
        *,
        priority: int,
        required: bool = False,
        merge: bool = True,
        group: str | None = None,
    ) -> str | None:
        if not content:
            return None
        # merge is kept on the message as builder-only metadata, rather than leaking into
        # the provider protocol. The operation kind already distinguishes generated users.
        ref = self._add(
            "user", LLMMessage(role="user", content=content), priority=priority,
            required=required, group=group,
        )
        assert ref is not None
        self._operations[-1].merge = merge
        return ref

    def tools(self, tools: list[LLMTool]) -> None:
        self._tools = deepcopy(tools)

    def alternative(self, group: str, caller=None) -> str:
        """Capture ordinary template operations as an alternative to an atomic group."""
        full = [op for op in self._operations if op.group == group]
        start = len(self._operations)
        tool_groups = dict(self._tool_groups)
        try:
            if caller is not None:
                rendered = caller()
                if rendered.strip():
                    raise ValueError("alternative program emitted stray text")
        finally:
            self._tool_groups = tool_groups
        operations = self._operations[start:]
        del self._operations[start:]
        if full and operations:
            sequence = max(op.sequence for op in full)
            first_ref = next(op.ref for op in full if op.kind in {"message", "user"})
            old_ref = next(op.ref for op in operations if op.kind in {"message", "user"})
            for op in operations:
                op.sequence = sequence
                op.group = group
                if op.ref == old_ref:
                    op.ref = first_ref
            self._group_alternatives[group] = operations
        return ""

    def build(
        self,
        token_limit: int | None,
        *,
        context_limit: int | None = None,
        reserved_output_tokens: int = 0,
    ):
        """Materialize operations, trimming only when a token limit is provided."""
        from ctx_weft.core.assembler.assembler import AssembledPrompt

        operations = list(self._operations)
        tools_tokens = estimate_tools_tokens(self._tools, self.token_counter)
        content_limit = None if token_limit is None else max(0, token_limit - tools_tokens)

        def group_key(op: PromptOperation) -> str:
            return op.group or f"operation:{op.sequence}"

        grouped: dict[str, list[PromptOperation]] = defaultdict(list)
        for operation in operations:
            grouped[group_key(operation)].append(operation)

        kept_groups = set(grouped)
        replacements: dict[str, list[PromptOperation]] = {}
        total = sum(op.token_count for op in operations)
        owner_group = {
            operation.ref: group_key(operation)
            for operation in operations
            if operation.kind in {"message", "user"} and operation.ref is not None
        }
        decoration_groups: dict[str, set[str]] = defaultdict(set)
        for operation in operations:
            if operation.kind in {"before", "after"} and operation.ref in owner_group:
                decoration_groups[owner_group[operation.ref]].add(group_key(operation))

        def discard_group(key: str) -> None:
            """Drop a group and decorations whose target disappears with it."""
            nonlocal total
            pending = [key]
            while pending:
                current = pending.pop()
                if current not in kept_groups:
                    continue
                kept_groups.discard(current)
                total -= sum(member.token_count for member in replacements.get(current, grouped[current]))
                pending.extend(decoration_groups.get(current, ()))

        candidates = []
        for key, members in grouped.items():
            required = any(member.required or member.priority == 0 for member in members)
            priority = max(member.priority for member in members)
            timestamp = min(
                (member.timestamp.isoformat()
                 if isinstance(member.timestamp, datetime) else str(member.timestamp)
                 for member in members
                 if member.timestamp is not None),
                default="",
            )
            tokens = sum(member.token_count for member in members)
            candidates.append((required, -priority, timestamp, -tokens, key))

        if content_limit is not None:
            for required, _negative_priority, _timestamp, _negative_tokens, key in sorted(candidates):
                if total <= content_limit:
                    break
                if required:
                    continue
                fallback = self._group_alternatives.get(key)
                if fallback:
                    old_tokens = sum(op.token_count for op in grouped[key])
                    new_tokens = sum(op.token_count for op in fallback)
                    if new_tokens < old_tokens:
                        replacements[key] = fallback
                        total -= old_tokens - new_tokens
            # Consider registered alternatives before dropping whole groups.
            for required, _negative_priority, _timestamp, _negative_tokens, key in sorted(candidates):
                if total <= content_limit:
                    break
                if required:
                    continue
                discard_group(key)

        if content_limit is not None and total > content_limit:
            assert token_limit is not None
            floor = [
                op for op in operations
                if group_key(op) in kept_groups and (op.required or op.priority == 0)
            ]
            images = sum(
                image_part_count(op.content.content)
                for op in floor
                if isinstance(op.content, LLMMessage)
            )
            raise ContextOverflowError(
                required=sum(op.token_count for op in floor) + tools_tokens,
                effective_limit=token_limit,
                context_limit=context_limit if context_limit is not None else token_limit,
                reserved_output_tokens=reserved_output_tokens,
                image_count=images,
            )

        kept = [op for op in operations if group_key(op) in kept_groups
                and group_key(op) not in replacements]
        kept.extend(op for key, replacement in replacements.items() if key in kept_groups
                    for op in replacement)
        kept_refs = {op.ref for op in kept if op.kind in {"message", "user"}}
        kept = [
            op for op in kept
            if op.kind not in {"before", "after"} or op.ref in kept_refs
        ]

        messages: list[LLMMessage] = []
        message_refs: list[str] = []
        trace: list[dict[str, Any]] = []
        ref_map: dict[str, int] = {}
        pending: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: {"before": [], "after": []})

        def decorate(index: int, ref: str, *, before: str = "", after: str = "") -> None:
            message = messages[index]
            content = message.content
            if before:
                content = content_with_prefix(content, before)
            if after:
                content = content_with_suffix(content, after)
            messages[index] = replace(message, content=content)
            if before or after:
                trace.append(
                    dict(operation="decorate", ref=ref, before=before, after=after))

        def append_message(
            message: LLMMessage, ref: str, *, merge: bool = False,
        ) -> int | None:
            if (
                message.content == ""
                and not message.tool_calls
                and not message.tool_call_id
                and message.reasoning_content is None
            ):
                return None
            last = messages[-1] if messages else None
            if (
                merge
                and last is not None
                and last.role == message.role == "user"
                and not last.tool_calls
                and not message.tool_calls
                and not last.tool_call_id
                and not message.tool_call_id
                and message.reasoning_content is None
            ):
                if not isinstance(message.content, str):
                    raise TypeError("merged append requires text")
                decorate(
                    len(messages) - 1,
                    message_refs[-1],
                    after="\n\n" + message.content,
                )
                trace.append(dict(operation="merge", ref=message_refs[-1]))
                return len(messages) - 1
            messages.append(deepcopy(message))
            message_refs.append(ref)
            trace.append(dict(operation="insert", ref=ref, index=len(messages) - 1))
            return len(messages) - 1

        def flush_decorations() -> None:
            for ref, values in list(pending.items()):
                if ref in ref_map:
                    decorate(
                        ref_map[ref],
                        ref,
                        before="".join(values["before"]),
                        after="".join(values["after"]),
                    )
                del pending[ref]

        for operation in sorted(kept, key=lambda op: op.sequence):
            if operation.kind == "system":
                continue
            if operation.kind == "message":
                flush_decorations()
                assert isinstance(operation.content, LLMMessage)
                assert operation.ref is not None
                index = append_message(operation.content, operation.ref)
                if index is not None:
                    ref_map[operation.ref] = index
            elif operation.kind == "user":
                flush_decorations()
                assert isinstance(operation.content, LLMMessage)
                assert operation.ref is not None
                index = append_message(
                    operation.content, operation.ref, merge=operation.merge)
                if index is not None:
                    ref_map[operation.ref] = index
            elif operation.kind in {"before", "after"} and operation.ref is not None:
                assert isinstance(operation.content, str)
                pending[operation.ref][operation.kind].append(operation.content)
        flush_decorations()

        system = "".join(
            op.content for op in sorted(kept, key=lambda op: op.sequence)
            if op.kind == "system" and isinstance(op.content, str)
        )
        if self.message_transform is not None:
            messages = [self.message_transform(message) for message in messages]
        self._last_refs = tuple(message_refs)
        self._last_trace = trace
        final_count = self.token_counter(system) + sum(
            estimate_message_tokens(message, self.token_counter)
            for message in messages
        ) + tools_tokens
        return AssembledPrompt(
            system=system,
            messages=messages,
            tools=deepcopy(self._tools),
            token_count=final_count,
            token_budget=token_limit if token_limit is not None else 0,
            metadata={"tools_reserved_tokens": tools_tokens},
        )

    @property
    def refs(self) -> tuple[str, ...]:
        """References from the most recent build, after trimming and merges."""
        return self._last_refs

    @property
    def trace(self) -> list[dict]:
        """Editing trace from the most recent build."""
        return list(self._last_trace)


def compile_history(records, *, token_counter=len, locale="en", task_id="", decorate=False):
    """Execute the shared history program without a task instruction or budget trim."""
    from ctx_weft.core.assembler.prompts import render_purpose
    from ctx_weft.core.assembler.sources._history import history_facts
    rows, facts = history_facts(records, task_id=task_id)
    builder = PromptBuilder(token_counter)
    render_purpose("history", locale=locale, prompt=builder, history=rows,
                   history_decorate=decorate, **facts)
    result = builder.build(None)
    result.metadata["history_refs"] = builder.refs
    result.metadata["history_sources"] = {row["ref"]: row for row in rows}
    result.metadata["alternatives"] = {
        group: [op.content for op in operations if op.kind in {"message", "user"}]
        for group, operations in builder._group_alternatives.items()
    }
    return result


def history_records(records, *, locale="en"):
    """Materialize an untrimmed history view with source IDs and blob references."""
    result = compile_history(records, locale=locale)
    sources = result.metadata["history_sources"]
    return [replace(sources[ref]["record"], content=message.content, role=message.role,
                    timestamp=sources[ref]["timestamp"])
            for ref, message in zip(result.metadata["history_refs"], result.messages, strict=True)]


def history_token_costs(records, *, token_counter, task_id="", known=None):
    """Return rendered history cost and reusable per-tree costs, without I/O.

    The memo is explicit input/output owned by the caller. Its keys contain only
    immutable record IDs; obsolete groups are dropped on every call.
    """
    from ctx_weft.core.assembler.sources._history import history_facts
    from ctx_weft.core.assembler.legacy_task_memory import capsule_origin
    from ctx_weft.protocols import MemoryScope
    known = known or {}
    rows, facts = history_facts(records, task_id=task_id)
    parents, keys, calls = {}, {}, {}

    def find(key):
        parents.setdefault(key, key)
        root = key
        while parents[root] != root:
            root = parents[root]
        while parents[key] != key:
            following = parents[key]
            parents[key] = root
            key = following
        return root

    def join(left, right):
        parents[find(left)] = find(right)

    for row in records:
        origin = row.address.task_id if row.scope == MemoryScope.TASK else capsule_origin(row)
        key = ("task", origin) if origin else ("record", row.id)
        keys[row.id] = key
        find(key)
    for task in facts["task_history"].values():
        join(("task", task["id"]), ("task", task["tree_key"][-1]))
    for row in rows:
        key = keys[row["record_id"]]
        ids = [c.get("id") for c in row["message"].tool_calls]
        ids += [row["message"].tool_call_id, row["marker"].get("group_id")]
        for call_id in filter(None, ids):
            if call_id in calls:
                join(key, calls[call_id])
            calls[call_id] = key
    groups = defaultdict(list)
    for row in records:
        groups[find(keys[row.id])].append(row)
    costs = {}
    for group in groups.values():
        key = tuple(r.id for r in group)
        costs[key] = known[key] if key in known else compile_history(
            group, token_counter=token_counter, task_id=task_id).token_count - token_counter("")
    return sum(costs.values()) + token_counter(""), costs
