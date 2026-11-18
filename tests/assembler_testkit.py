"""Small test helpers for the raw-block/Jinja assembler architecture."""

from __future__ import annotations

from tests.model_testkit import model_snapshot


from types import SimpleNamespace

from ctx_weft.core.assembler.builder import PromptBuilder, compile_history
from ctx_weft.core.assembler.context import PromptContext


class ListedCapabilities:
    def __init__(self, capabilities, *, description="", definitions=None):
        self.capabilities = capabilities
        self.description = description
        self.definitions = definitions or {}
        self.calls = 0

    async def list(self, ctx):
        self.calls += 1
        return list(self.capabilities)

    async def describe(self, ctx):
        from ctx_weft.protocols.capability import CapabilityProviderInfo

        return CapabilityProviderInfo(name="test", description=self.description,
                                      capability_count=len(self.capabilities))

    async def load_definition(self, name, ctx):
        return self.definitions.get(name)


def listed_providers(capabilities):
    groups = {}
    for cap in capabilities:
        groups.setdefault(cap.id.rsplit(":", 1)[0], []).append(cap)
    return {name: ListedCapabilities(caps) for name, caps in groups.items()}


def prompt_request(*, task_id: str = "t1", purpose: str = "act", **overrides):
    values = dict(
        purpose=purpose,
        intent=None,
        task=SimpleNamespace(
            id=task_id,
            title="Task",
            description="",
            user_prompt="",
            user_prompt_in_memory=True,
            outputs="",
            unattended=False,
        ),
        agent=SimpleNamespace(id="agent-1"),
        session=SimpleNamespace(),
        resolved_model=model_snapshot(100_000, 0),
        template=None,
        extra={},
        token_counter=len,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def history_messages(blocks, *, task_id: str = "t1"):
    """Project raw history blocks with the same mapper used before Jinja placement."""
    request = prompt_request(task_id=task_id)
    context = context_from_blocks(request, blocks)
    return compile_history(context._records.values(), task_id=task_id).messages


def compile_prompt(blocks, request=None, *, tools=None):
    request = request or prompt_request()
    return PromptBuilder(request.token_counter).compile(
        context_from_blocks(request, blocks), tools=tools or [], template=request.template)


def context_from_blocks(request, blocks):
    context = PromptContext.from_request(request)
    for block in blocks:
        context.add(block)
    return context


async def compiled_test_inputs(inputs, **kwargs):
    """Compiler half for tests whose assembler already returns a scripted prompt."""
    return getattr(inputs, "_scripted_prompt", inputs)


def scripted_inputs(prompt=None, *, system="", messages=(), tools=(), purpose="act"):
    """Keep step tests scripted while exercising the PromptContext boundary."""
    from ctx_weft.core.assembler.assembler import AssembledPrompt
    from ctx_weft.protocols import LLMTool

    if isinstance(prompt, PromptContext):
        return prompt
    if prompt is None:
        prompt = AssembledPrompt(system, list(messages), list(tools), 0)
    if not hasattr(prompt, "messages"):
        prompt.messages = []
    prompt.tools = [tool if isinstance(tool, LLMTool) else LLMTool(
        name=tool.name, description=getattr(tool, "description", ""),
        input_schema=getattr(tool, "input_schema", {})) for tool in getattr(prompt, "tools", [])]
    context = PromptContext(purpose=purpose)
    from ctx_weft.core.assembler.context import json_value
    context.references = [dict(position=i, id=f"test-{i}", source="test", kind="reference",
                               metadata={}, content=json_value(m.content), value_metadata={},
                               timestamp=None) for i, m in enumerate(prompt.messages)]
    context._scripted_prompt = prompt
    prompt.inputs = context
    return context


async def recalled_blocks(request, deps):
    """Inspect a Source's selected records after the explicit ID-read boundary."""
    from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource

    blocks = [b async for b in AgentRecallSource().fetch(request, deps)]
    ids = [b.memory_id for b in blocks if b.memory_id is not None]
    records = {r.id: r for r in await deps.memory.read_records(ids, deps.provider_ctx)} if ids else {}
    for block in blocks:
        if block.memory_id is not None:
            block.value = records[block.memory_id]
            block.metadata.setdefault("sequence", block.value.metadata.get("seq_no", 0))
    return blocks
