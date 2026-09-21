from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.core.models.session import Session
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import ImagePart, ProviderContext, TextPart
from ctx_weft.protocols.events import EventBlobStore
from ctx_weft.providers.llm.mock import MockLLMAdapter
from ctx_weft.providers.llm.provider import _FixedModelClient
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio


class _StubEventBlobStore(EventBlobStore):
    """携图路径的第三道门控（Task 4）要求宿主注册 EventBlobStore——最小可外部化桩。

    不 import 其他测试模块的等价实现（约定：测试模块之间不互相 import），故各文件
    各放一份最小拷贝。
    """

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        import hashlib
        ref = f"blob:{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


def _content():
    return [TextPart(text="看这张图"), ImagePart(data="ZGF0YQ==", media_type="image/png")]


async def test_session_start_params_accepts_parts():
    p = SessionStartParams.create(
        template_id="t", user_prompt=_content(), context_limit=1000,
    )
    assert p.user_prompt == _content()


async def test_session_start_params_still_accepts_str():
    p = SessionStartParams.create(
        template_id="t", user_prompt="纯文本", context_limit=1000,
    )
    assert p.user_prompt == "纯文本"


async def test_root_task_carries_full_content_session_carries_summary():
    sm = SessionRegistry(
        agent_lifecycle_manager=SimpleNamespace(),
        event_bus=SimpleNamespace(emit=AsyncMock()),
    )
    session = Session(
        id="s1", user_prompt=content_to_text(_content()), status="RUNNING",
        tenant_id="default", root_agent_id="a1", created_at=now_utc(),
    )
    # event 侧载荷由调用方（入口）从原始 content 算好后传入（blob-store 解耦 Task 3）：
    # 本用例只关心 Task/Session 各自承载什么，故给一份形态正确的最小载荷即可。
    task, _tm = await sm._make_root_task_manager(
        session, _content(), None, [{"type": "text", "text": "看这张图"}],
    )
    assert task.user_prompt == _content(), "Task 承载全量内容"
    assert isinstance(session.user_prompt, str), "Session 只承载文本摘要"
    # _make_root_task_manager 从不从 user_prompt 派生 description（源码恒为 ""）——
    # 断言确切值，而非宽松的 "" or isinstance(str) 判据，确保未来有人在此加派生逻辑时
    # 这条测试真的会盯着它是否走了 content_to_text。
    assert task.description == "", \
        "description 必须是 str——对 list 切片会静默产出截断的 part 列表"


async def test_run_single_task_root_task_description_is_text_not_truncated_parts():
    """run_single_task 内 description=user_prompt[:200] 是本任务点名的陷阱：对 list
    切片不报错，会得到截断的 part 列表。必须改走 content_to_text(user_prompt)[:200]。

    Phase 1 不碰装配链（core/assembler/），multimodal Task.user_prompt 走到 assembler
    的 task_spec source 还不被支持（那是后续任务的范围）——故这里在 _execute_task 处
    打桩截获 Task，只验证 run_single_task 自身构造 Task 的这一步，不跑真实 loop。
    """
    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    runtime = make_runtime(agent_provider=templates)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    runtime.providers.register_event_blob_store(_StubEventBlobStore())
    runtime.providers.register_llm_provider(
        SimpleNamespace(
            get_client=lambda account=None, model=None: _FixedModelClient(
                MockLLMAdapter(responses=[]), model or "mock-model", 128_000, 8_192,
                account=account or "acct-main",
            ),
        )
    )

    captured = {}

    async def _fake_execute_task(**kwargs):
        captured["task"] = kwargs["task"]
        # run_outcome=None：run 的结局是 run_single_task 收尾时要消费的（Task 4）。
        return SimpleNamespace(run_outcome=None), SimpleNamespace()

    runtime._execute_task = _fake_execute_task  # type: ignore[method-assign]

    await runtime.run_single_task(
        template_id="agent:tpl_echo", user_prompt=_content(),
    )

    task = captured["task"]
    assert task.description == content_to_text(_content())[:200]
    assert isinstance(task.description, str)
    assert task.user_prompt == _content(), "Task 承载全量内容"


# ── 漏传 event 侧载荷时不得静默降级（blob-store 解耦 Task 3 review Important 3）──
#
# 参数化穿线的通病：新参数默认 None，忘了传的调用方就静默把 `user_prompt: None` 发进
# SESSION_CREATED / SESSION_RESUMED，把 prompt 从重放流里抹掉。判据与 `push_task`
# 完全同形——纯文本回退用它自己，part 列表则响亮 raise。


class _CapturingBus:
    """记下每条 emit 的事件（SessionRegistry._emit 只用 bus.emit）。"""

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


def _session_registry(bus) -> SessionRegistry:
    from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
    from ctx_weft.core.registry import ProviderRegistry
    from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    reg = ProviderRegistry()
    reg.register_capability(templates)
    return SessionRegistry(
        agent_lifecycle_manager=AgentLifecycleManager(
            template_lookup=TemplateLookup(reg), event_bus=bus,
            model_resolver=lambda a, m: _FixedModelClient(
                MockLLMAdapter(responses=[]), "mdl_default", 200_000, 8192, account="acct_default"),
        ),
        event_bus=bus,
    )


async def test_create_session_without_jsonable_falls_back_to_the_text_prompt():
    """纯文本 prompt 漏传载荷 → 事件里仍是那段文本，不得变成 None。"""
    bus = _CapturingBus()
    sm = _session_registry(bus)

    await sm.create_session(
        template_id="agent:tpl_echo", user_prompt="你好", context_limit=1000,
    )

    created = next(e for e in bus.events if e.type == "SessionCreated")
    assert created.payload["user_prompt"] == "你好"
    task_created = next(e for e in bus.events if e.type == "TaskCreated")
    assert task_created.payload["task"]["user_prompt"] == "你好"


async def test_create_session_without_jsonable_raises_on_multimodal_prompt():
    """携图 prompt 漏传载荷 → 响亮 raise，绝不静默把 prompt 抹成 None。

    这里没有原始字节可用（`user_prompt` 可能已是 memory ref），唯一诚实的选择是拒绝。
    """
    bus = _CapturingBus()
    sm = _session_registry(bus)

    with pytest.raises(ValueError):
        await sm.create_session(
            template_id="agent:tpl_echo", user_prompt=_content(), context_limit=1000,
        )

    # 关键：拒绝必须发生在**发事件之前**。`push_task` 的守卫也会 raise，但那时
    # SESSION_CREATED 已经带着 `user_prompt: None` 落进事件流了——只断言 raise
    # 会把这个顺序缺陷放过去。
    assert not [e for e in bus.events if e.type == "SessionCreated"], (
        "校验失败不得先发出一条 user_prompt 被抹成 None 的 SessionCreated"
    )


async def test_resume_session_without_jsonable_falls_back_to_the_text_prompt():
    """resume 分支与 create 同一判据——两处都写事件，不能只加固一处。"""
    from unittest.mock import MagicMock

    from ctx_weft.protocols.events import Event
    from ctx_weft.core.utils.ids import generate_id
    from ctx_weft.providers.events import InMemoryEventStore

    bus = _CapturingBus()
    store = InMemoryEventStore()
    await append_one(store, Event(
        id=generate_id("evt"), run_id=None, sequence=1, session_id="ses-resume",
        type="SessionCreated", timestamp=now_utc(), tenant_id="default",
        payload={"template_id": "tpl", "user_prompt": "第一轮",
                 "root_agent_id": "agt_root", "context_limit": 1000},
    ))
    # resume_session 不调 instantiate，故 agent_lifecycle_manager 用桩即可（同
    # tests/unit/test_resume_context_limit.py 的既有做法）。
    sm = SessionRegistry(agent_lifecycle_manager=MagicMock(), event_bus=bus)

    await sm.resume_session(
        session_id="ses-resume", event_store=store, user_prompt="第二轮",
    )

    resumed = next(e for e in bus.events if e.type == "SessionResumed")
    assert resumed.payload["user_prompt"] == "第二轮"
