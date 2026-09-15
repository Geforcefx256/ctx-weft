"""Task 11: 端到端 golden 测试（情况 1–8 + 边界）。

黄金测试风格：构造 task 层事件序列 → 调 finalize_task_memory（或直接 _synthesize_dispatch_pair）
→ 断言 agent 层重建出的 AGENT_CONVERSATION_TURN 序列（role/内容要点/tool_calls/配对）。

覆盖范围（spec §4 / §5）：
  A1  单段正常 finish（无段摘要，只 finish 对）
  A3  ② 流中打断（段摘要含「被打断」、半截 assistant/cancelled tool 不进胶囊）
  A4  ① 编辑式打断（两 user 锚点相邻，其间无摘要）
  A6  fail 收尾（result="(无最终产出)"，tool "[outcome=fail]"，配对不悬挂）
  A10 short task 不合成胶囊（负向——断言无 AGENT_CONVERSATION_TURN 写入）
  H3  递归嵌套（同 agent 孙任务）
  H4  跨 agent 隔离（parent 看不到子 agent 的段摘要/user 锚点）
  H8  短同 agent 子任务未配对隐去

已在其他 test 文件覆盖（不重复）：
  A2  一次 HITL → test_capsule_interleaved.py::test_interleaved_capsule_order_and_roles
  finish 对结构/配对 → test_capsule_interleaved.py + test_capsule_render.py
  fold → test_root_subtree_fold.py
  gc_subtree → test_gc_preserve_dispatch.py
  H1/H2 同 agent 嵌套 → test_subtask_nesting.py
  H4 部分（bubble 内容） → test_subtask_nesting.py
"""
from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair, finalize_task_memory
from ctx_weft.protocols.capability import qualify
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)

SESSION = "s1"
FINISH_TASK_NAME = qualify("control:finish_task")


# ─── 辅助函数 ──────────────────────────────────────────────────────────────────

def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _task_scope(task_id: str, agent_id: str = "ag1") -> MemoryAddress:
    """task 层 scope（含 task_id）。"""
    return MemoryAddress(session_id=SESSION, task_id=task_id, agent_id=agent_id)


def _agent_scope(agent_id: str = "ag1") -> MemoryAddress:
    """agent 层 scope（task_id=None）。"""
    return MemoryAddress(session_id=SESSION, task_id=None, agent_id=agent_id)


def _ev(type_: MemoryEventType, scope: MemoryAddress, content: str, t: int,
        role: str | None = None, **meta) -> MemoryEvent:
    return MemoryEvent(
        type=type_, address=scope, content=content,
        timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta,
    )


def _make_task(task_id: str = "t1", agent_id: str = "ag1", prompt: str = "初始请求",
               parent_task_id: str | None = None, creator_agent_id: str | None = None,
               outputs: str | None = "最终答复", title: str = "测试任务") -> Task:
    return Task(
        id=task_id, session_id=SESSION, status="FINISHED", tenant_id="default",
        assigned_agent_id=agent_id,
        creator_agent_id=creator_agent_id or agent_id,
        parent_task_id=parent_task_id,
        title=title, description="", user_prompt=prompt,
        settings=NormalTaskSettings(),
        outputs=outputs,
    )


async def _caps(mem: InMemoryMemoryProvider, scope: MemoryAddress) -> list:
    """召回 agent 层 AGENT_CONVERSATION_TURN，按 timestamp 升序（oldest first）。"""
    recs = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    return list(reversed(recs))


class _FakeTM:
    """最简 TaskManager stub：没有子任务。"""
    def __init__(self, children: dict | None = None):
        self._children = children or {}

    def children_of(self, task_id: str) -> set[str]:
        return self._children.get(task_id, set())


def _state(task: Task, scope: MemoryAddress, loop_config: LoopConfig | None = None):
    agent = SimpleNamespace(id=scope.agent_id, loop_config=loop_config or LoopConfig())
    session = SimpleNamespace(id=SESSION, tenant_id="default")
    return SimpleNamespace(
        run_id="run1", sequence_counter=0,
        session=session, scope=scope, task=task, agent=agent,
    )


def _loop_ctx(mem: InMemoryMemoryProvider, tm=None):
    return SimpleNamespace(
        memory=mem, provider_ctx=_pctx(),
        task_manager=tm or _FakeTM(),
        llm=SimpleNamespace(tokenizer=HeuristicTokenizer()),
    )


# ─── A1: 单段正常 finish（无段摘要，只 finish 对） ──────────────────────────────

async def test_A1_single_segment_finish() -> None:
    """情况 1（task-resident）: task 层原始 UP + LLM/TOOL 处理（无段摘要），close 后
    agent 层 = 仅 finish 对（assistant act_recap + finish_task() 标记 + tool Process Report，2 条），
    **不镜像 body**；task 层 body（UP/LLM/TOOL）原样留、未被 supersede。
    反转契约（spec 2026-07-01）：finish_task 无参标记（答复由内联 body / blackboard 承载）。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    # task 层：原始 UP + 若干 LLM/TOOL（无段摘要）
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "把 auth 从 session 改成 JWT，并补测试", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "正在改…", 2, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_RESULT, tsc, "tool ok", 3, role="tool"), _pctx())

    task = _make_task(outputs="已切到 JWT：三处改完，8 测试全过")
    mem_content = "已切到 JWT：三处改完，8 测试全过\n\nProcess Report: 成功。读 session.py 确认 3 处改 jwt，补 8 测试。"

    await _synthesize_dispatch_pair(mem, asc, task, mem_content, "", "success", _pctx())

    caps = await _caps(mem, asc)

    # task-resident：agent 层只有 finish 对（2 条），body 不镜像
    assert len(caps) == 2, f"expected exactly 2 capsule turns (finish pair only, body stays in task layer), got {len(caps)}"

    # 两条即 finish pair（assistant finish_task + tool Process Report）
    finish_asst = caps[-2]
    finish_tool = caps[-1]
    assert finish_asst.role == "assistant", f"finish pair[0] must be assistant, got {finish_asst.role}"
    assert finish_tool.role == "tool", f"finish pair[1] must be tool, got {finish_tool.role}"

    tcs = finish_asst.metadata.get("tool_calls", [])
    assert len(tcs) == 1 and tcs[0]["name"].endswith("finish_task"), (
        f"finish assistant must have finish_task tool_call, got tool_calls={tcs}"
    )
    # 反转契约：finish_task 无参标记（input={}）；答复由内联 body / blackboard 承载，不在 finish 对重复
    assert tcs[0]["input"] == {}, (
        f"finish_task must be a no-arg marker (input={{}}); got {tcs[0]['input']!r}"
    )
    assert "Process Report:" in finish_tool.content, (
        f"finish tool must contain 'Process Report:'; got {finish_tool.content!r}"
    )

    # tool_call_id 配对完整
    assert finish_tool.metadata.get("tool_call_id") == tcs[0]["id"], (
        "finish pair tool_call_id must match assistant tool_calls[0].id"
    )

    # all turns have origin_task_id
    assert all(c.metadata.get("origin_task_id") == "t1" for c in caps), (
        "all capsule turns must have origin_task_id=t1"
    )

    # task-resident: task 层 body 原样留（未被 supersede）
    body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx())
    assert len(body) == 3, f"task-layer body must stay (not superseded); got {len(body)}"


# ─── A3: ② 流中打断（段摘要含「被打断」，半截 assistant/cancelled tool 不进胶囊） ──

async def test_A3_mid_stream_interrupt_annotation() -> None:
    """情况 3（task-resident）: 段①含 LLM_RESPONSE(interrupted=True) + cancelled TOOL_RESULT；
    后台 observe 产 TASK_COMPACT_SUMMARY（含「被打断」字样，apply_compact 折叠后幸存、interrupted
    raw 被 supersede）；打断后用户新指令为下一 USER_PROMPT 锚点。

    task-resident：close 不再镜像 body——agent 层只有 finish 对（2 条）；body（段摘要 + 两 user
    锚点、无 interrupted raw）留 task 层，召回时与 finish 对按时间戳归并。断言：
    - agent 层 = 仅 finish 对（assistant finish_task + tool Process Report）
    - task 层 body：段①摘要存在含「被打断」、打断后 UP 锚点存在、无裸 interrupted/cancelled raw
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    # 段①：interrupted assistant + cancelled tool（模拟流中打断）
    await mem.ingest(_ev(
        T.LLM_RESPONSE, tsc, "正用 PyJWT 写 login() 签发，写到一半…", 1,
        role="assistant", interrupted=True,
    ), _pctx())
    await mem.ingest(_ev(
        T.TOOL_RESULT, tsc, "[INTERRUPTED]", 2,
        role="tool", cancelled=True, status="INTERRUPTED",
    ), _pctx())

    # 模拟后台 observe：apply_compact 折叠 interrupted 段 → 产 TASK_COMPACT_SUMMARY（含被打断标注）
    # 并 supersede interrupted LLM/TOOL
    all_task_recs = await mem.recall_recent(tsc, [T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx())
    ids_to_supersede = [r.id for r in all_task_recs]
    summary_event = MemoryEvent(
        type=T.TASK_COMPACT_SUMMARY,
        address=tsc,
        content="〔段①·被用户打断〕正用 PyJWT 写 login() 签发，写到一半被打断。",
        timestamp=_BASE + timedelta(seconds=2),
        role="assistant",  # apply_compact 存 role=assistant
        metadata={},
    )
    await mem.ingest(summary_event, _pctx())
    if ids_to_supersede:
        await mem.supersede(ids_to_supersede, _pctx())

    # 原始 UP（打断前）
    up1 = MemoryEvent(
        type=T.USER_PROMPT, address=tsc, content="把 auth 从 session 改成 JWT，并补测试",
        timestamp=_BASE + timedelta(seconds=0), role="user", metadata={},
    )
    await mem.ingest(up1, _pctx())

    # 打断后新指令 UP
    up2 = MemoryEvent(
        type=T.USER_PROMPT, address=tsc, content="别用 PyJWT，用 authlib",
        timestamp=_BASE + timedelta(seconds=3), role="user", metadata={},
    )
    await mem.ingest(up2, _pctx())

    task = _make_task(prompt="把 auth 从 session 改成 JWT，并补测试",
                      outputs="改用 authlib 重写签发/校验，3 处改完")
    mem_content = "改用 authlib 重写签发/校验，3 处改完\n\nProcess Report: 成功。改用 authlib 重写签发/校验，3 处改完。"

    await _synthesize_dispatch_pair(mem, asc, task, mem_content, "", "success", _pctx())

    caps = await _caps(mem, asc)

    # task-resident：agent 层只有 finish 对（2 条）
    assert len(caps) == 2, f"expected only finish pair in agent layer; got {[(c.role, c.content[:40]) for c in caps]}"
    assert caps[-2].role == "assistant"
    assert caps[-1].role == "tool"
    assert "Process Report:" in caps[-1].content
    tcs = caps[-2].metadata.get("tool_calls", [])
    assert len(tcs) == 1 and tcs[0]["name"].endswith("finish_task")

    # body 留 task 层：召回 task 层确认段摘要 + 两 user 锚点幸存、无裸 interrupted raw
    body = await mem.recall_recent(
        tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY, T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx(),
    )

    # 段①摘要（role=assistant，含「被打断」）留 task 层
    summary_turns = [c for c in body if c.role == "assistant" and "被打断" in (c.content or "")]
    assert len(summary_turns) >= 1, (
        f"segment-summary with '被打断' must stay in task layer; got {[(c.role, (c.content or '')[:60]) for c in body]}"
    )

    # 无裸露 interrupted LLM / cancelled TOOL（已被 supersede）
    interrupted_raw = [
        c for c in body
        if c.metadata.get("interrupted") or c.metadata.get("cancelled") or
           c.metadata.get("status") == "INTERRUPTED"
    ]
    assert not interrupted_raw, (
        f"interrupted/cancelled raw must be superseded; found {interrupted_raw}"
    )

    # 两 user 锚点留 task 层（原始 + 打断后）
    user_turns = [c for c in body if c.role == "user"]
    assert any("JWT" in (c.content or "") or "session" in (c.content or "") for c in user_turns)
    assert any("authlib" in (c.content or "") for c in user_turns), (
        "post-interrupt USER_PROMPT must stay in task layer"
    )


# ─── A4: ① 编辑式打断（两 user 锚点相邻，其间无段摘要） ──────────────────────

async def test_A4_edit_interrupt_adjacent_anchors() -> None:
    """情况 4（task-resident）: task 层 = [USER_PROMPT 原始][USER_PROMPT edit-note][处理]。
    close 后 agent 层 = 仅 finish 对；body（两 user 锚点相邻、其间无段摘要）留 task 层。

    spec §3.7: interrupt_edit_note 产出一条 USER_PROMPT（「上一条…取消…改为」），
    原始 UP 仍在；二者之间无处理段（打断在出 token 前）。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    # 原始 UP（t=0）
    await mem.ingest(_ev(
        T.USER_PROMPT, tsc, "把 auth 模块从 session 改成 JWT", 0, role="user",
    ), _pctx())

    # edit-note UP（t=1）——模拟 interrupt_edit_note 产出的 USER_PROMPT
    await mem.ingest(_ev(
        T.USER_PROMPT, tsc,
        "(上一条已取消) 改为：把 auth 改成 OAuth2，不要 JWT",
        1, role="user",
    ), _pctx())

    # 后续处理（edit 后的处理段——此处无段摘要，直接 finish）
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "正在实现 OAuth2…", 2, role="assistant"), _pctx())

    task = _make_task(prompt="把 auth 模块从 session 改成 JWT",
                      outputs="按 OAuth2 实现完成")
    mem_content = "按 OAuth2 实现完成\n\nProcess Report: 成功。按 OAuth2 实现…"

    await _synthesize_dispatch_pair(mem, asc, task, mem_content, "", "success", _pctx())

    caps = await _caps(mem, asc)

    # task-resident：agent 层仅 finish 对（2 条）
    assert len(caps) == 2, f"expected only finish pair in agent layer; got {[(c.role, c.content[:40]) for c in caps]}"
    assert caps[-2].role == "assistant"
    assert caps[-1].role == "tool"
    assert "Process Report:" in caps[-1].content

    # body 留 task 层：两条 user 锚点相邻保留（原始在前 edit-note 在后），其间无段摘要 assistant
    body = list(reversed(await mem.recall_recent(
        tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY, T.LLM_RESPONSE], 500, _pctx(),
    )))  # newest-first → 时间序（oldest first）
    user_turns = [c for c in body if c.role == "user"]
    assert len(user_turns) >= 2, (
        f"expected at least 2 user anchor turns (original + edit-note) in task layer; got {len(user_turns)}"
    )

    original = user_turns[0]
    edit_note = user_turns[1]

    assert "JWT" in original.content or "session" in original.content, (
        f"first user turn should be original prompt; got {original.content!r}"
    )
    assert "取消" in edit_note.content or "改为" in edit_note.content, (
        f"second user turn should be edit-note; got {edit_note.content!r}"
    )
    assert "OAuth2" in edit_note.content, (
        f"edit-note should mention new direction (OAuth2); got {edit_note.content!r}"
    )

    # 两条 user 锚点之间无段摘要 assistant（task 层 body 中相邻）
    orig_idx = body.index(original)
    edit_idx = body.index(edit_note)
    between = body[orig_idx + 1: edit_idx]
    assistant_between = [c for c in between if c.role == "assistant"]
    assert not assistant_between, (
        f"no assistant segment-summary between the two adjacent user anchors (edit-interrupt); "
        f"found {[(c.role, c.content[:40]) for c in assistant_between]}"
    )


# ─── A6: fail 收尾（result 占位，tool "[outcome=fail]"，配对完整） ───────────────

async def test_A6_fail_outcome() -> None:
    """情况 6: outcome=fail，task.outputs 为空。
    断言 finish 对 = [assistant act_recap + finish_task()][tool "[outcome=fail] Process Report: …"]。
    反转契约：finish_task 无参标记；tool_call 配对不悬挂。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    await mem.ingest(_ev(
        T.USER_PROMPT, tsc, "把 auth 从 session 改成 JWT，并补测试", 0, role="user",
    ), _pctx())

    task = _make_task(outputs=None)  # fail → no outputs
    mem_content = "Process Report: 失败报告"  # 无 outputs，仅 process report

    await _synthesize_dispatch_pair(mem, asc, task, mem_content, "", "fail", _pctx())

    caps = await _caps(mem, asc)

    # task-resident：agent 层仅 finish 对（2 条）
    assert len(caps) == 2, f"expected only finish pair in agent layer; got {[(c.role, c.content[:40]) for c in caps]}"

    finish_asst = caps[-2]
    finish_tool = caps[-1]

    # finish pair roles
    assert finish_asst.role == "assistant", f"finish pair[0] must be assistant, got {finish_asst.role}"
    assert finish_tool.role == "tool", f"finish pair[1] must be tool, got {finish_tool.role}"

    # 反转契约：finish_task 无参标记（答复不在 finish 对；tool 槽承载 [outcome=fail] 过程报告）
    tcs = finish_asst.metadata.get("tool_calls", [])
    assert len(tcs) == 1, f"finish assistant must have exactly 1 tool_call; got {tcs}"
    assert tcs[0]["name"].endswith("finish_task"), f"tool name must end with finish_task; got {tcs[0]['name']}"
    assert tcs[0]["input"] == {}, (
        f"finish_task must be a no-arg marker (input={{}}); got {tcs[0]['input']!r}"
    )

    # tool content: "[outcome=fail] Process Report: ..."
    assert "[outcome=fail]" in finish_tool.content, (
        f"fail tool content must have '[outcome=fail]' prefix; got {finish_tool.content!r}"
    )
    assert "Process Report:" in finish_tool.content, (
        f"fail tool content must contain 'Process Report:'; got {finish_tool.content!r}"
    )

    # 配对完整（tool_call_id 匹配）
    assert finish_tool.metadata.get("tool_call_id") == tcs[0]["id"], (
        "fail finish pair: tool_call_id must match assistant tool_calls[0].id"
    )

    # origin_task_id 一致
    assert all(c.metadata.get("origin_task_id") == "t1" for c in caps)


# ─── A10: short task 不合成胶囊（负向） ────────────────────────────────────────

async def test_A10_short_task_synthesizes_finish_pair_keeps_body() -> None:
    """A10（task-resident）：短叶子 task finish 后也无条件合成 finish 对（取消 short 延迟），
    agent 层 = 仅 finish 对（2 条）；task 层 raw body 原样留（不被 supersede）。

    判定 short: LLM_RESPONSE turns ≤ turn_cap AND token_est ≤ threshold。
    LoopConfig 默认 short_task_turn_cap=2, short_task_token_threshold=1000。
    此处构造极短对话（is_short=True）——但 short 本 task 后不再 gate 合成/supersede。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    # 极短：1 轮 LLM（< turn_cap），极小 token
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "hello", 0, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "hi", 1, role="assistant"), _pctx())

    task = _make_task(outputs="hi", prompt="hello")
    # task 无 parent → is_own_root=True；task-resident 下无条件合成 finish 对

    loop_config = LoopConfig()  # 默认 short_task_turn_cap=2, short_task_token_threshold=1000

    await finalize_task_memory(
        mem, _state(task, tsc, loop_config),
        task, "hi\n\nProcess Report: 极短任务", "success", _loop_ctx(mem),
        act_recap="hi\n\nProcess Report: 极短任务", task_summary="",
    )

    # 断言：agent 层有 finish 对（2 条：assistant finish_task + tool Process Report）
    agent_turns = await _caps(mem, asc)
    assert len(agent_turns) == 2, (
        f"short task must synthesize finish pair (2 turns); "
        f"found {[(r.role, r.content[:40]) for r in agent_turns]}"
    )
    assert agent_turns[-2].role == "assistant"
    assert agent_turns[-1].role == "tool"
    tcs = agent_turns[-2].metadata.get("tool_calls", [])
    assert len(tcs) == 1 and tcs[0]["name"].endswith("finish_task")
    assert "Process Report:" in agent_turns[-1].content

    # task 层 raw body 原样留（task-resident：body 即胶囊，不 supersede）
    task_recs = await mem.recall_recent(tsc, [T.USER_PROMPT, T.LLM_RESPONSE], 500, _pctx())
    assert len(task_recs) == 2, (
        f"short task: task-layer body must stay (not superseded); got {len(task_recs)}"
    )


# ─── H3: 递归嵌套（同 agent 孙任务） ──────────────────────────────────────────

async def test_H3_recursive_nesting_grandchild() -> None:
    """H3: 同 agent 子任务再委派同 agent 孙任务（recursion depth=2）。
    断言孙胶囊平铺嵌在父 agent scope，各层 origin_task_id 归属正确，深度 ≤ 2 层均可见。

    场景：root(ag1) → child(ag1) → grandchild(ag1)
    grandchild close → 胶囊落 parent(child) scope = ag1 scope（同 agent）
                       且 bubble_to parent(child) = "Sub-task '孙子任务' scheduled."
    child close → 胶囊（含 grandchild 子胶囊）落 parent(root) scope = ag1 scope
    root 也在同 ag1 scope，所有胶囊都在 ag1 agent scope，靠 origin_task_id 区分。
    """
    mem = InMemoryMemoryProvider()

    # ── Grandchild（孙）──
    gc_id = "gc1"
    gc_parent_id = "child1"
    gc_scope = _task_scope(gc_id, "ag1")

    # 给孙任务写足够多 token 以脱离 short 判定
    await mem.ingest(_ev(T.USER_PROMPT, gc_scope, "孙子任务：收集数据", 1, role="user"), _pctx())
    for i in range(3):
        await mem.ingest(_ev(T.LLM_RESPONSE, gc_scope, "x " * 4000, i + 2, role="assistant"), _pctx())

    gc_task = _make_task(
        task_id=gc_id, agent_id="ag1",
        creator_agent_id="ag1", parent_task_id=gc_parent_id,
        prompt="孙子任务：收集数据", outputs="孙子收集完毕",
        title="孙子任务",
    )
    gc_task.origin_tool_call_id = "oc_gc"

    gc_mem_content = "孙子收集完毕\n\nProcess Report: 孙任务成功"
    await finalize_task_memory(
        mem, _state(gc_task, gc_scope),
        gc_task, gc_mem_content, "success", _loop_ctx(mem),
        act_recap=gc_mem_content, task_summary="",
    )

    # ── Child（子）──
    child_id = "child1"
    child_parent_id = "root1"
    child_scope = _task_scope(child_id, "ag1")

    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "子任务：研究 X", 10, role="user"), _pctx())
    for i in range(3):
        await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "y " * 4000, 11 + i, role="assistant"), _pctx())

    child_task = _make_task(
        task_id=child_id, agent_id="ag1",
        creator_agent_id="ag1", parent_task_id=child_parent_id,
        prompt="子任务：研究 X", outputs="子任务完成",
        title="子任务",
    )
    child_task.origin_tool_call_id = "oc_child"
    # child 知道自己有孙任务
    child_tm = _FakeTM({child_id: {gc_id}})

    child_mem_content = "子任务完成\n\nProcess Report: 子任务成功"
    await finalize_task_memory(
        mem, _state(child_task, child_scope),
        child_task, child_mem_content, "success", _loop_ctx(mem, child_tm),
        act_recap=child_mem_content, task_summary="",
    )

    # task-resident：finish 对都落 ag1 agent 层（AGENT_CONVERSATION_TURN 按 agent_id 召回、
    # 忽略 task_id），靠 origin_task_id 区分各层；body 留各自 task 层、不被镜像/supersede。
    ag1_scope = _agent_scope("ag1")
    all_caps = await mem.recall_recent(ag1_scope, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    caps_by_origin = {r.metadata.get("origin_task_id") for r in all_caps}

    # child / grandchild 两层 finish 对均在 ag1 agent 层（origin=child1 / gc1）
    assert child_id in caps_by_origin and gc_id in caps_by_origin, (
        f"both child and grandchild finish pairs expected in ag1 agent layer; got {caps_by_origin}"
    )

    # 每层 finish 对各 2 条（assistant finish_task + tool Process Report），无 body 镜像
    # §2.5: same-agent dispatch acks（配对 start_task 框的 tool 回合）排除后仅剩 finish 对
    # Task 2: also exclude minted start_task frames (dispatch infrastructure, not finish pairs)
    from ctx_weft.core.loop.steps.finalize import START_TASK_NAME
    start_task_tcids = {
        tc.get("id")
        for r in all_caps if r.role == "assistant"
        for tc in (r.metadata.get("tool_calls") or [])
        if tc.get("name") == START_TASK_NAME
    }
    for origin in (child_id, gc_id):
        layer_caps = [r for r in all_caps if r.metadata.get("origin_task_id") == origin]
        finish_pair = [r for r in layer_caps
                       if not (r.role == "tool" and r.metadata.get("tool_call_id") in start_task_tcids)
                       and not (r.role == "assistant"
                                and any(tc.get("name") == START_TASK_NAME
                                        for tc in (r.metadata.get("tool_calls") or [])))]
        # 有最终产出的 task → 三槽（recap / 答复+finish_task / 报告），见
        # finalize.build_finish_slots；无产出才退回两槽。
        assert len(finish_pair) == 3, (
            f"task-resident: each task must contribute exactly the finish slots (3 turns); "
            f"origin={origin} got {[(c.role, (c.content or '')[:40]) for c in finish_pair]}"
        )
        ordered = sorted(finish_pair, key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0)))
        assert [c.role for c in ordered] == ["assistant", "assistant", "tool"]
        finish_pair = ordered
        assert finish_pair[1].metadata.get("final_reply") is True, "答复槽在中间"
        assert any(tc.get("name") == FINISH_TASK_NAME
                   for tc in (finish_pair[1].metadata.get("tool_calls") or [])), (
            "finish_task 挂在答复那条")
        tool_turn = [c for c in finish_pair if c.role == "tool"][0]
        assert "Process Report:" in tool_turn.content

    # body 留各自 task 层：child / grandchild 的 user 锚点仍在各自 task 层（未被镜像/supersede）
    gc_body = await mem.recall_recent(gc_scope, [T.USER_PROMPT, T.LLM_RESPONSE], 500, _pctx())
    gc_user = [r for r in gc_body if r.role == "user"]
    assert gc_user and "孙子任务" in gc_user[0].content, (
        f"grandchild raw body (user anchor) must stay in its own task layer; got {gc_body!r}"
    )
    child_body = await mem.recall_recent(child_scope, [T.USER_PROMPT, T.LLM_RESPONSE], 500, _pctx())
    assert any(r.role == "user" and "子任务" in r.content for r in child_body), (
        "child raw body (user anchor) must stay in its own task layer"
    )


# ─── H4: 跨 agent 隔离（parent 看不到子 agent 段摘要/user 锚点） ────────────────

async def test_H4_cross_agent_isolation() -> None:
    """情况 8 / H4: root 委派跨 agent 子任务（use_subagent=True，creator≠assigned）。

    (a) parent scope 的 dispatch result = mem_content tool conversation turn（含 Process Report:），
        origin=delegating task(p1)（§2.3：dispatch 对 = 普通 conversation message）
    (b) parent agent scope 无 origin_task_id=child.id 的 AGENT_CONVERSATION_TURN
        （子 agent finish 对在 ag2 层，按 agent_id 对 parent 不可见）
    (c) 子 agent scope（ag2）有自己的 finish 对（task-resident：仅 finish 对，body 留子 task 层）

    这验证 spec §3.9 跨 agent = 黑盒。
    """
    mem = InMemoryMemoryProvider()

    child_id = "c_cross"
    child_agent = "ag2"
    parent_id = "p1"
    parent_agent = "ag1"

    child_scope = _task_scope(child_id, child_agent)

    # 子 agent 的 task 层对话（子 agent 内部，parent 不可见）
    # 须有足够内容令 is_short_leaf=False（LLM_RESPONSE turns > short_task_turn_cap or token > threshold）
    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "收集数据：2020-2024 X 领域", 1, role="user"), _pctx())
    for i in range(3):
        await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "z " * 4000, i + 2, role="assistant"), _pctx())
    # 模拟后台 observe 已折段摘要（TASK_COMPACT_SUMMARY），LLM_RESPONSE 被 supersede
    # 这里为简化直接写摘要+留 LLM_RESPONSE（不 supersede，保持对话可 snapshot）
    await mem.ingest(_ev(
        T.TASK_COMPACT_SUMMARY, child_scope,
        "〔子段①〕检索 5 源、下载并清洗…", 5, role="assistant",  # apply_compact 存 role=assistant
    ), _pctx())

    child_task = _make_task(
        task_id=child_id, agent_id=child_agent,
        creator_agent_id=parent_agent,  # cross-agent: creator ≠ assigned
        parent_task_id=parent_id,
        prompt="收集数据：2020-2024 X 领域", outputs="收集到 5 份数据集",
        title="收集数据",
    )
    child_task.origin_tool_call_id = "oc_cross"

    child_mem_content = "收集到 5 份数据集，已清洗。\n\nProcess Report: 成功，覆盖 5 年。"

    await finalize_task_memory(
        mem, _state(child_task, child_scope),
        child_task, child_mem_content, "success", _loop_ctx(mem),
        act_recap=child_mem_content, task_summary="",
    )

    parent_scope_task = _task_scope(parent_id, parent_agent)
    parent_scope_agent = _agent_scope(parent_agent)
    child_scope_agent = _agent_scope(child_agent)

    # (a) parent dispatch result = mem_content tool conversation turn（配对 oc_cross、归 p1 单元）
    parent_results = await mem.recall_recent(
        parent_scope_agent, [T.AGENT_CONVERSATION_TURN], 200, _pctx(),
    )
    cross_results = [r for r in parent_results
                     if r.role == "tool" and r.metadata.get("tool_call_id") == "oc_cross"]
    assert cross_results, "cross-agent child must bubble dispatch result (conversation turn) to parent"
    assert "Process Report:" in cross_results[0].content, (
        f"cross-agent bubble content must be mem_content (with Process Report:); "
        f"got {cross_results[0].content!r}"
    )
    assert cross_results[0].content == child_mem_content, (
        f"cross-agent bubble must equal full mem_content; got {cross_results[0].content!r}"
    )
    assert cross_results[0].metadata.get("origin_task_id") == parent_id, (
        "dispatch result 须归 delegating task(p1) 单元（与 finish 对同 origin、同命运）"
    )
    # 不再写 legacy TASK_DISPATCH_RESULT enum
    assert await mem.recall_recent(parent_scope_task, [T.TASK_DISPATCH_RESULT], 200, _pctx()) == []

    # (b) parent agent scope 无 origin=child.id 的 AGENT_CONVERSATION_TURN（隔离）
    parent_agent_turns = await mem.recall_recent(
        parent_scope_agent, [T.AGENT_CONVERSATION_TURN], 200, _pctx(),
    )
    child_in_parent = [r for r in parent_agent_turns if r.metadata.get("origin_task_id") == child_id]
    assert not child_in_parent, (
        f"cross-agent child capsule must NOT appear in parent agent scope; "
        f"found {[(r.role, r.content[:40]) for r in child_in_parent]}"
    )

    # (c) 子 agent scope（ag2）有自己的 finish 对（cross_agent = is_own_root → _synthesize_dispatch_pair）
    child_agent_turns = await mem.recall_recent(
        child_scope_agent, [T.AGENT_CONVERSATION_TURN], 200, _pctx(),
    )
    child_turns_by_origin = [r for r in child_agent_turns if r.metadata.get("origin_task_id") == child_id]
    assert len(child_turns_by_origin) == 3, (
        f"cross-agent child must have its own finish slots (3 turns) in child agent scope (ag2); "
        f"found {[(r.role, r.content[:40]) for r in child_agent_turns]}"
    )

    # finish 槽位（recap / 答复+finish_task / tool Process Report）
    child_turns_by_origin = sorted(child_turns_by_origin,
                                   key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0)))
    assert [r.role for r in child_turns_by_origin] == ["assistant", "assistant", "tool"]
    child_finish_tool = [r for r in child_turns_by_origin if r.role == "tool"]
    assert child_finish_tool and "Process Report:" in child_finish_tool[0].content

    # task-resident：子 agent finish 对**不含 body 镜像**（无 user 锚点/段摘要）
    assert not any(r.role == "user" for r in child_turns_by_origin), (
        "cross-agent finish pair must NOT mirror child body (no user anchor)"
    )

    # 子 raw body（user 锚点 + 段摘要）留子 task 层（ag2 的 c_cross scope），不进 agent finish 对
    child_body = await mem.recall_recent(
        child_scope, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY, T.LLM_RESPONSE], 200, _pctx(),
    )
    assert any(r.role == "user" and "收集数据" in (r.content or "") for r in child_body), (
        "child raw body (user anchor) must stay in child task layer"
    )

    # (c) 隔离验证：parent（ag1）召回看不到子 agent 内部段摘要 / finish 对（按 agent_id 隔离）
    child_summary_in_parent = [
        r for r in parent_agent_turns
        if r.role == "assistant" and "检索" in (r.content or "")
    ]
    assert not child_summary_in_parent, (
        f"parent must NOT see child agent's segment summary; "
        f"found {[(r.role, r.content[:40]) for r in child_summary_in_parent]}"
    )


# ─── H8: 短同 agent 子任务未配对隐去 ──────────────────────────────────────────

async def test_H8_short_same_agent_child_keeps_delegate_and_writes_ack() -> None:
    """H8（§2.5, 2026-07-03）：短同 agent 子任务 close → finalize 铸派发框 + 配对静态 ack，
    框与 ack 同锚 task.started_at；框名取真名 delegate_task；child 合成 finish 对；child raw body 留 child 层。

    场景：无 eager 框（delegate_task 不再 eager 写），短 child close。
    """
    from ctx_weft.core.capabilities.control_tools import DELEGATE_TASK_NAME
    mem = InMemoryMemoryProvider()

    parent_agent = "ag1"
    parent_id = "p1"
    child_id = "c_short"
    parent_agent_scope = _task_scope(parent_id, parent_agent)
    tc_short = "oc_short"

    # 短 child：极短对话（确保 is_short=True）
    child_scope = _task_scope(child_id, parent_agent)
    await mem.ingest(_ev(T.USER_PROMPT, child_scope, "短子任务", 5, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, child_scope, "ok", 6, role="assistant"), _pctx())

    child_task = _make_task(
        task_id=child_id, agent_id=parent_agent,
        creator_agent_id=parent_agent, parent_task_id=parent_id,
        prompt="短子任务", outputs="ok", title="短子任务",
    )
    child_task.origin_tool_call_id = tc_short
    child_task.origin_tool_name = DELEGATE_TASK_NAME  # 真名 → 框铸为 delegate_task
    started = _BASE + timedelta(seconds=1)  # started_at 早于 child body（更贴近真实顺序）
    child_task.started_at = started
    # task-resident：same-agent child → do_bubble=True（无条件，不再看 short）

    await finalize_task_memory(
        mem, _state(child_task, child_scope),
        child_task, "ok\n\nProcess Report: 短任务", "success", _loop_ctx(mem),
        act_recap="短任务", task_summary="",
    )

    # §2.5(2026-07-03)：铸派发框（真名 delegate_task）+ 配对静态 ack，二者同锚 started_at
    from ctx_weft.core.loop.steps.finalize import _dispatch_ack
    parent_caps = await mem.recall_recent(parent_agent_scope, [T.AGENT_CONVERSATION_TURN], 200, _pctx())
    frame = [r for r in parent_caps if r.role == "assistant"
             and any(tc.get("id") == tc_short and tc.get("name") == DELEGATE_TASK_NAME
                     for tc in (r.metadata.get("tool_calls") or []))]
    assert frame, "finalize 须铸派发框（真名 delegate_task）"
    ack = [r for r in parent_caps if r.role == "tool" and r.metadata.get("tool_call_id") == tc_short]
    assert ack and ack[0].content == _dispatch_ack(task_ref(child_task), "success"), (
        f"§2.5: static ack must be {_dispatch_ack(task_ref(child_task), 'success')!r}; got {[r.content for r in ack]}"
    )
    assert frame[0].timestamp == started == ack[0].timestamp, (
        f"框与 ack 须同锚 started_at；frame={frame[0].timestamp} ack={ack[0].timestamp} started={started}"
    )

    # child 自己合成 finish 对（同 agent scope）
    child_caps = parent_caps
    child_finish = [r for r in child_caps if r.metadata.get("origin_task_id") == child_id]
    assert child_finish, "short same-agent child must synthesize its own finish pair"

    # child raw body 留 child task 层（task-resident：不被 supersede）
    child_body = await mem.recall_recent(child_scope, [T.USER_PROMPT, T.LLM_RESPONSE], 200, _pctx())
    assert len(child_body) == 2, (
        f"short child raw body must stay in its task layer; got {len(child_body)}"
    )
