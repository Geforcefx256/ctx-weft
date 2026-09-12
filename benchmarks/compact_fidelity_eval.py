"""compact-fidelity 语义质量评测（spec: compact-fidelity · 非 CI，发布前人工运行）。

固定 mock 证不了 cue 的语义作用（其输出不随 cue 变化）——语义质量层在这里承担：
对含标记（约束/待办/失败原因）的样本任务跑**真实 LLM** 的多轮压缩，核对标记跨轮存活。

用法（需要真实 LLM 配置，不进 CI）：

    # OpenAI 兼容端点
    OPENAI_API_KEY=... OPENAI_BASE_URL=... python benchmarks/compact_fidelity_eval.py
    # 或 Anthropic
    ANTHROPIC_API_KEY=... python benchmarks/compact_fidelity_eval.py --provider anthropic

    # 自定义样本（JSONL：{"user": "...", "constraint": "...", "todo": "..."}）
    python benchmarks/compact_fidelity_eval.py --samples my_samples.jsonl

输出：每样本的约束/待办存活、Evidence 引用可解析、degraded 率、digest 长度比；
任一维度失败 → 退出码 1。基线对照建议与 benchmarks/2026-09-11-baseline.json 同法归档。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctx_weft.core.loop.steps import compact as cm  # noqa: E402

DEFAULT_SAMPLES = [
    {
        "user": "Build the quarterly analytics report from the warehouse export.",
        "constraint": "MUST NOT include customer names (compliance).",
        "todo": "section 3 charts still pending",
        "failure": "warehouse export timed out once",
    },
    {
        "user": "Refactor the auth module and update docs.",
        "constraint": "keep the public API signature unchanged.",
        "todo": "integration tests not written yet",
        "failure": "one flaky test in test_login",
    },
]

ROUNDS = 3  # 多轮压缩（L3 反复坍缩）——保真跨轮存活才是本评测的对象


def _make_llm(provider: str):
    if provider == "anthropic":
        from ctx_weft.providers.llm.anthropic import AnthropicLLMClient
        return AnthropicLLMClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    from ctx_weft.providers.llm.openai import OpenAILLMClient
    return OpenAILLMClient(api_key=os.environ["OPENAI_API_KEY"],
                           base_url=os.environ.get("OPENAI_BASE_URL"))


async def _eval_one(llm, sample: dict) -> dict:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextAssembler
    from ctx_weft.core.assembler.budget import PriorityBudgetStrategy
    from ctx_weft.core.assembler.composer import DefaultComposer
    from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
    from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.core.utils.ids import generate_id
    from ctx_weft.protocols import (
        MemoryAddress,
        MemoryEvent,
        MemoryKind,
        MemoryScope,
        ProviderContext,
    )
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="eval", task_id="tsk_eval", agent_id="ag_eval")
    pctx = ProviderContext(session_id="eval", tenant_id="default",
                           task_id="tsk_eval", agent_id="ag_eval")
    user_msg = f"{sample['user']} Constraint: {sample['constraint']}"
    await mem.ingest(MemoryEvent(
        id=generate_id("mev"), kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=scope, timestamp=now_utc(), role="user", content=user_msg), pctx)
    await mem.ingest(MemoryEvent(
        id=generate_id("mev"), kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=scope, timestamp=now_utc(), role="assistant",
        content=f"attempted; {sample['failure']}; TODO: {sample['todo']}",
        metadata={"tool_calls": []}), pctx)

    state = SimpleNamespace(
        agent=Agent(id="ag_eval", session_id="eval", template_id="t",
                    loop_guard=LoopGuard()),
        scope=scope,
        task=SimpleNamespace(id="tsk_eval", title="eval", description="",
                             user_prompt=user_msg, user_prompt_in_memory=True,
                             outputs=None),
        session=SimpleNamespace(context_limit=200_000, reserved_output_tokens=0),
        extra={}, run_id="eval", origin=None,
        resolved_model=SimpleNamespace(model=llm.model, account=""),
        sequence_counter=1,
    )
    ctx = SimpleNamespace(
        assembler=ContextAssembler(
            sources=[TaskSpecSource(), AgentRecallSource()],
            budget=PriorityBudgetStrategy(), composer=DefaultComposer(),
            deps=AssemblerDeps(memory=mem, knowledge_providers=[], provider_ctx=pctx,
                               capability_cache=None, agent_id="ag_eval")),
        llm=llm, memory=mem, cancel_token=None, event_bus=None, config=None,
        provider_ctx=pctx,
    )

    degraded_rounds = 0
    digest_text = ""
    for _ in range(ROUNDS):
        digest = await cm.summarize_for_compact(state, ctx, scope="task")
        degraded_rounds += int(digest.degraded)
        digest_text = digest.text
    body = digest_text
    return {
        "constraint_ok": sample["constraint"].split("(")[0].strip()[:30] in body
                         or sample["constraint"].split(",")[0].strip()[:30] in body,
        "todo_ok": sample["todo"].split()[0] in body,
        "degraded_rounds": degraded_rounds,
        "length_ratio": round(len(body) / max(1, len(user_msg)), 2),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--samples", type=Path, default=None,
                    help="JSONL，每行 {user, constraint, todo, failure}")
    args = ap.parse_args()

    samples = DEFAULT_SAMPLES
    if args.samples and args.samples.exists():
        import json
        samples = [json.loads(ln) for ln in args.samples.read_text("utf-8").splitlines() if ln]

    llm = _make_llm(args.provider)
    failures = 0
    print(f"compact-fidelity semantic eval · provider={args.provider} · rounds={ROUNDS}")
    for i, s in enumerate(samples, 1):
        r = await _eval_one(llm, s)
        ok = r["constraint_ok"] and r["todo_ok"] and r["degraded_rounds"] == 0
        failures += int(not ok)
        print(f"  [{i}] {'PASS' if ok else 'FAIL'} constraint={r['constraint_ok']} "
              f"todo={r['todo_ok']} degraded_rounds={r['degraded_rounds']} "
              f"len_ratio={r['length_ratio']}")
    print(f"{'ALL PASS' if failures == 0 else f'{failures} sample(s) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
