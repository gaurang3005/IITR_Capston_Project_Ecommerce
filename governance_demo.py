"""Task 15 transcript: least autonomy, risk class, runtime budget cap and how it was sized."""
import contextlib
import json
import warnings

import mock_llm
import rag_core
import review_stage
from agent_service import ask_agent
from crew_builder import build_crew
from governance import (EXPECTED_COMPLETION_TOKENS, PROMPT_OVERHEAD_TOKENS, QUESTION_COPIES,
                        RISK_LEVEL, BudgetExceeded, LeastAutonomyViolation, assert_least_autonomy,
                        enforce_budget, estimate_tokens, wire_tool)
from tools import OrderStatusTool
from cache import GenerationCache


def _text(messages) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(m.get("content") if isinstance(m, dict) else getattr(m, "content", ""))
                     for m in messages)


@contextlib.contextmanager
def metered(calls: list):
    """Measurement only: record (prompt, completion) of every model call of a request (crew LLM,
    RAG writer, both review agents), with an empty cache so the writer always runs. Restores the
    originals afterwards."""
    crew_call, review_create = mock_llm.MockReActLLM.call, review_stage.MockReviewClient.create
    writer, cache = rag_core.mock_extractive_writer, rag_core.CACHE

    def metered_crew(self, messages, *args, **kwargs):
        out = crew_call(self, messages, *args, **kwargs)
        calls.append((_text(messages), str(out)))
        return out

    async def metered_review(self, messages, **kwargs):
        result = await review_create(self, messages, **kwargs)
        schema = kwargs.get("json_output")          # ReviewVerdict: a real client sends it
        extra = json.dumps(schema.model_json_schema()) if isinstance(schema, type) else ""
        calls.append((_text(messages) + extra, str(result.content)))
        return result

    def metered_writer(query, contexts):          # a real writer adds its own instructions
        out = writer(query, contexts)
        calls.append((query + "\n" + "\n".join(c["text"] for c in contexts), out))
        return out

    mock_llm.MockReActLLM.call, review_stage.MockReviewClient.create = metered_crew, metered_review
    rag_core.mock_extractive_writer, rag_core.CACHE = metered_writer, GenerationCache()
    try:
        yield
    finally:
        mock_llm.MockReActLLM.call, review_stage.MockReviewClient.create = crew_call, review_create
        rag_core.mock_extractive_writer, rag_core.CACHE = writer, cache


crew = build_crew()
print("APPLICATION LAYER - least autonomy")
for a in crew.agents:
    print(f"   {a.role:<32} tools={[t.name for t in a.tools]} allow_delegation={a.allow_delegation}")
assert_least_autonomy(crew); print("   assert_least_autonomy(crew): PASS")
retrieval = crew.agents[0]
try:
    wire_tool(retrieval, OrderStatusTool())
except LeastAutonomyViolation as e:
    print("   wire_tool(retrieval_agent, check_order_status) -> BLOCKED:", e)
retrieval.tools.append(OrderStatusTool())          # bypass the helper on purpose ...
try:
    assert_least_autonomy(crew)
except LeastAutonomyViolation as e:
    print("   direct mutation then assert_least_autonomy -> BLOCKED:", e)

print(f"\nRISK CLASS: {RISK_LEVEL}")

print("\nRUNTIME LAYER - budget sizing: every model call of one request")
print("   calls prompt question x copies completion  total estimate  (est. tokens)")
rows = []
for i, q in enumerate(["How long is the return window for footwear?",
                       "What is the status of order ORD0029?",
                       "Can I return order ORD0021? It is footwear and the size is wrong.",
                       "Where is my order ORD9999?", "What is Nykaa's CEO's salary?"]):
    calls: list = []
    with metered(calls), warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the memory warning shows once, further down
        r = ask_agent(q, session_id=f"budget-size-{i}")
    asked = r["resolved_query"]
    prompt = sum(estimate_tokens(p) for p, _ in calls)
    completion = sum(estimate_tokens(c) for _, c in calls)
    copies = sum(p.count(asked) for p, _ in calls)
    estimate = enforce_budget(asked)["estimated_tokens"]
    rows.append((prompt - copies * estimate_tokens(asked), copies, completion,
                 prompt + completion, estimate))
    print(f"   {len(calls):>5} {prompt:>6} {estimate_tokens(asked):>8} x {copies:<6} "
          f"{completion:>10} {prompt + completion:>6} {estimate:>8}  {q}")
print(f"   largest: prompt besides the question {max(r[0] for r in rows)} -> "
      f"PROMPT_OVERHEAD_TOKENS={PROMPT_OVERHEAD_TOKENS}, copies {max(r[1] for r in rows)} -> "
      f"QUESTION_COPIES={QUESTION_COPIES}, completion {max(r[2] for r in rows)} -> "
      f"EXPECTED_COMPLETION_TOKENS={EXPECTED_COMPLETION_TOKENS}")
print("   estimate >= measured for every request:", all(r[4] >= r[3] for r in rows))

print("\nRUNTIME LAYER - per-request budget")
print("   normal   :", enforce_budget("How long is the return window for footwear?"))
try:
    enforce_budget("Please read this carefully. " * 400)
except BudgetExceeded as e:
    print("   oversized:", "REJECTED ->", e)
r = ask_agent("Please read this carefully. " * 400, session_id="budget-demo")
print("   via pipeline:", r["outcome"], "| crew was never called:", r["draft"] is None)
