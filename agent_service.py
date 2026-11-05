"""The single request pipeline. Every end-to-end demo and every entry point calls ask_agent(),
so the transcripts show exactly what the deployed service does."""
from __future__ import annotations

import time
import uuid
from typing import Any

import config
from crew_builder import run_crew
from governance import BudgetExceeded, enforce_budget
from guardrails import contains_pii, guard_request, output_guard
from logging_config import log_request
from memory_chat import build_chat, get_session_history, resolve_followup
from review_stage import review_draft


def _answer(masked_text: str, history: list, state: dict) -> str:
    """Runs inside RunnableWithMessageHistory, so `history` is this session's prior turns."""
    resolved = resolve_followup(masked_text, history)
    state["resolved_query"] = resolved
    history_text = " ".join(m.content for m in history)
    state["budget"] = enforce_budget(resolved, history_text)   # raises before any LLM or tool call

    response, evidence = run_crew(resolved)
    state["draft"] = response.answer
    og = output_guard(response.answer, evidence)
    state["output_guard"] = og
    if not og["grounded"]:
        state.update(grounded=False, escalated=True, sources=[])
        return config.SENTINEL_NO_ANSWER

    verdict, trace = review_draft(response.answer, evidence)
    final = verdict.final_answer
    if contains_pii(final):                                    # output-side PII lane
        final = config.SENTINEL_NO_ANSWER
    state.update(grounded=response.grounded, escalated=response.escalated,
                 sources=response.sources, review=verdict.model_dump(), review_trace=trace)
    return final


def ask_agent(user_text: str, session_id: str = "default", trace_id: str | None = None,
              channel: str = "cli") -> dict[str, Any]:
    trace_id = trace_id or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    guard = guard_request(user_text)
    state: dict[str, Any] = {}
    outcome, error = "answered", None
    try:
        if guard["blocked"]:
            outcome, answer = "blocked_injection", config.BLOCKED_ANSWER
        else:
            chat = build_chat(lambda text, hist: _answer(text, hist, state))
            # Only the masked text is passed on; the raw input stops here.
            answer = chat.invoke({"input": guard["text"]},
                                 config={"configurable": {"session_id": session_id}}).content
            if not state.get("grounded", True) or answer == config.SENTINEL_NO_ANSWER:
                outcome = "refused_ungrounded"
    except BudgetExceeded as exc:
        outcome, answer = "rejected_budget", f"Request rejected: {exc}"
        state["budget_error"] = str(exc)
    except Exception as exc:                 # still write the one log line, then re-raise below
        outcome, answer, error = f"error:{type(exc).__name__}", "", exc

    result = {
        "trace_id": trace_id, "session_id": session_id, "answer": answer, "outcome": outcome,
        "blocked": outcome == "blocked_injection", "pii_masked": guard["pii_masked"],
        "grounded": state.get("grounded", False), "escalated": state.get("escalated", False),
        "sources": state.get("sources", []), "review": state.get("review"),
        "draft": state.get("draft"), "resolved_query": state.get("resolved_query"),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    log_request(trace_id=trace_id, channel=channel, session_id=session_id,
                query_masked=guard["text"], pii_counts=guard["pii_counts"],
                injection=guard["injection"], resolved_query=state.get("resolved_query"),
                budget=state.get("budget") or state.get("budget_error"),
                sources=result["sources"], output_guard=state.get("output_guard"),
                review_approved=(state.get("review") or {}).get("approved"),
                review_trace=state.get("review_trace"),
                outcome=outcome, latency_ms=result["latency_ms"])
    if error is not None:
        raise error
    return result


def history_of(session_id: str) -> list[str]:
    return [f"{m.type}: {m.content[:80]}" for m in get_session_history(session_id).messages]