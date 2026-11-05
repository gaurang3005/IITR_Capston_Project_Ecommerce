"""The single request pipeline. Every end-to-end demo and every entry point calls ask_agent(),
so the transcripts show exactly what the deployed service does."""
from __future__ import annotations

from typing import Any

import config
from crew_builder import run_crew
from guardrails import contains_pii, guard_request, output_guard
from memory_chat import build_chat, get_session_history, resolve_followup


def _answer(masked_text: str, history: list, state: dict) -> str:
    """Runs inside RunnableWithMessageHistory, so `history` is this session's prior turns."""
    resolved = resolve_followup(masked_text, history)
    state["resolved_query"] = resolved

    response, evidence = run_crew(resolved)
    og = output_guard(response.answer, evidence)
    if not og["grounded"]:
        state.update(grounded=False, escalated=True, sources=[])
        return config.SENTINEL_NO_ANSWER

    final = response.answer
    if contains_pii(final):                                    # output-side PII lane
        final = config.SENTINEL_NO_ANSWER
    state.update(grounded=response.grounded, escalated=response.escalated,
                 sources=response.sources)
    return final


def ask_agent(user_text: str, session_id: str = "default") -> dict[str, Any]:
    guard = guard_request(user_text)
    state: dict[str, Any] = {}
    outcome = "answered"
    if guard["blocked"]:
        outcome, answer = "blocked_injection", config.BLOCKED_ANSWER
    else:
        chat = build_chat(lambda text, hist: _answer(text, hist, state))
        # Only the masked text is passed on; the raw input stops here.
        answer = chat.invoke({"input": guard["text"]},
                             config={"configurable": {"session_id": session_id}}).content
        if not state.get("grounded", True) or answer == config.SENTINEL_NO_ANSWER:
            outcome = "refused_ungrounded"
    return {
        "session_id": session_id, "answer": answer, "outcome": outcome,
        "blocked": outcome == "blocked_injection", "pii_masked": guard["pii_masked"],
        "grounded": state.get("grounded", False), "escalated": state.get("escalated", False),
        "sources": state.get("sources", []), "resolved_query": state.get("resolved_query"),
    }


def history_of(session_id: str) -> list[str]:
    return [f"{m.type}: {m.content[:80]}" for m in get_session_history(session_id).messages]
