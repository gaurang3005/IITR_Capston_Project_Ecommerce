"""The single request pipeline. Every end-to-end demo and every entry point calls ask_agent(),
so the transcripts show exactly what the deployed service does."""
from __future__ import annotations

from typing import Any

from crew_builder import run_crew
from memory_chat import build_chat, get_session_history, resolve_followup


def _answer(text: str, history: list, state: dict) -> str:
    """Runs inside RunnableWithMessageHistory, so `history` is this session's prior turns."""
    resolved = resolve_followup(text, history)
    state["resolved_query"] = resolved

    response = run_crew(resolved)
    state.update(grounded=response.grounded, escalated=response.escalated,
                 sources=response.sources)
    return response.answer


def ask_agent(user_text: str, session_id: str = "default") -> dict[str, Any]:
    state: dict[str, Any] = {}
    chat = build_chat(lambda text, hist: _answer(text, hist, state))
    answer = chat.invoke({"input": user_text},
                         config={"configurable": {"session_id": session_id}}).content
    return {
        "session_id": session_id, "answer": answer,
        "grounded": state.get("grounded", False), "escalated": state.get("escalated", False),
        "sources": state.get("sources", []), "resolved_query": state.get("resolved_query"),
    }


def history_of(session_id: str) -> list[str]:
    return [f"{m.type}: {m.content[:80]}" for m in get_session_history(session_id).messages]
