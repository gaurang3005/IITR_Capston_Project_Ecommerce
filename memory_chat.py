"""Task 8 — in-process, session-keyed conversation memory around the real pipeline."""
from __future__ import annotations

import re
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from dataset import CATEGORIES

_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    return _store.setdefault(session_id, InMemoryChatMessageHistory())


def reset_all() -> None:
    _store.clear()


_ORDER = re.compile(r"\bORD\d{4}\b", re.I)
_FOLLOW_UP = re.compile(r"^\s*(?:and\s+)?(?:what|how)\s+about\b|^\s*and\s+(?:for|the)\b", re.I)
_PRONOUN = re.compile(r"\b(?:it|that order|this order|my order|the order)\b", re.I)


def resolve_followup(user: str, history: list[BaseMessage]) -> str:
    """MOCK_LLM stand-in for an LLM 'rewrite the question using the chat history' step.
    1. 'What about Beauty?'      -> previous question with the category swapped
    2. 'Is it delayed?' (no id)  -> carries the last order id the customer typed in this session"""
    humans = [m.content for m in history if m.type == "human"]
    if not humans:
        return user
    q = user
    new_cat = next((c for c in CATEGORIES if re.search(rf"\b{c}\b", user, re.I)), None)
    if _FOLLOW_UP.search(user) and new_cat:
        for prev in reversed(humans):
            old_cat = next((c for c in CATEGORIES if re.search(rf"\b{c}\b", prev, re.I)), None)
            if old_cat:
                q = re.sub(rf"\b{old_cat}\b", new_cat, prev, flags=re.I)
                break
    if not _ORDER.search(q) and _PRONOUN.search(q):
        for prev in reversed(humans):
            m = _ORDER.search(prev)
            if m:
                q = f"{q} (order {m.group(0).upper()})"
                break
    return q


def build_chat(answer_fn: Callable[[str, list[BaseMessage]], str]) -> RunnableWithMessageHistory:
    """answer_fn(user_text, history) -> final answer text. History is loaded and appended for us."""
    chain = RunnableLambda(lambda p: AIMessage(content=answer_fn(p["input"], p.get("history", []))))
    return RunnableWithMessageHistory(chain, get_session_history,
                                      input_messages_key="input", history_messages_key="history")
