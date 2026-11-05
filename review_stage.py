"""Task 14 — 2-agent AutoGen RoundRobinGroupChat review of the crew's draft (autogen-agentchat 0.7.x).

Built on the Microsoft AutoGen 0.4+ API (`autogen-agentchat`, `autogen-core`), because that is
where the brief's RoundRobinGroupChat, max_turns, output_content_type and StructuredMessage live.
Lecture 43 uses AG2 (`from autogen import GroupChat`), which has a different API.
"""
from __future__ import annotations

import asyncio
import json
import re

import config
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

from guardrails import _split_sentences, sentence_supported
from schemas import ReviewVerdict

DRAFT_MARK, CTX_MARK = "DRAFT_ANSWER:", "RETRIEVED_CONTEXT:"


def _text(messages) -> str:
    return "\n".join(m.content for m in messages if isinstance(getattr(m, "content", None), str))


def _parse_task(text: str) -> tuple[str, str]:
    m = re.search(rf"{DRAFT_MARK}\s*(.*?)\s*{CTX_MARK}\s*(.*?)(?:\n\[REVIEWER\]|\Z)", text, re.S)
    return (m.group(1).strip(), m.group(2).strip()) if m else (text, "")


def review_sentences(draft: str, context: str) -> tuple[list[str], list[str]]:
    kept, dropped = [], []
    for s in _split_sentences(draft):
        (kept if sentence_supported(s, context) else dropped).append(s)
    return kept, dropped


class MockReviewClient(ChatCompletionClient):
    """Deterministic ChatCompletionClient (MOCK_LLM seam for AutoGen). Reads the real draft and
    context from the conversation and checks every draft sentence (guardrails._split_sentences)
    against the context."""

    def __init__(self, role: str) -> None:
        assert role in ("reviewer", "editor")
        self.role, self.calls = role, 0
        self._last = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(self, messages, *, tools=[], tool_choice="auto", json_output=None,
                     extra_create_args={}, cancellation_token=None) -> CreateResult:
        self.calls += 1
        convo = _text(messages)
        draft, context = _parse_task(convo)
        kept, dropped = review_sentences(draft, context)
        if self.role == "reviewer":
            content = ("[REVIEWER] " + (f"UNSUPPORTED: {json.dumps(dropped)}" if dropped
                                        else "All draft sentences are supported by the context."))
        else:   # editor -> must return JSON that parses as ReviewVerdict (json_output=ReviewVerdict)
            v = (ReviewVerdict(approved=True, final_answer=draft,
                               reason="Every sentence is supported by the retrieved context.")
                 if not dropped else
                 ReviewVerdict(approved=False,
                               final_answer=" ".join(kept) or config.SENTINEL_NO_ANSWER,
                               reason=f"Removed {len(dropped)} unsupported claim(s): {dropped}"))
            content = v.model_dump_json()
        self._last = RequestUsage(prompt_tokens=len(convo) // 4, completion_tokens=len(content) // 4)
        self._total = RequestUsage(prompt_tokens=self._total.prompt_tokens + self._last.prompt_tokens,
                                   completion_tokens=self._total.completion_tokens + self._last.completion_tokens)
        return CreateResult(content=content, finish_reason="stop", usage=self._last, cached=False)

    async def create_stream(self, messages, **kwargs):
        yield await self.create(messages, **kwargs)

    async def close(self) -> None: ...
    def actual_usage(self) -> RequestUsage: return self._last
    def total_usage(self) -> RequestUsage: return self._total
    def count_tokens(self, messages, *, tools=[]) -> int: return len(_text(messages)) // 4
    def remaining_tokens(self, messages, *, tools=[]) -> int: return 8192 - self.count_tokens(messages)

    @property
    def capabilities(self):                          # deprecated alias, still abstract in 0.7
        return self.model_info

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(vision=False, function_calling=False, json_output=True,
                         structured_output=True, family="unknown", multiple_system_messages=True)


def build_review_team() -> RoundRobinGroupChat:
    # Agent names must be valid Python identifiers in autogen-agentchat ("Final-Editor" crashes).
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=MockReviewClient("reviewer"),
        system_message="Check every sentence of the draft against the retrieved context. "
                       "List any sentence the context does not support.")
    editor = AssistantAgent(
        "Final_Editor", model_client=MockReviewClient("editor"),
        system_message="Approve the draft unchanged if it is fully supported, otherwise remove "
                       "the unsupported claims. Return a ReviewVerdict.",
        output_content_type=ReviewVerdict)                       # structured output
    return RoundRobinGroupChat(
        [reviewer, editor],
        max_turns=2,                                             # NOT max_iterations
        # alternative: termination_condition=MaxMessageTermination(3)  (task msg counts as 1)
        custom_message_types=[StructuredMessage[ReviewVerdict]])  # else: "not registered"


async def areview(draft: str, contexts: list[str]) -> tuple[ReviewVerdict, list[str]]:
    task = f"{DRAFT_MARK}\n{draft}\n{CTX_MARK}\n" + "\n".join(contexts)
    result = await build_review_team().run(task=task)
    trace = [f"{type(m).__name__}:{getattr(m, 'source', '?')}" for m in result.messages]
    for m in reversed(result.messages):
        if isinstance(m, StructuredMessage) and isinstance(m.content, ReviewVerdict):
            return m.content, trace
    raise RuntimeError(f"review produced no structured verdict: {trace}")   # fail loudly


def review_draft(draft: str, contexts: list[str]) -> tuple[ReviewVerdict, list[str]]:
    """Sync entry point. Call it from sync code only (FastAPI runs sync endpoints in a thread,
    and the WebSocket handler offloads with asyncio.to_thread), so asyncio.run is safe."""
    return asyncio.run(areview(draft, contexts))
