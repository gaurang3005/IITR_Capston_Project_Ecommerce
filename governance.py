"""Task 15 — least autonomy (Application layer), risk class, per-request budget (Runtime layer)."""
from __future__ import annotations

import config

# ---- Application layer: least autonomy ---------------------------------------------
LOOKUP_TOOL_NAME = "check_order_status"
AUTHORISED_AGENTS = frozenset({"Nykaa Order Lookup Agent"})


class LeastAutonomyViolation(PermissionError):
    pass


def assert_least_autonomy(crew) -> None:
    """Executable invariant: only the Lookup Agent holds the order tool, and nobody can
    delegate (delegation would let another agent reach the tool indirectly)."""
    for agent in crew.agents:
        names = {getattr(t, "name", "") for t in (agent.tools or [])}
        if LOOKUP_TOOL_NAME in names and agent.role not in AUTHORISED_AGENTS:
            raise LeastAutonomyViolation(f"{agent.role!r} may not hold {LOOKUP_TOOL_NAME!r}")
        if agent.allow_delegation:
            raise LeastAutonomyViolation(f"{agent.role!r} has allow_delegation=True")


def wire_tool(agent, tool) -> None:
    """The only sanctioned way to add a tool at runtime; refuses the order tool for others."""
    if getattr(tool, "name", "") == LOOKUP_TOOL_NAME and agent.role not in AUTHORISED_AGENTS:
        raise LeastAutonomyViolation(f"{agent.role!r} is not authorised to use {tool.name!r}")
    agent.tools = list(agent.tools or []) + [tool]


RISK_LEVEL = "Medium"

# ---- Runtime layer: per-request token / cost budget ---------------------------------
# Sized by governance_demo.py, which meters every model call of one request (crew 4-5, RAG writer
# 0-1, review 2; see transcripts/15_governance.txt). Under MOCK_LLM the largest request sent 3,740
# prompt tokens besides the question, repeated the question 7 times and got 681 completion tokens back.
# The constants round those up. A real model would need re-measuring from the usage it reports.
PROMPT_OVERHEAD_TOKENS = 3900        # all prompts of one request, minus the question itself
QUESTION_COPIES = 7                  # times the prompts repeat the question (crew 6, writer 1)
EXPECTED_COMPLETION_TOKENS = 700     # all completions of one request


class BudgetExceeded(Exception):
    pass


def estimate_tokens(text: str) -> int:
    """~4 characters per token (same fallback CrewAI's BaseLLM uses). Offline, no tokenizer download."""
    return (len(text) + 3) // 4


def enforce_budget(user_text: str, history_text: str = "") -> dict:
    """Reject before any LLM or tool call. Never truncate silently. History is counted once, as
    headroom: the pipeline only reads it to resolve the order id and never sends it to a model."""
    tokens = (PROMPT_OVERHEAD_TOKENS + QUESTION_COPIES * estimate_tokens(user_text)
              + estimate_tokens(history_text) + EXPECTED_COMPLETION_TOKENS)
    inr = round(tokens / 1000 * config.INR_PER_1K_TOKENS, 4)
    report = {"estimated_tokens": tokens, "token_budget": config.MAX_TOKENS_PER_REQUEST,
              "estimated_inr": inr, "inr_budget": config.MAX_INR_PER_REQUEST}
    if tokens > config.MAX_TOKENS_PER_REQUEST or inr > config.MAX_INR_PER_REQUEST:
        raise BudgetExceeded(f"request needs ~{tokens} tokens (₹{inr}); budget is "
                             f"{config.MAX_TOKENS_PER_REQUEST} tokens (₹{config.MAX_INR_PER_REQUEST})")
    return report
