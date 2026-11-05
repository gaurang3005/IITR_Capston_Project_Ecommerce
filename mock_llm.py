"""MOCK_LLM for CrewAI — a deterministic, offline subclass of crewai.llms.base_llm.BaseLLM.

What CrewAI actually sends a BaseLLM that returns supports_function_calling() == False
(verified on crewai 1.15.30 by printing `messages`):
  * tools=None and available_functions=None. The tool list is NOT passed as an argument.
    It is printed into the SYSTEM message as "Tool Name: X / Tool Arguments: {json schema}".
  * for an agent with tools, the same system m  essage contains the ReAct template, including
    the literal line "Observation: the result of the action"         <- Pitfall 1 bait
  * after a tool runs, CrewAI appends an ASSISTANT message = our own text +
    "\\nObservation: <real tool output>"
  * the Composer (no tools) gets no tool list and no ReAct template; its call carries
    response_model=CrewResponse and its user message holds both upstream task outputs.

So:  Pitfall 1 -> read Observation only from assistant messages (the model's own turns).
     Pitfall 2 -> parse each tool's declared argument names from "Tool Arguments" and
                  route by parameter (record_id vs query), never by tool name.
Routing reads only the messages (_calls and _tool_actions are transcript counters that are
never read back), so one instance can safely serve every agent.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config  # first: sets the telemetry env vars before crewai is imported
from crewai.llms.base_llm import BaseLLM
from pydantic import PrivateAttr

COMPOSE_MARKER = "[[COMPOSE_CREW_RESPONSE]]"
_RECORD_ID = re.compile(r"\bORD\d{4}\b", re.IGNORECASE)
_TOOL_BLOCK = re.compile(r"Tool Name:\s*(?P<name>\S+)\s*\nTool Arguments:\s*(?P<args>\{.*?\})\s*\n"
                         r"Tool Description:", re.S)


def _content(m: Any) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    if isinstance(c, list):                                    # multimodal-style content parts
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c or "")


def _role(m: Any) -> str:
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) or ""


class MockReActLLM(BaseLLM):
    """Deterministic ReAct writer. Zero network, zero keys."""

    _calls: int = PrivateAttr(default=0)
    _tool_actions: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, model: str = "mock-react", **kwargs: Any) -> None:
        super().__init__(model=model, temperature=0.0, **kwargs)

    # ---- CrewAI contract -----------------------------------------------------------
    def supports_function_calling(self) -> bool:
        return False                              # use the ReAct text path (where both pitfalls live)

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    def get_input_tokens_count(self, messages, tools=None, response_model=None) -> int:
        # Same 4-chars-per-token estimate as BaseLLM, minus its per-call warning.
        text = messages if isinstance(messages, str) else " ".join(_content(m) for m in messages)
        return (len(text) + 3) // 4

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             **kwargs: Any) -> str:
        self._calls += 1
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        system = "\n".join(_content(m) for m in messages if _role(m) == "system")
        user = "\n".join(_content(m) for m in messages if _role(m) == "user")
        own_turns = [_content(m) for m in messages if _role(m) == "assistant"]

        task = self._current_task(user)
        observation = self._last_observation(own_turns)          # Pitfall 1 guard
        params = self._declared_tools(system)                    # Pitfall 2 input

        if COMPOSE_MARKER in task:                               # Composer (no tools)
            return "Thought: I can compose the response.\nFinal Answer: " + self._compose(task)
        if observation is not None:                              # a real tool already ran
            return "Thought: I now know the final answer\nFinal Answer: " + observation
        if not params:
            return "Thought: I have no tools.\nFinal Answer: " + task[:500]

        record_tool = next((n for n, ps in params.items() if "record_id" in ps), None)
        query_tool = next((n for n, ps in params.items() if "query" in ps), None)
        question = self._question(task)
        rid = _RECORD_ID.search(question)
        if record_tool and rid:
            return self._act(record_tool, {"record_id": rid.group(0).upper()})
        if query_tool:
            return self._act(query_tool, {"query": question})
        return "Thought: Nothing to look up.\nFinal Answer: NO_ORDER_ID_GIVEN"

    # ---- parsing helpers -----------------------------------------------------------
    def _act(self, tool: str, args: dict) -> str:
        self._tool_actions.append(tool)
        return (f"Thought: I should use {tool}.\nAction: {tool}\n"
                f"Action Input: {json.dumps(args, ensure_ascii=False)}")

    @staticmethod
    def _declared_tools(system: str) -> dict[str, list[str]]:
        """{tool_name: [declared argument names]} read from CrewAI's own tool listing."""
        out = {}
        for m in _TOOL_BLOCK.finditer(system):
            try:
                schema = json.loads(m.group("args"))
                props = schema.get("properties", schema)          # json-schema or flat dict
                out[m.group("name")] = list(props.keys())
            except json.JSONDecodeError:
                out[m.group("name")] = []
        return out

    @staticmethod
    def _last_observation(own_turns: list[str]) -> str | None:
        for turn in reversed(own_turns):
            if "\nObservation:" in turn or turn.startswith("Observation:"):
                return turn.split("Observation:", 1)[1].strip()
        return None

    @staticmethod
    def _current_task(user: str) -> str:
        i = user.find("Current Task:")
        task = user[i + len("Current Task:"):] if i != -1 else user
        j = task.find("\nBegin!")
        return (task[:j] if j != -1 else task).strip()

    @staticmethod
    def _question(task: str) -> str:
        m = re.search(r"CUSTOMER_QUESTION:\s*(.+?)\s*(?:\n|$)", task)
        return m.group(1).strip() if m else task.split("\n", 1)[0].strip()

    # ---- the Composer --------------------------------------------------------------
    def _compose(self, task: str) -> str:
        """Build CrewResponse JSON from the upstream task outputs CrewAI injects as context."""
        policy = re.search(r"Based on Nykaa policy documents:\n(?P<body>(?:- \(kb-\d+\).*\n?)+)", task)
        order = re.search(r"\{[^{}]*\"escalation_score\"[^{}]*\}|\{[^{}]*\"found\": false[^{}]*\}", task)
        lines, sources, grounded, escalated = [], [], False, False
        if order:
            o = json.loads(order.group(0))
            # Templated from the record's own fields and values, so each sentence can be checked.
            if o.get("found"):
                lines.append(f"Order {o['record_id']} ({o['category']}, order value {o['order_value_inr']} INR) "
                             f"is {o['status']}, created {o['days_since_created']} days ago"
                             + (", and the shipment is delayed." if o["delayed_shipment"] else "."))
                sources.append(o["record_id"])
                if o["recommend_escalation"]:
                    escalated = True
                    lines.append(f"This order needs escalation: escalation score {o['escalation_score']} "
                                 f"is above the threshold {o['escalation_threshold']}.")
            else:
                lines.append(f"No such order {o['record_id']} was found.")
                escalated = True
        if policy:
            for row in policy.group("body").strip().splitlines():
                m = re.match(r"- \((kb-\d+)\) (.*)", row)
                if m:
                    sources.append(m.group(1))
                    lines.append(m.group(2))
            grounded = True
        if not lines:
            return json.dumps({"answer": config.SENTINEL_NO_ANSWER, "sources": [],
                               "grounded": False, "escalated": True, "confidence": 0.0})
        return json.dumps({"answer": " ".join(lines), "sources": sources,
                           "grounded": grounded or bool(order and json.loads(order.group(0)).get("found")),
                           "escalated": escalated,
                           "confidence": 0.8 if grounded else 0.6}, ensure_ascii=False)

    # handy for transcripts
    @property
    def tool_actions(self) -> list[str]:
        return list(self._tool_actions)


def get_crew_llm() -> BaseLLM:
    """MOCK_LLM by default; a real model only when MOCK_LLM=false is set explicitly."""
    if config.MOCK_LLM:
        return MockReActLLM()
    from crewai import LLM
    return LLM(model=config.REAL_LLM_MODEL, temperature=0.0)
