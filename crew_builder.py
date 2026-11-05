"""Task 7 — Retrieval + Lookup + Composer crew, run with .kickoff()."""
from __future__ import annotations

import re

import config  # noqa: F401  (telemetry env vars before crewai import)
from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM

from mock_llm import COMPOSE_MARKER, get_crew_llm
from schemas import CrewResponse
from tools import OrderStatusTool, RAGLookupTool


def build_crew(llm: BaseLLM | None = None) -> Crew:
    llm = llm or get_crew_llm()
    retrieval_agent = Agent(
        role="Nykaa Policy Retrieval Agent",
        goal="Answer policy questions strictly from the knowledge base.",
        backstory="You only know Nykaa's written policies; you never invent rules.",
        llm=llm, tools=[RAGLookupTool()], allow_delegation=False, max_iter=3, verbose=False)
    lookup_agent = Agent(
        role="Nykaa Order Lookup Agent",
        goal="Report a specific order's status, value and escalation score.",
        backstory="You read order records; you never guess an order's state.",
        llm=llm, tools=[OrderStatusTool()], allow_delegation=False, max_iter=3, verbose=False)
    composer_agent = Agent(
        role="Customer Response Composer",
        goal="Combine the retrieval and lookup results into one customer answer.",
        backstory="You write the final reply and add no facts of your own.",
        llm=llm, tools=[], allow_delegation=False, max_iter=2, verbose=False)

    retrieval_task = Task(
        description="Find the Nykaa policy text that answers the customer.\n"
                    "CUSTOMER_QUESTION: {question}",
        expected_output="Grounded policy text with kb doc ids, or the no-information sentence.",
        agent=retrieval_agent)
    lookup_task = Task(
        description="If the customer names an order id, look it up; otherwise reply "
                    "NO_ORDER_ID_GIVEN.\nCUSTOMER_QUESTION: {question}",
        expected_output="The order record JSON, or NO_ORDER_ID_GIVEN.",
        agent=lookup_agent)
    composer_task = Task(
        description=f"{COMPOSE_MARKER}\nCombine the policy text and the order record from the "
                    "context into ONE customer answer. Invent nothing.\nCUSTOMER_QUESTION: {question}",
        expected_output="A JSON object matching the CrewResponse schema.",
        agent=composer_agent,
        context=[retrieval_task, lookup_task],   # upstream outputs are injected into this prompt
        output_pydantic=CrewResponse)            # CrewAI parses it; we also validate it in code

    crew = Crew(agents=[retrieval_agent, lookup_agent, composer_agent],
                tasks=[retrieval_task, lookup_task, composer_task],
                process=Process.sequential, memory=False, cache=False, verbose=False)
    return crew


def run_crew(question: str, llm: BaseLLM | None = None) -> CrewResponse:
    """kickoff + Task 9 validation in code."""
    out = build_crew(llm).kickoff(inputs={"question": question})
    if out.pydantic is not None:                             # CrewAI already parsed it
        payload = out.pydantic.model_dump_json()
    else:                                                    # fall back to the raw text's JSON
        found = re.search(r"\{.*\}", out.raw, re.S)
        payload = found.group(0) if found else "{}"
    return CrewResponse.model_validate_json(payload)         # raises ValidationError if not
