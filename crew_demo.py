"""Task 7 + 9 transcript: both tools fire on different queries via .kickoff(); responses validate."""
from pydantic import ValidationError
from crew_builder import run_crew
from mock_llm import MockReActLLM
from schemas import CrewResponse

for q in ["How long is the return window for footwear?", "What is the status of order ORD0029?",
          "Where is my order ORD9999?"]:
    llm = MockReActLLM()
    resp,_ = run_crew(q, llm)
    print(f"\nQ: {q}\n   tools invoked via kickoff(): {llm.tool_actions}")
    print("   CrewResponse (validated):", resp.model_dump_json())

print("\nTask 9 negative test (the validator is load-bearing):")
try:
    CrewResponse.model_validate({"answer": "", "sources": [], "grounded": True,
                                 "escalated": False, "confidence": 2.0})
except ValidationError as e:
    print(f"   REJECTED: {e.error_count()} validation errors ->",
          [(err['loc'][0], err['type']) for err in e.errors()])

