"""Task 14 transcript: the AutoGen team reviews the CrewAI Composer's own draft (from .kickoff()) plus
the retrieved context, on two different queries: A is approved unchanged, B is revised after an
ungrounded claim is deliberately appended to the draft. Both verdicts are structured ReviewVerdicts."""
from crew_builder import run_crew
from review_stage import review_draft


def review(label: str, query: str, draft: str, evidence: list[str]):
    verdict, trace = review_draft(draft, evidence)
    print(f"{label}) query: {query}")
    print(f"   draft         : {draft}")
    print(f"   message trace : {trace}")
    print(f"   verdict       : {verdict.model_dump()}")
    return verdict


qa = "How long is the return window for footwear?"
draft_a, evidence_a = run_crew(qa)                    # the Composer's validated CrewResponse
va = review("A", qa, draft_a.answer, evidence_a)
print(f"   approved unchanged: {va.approved and va.final_answer == draft_a.answer}\n")

qb = "When will I get my COD refund?"
draft_b, evidence_b = run_crew(qb)
# TEST ARTEFACT: an ungrounded claim appended to the real draft (the crew never writes it).
injected = draft_b.answer + " COD refunds are always credited within 24 hours."
vb = review("B", qb, injected, evidence_b)
print(f"   revised: {not vb.approved and vb.final_answer != injected}"
      f" | injected claim removed: {'24 hours' not in vb.final_answer}")
