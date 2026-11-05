"""Task 14 transcript: one approve-unchanged case and one revise case, both structured verdicts."""
from rag_core import generate
from review_stage import review_draft

q = "How long is the return window for footwear?"
g = generate(q)
contexts = [c["text"] for c in g["contexts"]]
clean = " ".join(contexts[:1])                        # a draft built only from retrieved text

v1, trace1 = review_draft(clean, contexts)
print(f"A) query: {q}\n   draft: {clean}")
print(f"   message trace: {trace1}")
print(f"   verdict: {v1.model_dump()}")
print(f"   approved unchanged: {v1.approved and v1.final_answer == clean}")

# TEST ARTEFACT: a deliberately injected ungrounded claim (not real crew output).
injected = clean + " Electronics can also be returned within 60 days for any reason."
v2, trace2 = review_draft(injected, contexts)
print(f"\nB) draft with injected claim: {injected}")
print(f"   message trace: {trace2}")
print(f"   verdict: {v2.model_dump()}")
print(f"   revised: {not v2.approved and v2.final_answer != injected}")
