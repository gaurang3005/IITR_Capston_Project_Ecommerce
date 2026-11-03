"""Task 5 — document-level precision/recall for both chunking strategies, per-query arithmetic."""
from __future__ import annotations

import config
from rag_core import retrieve

# Gold labels are the parent docs that should answer each query. Same 5 queries as Task 4.
# The labels are strict: a warranty doc is not a correct answer to a return-window question.
QUERIES: dict[str, set[str]] = {
    "How long is the return window for footwear?": {"kb-01"},
    "When will I get my COD refund?": {"kb-02"},
    "What is the delivery time for metro cities?": {"kb-03"},
    "Can I cancel an order after it has shipped?": {"kb-06"},
    "How much are 500 loyalty points worth?": {"kb-07"},
}
K_CHUNKS = config.TOP_K      # same chunk budget for both collections
MAX_DOCS = 3                 # same document budget after de-duplication


def retrieved_docs(query: str, collection: str) -> tuple[list[str], list[str]]:
    hits = retrieve(query, collection, k=K_CHUNKS)
    chunk_docs = [h["doc_id"] for h in hits]
    docs: list[str] = []
    for d in chunk_docs:                       # dedupe, keep rank order
        if d not in docs:
            docs.append(d)
    return chunk_docs, docs[:MAX_DOCS]


def run() -> dict[str, dict[str, float]]:
    summary = {}
    for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE):
        print(f"\n=== {name}  (top-{K_CHUNKS} chunks -> unique parent docs, cap {MAX_DOCS}) ===")
        ps, rs = [], []
        for q, gold in QUERIES.items():
            chunk_docs, got = retrieved_docs(q, name)
            tp = [d for d in got if d in gold]
            p = len(tp) / len(got) if got else 0.0
            r = len(tp) / len(gold)
            ps.append(p); rs.append(r)
            print(f"\nQ: {q}")
            print(f"   chunks -> docs : {chunk_docs} -> {got}")
            print(f"   gold           : {sorted(gold)}   relevant retrieved: {tp}")
            print(f"   precision = {len(tp)}/{len(got)} = {p:.2f}")
            print(f"   recall    = {len(tp)}/{len(gold)} = {r:.2f}")
        mp, mr = sum(ps) / len(ps), sum(rs) / len(rs)
        print(f"\n   mean precision = ({' + '.join(f'{x:.2f}' for x in ps)}) / {len(ps)} = {mp:.3f}")
        print(f"   mean recall    = ({' + '.join(f'{x:.2f}' for x in rs)}) / {len(rs)} = {mr:.3f}")
        summary[name] = {"precision": mp, "recall": mr}
    return summary


if __name__ == "__main__":
    s = run()
    best = max(s, key=lambda n: (s[n]["precision"] + s[n]["recall"], s[n]["precision"]))
    print(f"\nHigher P+R on this sample: {best}")
