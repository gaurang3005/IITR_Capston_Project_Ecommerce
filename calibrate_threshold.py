"""Task 4, step 1: measure top-1 similarity, then choose the threshold between the clusters.
Queries are embedded exactly as the answer path embeds them (rag_core.search_text: brand name
removed); for a query that contains the brand, its similarity with the brand is shown too."""
import config
from rag_core import build_both, search_text, top1_similarity

IN_SCOPE = [
    "How long is the return window for footwear?",
    "When will I get my COD refund?",
    "What is the delivery time for metro cities?",
    "Can I cancel an order after it has shipped?",
    "How much are 500 loyalty points worth?",
]
OUT_OF_SCOPE = [
    "What is your CEO's salary?",
    "Book me a flight to Delhi tomorrow.",
    "Recommend a good laptop for gaming.",      # shares shopping vocabulary: the hard case
    # With the brand name in the embedded text these scored 0.47-0.53 and were answered.
    # ("What is Nykaa's CEO's salary?" is kept out of calibration: it is a Task 13 test case.)
    "Who founded Nykaa?",
    "Does Nykaa sell groceries?",
    "What is Nykaa's share price today?",
]


def _measure(queries: list[str], collection: str) -> list[tuple[float, str, str]]:
    rows = []
    for q in queries:
        note = ""
        if search_text(q) != q:
            note = f"   (with the brand name: {top1_similarity(q, collection, strip_brand=False):.3f})"
        rows.append((top1_similarity(q, collection), q, note))
    return rows


def calibrate(collection: str) -> dict:
    ins, outs = _measure(IN_SCOPE, collection), _measure(OUT_OF_SCOPE, collection)
    lo_in, hi_out = min(s for s, _, _ in ins), max(s for s, _, _ in outs)
    print(f"\n=== {collection} ===")
    print("IN-SCOPE top-1 similarity")
    for s, q, note in ins:
        print(f"  {s:.3f}  {q}{note}")
    print("OUT-OF-SCOPE top-1 similarity")
    for s, q, note in outs:
        print(f"  {s:.3f}  {q}{note}")
    if lo_in > hi_out:
        thr = round((lo_in + hi_out) / 2, 3)
        print(f"clusters separate: in-scope min {lo_in:.3f} > out-of-scope max {hi_out:.3f} "
              f"(gap {lo_in - hi_out:.3f}) -> threshold = midpoint = {thr}")
    else:
        # Overlap: pick the cut that misclassifies the fewest sample queries and flag it.
        cands = sorted({s for s, _, _ in ins + outs})
        thr = max(cands, key=lambda t: sum(s >= t for s, _, _ in ins) + sum(s < t for s, _, _ in outs))
        thr = round(thr, 3)
        print(f"clusters OVERLAP (in-scope min {lo_in:.3f} <= out-of-scope max {hi_out:.3f}); "
              f"best-separating cut on this sample = {thr:.3f} (some queries will be misclassified).")
    return {"in_min": lo_in, "out_max": hi_out, "threshold": thr}


if __name__ == "__main__":
    build_both()
    res = {c: calibrate(c) for c in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE)}
    print("\nThresholds for config.SIMILARITY_THRESHOLD:")
    print("  {" + ", ".join(f'"{c}": {r["threshold"]}' for c, r in res.items()) + "}")
    print("  currently in config.py:", config.SIMILARITY_THRESHOLD)
