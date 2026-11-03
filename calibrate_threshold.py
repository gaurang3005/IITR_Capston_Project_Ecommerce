"""Task 4, step 1: measure top-1 similarity, then choose the threshold between the clusters."""
import config
from rag_core import build_both, top1_similarity

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
]


def calibrate(collection: str) -> dict:
    ins = [(top1_similarity(q, collection), q) for q in IN_SCOPE]
    outs = [(top1_similarity(q, collection), q) for q in OUT_OF_SCOPE]
    lo_in, hi_out = min(s for s, _ in ins), max(s for s, _ in outs)
    print(f"\n=== {collection} ===")
    print("IN-SCOPE top-1 similarity")
    for s, q in ins:
        print(f"  {s:.3f}  {q}")
    print("OUT-OF-SCOPE top-1 similarity")
    for s, q in outs:
        print(f"  {s:.3f}  {q}")
    if lo_in > hi_out:
        thr = round((lo_in + hi_out) / 2, 3)
        print(f"clusters separate: in-scope min {lo_in:.3f} > out-of-scope max {hi_out:.3f} "
              f"(gap {lo_in - hi_out:.3f}) -> threshold = midpoint = {thr}")
    else:
        # Overlap: pick the cut that misclassifies the fewest sample queries and flag it.
        cands = sorted({s for s, _ in ins + outs})
        thr = max(cands, key=lambda t: sum(s >= t for s, _ in ins) + sum(s < t for s, _ in outs))
        thr = round(thr, 3)
        print(f"clusters OVERLAP (in-scope min {lo_in:.3f} <= out-of-scope max {hi_out:.3f}); "
              f"best-separating cut on this sample = {thr:.3f} (some queries will be misclassified).")
    return {"in_min": lo_in, "out_max": hi_out, "threshold": thr}


if __name__ == "__main__":
    build_both()
    res = {c: calibrate(c) for c in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE)}
    print("\nThresholds for config.SIMILARITY_THRESHOLD:")
    print("  {" + ", ".join(f'"{c}": {r["threshold"]}' for c, r in res.items()) + "}")
