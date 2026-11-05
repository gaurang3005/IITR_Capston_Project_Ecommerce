"""Task 16 transcript: repeated query -> cache hit that skips retrieval and the writer (LLM) call."""
import time
from rag_core import CACHE, STATS, generate

CACHE.clear()
for label, q in [("first", "How long is the return window for footwear?"),
                 ("repeat", "How long is the return window for footwear?"),
                 ("same after normalisation", "  how long is the RETURN window for footwear??  ")]:
    before = dict(STATS)
    t0 = time.perf_counter(); r = generate(q); ms = (time.perf_counter() - t0) * 1000
    print(f"{label:<26} cache_hit={r['cache_hit']!s:<5} {ms:8.2f} ms   "
          f"retrieve_calls +{STATS['retrieve_calls'] - before['retrieve_calls']}  "
          f"writer(LLM)_calls +{STATS['writer_calls'] - before['writer_calls']}")
print("cache stats:", CACHE.stats())
