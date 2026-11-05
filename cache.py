"""Task 16 — in-memory cache for the grounded-generation step, keyed by normalised query text."""
from __future__ import annotations

import re
from collections import OrderedDict

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    """'  How long is the return window for FOOTWEAR?? ' -> 'how long is the return window for footwear'"""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


class GenerationCache:
    """LRU cache. Key = (collection, normalised query). Policy answers only: order status is
    live data and is never cached (a cached 'Shipped' would go stale)."""

    def __init__(self, maxsize: int = 256) -> None:
        self._store: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self.maxsize = maxsize
        self.hits = self.misses = 0

    def get(self, collection: str, query: str) -> dict | None:
        key = (collection, normalise(query))
        if key in self._store:
            self._store.move_to_end(key)
            self.hits += 1
            return self._store[key]
        self.misses += 1
        return None

    def put(self, collection: str, query: str, value: dict) -> None:
        self._store[(collection, normalise(query))] = value
        self._store.move_to_end((collection, normalise(query)))
        while len(self._store) > self.maxsize:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "size": len(self._store),
                "hit_rate": round(self.hits / total, 3) if total else 0.0}
