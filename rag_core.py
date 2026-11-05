"""Tasks 3, 4 and 16 — chunking, embedding, two Chroma collections, grounded generation, cache."""
from __future__ import annotations

import re
import sys
import threading
from typing import Callable

import config  # first: sets the Hugging Face offline switch before sentence_transformers loads
import chromadb
from sentence_transformers import SentenceTransformer

from cache import GenerationCache
from knowledge_base import DOCUMENTS


# ---- embedder (loaded once) ---------------------------------------------------
_embedder: SentenceTransformer | None = None


def embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """Normalised vectors, same model for documents and queries (lecture 18 'same model rule')."""
    return embedder().encode(texts, normalize_embeddings=True, convert_to_numpy=True).tolist()


# ---- chunking strategies --------------------------------------------------------
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_fixed(text: str, size: int = config.CHUNK_WORDS,
                overlap: int = config.OVERLAP_WORDS) -> list[str]:
    """Sliding window of `size` words; consecutive windows share `overlap` words.
    Requires 0 <= overlap < size (overlap >= size would never advance the window)."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be smaller than chunk size")
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += size - overlap
    return chunks


def chunk_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def build_chunks(strategy: str) -> list[dict]:
    out = []
    for doc in DOCUMENTS:
        pieces = chunk_fixed(doc.text) if strategy == "fixed" else chunk_sentences(doc.text)
        for i, piece in enumerate(pieces):
            out.append({"chunk_id": f"{doc.doc_id}::{strategy}::{i}", "doc_id": doc.doc_id,
                        "topic": doc.topic, "source": doc.source, "strategy": strategy,
                        "text": piece, "chunk_index": i})
    return out


# ---- indexing -------------------------------------------------------------------
_client: chromadb.ClientAPI | None = None


def client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def _collection(name: str):
    # We always pass our own vectors (embeddings= on upsert, query_embeddings= on query), so
    # Chroma never embeds text and never downloads its default ONNX model (that download would
    # break the zero-network rule). embedding_function=None (as in lectures 18/20) does not block
    # that download on chromadb 1.5.x: a text-only call such as query_texts= still falls back
    # to the default model, which is why every call here passes vectors.
    # cosine space: distance = 1 - cosine_similarity. Chroma's default is squared L2,
    # where 1 - distance is not a similarity.
    return client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def index_collection(name: str, strategy: str) -> int:
    """Rebuild one collection from scratch so edited/removed KB text leaves no stale chunks."""
    try:
        client().delete_collection(name)
    except Exception:                      # first run: nothing to delete
        pass
    chunks = build_chunks(strategy)
    _collection(name).upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed([c["text"] for c in chunks]),
        metadatas=[{k: c[k] for k in ("doc_id", "topic", "source", "strategy", "chunk_index")}
                   for c in chunks],
    )
    return len(chunks)


def build_both() -> dict[str, int]:
    global _index_ready
    with _index_lock:
        counts = {config.COLLECTION_FIXED: index_collection(config.COLLECTION_FIXED, "fixed"),
                  config.COLLECTION_SENTENCE: index_collection(config.COLLECTION_SENTENCE, "sentence")}
        CACHE.clear()                      # a KB change invalidates every cached answer
        _index_ready = True
    return counts


# ---- keeping the index in step with knowledge_base/ -----------------------------
# chroma_store/ is not in git, so a fresh clone starts with no index, and a document added through
# POST /add-document stays in chroma_store/ after its process ends. So before the first query of
# every process, the stored chunks are compared with the chunks the current knowledge base produces,
# and both collections are rebuilt if they differ. Results no longer depend on which script ran first.
_index_ready = False
_index_lock = threading.RLock()


def index_is_current() -> bool:
    for name, strategy in ((config.COLLECTION_FIXED, "fixed"), (config.COLLECTION_SENTENCE, "sentence")):
        stored = _collection(name).get(include=["documents"])
        wanted = {c["chunk_id"]: c["text"] for c in build_chunks(strategy)}
        if dict(zip(stored["ids"], stored["documents"])) != wanted:
            return False
    return True


def ensure_index() -> None:
    global _index_ready
    if _index_ready:
        return
    with _index_lock:
        if not _index_ready and not index_is_current():
            counts = build_both()
            print(f"[rag_core] index missing or out of date -> rebuilt from knowledge_base/: {counts}",
                  file=sys.stderr)
        _index_ready = True


# ---- retrieval ------------------------------------------------------------------
STATS = {"retrieve_calls": 0, "writer_calls": 0}      # proves cache hits skip real work
_BRAND = re.compile(r"\b(?:%s)(?:['’]s)?\b" % "|".join(map(re.escape, config.BRAND_TERMS)),
                    re.IGNORECASE)


def search_text(query: str) -> str:
    """The text that is embedded for a query: the brand name removed (see config.BRAND_TERMS)."""
    return " ".join(_BRAND.sub(" ", query).split()) or query


def retrieve(query: str, collection_name: str, k: int = config.TOP_K, *,
             strip_brand: bool = True) -> list[dict]:
    ensure_index()
    STATS["retrieve_calls"] += 1
    res = _collection(collection_name).query(
        query_embeddings=embed([search_text(query) if strip_brand else query]), n_results=k,
        include=["documents", "metadatas", "distances"])
    return [{"chunk_id": cid, "text": text, "doc_id": meta["doc_id"], "topic": meta["topic"],
             "distance": dist, "similarity": 1.0 - dist}
            for cid, text, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0])]


def top1_similarity(query: str, collection_name: str, *, strip_brand: bool = True) -> float:
    return retrieve(query, collection_name, k=1, strip_brand=strip_brand)[0]["similarity"]


# ---- grounded generation --------------------------------------------------------
GroundedWriter = Callable[[str, list[dict]], str]


def mock_extractive_writer(query: str, contexts: list[dict]) -> str:
    """MOCK_LLM writer: quotes every supporting chunk verbatim, best first, tagged with its doc id.
    (Quoting only each document's top chunk often left out the sentence that held the answer.)"""
    STATS["writer_calls"] += 1
    lines = ["Based on Nykaa policy documents:"]
    for c in contexts:
        text = c["text"][2:] if c["text"].startswith("- ") else c["text"]   # drop the KB bullet mark
        lines.append(f"- ({c['doc_id']}) {text}")
    return "\n".join(lines)


def _writer() -> GroundedWriter:
    if not config.MOCK_LLM:
        raise NotImplementedError("MOCK_LLM=false is not supported: this submission ships only the "
                                  "MOCK_LLM grounded writer. Unset MOCK_LLM to run.")
    return mock_extractive_writer


CACHE = GenerationCache()


def generate(query: str, collection_name: str = config.RECOMMENDED_COLLECTION) -> dict:
    """Retrieve -> similarity gate -> write. Cached per (collection, normalised query) (Task 16)."""
    hit = CACHE.get(collection_name, query)
    if hit is not None:
        return {**hit, "cache_hit": True}
    hits = retrieve(query, collection_name)
    top = hits[0]["similarity"] if hits else 0.0
    threshold = config.SIMILARITY_THRESHOLD[collection_name]
    grounded = top >= threshold
    # Only chunks that themselves clear the threshold are allowed into the answer.
    supporting = [h for h in hits if h["similarity"] >= threshold] if grounded else []
    result = {
        "answer": _writer()(query, supporting) if grounded else config.SENTINEL_NO_ANSWER,
        "grounded": grounded, "top_similarity": round(top, 4), "threshold": threshold,
        "contexts": supporting, "all_hits": hits, "cache_hit": False,
    }
    CACHE.put(collection_name, query, result)
    return result


if __name__ == "__main__":
    counts = build_both()
    print(f"Embedding model: {config.EMBED_MODEL_NAME}")
    for name, n in counts.items():
        print(f"{name:<12}: {n} chunks indexed")
    q = "How long is the return window for footwear?"
    print(f'\nSample query: "{q}"')
    for name in counts:
        print(f"{name} top-3:")
        for h in retrieve(q, name, k=3):
            print(f"  {h['similarity']:.3f}  {h['doc_id']}  {h['text'][:70]}...")
