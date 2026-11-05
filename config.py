"""Project settings, grouped by the task that introduced them. Modules read every path, threshold
and limit from here instead of hard-coding it. Import this module before crewai and
sentence_transformers: it sets environment variables those libraries read when they are imported."""
import os
import sys
from pathlib import Path

# On Windows, output redirected to a file (python x.py > transcripts/x.txt) uses the ANSI code page,
# which has no ₹ sign. UTF-8 output keeps the transcripts readable on every OS.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_store"

# --- chunking, embedding and retrieval (Task 3) ---
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_WORDS = 40
OVERLAP_WORDS = 10                                        # 25% overlap
COLLECTION_FIXED = "kb_fixed"
COLLECTION_SENTENCE = "kb_sentence"
TOP_K = 4

# Zero network for embeddings: SentenceTransformers asks the Hugging Face Hub for model updates on
# every load. Once the model is in the local cache (the first run downloads it), offline mode stops
# that call, and the "Loading weights" progress bar is switched off to keep transcripts readable.
_HF_HUB_CACHE = Path(os.getenv("HF_HUB_CACHE")
                     or Path(os.getenv("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub")
if (_HF_HUB_CACHE / f"models--sentence-transformers--{EMBED_MODEL_NAME}").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# --- grounded generation (Task 4) ---
# Measured with calibrate_threshold.py (transcripts/04_threshold_calibration.txt). There is one
# value per collection because the two chunkings give different similarity distributions.

SIMILARITY_THRESHOLD = {
    COLLECTION_FIXED: 0.379,     # midpoint of in-scope min 0.422 and out-of-scope max 0.336
    COLLECTION_SENTENCE: 0.403,  # midpoint of in-scope min 0.497 and out-of-scope max 0.309
}


# Every KB document is about Nykaa, so the brand name carries no topic signal, yet it lifted off-topic
# questions over the threshold ("Does Nykaa sell groceries?" scored 0.534 and was answered).
# rag_core.search_text() removes these words before a query is embedded.
BRAND_TERMS = ("Nykaa",)

SENTINEL_NO_ANSWER = ("I don't have grounded information to answer that. "
                      "Please contact Nykaa customer support so a human can help.")
# MOCK_LLM is the default and the only supported mode: no API keys, no network. MOCK_LLM=false
# raises NotImplementedError at the grounded writer (rag_core._writer).
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() != "false"


# --- chunking choice (Task 5) ---
# kb_sentence had the higher precision at equal recall (transcripts/05_chunking_comparison.txt).
# generate() searches this collection unless it is given another one.
RECOMMENDED_COLLECTION = COLLECTION_SENTENCE

# --- CrewAI (Task 7) ---
# Telemetry is switched off here, and every module that uses CrewAI imports config before crewai,
# so a MOCK_LLM run makes no network calls.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
# Seam for a future real crew model (mock_llm.get_crew_llm), e.g. "groq/llama-3.1-8b-instant".
# Not used by this submission: a MOCK_LLM run never reads it.
REAL_LLM_MODEL = os.getenv("REAL_LLM_MODEL", "")


# --- guardrails (Task 10) ---
BLOCKED_ANSWER = "I can't help with that request."

# --- structured logging (Task 12) ---
LOG_DIR = ROOT / "logs"

# --- governance (Task 15) ---
MAX_TOKENS_PER_REQUEST = 6000        # whole request, every model call; largest measured: 4,540
INR_PER_1K_TOKENS = 0.50             # illustrative rate for a cost estimate (not a real price)
MAX_INR_PER_REQUEST = 3.00           # = 6000 tokens at the rate above
