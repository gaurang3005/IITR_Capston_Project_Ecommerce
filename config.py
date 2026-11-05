"""Project settings, grouped by the task that introduced them. Modules read every path, threshold
and limit from here instead of hard-coding it."""
import sys
from pathlib import Path
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

# --- grounded generation (Task 4) ---
# Measured with calibrate_threshold.py (transcripts/04_threshold_calibration.txt). There is one
# value per collection because the two chunkings give different similarity distributions.

SIMILARITY_THRESHOLD = {
    COLLECTION_FIXED: 0.359,     # midpoint of in-scope min 0.422 and out-of-scope max 0.297
    COLLECTION_SENTENCE: 0.382,  # midpoint of in-scope min 0.497 and out-of-scope max 0.266
}


SENTINEL_NO_ANSWER = ("I don't have grounded information to answer that. "
                      "Please contact Nykaa customer support so a human can help.")
# MOCK_LLM is the default: no API keys, no network. A real model has to be enabled explicitly.
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
# The model the crew runs on when MOCK_LLM=false, e.g. "groq/llama-3.1-8b-instant". It is empty by
# default, and a MOCK_LLM run never reads it.
REAL_LLM_MODEL = os.getenv("REAL_LLM_MODEL", "")


# --- guardrails (Task 10) ---
BLOCKED_ANSWER = "I can't help with that request."

# --- structured logging (Task 12) ---
LOG_DIR = ROOT / "logs"
