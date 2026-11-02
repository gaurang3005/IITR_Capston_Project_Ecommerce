"""Project settings, grouped by the task that introduced them. Modules read every path, threshold
and limit from here instead of hard-coding it."""
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
