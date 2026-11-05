"""Task 2 — the Nykaa policy knowledge base, loaded from the knowledge_base/ folder.
offline files created, this dose following. 

Loads the folder: it reads every .md file in sorted order and skips # title lines. It joins the text into one paragraph and reads the files as UTF-8, so ₹ stays intact.
Assigns ids: the file name sets the id and topic, so 01_return_window_by_category.md becomes doc_id = "kb-01" with topic = "return_window_by_category"-> for use in later task. 
Catches mistakes: a badly named file, an empty file or a repeated number stops it with a clear error.
self-check: checks that there are at least 12 docs with 2 to 5 sentences each, and prints PASS or FAIL
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
_FILE_RE = re.compile(r"^(\d{2,})_([a-z0-9_]+)$")         # stem -> (number, topic)


@dataclass(frozen=True)
class Document:
    doc_id: str
    topic: str
    text: str
    source: str = ""                                       # file name, kept as chunk metadata


def load_documents(folder: Path = KB_DIR) -> list[Document]:
    """Read every *.md file in `folder`, sorted by name so doc order is stable on every OS."""
    files = sorted(Path(folder).glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files in {folder}")
    docs = []
    for path in files:
        match = _FILE_RE.match(path.stem)
        if not match:
            raise ValueError(f"{path.name}: name it like 01_return_window_by_category.md")
        lines = path.read_text(encoding="utf-8").splitlines()   # utf-8: Windows default would break ₹
        body = " ".join(line for line in lines if not line.lstrip().startswith("#"))
        text = " ".join(body.split())                       # one paragraph, single spaces
        if not text:
            raise ValueError(f"{path.name} has no text")
        doc_id = f"kb-{match.group(1)}"                      # 01_... -> kb-01
        docs.append(Document(doc_id, match.group(2), text, path.name))
    ids = [d.doc_id for d in docs]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate doc_id across files: {dupes}")
    return docs


DOCUMENTS: list[Document] = load_documents()

def get_document(doc_id: str) -> Document:
    return next(d for d in DOCUMENTS if d.doc_id == doc_id)


def get_document(doc_id: str) -> Document:
    return next(d for d in DOCUMENTS if d.doc_id == doc_id)


REQUIRED_TOPICS = [d.topic for d in DOCUMENTS]          # one doc per required brief topic


def sentences(text: str) -> list[str]:
    # Same splitter the sentence chunker uses, so the 2-5 check and the chunks agree.
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


if __name__ == "__main__":
    print(f"{len(DOCUMENTS)} documents loaded from {KB_DIR.name}/")
    bad = []
    for d in DOCUMENTS:
        n = len(sentences(d.text))
        ok = 2 <= n <= 5
        bad += [] if ok else [d.doc_id]
        print(f"  {d.source:<46} -> {d.doc_id}  {n} sentences  {'OK' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if len(DOCUMENTS) >= 12 and not bad else f"FAIL {bad}")