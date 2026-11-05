"""Task 2 — the Nykaa policy knowledge base, loaded from the knowledge_base/ folder.

- Loads the folder: reads every .md file in sorted order as UTF-8 (so ₹ stays intact), skips # title
  lines and joins the rest into one paragraph.
- Assigns ids: the file name sets the id and topic, so 01_return_window_by_category.md becomes
  doc_id="kb-01" with topic="return_window_by_category".
- Catches mistakes: a badly named file, an empty file or a repeated number stops it with a clear error.
- Self-check (python knowledge_base.py): at least 12 documents, 2-5 sentences each, and a document for
  every topic the brief requires; prints PASS or FAIL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
_FILE_RE = re.compile(r"^(\d{2,})_([a-z0-9_]+)$")         # stem -> (number, topic)

# The brief's 12 required topics -> the file topic that must cover each one.
REQUIRED_TOPICS = {
    "Return window by product category": "return_window_by_category",
    "COD refund timelines": "cod_refund_timelines",
    "Delivery SLAs": "delivery_slas",
    "Reverse-pickup eligibility": "reverse_pickup_eligibility",
    "Warranty terms by category": "warranty_terms_by_category",
    "Order-cancellation policy": "order_cancellation_policy",
    "Loyalty-points redemption policy": "loyalty_points_redemption",
    "Payment-failure/retry policy": "payment_failure_retry",
    "Size-exchange policy": "size_exchange_policy",
    "Damaged-item claim process": "damaged_item_claim_process",
    "International shipping restrictions": "international_shipping_restrictions",
    "Customer-support escalation matrix": "support_escalation_matrix",
}


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

    print("\nRequired topics (from the brief):")
    by_topic = {d.topic: d.doc_id for d in DOCUMENTS}
    missing = [name for name, topic in REQUIRED_TOPICS.items() if topic not in by_topic]
    for name, topic in REQUIRED_TOPICS.items():
        print(f"  {name:<38} -> {by_topic.get(topic, 'MISSING')}")

    ok = len(DOCUMENTS) >= 12 and not bad and not missing
    print("\nRESULT:", "PASS" if ok else f"FAIL sentences={bad} missing_topics={missing}")
