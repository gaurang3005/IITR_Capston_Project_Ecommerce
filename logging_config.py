"""Task 12 — exactly one JSON-Lines entry per request, with trace id and timing."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import config
from guardrails import mask_pii          # the same masker the model sees (brief requirement)

config.LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = config.LOG_DIR / "app.jsonl"

_logger = logging.getLogger("nykaa.requests")
_logger.setLevel(logging.INFO)
_logger.propagate = False
if not _logger.handlers:
    _h = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_h)


def _scrub(obj):
    """Defence in depth: re-mask every string before it reaches disk."""
    if isinstance(obj, str):
        return mask_pii(obj)[0]
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub(v) for v in obj]
    return obj


def log_request(**fields) -> dict:
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **_scrub(fields)}
    _logger.info(json.dumps(entry, ensure_ascii=False))
    for h in _logger.handlers:
        h.flush()
    return entry
