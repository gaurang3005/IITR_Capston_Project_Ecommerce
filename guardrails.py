"""Task 10 — input PII masking, prompt-injection detection, output groundedness (+ output PII scan).

mask_pii() is the project's only PII masker: anything that needs masked text calls it.
"""
from __future__ import annotations

import re

# ---- PII: fixed-format fields only (brief scope) -------------------------------
# Indian mobile: optional +91/91/0 prefix, 10 digits starting 6-9, optional space/hyphen groups.
# (?<![\d+]) / (?!\d) stop it matching inside a longer digit run (amounts, card numbers)
# or straight after a '+' that does not start +91 (a foreign number).
_PHONE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
                    r"|(?<![\d+])(?:\+?91[\s-]?|0)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
# Card last-4 in the phrasings customers actually use; group(1) is the 4 digits.
_CARD4 = re.compile(r"(?i)(?:ending(?:\s+(?:in|with))?|last\s*(?:4|four)(?:\s*digits?)?|x{4}|\*{4})"
                    r"[\s:\-]*(\d{4})(?!\d)")
# Best practice beyond the brief: a full card number (13-19 digits) must never reach a log either.
_PAN = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

PHONE_TOKEN, CARD_TOKEN, PAN_TOKEN = "[PHONE_REDACTED]", "[CARD_LAST4_REDACTED]", "[CARD_NUMBER_REDACTED]"
OUT_OF_SCOPE_PII = ("customer name", "delivery address")   # free text: acknowledged, not masked


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Return (masked_text, counts). Counts only: raw values are never stored anywhere."""
    counts = {"card_number": 0, "card_last4": 0, "phone": 0}

    def _card4(m: re.Match) -> str:                  # keep the cue ("ending in", "last 4"), mask the digits
        counts["card_last4"] += 1
        return m.group(0)[: m.start(1) - m.start(0)] + CARD_TOKEN

    def _count(field: str, token: str):
        def _r(_m: re.Match) -> str:
            counts[field] += 1
            return token
        return _r

    text = _PAN.sub(_count("card_number", PAN_TOKEN), text)   # longest pattern first
    text = _CARD4.sub(_card4, text)
    text = _PHONE.sub(_count("phone", PHONE_TOKEN), text)
    return text, counts


def contains_pii(text: str) -> bool:
    return any(p.search(text) for p in (_PAN, _CARD4, _PHONE))


# ---- prompt injection -----------------------------------------------------------
_INJECTION = re.compile("|".join([
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts?)",
    r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above|system)",
    r"(?:reveal|show|print|repeat|leak)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+|hidden\s+)?(?:prompt|instructions)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"\bdo\s+anything\s+now\b|\bjailbreak\b|\bdeveloper\s+mode\b",
    r"pretend\s+(?:you\s+are|to\s+be)\b",
    r"(?:override|bypass)\s+(?:your\s+|the\s+)?(?:rules|policy|policies|guardrails|safety)",
    r"^\s*(?:system|assistant)\s*:",               # forged role turn at line start
]), re.IGNORECASE | re.MULTILINE)


def detect_injection(text: str) -> dict:
    m = _INJECTION.search(text)
    return {"is_injection": bool(m), "pattern": m.group(0) if m else None}


def guard_request(user_text: str) -> dict:
    """Single input funnel. Everything downstream sees only the masked `text`."""
    masked, counts = mask_pii(user_text)
    inj = detect_injection(masked)
    return {"text": masked, "pii_counts": counts, "pii_masked": any(counts.values()),
            "injection": inj, "blocked": inj["is_injection"]}


# ---- output groundedness ----------------------------------------------------------
_STOP = set("""a an the is are was were be been to of and or for in on at by with from as it its this
that these those you your we our us i my me can may will shall do does did how what when which who
whom why if then than so not no only also any all each per based nykaa policy documents""".split())
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _split_sentences(text: str) -> list[str]:
    """Split on sentence ends and newlines; fragments with < 2 content words are skipped, not checked."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -") for p in parts if len(_tokens(p)) >= 2]


def sentence_supported(sentence: str, context: str, min_overlap: float = 0.6) -> bool:
    """Supported = every number in it appears in context AND >= min_overlap of its content words do."""
    ctx_nums = set(_NUM.findall(context))
    if any(n not in ctx_nums for n in _NUM.findall(sentence)):
        return False                                    # an invented number is never grounded
    toks = _tokens(sentence)
    return bool(toks) and len(toks & _tokens(context)) / len(toks) >= min_overlap


def output_guard(answer: str, contexts: list[str], min_overlap: float = 0.6) -> dict:
    """Refuse (grounded=False) unless the answer has a checkable sentence, every one is supported
    by the evidence (policy chunks + order record), and the answer contains no PII."""
    ctx = "\n".join(contexts)
    sents = _split_sentences(answer)
    unsupported = [s for s in sents if not sentence_supported(s, ctx, min_overlap)]
    ratio = 1 - len(unsupported) / len(sents) if sents else 0.0
    return {"grounded": bool(sents) and not unsupported and not contains_pii(answer),
            "supported_ratio": round(ratio, 3), "unsupported": unsupported,
            "pii_in_output": contains_pii(answer)}
