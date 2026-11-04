"""Task 6 — check_order_status and its designed escalation score."""
from __future__ import annotations

import json
from typing import Any

from dataset import ORDERS

# ---- escalation score (Task 6) ----------------------------------------------------
W_DELAY, W_AGE = 0.6, 0.4
DAYS_CAP = 30                                  # brief: days_since_created is 0-30
AGE_PERCENTILE = 0.75


def _percentile(values: list[int], p: float) -> float:
    """Linear interpolation (same as numpy's default), no numpy needed."""
    xs = sorted(values)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    """0.6 * delayed + 0.4 * (clamp(days, 0, 30) / 30). Delayed: 0.6-1.0, on-time: 0.0-0.4."""
    age = min(max(days_since_created, 0), DAYS_CAP) / DAYS_CAP
    return round(W_DELAY * int(delayed_shipment) + W_AGE * age, 4)


P75_DAYS = _percentile([o["days_since_created"] for o in ORDERS], AGE_PERCENTILE)
ESCALATION_THRESHOLD = round(W_DELAY + W_AGE * P75_DAYS / DAYS_CAP, 4)
ORDERS_BY_ID = {o["record_id"]: o for o in ORDERS}       # O(1) lookup by record id


def check_order_status(record_id: str) -> dict[str, Any]:
    """Look up one order; return status, value and the designed escalation score."""
    order = ORDERS_BY_ID.get(record_id.strip().upper())
    if order is None:
        return {"found": False, "record_id": record_id, "error": "no such order"}
    score = escalation_score(order["delayed_shipment"], order["days_since_created"])
    return {"found": True, "record_id": order["record_id"], "status": order["status"],
            "category": order["category"], "order_value_inr": order["order_value_inr"],
            "days_since_created": order["days_since_created"],
            "delayed_shipment": order["delayed_shipment"], "escalation_score": score,
            "escalation_threshold": ESCALATION_THRESHOLD,
            "recommend_escalation": score >= ESCALATION_THRESHOLD}


if __name__ == "__main__":
    ages = sorted(o["days_since_created"] for o in ORDERS)
    print(f"days_since_created sorted: {ages}")
    print(f"p75(days) = {P75_DAYS:.2f}")
    print(f"ESCALATION_THRESHOLD = 0.6 + 0.4 * ({P75_DAYS:.2f} / 30) = {ESCALATION_THRESHOLD}")
    delayed = [o for o in ORDERS if o["delayed_shipment"]]
    flagged = [o["record_id"] for o in delayed
               if escalation_score(True, o["days_since_created"]) >= ESCALATION_THRESHOLD]
    print(f"delayed orders: {len(delayed)}; escalated by the score: {len(flagged)} -> {flagged}")
    for rid in flagged + ["ORD0022", "ORD0001", "ORD9999"]:
        print(rid, json.dumps(check_order_status(rid)))
