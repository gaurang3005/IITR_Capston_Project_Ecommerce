"""Task 1 — deterministic, seeded Nykaa order dataset.

Design choices (also in README.md). With seed 42 the achieved category counts are Apparel 7,
Electronics 7, Home 10, Footwear 12, Beauty 12; the weights are only expectations:
  seed=42, N=48,
  category weights Apparel .30 / Electronics .18 / Home .16 / Footwear .16 / Beauty .20,
  status weights Placed .25 / Shipped .25 / Delivered .30 / Returned .10 / Refunded .10,
  value range Rs.199-49,999, days_since_created ~ U{0..30}, p(delayed_shipment)=0.20.
Nothing here is hand-edited; changing the seed re-rolls everything.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

# ---- the dataset design: these values alone reproduce the dataset -----------
SEED = 42
N_ORDERS = 48
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_WEIGHTS = [0.30, 0.18, 0.16, 0.16, 0.20]   # order matches CATEGORIES
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
STATUS_WEIGHTS = [0.25, 0.25, 0.30, 0.10, 0.10]
AMOUNT_MIN_INR, AMOUNT_MAX_INR = 199, 49_999
DAYS_MIN, DAYS_MAX = 0, 30
P_DELAYED = 0.20

# Structural floors from the brief.
MIN_PER_CATEGORY = 3
MIN_PER_STATUS = 1
DELAY_BAND = (10.0, 30.0)          # inclusive percent


def _build_orders() -> list[dict[str, Any]]:
    """Build the dataset. One RNG, fixed call order => byte-stable output."""
    rng = random.Random(SEED)
    orders: list[dict[str, Any]] = []
    for i in range(N_ORDERS):
        # The draw order is fixed: reordering these calls would produce a different dataset.
        category = rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]
        status = rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]
        value = rng.randint(AMOUNT_MIN_INR, AMOUNT_MAX_INR)
        days = rng.randint(DAYS_MIN, DAYS_MAX)
        delayed = rng.random() < P_DELAYED

        orders.append({
            "record_id": f"ORD{i + 1:04d}",
            "category": category,
            "status": status,
            "order_value_inr": value,
            "days_since_created": days,
            "delayed_shipment": delayed,
        })
    return orders


ORDERS: list[dict[str, Any]] = _build_orders()


# ---------------------------------------------------------------------------
# Validation / reporting
# ---------------------------------------------------------------------------
def validate() -> dict[str, Any]:
    """Check every structural threshold and return a report. Never edits data."""
    cat_counts = Counter(o["category"] for o in ORDERS)
    status_counts = Counter(o["status"] for o in ORDERS)
    delayed = sum(1 for o in ORDERS if o["delayed_shipment"])
    delayed_pct = 100.0 * delayed / len(ORDERS)

    problems: list[str] = []
    if len(ORDERS) < 40:
        problems.append(f"need >=40 orders, have {len(ORDERS)}")
    for c in CATEGORIES:
        if cat_counts.get(c, 0) < MIN_PER_CATEGORY:
            problems.append(f"category {c}: {cat_counts.get(c, 0)} < {MIN_PER_CATEGORY}")
    for s in STATUSES:
        if status_counts.get(s, 0) < MIN_PER_STATUS:
            problems.append(f"status {s}: {status_counts.get(s, 0)} < {MIN_PER_STATUS}")
    if not (DELAY_BAND[0] <= delayed_pct <= DELAY_BAND[1]):
        problems.append(f"delayed {delayed_pct:.1f}% outside {DELAY_BAND}")

    return {
        "n": len(ORDERS),
        "per_category": {c: cat_counts.get(c, 0) for c in CATEGORIES},
        "per_status": {s: status_counts.get(s, 0) for s in STATUSES},
        "delayed_count": delayed,
        "delayed_pct": round(delayed_pct, 1),
        "problems": problems,
        "ok": not problems,
    }


if __name__ == "__main__":
    rep = validate()
    print(f"ORDERS: {rep['n']} records (seed={SEED})")
    print("\nCount per category (need >=3 each):")
    for c, n in rep["per_category"].items():
        print(f"  {c:<12} {n:>3}  {'OK' if n >= MIN_PER_CATEGORY else 'FAIL'}")
    print("\nCount per status (need >=1 each):")
    for s, n in rep["per_status"].items():
        print(f"  {s:<12} {n:>3}  {'OK' if n >= 1 else 'FAIL'}")
    print(f"\nDelayed shipments: {rep['delayed_count']}/{rep['n']} "
          f"= {rep['delayed_pct']}%  (band 10-30%)")
    print("\nRESULT:", "PASS" if rep["ok"] else f"FAIL {rep['problems']}")
    print("\nFirst record:", ORDERS[0])
