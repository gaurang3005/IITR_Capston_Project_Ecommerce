I completed the Nykaa (E-commerce & Retail) track.

Telemetry: `config.py` sets `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED` and `CREWAI_TRACING_ENABLED` before anything imports CrewAI, so no outbound network call is made during `crew.kickoff()`. Import `config` before any CrewAI import in every entry point.

Mode: everything runs under `MOCK_LLM` (the default) with zero API keys and zero network access.

Task 1 — deterministic, seeded Nykaa order dataset.
----------------------------------------------------
Design choices (seed 42 gives Apparel 7, Electronics 7, Home 10, Footwear 12, Beauty 12 -- these are the achieved counts, not the weights):

    seed=42, N=48,
    category weights Apparel .30 / Electronics .18 / Home .16 / Footwear .16 / Beauty .20,
    status weights Placed .25 / Shipped .25 / Delivered .30 / Returned .10 / Refunded .10,
    value range Rs.199-49,999, days_since_created ~ U{0..30}, p(delayed_shipment)=0.20

Price-range reasoning: Rs.199 to Rs.49,999 spans low-cost beauty and apparel items up to higher-priced electronics and footwear, so the values look like a mixed e-commerce catalogue.

Nothing is hand-edited; changing the seed re-rolls everything.

Reproduce: `python dataset.py` (prints the report below).

Output:

    ORDERS: 48 records (seed=42)

    Count per category (need >=3 each):
    Apparel        7  OK
    Electronics    7  OK
    Home          10  OK
    Footwear      12  OK
    Beauty        12  OK

    Count per status (need >=1 each):
    Placed        11  OK
    Shipped       14  OK
    Delivered     11  OK
    Returned       7  OK
    Refunded       5  OK

    Delayed shipments: 10/48 = 20.8%  (band 10-30%)

    RESULT: PASS

    First record: {'record_id': 'ORD0001', 'category': 'Home', 'status': 'Placed', 'order_value_inr': 18223, 'days_since_created': 7, 'delayed_shipment': False}
