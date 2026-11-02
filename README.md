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

Reproduce: `python dataset.py ('prints the report below).

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

Task-2. Write your knowledge base : checks 
----------------------------------------------------------
Loads the folder: it reads every .md file in sorted order and skips # title lines. It joins the text into one paragraph and reads the files as UTF-8, so ₹ stays intact.
Assigns ids: the file name sets the id and topic, so 01_return_window_by_category.md becomes doc_id = "kb-01" with topic = "return_window_by_category"-> for use in later task. 
Catches mistakes: a badly named file, an empty file or a repeated number stops it with a clear error.
self-check: python knowledge_base.py checks that there are at least 12 docs with 2–5 sentences each, and prints PASS or FAIL
output:-
12 documents loaded from knowledge_base/
  01_return_window_by_category.md                -> kb-01  5 sentences  OK
  02_cod_refund_timelines.md                     -> kb-02  5 sentences  OK
  03_delivery_slas.md                            -> kb-03  5 sentences  OK
  04_reverse_pickup_eligibility.md               -> kb-04  5 sentences  OK
  05_warranty_terms_by_category.md               -> kb-05  5 sentences  OK
  06_order_cancellation_policy.md                -> kb-06  5 sentences  OK
  07_loyalty_points_redemption.md                -> kb-07  5 sentences  OK
  08_payment_failure_retry.md                    -> kb-08  5 sentences  OK
  09_size_exchange_policy.md                     -> kb-09  5 sentences  OK
  10_damaged_item_claim_process.md               -> kb-10  5 sentences  OK
  11_international_shipping_restrictions.md      -> kb-11  5 sentences  OK
  12_support_escalation_matrix.md                -> kb-12  5 sentences  OK