*************************
*PART-I
************************


I completed the Nykaa (E-commerce & Retail) track.

Telemetry: `config.py` sets `CREWAI_DISABLE_TELEMETRY`, `OTEL_SDK_DISABLED` and `CREWAI_TRACING_ENABLED` before anything imports CrewAI, so no outbound network call is made during `crew.kickoff()`. Import `config` before any CrewAI import in every entry point.

Mode: everything runs under `MOCK_LLM` (the default) with zero API keys and zero network access.
-----------------------------------------------------
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
----------------------------------------------------------
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
-----------------------------------------------------------------
Task 3 — Two chunking strategies, embedded and indexed
--------------------------------------------------------------
*calibrate_threshold.py output: 
Embedding model: all-MiniLM-L6-v2
kb_fixed    : 50 chunks indexed
kb_sentence : 60 chunks indexed

Sample query: "How long is the return window for footwear?"
kb_fixed top-3:
  0.422  kb-01  cosmetics cannot be resold for hygiene reasons. - The window is counte...
  0.418  kb-05  large Home appliances carry 6 months on the motor or heating element. ...
  0.416  kb-01  Nykaa accepts returns within a window that depends on what was bought,...
kb_sentence top-3:
  0.590  kb-01  - Apparel and Footwear may be returned within 14 days of delivery, Hom...
  0.478  kb-05  - Apparel and Footwear have no ongoing warranty, but a manufacturing d...
  0.376  kb-05  Warranty is separate from the return window and is honoured by the bra...
(.venv)  > & e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\.venv\Scripts\python.exe e:/Goole_Drive/Git/IITR_Capston_Project_Ecommerce/calibrate_threshold.py
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3306.04it/s]

=== kb_fixed ===
IN-SCOPE top-1 similarity
  0.422  How long is the return window for footwear?
  0.655  When will I get my COD refund?
  0.678  What is the delivery time for metro cities?
  0.747  Can I cancel an order after it has shipped?
  0.677  How much are 500 loyalty points worth?
OUT-OF-SCOPE top-1 similarity
  0.169  What is your CEO's salary?
  0.297  Book me a flight to Delhi tomorrow.
  0.156  Recommend a good laptop for gaming.
clusters separate: in-scope min 0.422 > out-of-scope max 0.297 (gap 0.125) -> threshold = midpoint = 0.359

=== kb_sentence ===
IN-SCOPE top-1 similarity
  0.590  How long is the return window for footwear?
  0.725  When will I get my COD refund?
  0.497  What is the delivery time for metro cities?
  0.727  Can I cancel an order after it has shipped?
  0.591  How much are 500 loyalty points worth?
OUT-OF-SCOPE top-1 similarity
  0.197  What is your CEO's salary?
  0.266  Book me a flight to Delhi tomorrow.
  0.148  Recommend a good laptop for gaming.
clusters separate: in-scope min 0.497 > out-of-scope max 0.266 (gap 0.231) -> threshold = midpoint = 0.382

Thresholds for config.SIMILARITY_THRESHOLD:
  {"kb_fixed": 0.359, "kb_sentence": 0.382}


*output of rag_core.py:-
Loading weights: 100%|███████████████████████| 103/103 [00:00<00:00, 4396.90it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Embedding model: all-MiniLM-L6-v2
kb_fixed    : 50 chunks indexed
kb_sentence : 60 chunks indexed

Sample query: "How long is the return window for footwear?"
kb_fixed top-3:
  0.422  kb-01  cosmetics cannot be resold for hygiene reasons. - The window is counte...
  0.418  kb-05  large Home appliances carry 6 months on the motor or heating element. ...
  0.416  kb-01  Nykaa accepts returns within a window that depends on what was bought,...
kb_sentence top-3:
  0.590  kb-01  - Apparel and Footwear may be returned within 14 days of delivery, Hom...
  0.478  kb-05  - Apparel and Footwear have no ongoing warranty, but a manufacturing d...
  0.376  kb-05  Warranty is separate from the return window and is honoured by the bra...

----------------------------------------------------------------------------
Task 4 — Grounded generation and your measured I don't know threshold
----------------------------------------------------------------------------
chosen threshold - as per calibration_threshold.py. -- Task-3. 
SIMILARITY_THRESHOLD = {
    COLLECTION_FIXED: 0.359,     # midpoint of in-scope min 0.422 and out-of-scope max 0.297
    COLLECTION_SENTENCE: 0.382,  # midpoint of in-scope min 0.497 and out-of-scope max 0.266
}

*output of grounded_demo.py

Q: How long is the return window for footwear?
   top-1 similarity 0.5898 vs threshold 0.382 -> grounded=True
   Based on Nykaa policy documents:
- (kb-01) - Apparel and Footwear may be returned within 14 days of delivery, Home items within 10 days, and Electronics within 7 days.
- (kb-05) - Apparel and Footwear have no ongoing warranty, but a manufacturing defect such as a split sole or a failed seam can be claimed within 30 days of delivery.

Q: When will I get my COD refund?
   top-1 similarity 0.7248 vs threshold 0.382 -> grounded=True
   Based on Nykaa policy documents:
- (kb-02) - If a COD refund has not arrived 10 business days after the quality check cleared, support raises a payment trace with the bank.
- (kb-06) - A prepaid cancellation is refunded in full to the original instrument within 3 to 5 business days, while a COD cancellation needs no refund because nothing was paid.
- (kb-10) - An approved claim is settled as a free replacement where stock exists, or as a full refund including any shipping fee, within 3 to 5 business days of approval.

Q: What is the delivery time for metro cities?
   top-1 similarity 0.497 vs threshold 0.382 -> grounded=True
   Based on Nykaa policy documents:
- (kb-03) - Metro pincodes are served in 2 to 3 business days, tier-2 and tier-3 towns in 4 to 6 business days, and remote or hill pincodes in 7 to 9 business days.

Q: Can I cancel an order after it has shipped?
   top-1 similarity 0.7268 vs threshold 0.382 -> grounded=True
   Based on Nykaa policy documents:
- (kb-06) A customer can cancel an order at no cost at any time while its status is still Placed.
- (kb-02) A cash-on-delivery order is paid in cash at the door, so its refund cannot be reversed to a card and must be sent to a bank account the customer confirms.

Q: How much are 500 loyalty points worth?
   top-1 similarity 0.5909 vs threshold 0.382 -> grounded=True
   Based on Nykaa policy documents:
- (kb-07) Customers earn 1 loyalty point for every Rs.100 of order value that is actually paid and retained.

Q: Book me a flight to Delhi tomorrow.
   top-1 similarity 0.2663 vs threshold 0.382 -> grounded=False
   I don't have grounded information to answer that. Please contact Nykaa customer support so a human can help.
--------------------------------------------------------
Task 5 — Precision/recall of both chunking strategies
--------------------------------------------------------
output of rage_compare.py

The win is narrow. Precision is 0.633 against 0.600, with equal recall, and the per-query results are mixed. Sentence chunks win on metro delivery (1.00 against 0.50) but lose on footwear and COD (0.33 against 0.50), because the top 4 sentences spread over neighbouring docs

=== kb_fixed  (top-4 chunks -> unique parent docs, cap 3) ===

Q: How long is the return window for footwear?
   chunks -> docs : ['kb-01', 'kb-05', 'kb-01', 'kb-01'] -> ['kb-01', 'kb-05']
   gold           : ['kb-01']   relevant retrieved: ['kb-01']
   precision = 1/2 = 0.50
   recall    = 1/1 = 1.00

Q: When will I get my COD refund?
   chunks -> docs : ['kb-02', 'kb-06', 'kb-02', 'kb-06'] -> ['kb-02', 'kb-06']
   gold           : ['kb-02']   relevant retrieved: ['kb-02']
   precision = 1/2 = 0.50
   recall    = 1/1 = 1.00

Q: What is the delivery time for metro cities?
   chunks -> docs : ['kb-03', 'kb-03', 'kb-03', 'kb-10'] -> ['kb-03', 'kb-10']
   gold           : ['kb-03']   relevant retrieved: ['kb-03']
   precision = 1/2 = 0.50
   recall    = 1/1 = 1.00

Q: Can I cancel an order after it has shipped?
   chunks -> docs : ['kb-06', 'kb-06', 'kb-06', 'kb-02'] -> ['kb-06', 'kb-02']
   gold           : ['kb-06']   relevant retrieved: ['kb-06']
   precision = 1/2 = 0.50
   recall    = 1/1 = 1.00

Q: How much are 500 loyalty points worth?
   chunks -> docs : ['kb-07', 'kb-07', 'kb-07', 'kb-07'] -> ['kb-07']
   gold           : ['kb-07']   relevant retrieved: ['kb-07']
   precision = 1/1 = 1.00
   recall    = 1/1 = 1.00

   mean precision = (0.50 + 0.50 + 0.50 + 0.50 + 1.00) / 5 = 0.600
   mean recall    = (1.00 + 1.00 + 1.00 + 1.00 + 1.00) / 5 = 1.000

=== kb_sentence  (top-4 chunks -> unique parent docs, cap 3) ===

Q: How long is the return window for footwear?
   chunks -> docs : ['kb-01', 'kb-05', 'kb-05', 'kb-11'] -> ['kb-01', 'kb-05', 'kb-11']
   gold           : ['kb-01']   relevant retrieved: ['kb-01']
   precision = 1/3 = 0.33
   recall    = 1/1 = 1.00

Q: When will I get my COD refund?
   chunks -> docs : ['kb-02', 'kb-06', 'kb-02', 'kb-10'] -> ['kb-02', 'kb-06', 'kb-10']
   gold           : ['kb-02']   relevant retrieved: ['kb-02']
   precision = 1/3 = 0.33
   recall    = 1/1 = 1.00

Q: What is the delivery time for metro cities?
   chunks -> docs : ['kb-03', 'kb-03', 'kb-03', 'kb-03'] -> ['kb-03']
   gold           : ['kb-03']   relevant retrieved: ['kb-03']
   precision = 1/1 = 1.00
   recall    = 1/1 = 1.00

Q: Can I cancel an order after it has shipped?
   chunks -> docs : ['kb-06', 'kb-06', 'kb-06', 'kb-02'] -> ['kb-06', 'kb-02']
   gold           : ['kb-06']   relevant retrieved: ['kb-06']
   precision = 1/2 = 0.50
   recall    = 1/1 = 1.00

Q: How much are 500 loyalty points worth?
   chunks -> docs : ['kb-07', 'kb-07', 'kb-07', 'kb-07'] -> ['kb-07']
   gold           : ['kb-07']   relevant retrieved: ['kb-07']
   precision = 1/1 = 1.00
   recall    = 1/1 = 1.00

   mean precision = (0.33 + 0.33 + 1.00 + 0.50 + 1.00) / 5 = 0.633
   mean recall    = (1.00 + 1.00 + 1.00 + 1.00 + 1.00) / 5 = 1.000

Higher P+R on this sample: kb_sentence

*************************
*PART-II
************************

