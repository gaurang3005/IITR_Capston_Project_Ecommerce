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

---------------------------------------------------------------
Task 6 — Tools.py --> check_order_status with a designed escalation score
---------------------------------------------------------------
I recommend escalation when the score is at or above 0.87. That is a delayed order at the 75th percentile of days_since_created in my data (20.25 days): 0.6 + 0.4 * (20.25 / 30) = 0.87. On-time orders cannot exceed 0.40, so the threshold escalates 2 of the 10 delayed orders (ORD0029 and ORD0039) instead of all 10 (transcripts/06_order_tool.txt). 


age              = clip(days_since_created, 0, 30) / 30
escalation_score = round(0.6 * delayed_shipment + 0.4 * age, 4)      # delayed_shipment as 1 or 0

*output of my tool.py

days_since_created sorted: [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 9, 9, 11, 12, 14, 14, 16, 16, 16, 17, 18, 18, 19, 20, 20, 20, 20, 21, 23, 23, 24, 24, 24, 27, 27, 27, 28, 29, 29]
p75(days) = 20.25
ESCALATION_THRESHOLD = 0.6 + 0.4 * (20.25 / 30) = 0.87
delayed orders: 10; escalated by the score: 2 -> ['ORD0029', 'ORD0039']
ORD0029 {"found": true, "record_id": "ORD0029", "status": "Placed", "category": "Home", "order_value_inr": 24703, "days_since_created": 24, "delayed_shipment": true, "escalation_score": 0.92, "escalation_threshold": 0.87, "recommend_escalation": true}
ORD0039 {"found": true, "record_id": "ORD0039", "status": "Placed", "category": "Footwear", "order_value_inr": 4616, "days_since_created": 28, "delayed_shipment": true, "escalation_score": 0.9733, "escalation_threshold": 0.87, "recommend_escalation": true}
ORD0022 {"found": true, "record_id": "ORD0022", "status": "Returned", "category": "Footwear", "order_value_inr": 10215, "days_since_created": 20, "delayed_shipment": true, "escalation_score": 0.8667, "escalation_threshold": 0.87, "recommend_escalation": false}
ORD0001 {"found": true, "record_id": "ORD0001", "status": "Placed", "category": "Home", "order_value_inr": 18223, "days_since_created": 7, "delayed_shipment": false, "escalation_score": 0.0933, "escalation_threshold": 0.87, "recommend_escalation": false}
ORD9999 {"found": false, "record_id": "ORD9999", "error": "no such order"}


---------------------------------------------------------------
Task-7 Build a CrewAI crew
---------------------------------------------------------------
A three-agent crew run with kickoff(); a mock ReAct LLM drives it offline

main files: crew_builder.py, mock_llm.py, crew_demo.py, schemas.py

In, Mock_llm.py, 
1. _last_observation() reads tool output only from assistant messages, so the ReAct template's fake Observation: (in the system message) can never be read as a result.
2. Routing keeps no state: one mock instance can serve every agent, and nothing from one agent's steps leaks into the next agent's decision.
3. Dispatch reads the declared parameter names (_declared_tools() parses Tool Arguments:), so the substring lookup in rag_lookup is irrelevant.
4. _current_task() trims the prompt scaffolding and _question() takes the CUSTOMER_QUESTION: line, so a tool receives the customer's question, not the prompt.

*output of crew_demo.py

Q: How long is the return window for footwear?
   tools invoked via kickoff(): ['rag_lookup']
   CrewResponse (validated): {"answer":"- Apparel and Footwear may be returned within 14 days of delivery, Home items within 10 days, and Electronics within 7 days. - Apparel and Footwear have no ongoing warranty, but a manufacturing defect such as a split sole or a failed seam can be claimed within 30 days of delivery.","sources":["kb-01","kb-05"],"grounded":true,"escalated":false,"confidence":0.8}

Q: What is the status of order ORD0029?
   tools invoked via kickoff(): ['rag_lookup', 'check_order_status']
   CrewResponse (validated): {"answer":"Order ORD0029 (Home, order value 24703 INR) is Placed, created 24 days ago, and the shipment is delayed. This order needs escalation: escalation score 0.92 is above the threshold 0.87.","sources":["ORD0029"],"grounded":true,"escalated":true,"confidence":0.6}

Q: Where is my order ORD9999?
   tools invoked via kickoff(): ['rag_lookup', 'check_order_status']
   CrewResponse (validated): {"answer":"No such order ORD9999 was found.","sources":[],"grounded":false,"escalated":true,"confidence":0.6}

Task 9 negative test (the validator is load-bearing):
   REJECTED: 2 validation errors -> [('answer', 'string_too_short'), ('confidence', 'less_than_equal')]


*-The Lookup agent calls check_order_status (it declares record_id) only when the question carries an order id, and otherwise answers NO_ORDER_ID_GIVEN. A name-substring router would have sent the order id to rag_lookup. An unknown order (ORD9999) is reported as not found and marked escalated, so a human checks it.


----------------------------------------------------------------------------
Task-8 Session memory with RunnableWithMessageHistory, around the real agent
-----------------------------------------------------------------------------
LangChain session memory: a follow-up is understood within its session and not in a fresh one

file: memory_chat.py, memory_demo.py, agent_service.py

*output of memory_demo.py
=== Session A (multi-turn) ===
e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\agent_service.py:24: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
  chat = build_chat(lambda text, hist: _answer(text, hist, state))
e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\memory_chat.py:18: LangChainDeprecationWarning: The class `InMemoryChatMessageHistory` was deprecated in LangChain 1.6.4 and will be removed in 2.0.0 See the short-term memory documentation for recommended alternatives: https://docs.langchain.com/oss/python/langchain/short-term-memory
  return _store.setdefault(session_id, InMemoryChatMessageHistory())
Loading weights: 100%|█████████████████████████| 103/103 [00:00<00:00, 7372.87it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
user: What is the return window for footwear?   [resolved: What is the return window for footwear?]
  agent: - Apparel and Footwear may be returned within 14 days of delivery, Home items within 10 days, and Electronics within 7 days. - Apparel and Footwear have no ongo
  sources: ['kb-01', 'kb-05']
e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\agent_service.py:24: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
  chat = build_chat(lambda text, hist: _answer(text, hist, state))
user: And what about Beauty?   [resolved: What is the return window for Beauty?]
  agent: - Beauty products are returnable within 7 days only while the seal, shrink-wrap and batch sticker are intact, since opened cosmetics cannot be resold for hygien
  sources: ['kb-01', 'kb-04', 'kb-10', 'kb-05']
user: Check order ORD0039 please   [resolved: Check order ORD0039 please]
  agent: Order ORD0039 (Footwear, order value 4616 INR) is Placed, created 28 days ago, and the shipment is delayed. This order needs escalation: escalation score 0.9733
  sources: ['ORD0039']
user: Is it delayed?   [resolved: Is it delayed? (order ORD0039)]
  agent: Order ORD0039 (Footwear, order value 4616 INR) is Placed, created 28 days ago, and the shipment is delayed. This order needs escalation: escalation score 0.9733
  sources: ['ORD0039', 'kb-03', 'kb-02']
session-A history: 8 messages

=== Session B (fresh conversation, same follow-up words) ===
user: And what about Beauty?   [resolved: And what about Beauty?]
  agent: - Beauty is the one category where a seal-broken or leaking unit is accepted back despite the hygiene rule, because the fault is ours. - Beauty products carryn
  sources: ['kb-10', 'kb-05']
user: Is it delayed?   [resolved: Is it delayed?]
  agent: - An order is flagged as a delayed shipment once it crosses its promised window without a delivery scan.
  sources: ['kb-03']
session-B history: 4 messages (no footwear, no ORD0039)

----------------------------------------------------------------------------
Task-9: Add structured output schema. 
-----------------------------------------------------------------------------

Every crew answer must validate against `CrewResponse` in `schemas.py`: `answer` (non-empty), `sources`, `grounded`, `escalated` and `confidence` (0–1). The Composer's task sets `output_pydantic=CrewResponse`, and `run_crew()` validates the result again in code, so a bad response raises `ValidationError`. 

task-7 shows an invalid payload rejected with 2 validation errors.

----------------------------------------------------------------------------
Task-10: Add guardrails
-----------------------------------------------------------------------------
Three guards, each demonstrated firing on a deliberate test case. All of them  in guardrails.py.

1)Input — PII masking (fixed-format fields only) 2). Input — prompt-injection detection 3). Output — groundedness gate

- The refusal text for a blocked request added in a new section at the end of config.py:
- crew_builder.py--> run_crew() now returns the Retrieval and Lookup task outputs
- agent_service.py masks and screens the input before memory sees it, and checks the answer before it is returned. It also reports an outcome for every request. 


*output of guardrails_demo.py
1) INPUT PII MASKING (phone + card last-4; name/address declared out of scope)
   model sees: Hi, I'm Test User. Call me on [PHONE_REDACTED]; my card ending [CARD_LAST4_REDACTED] was charged twice. 
   counts: {'card_number': 0, 'card_last4': 1, 'phone': 1}

2) PROMPT-INJECTION DETECTION
   {'is_injection': True, 'pattern': 'Ignore all previous instructions'}
   pipeline outcome: blocked_injection | answer: I can't help with that request.
   false positive on a customer named Dan: False

3) OUTPUT GROUNDEDNESS
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3641.66it/s]
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
   supported  : {'grounded': True, 'supported_ratio': 1.0, 'unsupported': [], 'pii_in_output': False}
   invented   : {'grounded': False, 'supported_ratio': 0.0, 'unsupported': ['You may return electronics within 90 days for any reason.'], 'pii_in_output': False} -> refused
e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\agent_service.py:39: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
  chat = build_chat(lambda text, hist: _answer(text, hist, state))
e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\memory_chat.py:18: LangChainDeprecationWarning: The class `InMemoryChatMessageHistory` was deprecated in LangChain 1.6.4 and will be removed in 2.0.0 See the short-term memory documentation for recommended alternatives: https://docs.langchain.com/oss/python/langchain/short-term-memory
  return _store.setdefault(session_id, InMemoryChatMessageHistory())
   out-of-scope via pipeline: refused_ungrounded | I don't have grounded information to answer that. Please contact Nykaa customer support so a human can help.


*************************
*PART-III
************************

----------------------------------------------------------------------------
Task-11: Deploy via FastAPI (2+ HTTP endpoints + 1 WebSocket)
-----------------------------------------------------------------------------
file: api.py ; api_demo.py

- session_id is created per WebSocket connection, not passed by the client. That is deliberate: a client cannot hijack or collide with another conversation's memory.
- /add-document is in-process only: it appends to DOCUMENTS and re-indexes, but does not write a file, so a restart reloads knowledge_base/ without the new doc. (chroma_store keeps its chunks on disk until the next re-index.) The durable path is a new knowledge_base/<NN>_<topic>.md file plus a re-index.
- AskRequest.message has max_length=50_000, only as an abuse cap. The real control on request size is the token budget in Task 15, and the cap has to stay above it (a 2,000-character cap)
- A malformed WebSocket frame (bad JSON, a binary frame, or JSON that is not an object) gets an error reply, and the chat continues. So does a turn that fails. A disconnect, or a send after the client vanished (RuntimeError), ends only that client's handler. The server keeps serving everyone else.

*output of api_demo.py

   GET /health -> {'status': 'ok', 'documents': 12, 'mock_llm': True}
   e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\agent_service.py:39: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
   chat = build_chat(lambda text, hist: _answer(text, hist, state))
   e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\memory_chat.py:18: LangChainDeprecationWarning: The class `InMemoryChatMessageHistory`was deprecated in LangChain 1.6.4 and will be removed in 2.0.0 See the short-term memory documentation for recommended alternatives: https://docs.langchain.com/oss/python/langchain/short-term-memory
   return _store.setdefault(session_id, InMemoryChatMessageHistory())
   Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 4671.27it/s]
   POST /ask -> 200 {'outcome': 'answered', 'sources': ['kb-07'], 'grounded': True}
   e:\Goole_Drive\Git\IITR_Capston_Project_Ecommerce\agent_service.py:39: LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.
   chat = build_chat(lambda text, hist: _answer(text, hist, state))
   POST /ask (PII) -> 200 answered pii_masked= True
   POST /ask (empty) -> 422 (Pydantic validation)
   POST /add-document -> 200 {'doc_id': 'kb-13', 'chunks_indexed': {'kb_fixed': 51, 'kb_sentence': 62}}

   WebSocket: client 1 does two turns (memory), then disconnects mid-conversation
      {'type': 'hello', 'session_id': 'ws-6173258e'}
      turn1: ['kb-01', 'kb-05']
      turn2: ['kb-01', 'kb-04', 'kb-10', 'kb-13']
      bad frame: {'type': 'error', 'detail': 'send JSON like {"message": "..."}'}
      client 1 closed the socket
      client 2 still served: answered

*output of ws_client.py

   run on server: uvicorn api:app --port 8000

   c1 {'type': 'hello', 'session_id': 'ws-be2303ea'}
   c1 answer: answered
   c1 disconnected mid-conversation
   c2 answer: answered

The second client was still served after the first one vanished mid-turn. That is the "survives a disconnect" evidence.

