"""Tasks 11 + 12 transcript: HTTP endpoints with server-issued sessions, WebSocket multi-turn and
disconnect, then a read-back of logs/app.jsonl: one entry per request, trace id + timing, no raw PII."""
import json

from fastapi.testclient import TestClient

import config
from api import app

LOG = config.LOG_DIR / "app.jsonl"
LOG.write_text("", encoding="utf-8")                 # fresh log for this demo
sent = {"http": 0, "ws": 0}                          # requests sent, to check one log line each


def call(method: str, path: str, body: dict | None = None):
    sent["http"] += 1
    return c.get(path) if method == "GET" else c.post(path, json=body)


with TestClient(app) as c:                           # `with` also runs the startup hook (index check)
    print("GET /health ->", call("GET", "/health").json())

    print("\nHTTP sessions are server-issued")
    a1 = call("POST", "/ask", {"message": "What is the return window for footwear?"}).json()
    a2 = call("POST", "/ask", {"message": "And what about Beauty?", "session_id": a1["session_id"]}).json()
    print(f"   customer A, turn 1 -> {a1['outcome']}, session_id={a1['session_id']}, sources={a1['sources']}")
    print(f"   customer A, turn 2 'And what about Beauty?' (same session_id) -> sources={a2['sources']}")
    b1 = call("POST", "/ask", {"message": "Check order ORD0039 please"}).json()
    c1 = call("POST", "/ask", {"message": "Is it delayed?"}).json()       # no session_id: new conversation
    print(f"   customer B 'Check order ORD0039 please' -> sources={b1['sources']}")
    print(f"   customer C 'Is it delayed?' (no session_id) -> sources={c1['sources']}, "
          f"sees B's order: {'ORD0039' in c1['answer']}")
    r = call("POST", "/ask", {"message": "Is it delayed?", "session_id": "http-guessed-id"})
    print("   session_id the server never issued ->", r.status_code, r.json()["detail"])

    print("\nOther HTTP cases")
    r = call("POST", "/ask", {"message": "Call me on 9876543210 about order ORD0029, card ending 4321"})
    print("   POST /ask (PII) ->", r.status_code, r.json()["outcome"], "pii_masked =", r.json()["pii_masked"])
    r = call("POST", "/ask", {"message": ""})
    print("   POST /ask (empty) ->", r.status_code, "(Pydantic validation)")
    r = call("POST", "/ask", {"message": "x " * 4000})
    print("   POST /ask (oversized) ->", r.status_code, r.json()["detail"][:90])
    doc = {"doc_id": "kb-13", "topic": "gift_wrap_policy",
           "text": "Gift wrapping costs ₹49 per order. It is available for Beauty and Apparel only."}
    r = call("POST", "/add-document", doc)
    print("   POST /add-document ->", r.status_code, r.json())
    r = call("POST", "/add-document", doc)
    print("   POST /add-document (same doc_id again) ->", r.status_code, r.json()["detail"])

    print("\nWebSocket: client 1 does two turns (memory), sends a bad frame, then disconnects")
    with c.websocket_connect("/ws/chat") as ws:
        print("  ", ws.receive_json())
        ws.send_json({"message": "What is the return window for footwear?"}); sent["ws"] += 1
        print("   turn 1:", ws.receive_json()["sources"])
        ws.send_json({"message": "And what about Beauty?"}); sent["ws"] += 1
        print("   turn 2:", ws.receive_json()["sources"])
        ws.send_text("not json"); sent["ws"] += 1
        print("   bad frame:", ws.receive_json())
    print("   client 1 closed the socket")
    with c.websocket_connect("/ws/chat") as ws2:
        ws2.receive_json()
        ws2.send_json({"message": "Check order ORD0039"}); sent["ws"] += 1
        print("   client 2 still served:", ws2.receive_json()["outcome"])

lines = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
http = [x for x in lines if x["channel"] == "http"]
ws_messages = [x for x in lines if x["channel"] == "ws"
               and x.get("event") not in ("ws_disconnect", "ws_send_after_close")]
print(f"\nlogs/app.jsonl: {len(lines)} lines")
print(f"   HTTP requests sent: {sent['http']} -> HTTP log lines: {len(http)}")
print(f"   WebSocket messages sent: {sent['ws']} -> WS message log lines: {len(ws_messages)}, "
      f"disconnect lines: {sum(x.get('event') == 'ws_disconnect' for x in lines)}")
print("   every line has trace_id and timing:",
      all("trace_id" in x and ("latency_ms" in x or "duration_ms" in x) for x in lines))
print("   HTTP status codes logged:", sorted({x["status_code"] for x in http}))
raw = LOG.read_text(encoding="utf-8")
print("   raw '9876543210' on disk:", "9876543210" in raw, "| raw 'ending 4321' on disk:", "ending 4321" in raw)
pii = next(x for x in http if "PHONE_REDACTED" in (x.get("query_masked") or ""))
print("   masked line:", json.dumps({k: pii[k] for k in ("trace_id", "method", "path", "status_code",
                                                         "query_masked", "pii_counts", "outcome", "latency_ms")}))
