"""Task 11 transcript: HTTP endpoints, WebSocket multi-turn + disconnect."""
from fastapi.testclient import TestClient
from api import app

c = TestClient(app)

print("GET /health ->", c.get("/health").json())
r = c.post("/ask", json={"message": "How much are 500 loyalty points worth?"})
print("POST /ask ->", r.status_code, {k: r.json()[k] for k in ("outcome", "sources", "grounded")})
r = c.post("/ask", json={"message": "Call me on 9876543210 about order ORD0029, card ending 4321"})
print("POST /ask (PII) ->", r.status_code, r.json()["outcome"], "pii_masked=", r.json()["pii_masked"])
r = c.post("/ask", json={"message": ""})
print("POST /ask (empty) ->", r.status_code, "(Pydantic validation)")
r = c.post("/add-document", json={"doc_id": "kb-13", "topic": "gift_wrap_policy",
           "text": "Gift wrapping costs ₹49 per order. It is available for Beauty and Apparel only."})
print("POST /add-document ->", r.status_code, r.json())

print("\nWebSocket: client 1 does two turns (memory), then disconnects mid-conversation")
with c.websocket_connect("/ws/chat") as ws:
    hello = ws.receive_json(); print("  ", hello)
    ws.send_json({"message": "What is the return window for footwear?"}); print("   turn1:", ws.receive_json()["sources"])
    ws.send_json({"message": "And what about Beauty?"}); print("   turn2:", ws.receive_json()["sources"])
    ws.send_text("not json"); print("   bad frame:", ws.receive_json())
print("   client 1 closed the socket")
with c.websocket_connect("/ws/chat") as ws2:
    ws2.receive_json()
    ws2.send_json({"message": "Check order ORD0039"}); print("   client 2 still served:", ws2.receive_json()["outcome"])
