"""Task 11 — FastAPI: POST /ask, POST /add-document, GET /health, WS /ws/chat.
Task 12 — exactly one JSON-Lines entry per HTTP request (middleware) and per WebSocket message."""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

import config  # first: telemetry off before crewai is imported
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import rag_core
from agent_service import ask_agent
from knowledge_base import DOCUMENTS, Document
from logging_config import log_request


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await asyncio.to_thread(rag_core.ensure_index)       # index ready before the first request
    yield


app = FastAPI(title="Nykaa Support Agent", version="1.0.0", lifespan=lifespan)

# HTTP conversation ids are server-issued, like the WebSocket ones: omit session_id to start a
# conversation, then send back the id you received to continue it. An id the server never issued is
# refused, so one customer can neither land in nor guess their way into another customer's memory.
_http_sessions: set[str] = set()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class AskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=50_000)   # abuse cap; Task 15's budget is the real limit
    session_id: str | None = Field(default=None, max_length=64,
                                   description="Omit to start a conversation; send back the returned id to continue.")


class ReviewOut(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    outcome: str
    grounded: bool
    escalated: bool
    sources: list[str]
    pii_masked: bool
    review: ReviewOut | None = None


class AddDocumentRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^kb-\d{2,}$")
    topic: str = Field(..., pattern=r"^[a-z0-9_]{3,64}$")      # same slug rule as KB file names
    text: str = Field(..., min_length=20, max_length=2000)


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_indexed: dict[str, int]


_STATUS_OUTCOME = {404: "not_found", 409: "conflict", 413: "rejected_budget", 422: "rejected_validation"}


@app.middleware("http")
async def log_every_http_request(request: Request, call_next):
    """Task 12: one JSON-Lines entry per HTTP request, with trace id, timing and status, including
    /health, 4xx and 5xx. Endpoints add their own fields to request.state.log."""
    request.state.trace_id, request.state.log = uuid.uuid4().hex[:12], {}
    t0, status, error = time.perf_counter(), 500, None
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-Id"] = request.state.trace_id
        return response
    except Exception as exc:                          # logged here; FastAPI then returns the 500
        error = type(exc).__name__
        raise
    finally:
        entry = {"trace_id": request.state.trace_id, "channel": "http", **request.state.log,
                 "method": request.method, "path": request.url.path, "status_code": status,
                 "latency_ms": _ms(t0)}
        entry.setdefault("outcome", "ok" if status < 400 else _STATUS_OUTCOME.get(status, f"http_{status}"))
        if error:
            entry["error"] = error
        log_request(**entry)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:   # sync def -> FastAPI runs it in a worker thread
    session_id = req.session_id
    if session_id is None:                                   # a new conversation
        session_id = f"http-{uuid.uuid4().hex[:12]}"
        _http_sessions.add(session_id)
    elif session_id not in _http_sessions:
        request.state.log["outcome"] = "unknown_session"
        raise HTTPException(status_code=404, detail="unknown session_id: omit it to start a new conversation")
    result = ask_agent(req.message, session_id=session_id, trace_id=request.state.trace_id,
                       channel="http", log_sink=request.state.log)
    if result["outcome"] == "rejected_budget":
        raise HTTPException(status_code=413, detail=result["answer"])
    return AskResponse(**{k: result[k] for k in AskResponse.model_fields})


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest, request: Request) -> AddDocumentResponse:
    request.state.log.update(event="add_document", doc_id=req.doc_id)
    if any(d.doc_id == req.doc_id for d in DOCUMENTS):
        request.state.log["outcome"] = "duplicate_doc_id"
        raise HTTPException(status_code=409, detail=f"{req.doc_id} already exists")
    # In-process only: no file is written to knowledge_base/, so a restart reloads the folder without
    # it, and the next process rebuilds the index without its chunks (rag_core.ensure_index).
    # Permanent documents are added as knowledge_base/<NN>_<topic>.md files.
    DOCUMENTS.append(Document(req.doc_id, req.topic, req.text, source="api:add-document"))
    counts = rag_core.build_both()                                # re-index both collections
    request.state.log.update(chunks=counts, outcome="indexed")
    return AddDocumentResponse(doc_id=req.doc_id, chunks_indexed=counts)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(DOCUMENTS), "mock_llm": config.MOCK_LLM}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"        # server-issued: one connection = one memory
    opened = time.perf_counter()
    await ws.send_json({"type": "hello", "session_id": session_id})

    def log_rejected(event: str, outcome: str, t0: float) -> None:   # a message the pipeline never saw
        log_request(trace_id=uuid.uuid4().hex[:12], channel="ws", event=event, session_id=session_id,
                    outcome=outcome, latency_ms=_ms(t0))

    try:
        while True:
            try:
                payload = await ws.receive_json()
            except (ValueError, KeyError):           # not JSON text (bad JSON or a binary frame)
                payload = None
            t0 = time.perf_counter()
            if not isinstance(payload, dict):        # malformed frame: tell the client, keep going
                await ws.send_json({"type": "error", "detail": "send JSON like {\"message\": \"...\"}"})
                log_rejected("ws_bad_frame", "rejected_bad_frame", t0)
                continue
            text = str(payload.get("message", "")).strip()
            if not text:
                await ws.send_json({"type": "error", "detail": "empty message"})
                log_rejected("ws_empty_message", "rejected_empty_message", t0)
                continue
            # The pipeline is sync/CPU-bound; keep the event loop free for other clients.
            try:
                result = await asyncio.to_thread(ask_agent, text, session_id, channel="ws")
            except Exception as exc:                 # one bad turn must not end the chat (already logged)
                await ws.send_json({"type": "error", "detail": type(exc).__name__})
                continue
            await ws.send_json({"type": "answer", **{k: result[k] for k in
                                ("trace_id", "answer", "outcome", "grounded", "sources")}})
    except WebSocketDisconnect as exc:
        log_request(trace_id=session_id, channel="ws", event="ws_disconnect", session_id=session_id,
                    close_code=exc.code, outcome="client_disconnected", duration_ms=_ms(opened))
    except RuntimeError as exc:                     # client vanished while we were sending
        log_request(trace_id=session_id, channel="ws", event="ws_send_after_close",
                    session_id=session_id, error=str(exc), outcome="client_disconnected",
                    duration_ms=_ms(opened))
