"""Task 11 — FastAPI: POST /ask, POST /add-document, GET /health, WS /ws/chat."""
from __future__ import annotations

import asyncio
import uuid

import config  # first: telemetry off before crewai is imported
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

import rag_core
from agent_service import ask_agent
from knowledge_base import DOCUMENTS, Document
from logging_config import log_request

app = FastAPI(title="Nykaa Support Agent", version="1.0.0")


class AskRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=50_000)   # abuse cap, far above a real question
    session_id: str = Field(default="http-default", max_length=64)


class ReviewOut(BaseModel):
    approved: bool
    final_answer: str
    reason: str
class AskResponse(BaseModel):
    trace_id: str
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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:          # sync def -> FastAPI runs it in a worker thread
    result = ask_agent(req.message, session_id=req.session_id, channel="http")
    if result["outcome"] == "rejected_budget":
        raise HTTPException(status_code=413, detail=result["answer"])
    return AskResponse(**{k: result[k] for k in AskResponse.model_fields})


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    if any(d.doc_id == req.doc_id for d in DOCUMENTS):
        raise HTTPException(status_code=409, detail=f"{req.doc_id} already exists")
    trace = uuid.uuid4().hex[:12]
    # In-process only: no file is written to knowledge_base/, so a restart reloads the folder
    # without it (chroma_store keeps its chunks until the next build_both()). Permanent documents
    # are added as knowledge_base/<NN>_<topic>.md files, followed by a re-index.
    DOCUMENTS.append(Document(req.doc_id, req.topic, req.text, source="api:add-document"))
    counts = rag_core.build_both()                                # re-index both collections
    log_request(trace_id=trace, channel="http", event="add_document", doc_id=req.doc_id,
                chunks=counts, outcome="indexed")
    return AddDocumentResponse(doc_id=req.doc_id, chunks_indexed=counts)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(DOCUMENTS), "mock_llm": config.MOCK_LLM}


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket) -> None:
    await ws.accept()
    session_id = f"ws-{uuid.uuid4().hex[:8]}"        # server-issued: one connection = one memory
    await ws.send_json({"type": "hello", "session_id": session_id})
    try:
        while True:
            try:
                payload = await ws.receive_json()
            except (ValueError, KeyError):           # not JSON text (bad JSON or a binary frame)
                payload = None
            if not isinstance(payload, dict):        # malformed frame: tell the client, keep going
                await ws.send_json({"type": "error", "detail": "send JSON like {\"message\": \"...\"}"})
                continue
            text = str(payload.get("message", "")).strip()
            if not text:
                await ws.send_json({"type": "error", "detail": "empty message"})
                continue
            # The pipeline is sync/CPU-bound; keep the event loop free for other clients.
            try:
                result = await asyncio.to_thread(ask_agent, text, session_id, channel="ws")
            except Exception as exc:                 # one bad turn must not end the chat
                await ws.send_json({"type": "error", "detail": type(exc).__name__})
                continue
            await ws.send_json({"type": "answer", **{k: result[k] for k in
                                ("trace_id", "answer", "outcome", "grounded", "sources")}})
    except WebSocketDisconnect as exc:
        log_request(trace_id=session_id, channel="ws", event="ws_disconnect",
                    session_id=session_id, close_code=exc.code, outcome="client_disconnected")
    except RuntimeError as exc:                     # client vanished while we were sending
        log_request(trace_id=session_id, channel="ws", event="ws_send_after_close",
                    session_id=session_id, error=str(exc), outcome="client_disconnected")
