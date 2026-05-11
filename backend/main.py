"""
NoseKnows FastAPI backend — v1 (Phase 1: PostgreSQL + session management).

New endpoints:
  POST   /sessions                  → create session
  GET    /sessions                  → list sessions
  DELETE /sessions/{id}             → delete session + cascade
  GET    /sessions/{id}/messages    → full message history
  POST   /sessions/{id}/chat        → send message (replaces /chat)
  POST   /messages/{id}/feedback    → rate message (replaces /feedback)
  GET    /health                    → liveness probe (unchanged)

Deprecated (kept for transition):
  POST   /chat      → stateless, no DB writes
  POST   /feedback  → writes to feedback_log.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()

app = FastAPI(
    title="NoseKnows API",
    description="Fragrance recommendation agent — HyDE + ChromaDB + SLM + PostgreSQL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

from src.db.session import get_db  # noqa: E402
from src.db.models import Session as DBSession, Message as DBMessage, Feedback as DBFeedback  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy agent loader
# ---------------------------------------------------------------------------

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from src.agent.agent import build_agent
        _agent = build_agent(verbose=False)
    return _agent


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class FiltersModel(BaseModel):
    gender: str | None = Field(None, description="women | men | unisex")
    brand: str | None = None
    accord: str | None = None
    top_k: int = Field(5, ge=1, le=20)
    use_hyde: bool = True


class PerfumeCard(BaseModel):
    perfume: str
    brand: str
    top_notes: list[str]
    heart_notes: list[str]
    base_notes: list[str]
    accords: list[str]
    rating: float | None
    url: str


# ── Session schemas ──────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str | None = None


class SessionResponse(BaseModel):
    id: _uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ── Message schemas ──────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: _uuid.UUID
    session_id: _uuid.UUID
    role: str
    content: str
    intent: str | None
    hyde_doc: str | None
    perfumes: list[dict] | None
    latency_ms: int | None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Chat schemas ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: FiltersModel = Field(default_factory=FiltersModel)


class ChatResponse(BaseModel):
    """Stateless /chat response (deprecated)."""
    response: str
    perfumes: list[PerfumeCard]
    hyde_doc: str | None
    intent: str | None


class SessionChatResponse(BaseModel):
    """Response from POST /sessions/{id}/chat — includes DB message IDs."""
    message_id: _uuid.UUID        # assistant message ID (use for feedback)
    response: str
    perfumes: list[PerfumeCard]
    hyde_doc: str | None
    intent: str | None


# ── Feedback schemas ─────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """New per-message feedback: thumbs up (+1) or thumbs down (-1)."""
    score: Literal[-1, 1]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    logged: bool
    message: str


# ── Deprecated feedback schema (old /feedback endpoint) ─────────────────────

class LegacyFeedbackRequest(BaseModel):
    query: str
    response: str
    score: int = Field(..., ge=1, le=5)
    comment: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_notes_str(raw: str) -> list[str]:
    if not raw:
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def _extract_perfume_cards(intermediate_steps: list) -> list[PerfumeCard]:
    cards: list[PerfumeCard] = []
    for action, observation in intermediate_steps:
        if getattr(action, "tool", "") != "retrieve_fragrances":
            continue
        blocks = [b.strip() for b in observation.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            card: dict[str, Any] = {
                "perfume": "", "brand": "",
                "top_notes": [], "heart_notes": [], "base_notes": [],
                "accords": [], "rating": None, "url": "",
            }
            for line in lines:
                if line.startswith("1.") or (len(line) > 2 and line[1] == "."):
                    parts = line.split(". ", 1)[-1].split(" — ", 1)
                    card["perfume"] = parts[0].strip()
                    card["brand"] = parts[1].strip() if len(parts) > 1 else ""
                elif line.strip().startswith("Top:"):
                    card["top_notes"] = _parse_notes_str(line.split(":", 1)[-1])
                elif line.strip().startswith("Heart:"):
                    card["heart_notes"] = _parse_notes_str(line.split(":", 1)[-1])
                elif line.strip().startswith("Base:"):
                    card["base_notes"] = _parse_notes_str(line.split(":", 1)[-1])
                elif line.strip().startswith("Accords:"):
                    card["accords"] = _parse_notes_str(line.split(":", 1)[-1])
                elif "Rating:" in line:
                    try:
                        card["rating"] = float(line.split("Rating:")[1].split()[0])
                    except (ValueError, IndexError):
                        pass
            if card["perfume"]:
                cards.append(PerfumeCard(**card))
    return cards


def _extract_intent(intermediate_steps: list) -> str | None:
    for action, observation in intermediate_steps:
        if getattr(action, "tool", "") == "classify_intent":
            return observation
    return None


def _run_agent_sync(query: str, filters: dict | None) -> dict:
    from src.agent.agent import run_agent
    agent = _get_agent()
    return run_agent(agent, query, filters=filters)


def _log_feedback_file(entry: dict) -> None:
    log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_filters(f: FiltersModel) -> dict | None:
    if any([f.gender, f.brand, f.accord]):
        return {k: v for k, v in {"gender": f.gender, "brand": f.brand, "accord": f.accord}.items() if v}
    return None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "nosknows-api", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreate = SessionCreate(),
    db: AsyncSession = Depends(get_db),
) -> SessionResponse:
    session = DBSession(title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return SessionResponse.model_validate(session)


@app.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> list[SessionResponse]:
    result = await db.execute(
        select(DBSession).order_by(DBSession.updated_at.desc())
    )
    sessions = result.scalars().all()
    return [SessionResponse.model_validate(s) for s in sessions]


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, db: AsyncSession = Depends(get_db)) -> None:
    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    result = await db.execute(select(DBSession).where(DBSession.id == sid))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await db.delete(session)
    await db.commit()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: str, db: AsyncSession = Depends(get_db)
) -> list[MessageResponse]:
    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")
    result = await db.execute(select(DBSession).where(DBSession.id == sid))
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = await db.execute(
        select(DBMessage)
        .where(DBMessage.session_id == sid)
        .order_by(DBMessage.created_at.asc())
    )
    return [MessageResponse.model_validate(m) for m in msgs.scalars().all()]


# ---------------------------------------------------------------------------
# Chat (new — session-aware)
# ---------------------------------------------------------------------------

@app.post("/sessions/{session_id}/chat", response_model=SessionChatResponse)
async def session_chat(
    session_id: str,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> SessionChatResponse:
    try:
        sid = _uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session ID format")

    result = await db.execute(select(DBSession).where(DBSession.id == sid))
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Auto-set title from first user message
    if session.title is None:
        session.title = request.query[:60]

    # Save user message
    user_msg = DBMessage(
        session_id=sid,
        role="user",
        content=request.query,
    )
    db.add(user_msg)
    await db.flush()

    # Run agent (blocking — offload to thread)
    filters = _build_filters(request.filters)
    t0 = time.monotonic()
    try:
        agent_result = await asyncio.to_thread(_run_agent_sync, request.query, filters)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - t0) * 1000)

    perfume_cards = _extract_perfume_cards(agent_result.get("intermediate_steps", []))
    intent = _extract_intent(agent_result.get("intermediate_steps", []))
    hyde_doc = agent_result.get("hyde_doc")
    response_text = agent_result["output"]

    # Save assistant message
    asst_msg = DBMessage(
        session_id=sid,
        role="assistant",
        content=response_text,
        intent=intent,
        hyde_doc=hyde_doc,
        perfumes=[c.model_dump() for c in perfume_cards],
        latency_ms=latency_ms,
    )
    db.add(asst_msg)
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(asst_msg)

    return SessionChatResponse(
        message_id=asst_msg.id,
        response=response_text,
        perfumes=perfume_cards,
        hyde_doc=hyde_doc,
        intent=intent,
    )


# ---------------------------------------------------------------------------
# Feedback (new — per message)
# ---------------------------------------------------------------------------

@app.post("/messages/{message_id}/feedback", response_model=FeedbackResponse)
async def message_feedback(
    message_id: str,
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    try:
        mid = _uuid.UUID(message_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid message ID format")

    result = await db.execute(select(DBMessage).where(DBMessage.id == mid))
    msg = result.scalar_one_or_none()
    if msg is None:
        raise HTTPException(status_code=404, detail="Message not found")

    fb = DBFeedback(
        message_id=mid,
        session_id=msg.session_id,
        score=request.score,
        comment=request.comment,
    )
    db.add(fb)
    await db.commit()
    return FeedbackResponse(logged=True, message="Feedback recorded. Thank you!")


# ---------------------------------------------------------------------------
# Deprecated pass-throughs
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, deprecated=True)
async def chat_legacy(request: ChatRequest) -> ChatResponse:
    """Stateless legacy endpoint — no DB writes. Use POST /sessions/{id}/chat instead."""
    filters = _build_filters(request.filters)
    try:
        result = await asyncio.to_thread(_run_agent_sync, request.query, filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        response=result["output"],
        perfumes=_extract_perfume_cards(result.get("intermediate_steps", [])),
        hyde_doc=result.get("hyde_doc"),
        intent=_extract_intent(result.get("intermediate_steps", [])),
    )


@app.post("/feedback", response_model=FeedbackResponse, deprecated=True)
async def feedback_legacy(request: LegacyFeedbackRequest) -> FeedbackResponse:
    """Legacy endpoint — writes to feedback_log.jsonl. Use POST /messages/{id}/feedback instead."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "response": request.response,
        "score": request.score,
        "comment": request.comment,
    }
    try:
        await asyncio.to_thread(_log_feedback_file, entry)
        return FeedbackResponse(logged=True, message="Feedback recorded. Thank you!")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {exc}") from exc
