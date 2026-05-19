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
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
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

from src.db.session import get_db, AsyncSessionLocal  # noqa: E402
from src.db.models import Session as DBSession, Message as DBMessage, Feedback as DBFeedback, IngestJob  # noqa: E402
from src.tools.guard import _blocklist_check  # noqa: E402

# ---------------------------------------------------------------------------
# Graph runner (replaces hand-rolled ReAct agent)
# ---------------------------------------------------------------------------


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


def _parse_retrieved_text(text: str) -> list[PerfumeCard]:
    """Parse the formatted retrieval text block into PerfumeCard objects."""
    import re as _re
    if not text or "No matching perfumes found" in text:
        return []
    cards: list[PerfumeCard] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        card: dict[str, Any] = {
            "perfume": "", "brand": "",
            "top_notes": [], "heart_notes": [], "base_notes": [],
            "accords": [], "rating": None, "url": "",
        }
        for line in block.splitlines():
            line = line.strip()
            if _re.match(r"^\d+\.", line):
                header = _re.sub(r"^\d+\.\s*", "", line)
                parts = header.split(" — ", 1)
                card["perfume"] = parts[0].strip()
                card["brand"] = parts[1].strip() if len(parts) > 1 else ""
            elif line.startswith("Top:"):
                card["top_notes"] = _parse_notes_str(line[4:])
            elif line.startswith("Heart:"):
                card["heart_notes"] = _parse_notes_str(line[6:])
            elif line.startswith("Base:"):
                card["base_notes"] = _parse_notes_str(line[5:])
            elif line.startswith("Accords:"):
                card["accords"] = _parse_notes_str(line[8:])
            elif line.startswith("Rating:"):
                try:
                    card["rating"] = float(line[7:].split()[0])
                except (ValueError, IndexError):
                    pass
        if card["perfume"]:
            cards.append(PerfumeCard(**card))
    return cards


def _run_graph_sync(session_id: str, query: str, filters: dict | None) -> dict:
    from src.agent.graph import run_graph
    return run_graph(session_id, query, filters)


def _log_feedback_file(entry: dict) -> None:
    log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _build_filters(f: FiltersModel) -> dict:
    result: dict = {"top_k": f.top_k, "use_hyde": f.use_hyde}
    if f.gender: result["gender"] = f.gender
    if f.brand: result["brand"] = f.brand
    if f.accord: result["accord"] = f.accord
    return result


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

    # Input guard — reject queries containing prohibited language before touching the LLM
    if _blocklist_check(request.query):
        raise HTTPException(
            status_code=400,
            detail="Your message contains prohibited language. Please rephrase your query.",
        )

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

    # Run graph (blocking — offload to thread)
    filters = _build_filters(request.filters)
    t0 = time.monotonic()
    try:
        agent_result = await asyncio.to_thread(_run_graph_sync, str(sid), request.query, filters)
    except Exception as exc:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    latency_ms = int((time.monotonic() - t0) * 1000)

    all_cards = _parse_retrieved_text(agent_result.get("retrieved", ""))
    intent = agent_result.get("intent")
    hyde_doc = agent_result.get("hyde_doc")
    response_text = agent_result["output"]

    # For follow-up queries the retrieved set is the full previous turn's results;
    # filter to only the cards the model explicitly mentioned in its response.
    # For all other intents show every retrieved card (preserves the user's top-k choice).
    if intent == "follow_up":
        resp_lower = response_text.lower()
        mentioned = [c for c in all_cards if c.perfume.lower() in resp_lower]
        perfume_cards = mentioned if mentioned else all_cards
    else:
        perfume_cards = all_cards

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

    # Upsert: update if feedback already exists for this message, else insert.
    existing = await db.execute(select(DBFeedback).where(DBFeedback.message_id == mid))
    fb = existing.scalar_one_or_none()
    if fb is not None:
        fb.score = request.score
        fb.comment = request.comment
    else:
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
# Perfumes (ChromaDB browse)
# ---------------------------------------------------------------------------

class PerfumeListItem(BaseModel):
    id: str
    perfume: str
    brand: str
    gender: str | None
    year: int | None
    top_notes: list[str]
    heart_notes: list[str]
    base_notes: list[str]
    accords: list[str]
    rating: float | None
    rating_count: int | None
    url: str


class PerfumeListResponse(BaseModel):
    total: int
    perfumes: list[PerfumeListItem]


def _list_perfumes_sync(limit: int, offset: int, search: str, gender: str) -> dict:  # gender = comma-separated
    import os
    import chromadb

    from chromadb.config import Settings as ChromaSettings
    db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        col = client.get_collection("fragrances")
    except Exception:
        return {"total": 0, "perfumes": []}

    total = col.count()

    # When a text search is provided we must fetch all records and filter in Python
    # because ChromaDB has no full-text search.  Otherwise, use ChromaDB paging directly.
    needs_python_filter = bool(search)
    fetch_limit = total if needs_python_filter else limit
    fetch_offset = 0 if needs_python_filter else offset

    where: dict = {}
    if gender and gender not in ("any", ""):
        genders = [g.strip() for g in gender.split(",") if g.strip()]
        if len(genders) == 1:
            where["gender"] = {"$eq": genders[0]}
        elif len(genders) > 1:
            where["gender"] = {"$in": genders}

    kwargs: dict = {"limit": fetch_limit, "offset": fetch_offset, "include": ["metadatas"]}
    if where:
        kwargs["where"] = where

    try:
        res = col.get(**kwargs)
    except Exception:
        res = col.get(limit=fetch_limit, offset=fetch_offset, include=["metadatas"])

    def _split(s: str) -> list[str]:
        return [n.strip() for n in (s or "").split(",") if n.strip()]

    items = []
    for mid, meta in zip(res["ids"], res["metadatas"]):
        try:
            year = int(meta.get("year", 0)) or None
        except (ValueError, TypeError):
            year = None
        try:
            rating = float(meta.get("rating", 0)) or None
        except (ValueError, TypeError):
            rating = None
        try:
            rating_count = int(meta.get("rating_count", 0)) or None
        except (ValueError, TypeError):
            rating_count = None

        perfume_name = meta.get("perfume", "")
        brand = meta.get("brand", "")
        if search:
            sl = search.lower()
            if sl not in perfume_name.lower() and sl not in brand.lower():
                continue

        items.append({
            "id": mid,
            "perfume": perfume_name,
            "brand": brand,
            "gender": meta.get("gender"),
            "year": year,
            "top_notes": _split(meta.get("top_notes", "")),
            "heart_notes": _split(meta.get("heart_notes", "")),
            "base_notes": _split(meta.get("base_notes", "")),
            "accords": _split(meta.get("accords", "")),
            "rating": rating,
            "rating_count": rating_count,
            "url": meta.get("url", ""),
        })

    # For text-search results, paginate in Python after filtering.
    if needs_python_filter:
        filtered_total = len(items)
        items = items[offset : offset + limit]
        return {"total": filtered_total, "perfumes": items}

    return {"total": total, "perfumes": items}


# ---------------------------------------------------------------------------
# Brand autocomplete
# ---------------------------------------------------------------------------

_brand_cache: list[str] = []


def _load_brands_sync() -> list[str]:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
    try:
        client = chromadb.PersistentClient(
            path=db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        col = client.get_collection("fragrances")
    except Exception:
        return []

    total = col.count()
    seen: set[str] = set()
    batch = 1000
    for offset in range(0, total, batch):
        res = col.get(limit=batch, offset=offset, include=["metadatas"])
        for meta in res["metadatas"]:
            b = (meta.get("brand") or "").strip()
            if b:
                seen.add(b)

    return sorted(seen, key=str.lower)


@app.get("/brands", response_model=list[str])
async def list_brands(q: str = "") -> list[str]:
    global _brand_cache
    if not _brand_cache:
        _brand_cache = await asyncio.to_thread(_load_brands_sync)
    if not q:
        return _brand_cache[:80]
    ql = q.strip().lower()
    return [b for b in _brand_cache if ql in b.lower()][:50]


@app.get("/perfumes", response_model=PerfumeListResponse)
async def list_perfumes(
    limit: int = 60,
    offset: int = 0,
    search: str = "",
    gender: str = "",
) -> PerfumeListResponse:
    result = await asyncio.to_thread(_list_perfumes_sync, limit, offset, search, gender)
    return PerfumeListResponse(**result)


# ---------------------------------------------------------------------------
# Feedback list (for Feedback browser page)
# ---------------------------------------------------------------------------

class FeedbackListItem(BaseModel):
    id: _uuid.UUID
    message_id: _uuid.UUID
    session_id: _uuid.UUID
    score: int
    comment: str | None
    created_at: datetime
    message_content: str | None
    query: str | None
    intent: str | None

    class Config:
        from_attributes = True


class FeedbackListResponse(BaseModel):
    total: int
    items: list[FeedbackListItem]


@app.delete("/feedback/{feedback_id}", status_code=204)
async def delete_feedback(
    feedback_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    try:
        fid = _uuid.UUID(feedback_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid feedback ID format")
    result = await db.execute(select(DBFeedback).where(DBFeedback.id == fid))
    fb = result.scalar_one_or_none()
    if fb is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    await db.delete(fb)
    await db.commit()


@app.get("/feedback/list", response_model=FeedbackListResponse)
async def list_feedback(
    limit: int = 50,
    offset: int = 0,
    score: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> FeedbackListResponse:
    from sqlalchemy import func, join

    q = select(DBFeedback).order_by(DBFeedback.created_at.desc())
    if score is not None:
        q = q.where(DBFeedback.score == score)
    total_res = await db.execute(select(func.count()).select_from(DBFeedback))
    total = total_res.scalar_one()

    fb_res = await db.execute(q.limit(limit).offset(offset))
    feedback_rows = fb_res.scalars().all()

    items = []
    for fb in feedback_rows:
        msg_res = await db.execute(select(DBMessage).where(DBMessage.id == fb.message_id))
        msg = msg_res.scalar_one_or_none()
        query_content: str | None = None
        if msg:
            prev = await db.execute(
                select(DBMessage)
                .where(DBMessage.session_id == msg.session_id, DBMessage.role == "user")
                .where(DBMessage.created_at < msg.created_at)
                .order_by(DBMessage.created_at.desc())
                .limit(1)
            )
            user_msg = prev.scalar_one_or_none()
            query_content = user_msg.content if user_msg else None

        items.append(FeedbackListItem(
            id=fb.id,
            message_id=fb.message_id,
            session_id=fb.session_id,
            score=fb.score,
            comment=fb.comment,
            created_at=fb.created_at,
            message_content=msg.content if msg else None,
            query=query_content,
            intent=msg.intent if msg else None,
        ))

    return FeedbackListResponse(total=total, items=items)


# ---------------------------------------------------------------------------
# Ingestion API (Phase 4)
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
_REQUIRED_COLS = {"Perfume", "Brand"}
_job_progress: dict[str, int] = {}  # in-memory progress tracking per job


class IngestRunRequest(BaseModel):
    filename: str


def _parse_upload_preview(file_path: str) -> dict:
    import pandas as pd

    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            head = pd.read_csv(file_path, sep=";", nrows=6, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        head = pd.read_csv(file_path, sep=";", nrows=6, encoding="utf-8", encoding_errors="replace")

    columns = [c.strip() for c in head.columns.tolist()]
    head.columns = columns  # align preview dict keys with detected_columns
    col_set = set(columns)
    missing = sorted(_REQUIRED_COLS - col_set)

    with open(file_path, encoding="utf-8", errors="replace") as fh:
        total_rows = max(sum(1 for _ in fh) - 1, 0)  # subtract header line

    preview = head.fillna("").head(5).astype(str).to_dict("records")
    return {
        "filename": Path(file_path).name,
        "detected_columns": columns,
        "missing_required": missing,
        "row_count": max(total_rows, 0),
        "preview": preview,
        "valid": len(missing) == 0,
    }


def _pipeline_step(file_path: str, chunks_path: str) -> int:
    from src.data_pipeline import run as dp_run
    chunks = dp_run(file_path, chunks_path)
    return len(chunks)


def _index_step(chunks_path: str, db_path: str) -> int:
    from src.rag.build_index import build_index
    collection = build_index(chunks_path, db_path)
    return collection.count()


async def _run_ingest_bg(job_id: str, file_path: str) -> None:
    """Background task: run data pipeline then rebuild ChromaDB index."""
    jid = _uuid.UUID(job_id)

    async with AsyncSessionLocal() as db:
        job = (await db.execute(select(IngestJob).where(IngestJob.id == jid))).scalar_one_or_none()
        if job is None:
            return
        job.status = "running"
        await db.commit()

    _job_progress[job_id] = 5

    try:
        chunks_path = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
        total_rows = await asyncio.to_thread(_pipeline_step, file_path, chunks_path)
        _job_progress[job_id] = 40

        chroma_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
        processed = await asyncio.to_thread(_index_step, chunks_path, chroma_path)
        _job_progress[job_id] = 100

        async with AsyncSessionLocal() as db:
            job = (await db.execute(select(IngestJob).where(IngestJob.id == jid))).scalar_one_or_none()
            if job:
                job.status = "done"
                job.total_rows = total_rows
                job.processed_rows = processed
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

    except Exception as exc:
        async with AsyncSessionLocal() as db:
            job = (await db.execute(select(IngestJob).where(IngestJob.id == jid))).scalar_one_or_none()
            if job:
                job.status = "error"
                job.message = str(exc)[:500]
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()
        _job_progress.pop(job_id, None)


@app.post("/ingest/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / (file.filename or "upload.csv")
    content = await file.read()
    with open(dest, "wb") as fh:
        fh.write(content)
    try:
        result = await asyncio.to_thread(_parse_upload_preview, str(dest))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc
    return result


@app.post("/ingest/run")
async def run_ingest(
    body: IngestRunRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> dict:
    file_path = UPLOAD_DIR / body.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{body.filename}' not found in uploads")
    job = IngestJob(filename=body.filename, status="pending")
    db.add(job)
    await db.commit()
    await db.refresh(job)
    job_id = str(job.id)
    _job_progress[job_id] = 0
    background_tasks.add_task(_run_ingest_bg, job_id, str(file_path))
    return {"job_id": job_id, "status": "pending"}


@app.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        jid = _uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid job ID")
    job = (await db.execute(select(IngestJob).where(IngestJob.id == jid))).scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    progress = 100 if job.status == "done" else _job_progress.get(job_id, 0)
    return {
        "id": str(job.id),
        "filename": job.filename,
        "status": job.status,
        "total_rows": job.total_rows,
        "processed_rows": job.processed_rows,
        "progress_pct": progress,
        "message": job.message,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@app.get("/ingest/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)) -> list[dict]:
    result = await db.execute(select(IngestJob).order_by(IngestJob.created_at.desc()))
    return [
        {
            "id": str(j.id),
            "filename": j.filename,
            "status": j.status,
            "total_rows": j.total_rows,
            "created_at": j.created_at.isoformat(),
        }
        for j in result.scalars().all()
    ]


# ---------------------------------------------------------------------------
# RLHF (Phase 6)
# ---------------------------------------------------------------------------

@app.get("/rlhf/stats")
async def rlhf_stats(db: AsyncSession = Depends(get_db)) -> dict:
    """Return feedback stats useful for deciding when to run DPO export."""
    from sqlalchemy import func

    total_res = await db.execute(select(func.count()).select_from(DBFeedback))
    total = total_res.scalar_one()

    pos_res = await db.execute(
        select(func.count()).select_from(DBFeedback).where(DBFeedback.score == 1)
    )
    positive = pos_res.scalar_one()
    negative = total - positive

    # Count distinct prompts that have BOTH a positive and negative response
    # (these are the ones that will yield DPO pairs)
    from sqlalchemy import distinct
    pos_msgs = (
        select(DBMessage.session_id, DBMessage.content)
        .join(DBFeedback, DBFeedback.message_id == DBMessage.id)
        .where(DBFeedback.score == 1)
        .where(DBMessage.role == "assistant")
    ).subquery()

    neg_msgs = (
        select(DBMessage.session_id, DBMessage.content)
        .join(DBFeedback, DBFeedback.message_id == DBMessage.id)
        .where(DBFeedback.score == -1)
        .where(DBMessage.role == "assistant")
    ).subquery()

    # Rough DPO-pair estimate: queries with both pos and neg (done in Python below)
    all_fb = (
        await db.execute(
            select(DBFeedback.score, DBMessage.session_id, DBMessage.content)
            .join(DBMessage, DBMessage.id == DBFeedback.message_id)
            .where(DBMessage.role == "assistant")
        )
    ).all()

    # Group by session to find sessions that have both polarities
    from collections import defaultdict
    session_scores: dict = defaultdict(set)
    for score, session_id, _ in all_fb:
        session_scores[str(session_id)].add(score)

    paired_sessions = sum(
        1 for scores in session_scores.values() if 1 in scores and -1 in scores
    )

    dpo_pairs_path = os.getenv("DPO_PAIRS_PATH", "data/dpo_pairs.jsonl")
    pairs_on_disk = 0
    if Path(dpo_pairs_path).exists():
        with open(dpo_pairs_path, encoding="utf-8") as fh:
            pairs_on_disk = sum(1 for _ in fh)

    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "sessions_with_contrast": paired_sessions,
        "estimated_dpo_pairs": paired_sessions,
        "dpo_pairs_on_disk": pairs_on_disk,
        "dpo_pairs_path": dpo_pairs_path,
        "ready_for_export": paired_sessions >= 5,
    }


@app.post("/rlhf/export")
async def rlhf_export(
    background_tasks: BackgroundTasks,
    output: str | None = None,
) -> dict:
    """Trigger DPO pair export as a background task. Returns immediately."""
    out_path = output or os.getenv("DPO_PAIRS_PATH", "data/dpo_pairs.jsonl")

    async def _run() -> None:
        from src.rlhf.export_dpo_pairs import main as export_main
        await export_main(out_path, min_pairs=5, dry_run=False)

    background_tasks.add_task(_run)
    return {
        "status": "started",
        "output": out_path,
        "message": "DPO export running in background. Check the output file when complete.",
    }


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------

EVAL_RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH", "data/eval_results.jsonl"))


@app.get("/eval/results")
async def get_eval_results() -> list[dict]:
    """Return all rows from eval_results.jsonl, newest run last."""
    if not EVAL_RESULTS_PATH.exists():
        return []
    rows = []
    with EVAL_RESULTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ---------------------------------------------------------------------------
# Deprecated pass-throughs
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse, deprecated=True)
async def chat_legacy(request: ChatRequest) -> ChatResponse:
    """Stateless legacy endpoint — no DB writes. Use POST /sessions/{id}/chat instead."""
    import uuid as _uuid_mod
    filters = _build_filters(request.filters)
    # Use a throwaway session ID — state won't be persisted across calls
    ephemeral_id = str(_uuid_mod.uuid4())
    try:
        result = await asyncio.to_thread(_run_graph_sync, ephemeral_id, request.query, filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        response=result["output"],
        perfumes=_parse_retrieved_text(result.get("retrieved", "")),
        hyde_doc=result.get("hyde_doc"),
        intent=result.get("intent"),
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
