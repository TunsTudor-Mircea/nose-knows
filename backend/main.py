"""
NoseKnows FastAPI backend.

Endpoints:
  GET  /health             → liveness probe
  POST /chat               → run the agent, return recommendation + perfume cards
  POST /feedback           → log a user preference rating (RLHF seed)

Start with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="NoseKnows API",
    description="Fragrance recommendation agent powered by HyDE + ChromaDB + SLM",
    version="0.1.0",
)

# Allow the Next.js dev server (port 3000) and any production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy-loaded agent (initialised on first request to avoid slow startup)
# ---------------------------------------------------------------------------
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from src.agent.agent import build_agent
        _agent = build_agent(verbose=False)
    return _agent


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FiltersModel(BaseModel):
    gender: str | None = Field(None, description="women | men | unisex")
    brand: str | None = None
    accord: str | None = None
    top_k: int = Field(5, ge=1, le=20)
    use_hyde: bool = True


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    filters: FiltersModel = Field(default_factory=FiltersModel)


class PerfumeCard(BaseModel):
    perfume: str
    brand: str
    top_notes: list[str]
    heart_notes: list[str]
    base_notes: list[str]
    accords: list[str]
    rating: float | None
    url: str


class ChatResponse(BaseModel):
    response: str
    perfumes: list[PerfumeCard]
    hyde_doc: str | None
    intent: str | None


class FeedbackRequest(BaseModel):
    query: str
    response: str
    score: int = Field(..., ge=1, le=5, description="1=very bad, 5=very good")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    logged: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_notes_str(raw: str) -> list[str]:
    if not raw:
        return []
    return [n.strip() for n in raw.split(",") if n.strip()]


def _extract_perfume_cards(intermediate_steps: list) -> list[PerfumeCard]:
    """
    Extract perfume cards from the agent's intermediate steps by looking for
    the retrieve_fragrances observation and re-querying ChromaDB for metadata.
    """
    # We re-use the HyDERetriever directly with the raw observations
    cards: list[PerfumeCard] = []
    for action, observation in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        if tool_name != "retrieve_fragrances":
            continue
        # Parse the formatted text block returned by retrieve_fragrances
        blocks = [b.strip() for b in observation.split("\n\n") if b.strip()]
        for block in blocks:
            lines = block.splitlines()
            card: dict[str, Any] = {
                "perfume": "",
                "brand": "",
                "top_notes": [],
                "heart_notes": [],
                "base_notes": [],
                "accords": [],
                "rating": None,
                "url": "",
            }
            for line in lines:
                if line.startswith("1.") or (len(line) > 2 and line[1] == "."):
                    # "1. PerfumeName — Brand"
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
                        rating_str = line.split("Rating:")[1].split()[0]
                        card["rating"] = float(rating_str)
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


def _log_feedback(entry: dict) -> None:
    log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "data/feedback_log.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "nosknows-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the NoseKnows agent for the given query and return a recommendation
    along with structured perfume cards for the frontend.
    """
    filters: dict | None = None
    f = request.filters
    if any([f.gender, f.brand, f.accord]):
        filters = {k: v for k, v in {"gender": f.gender, "brand": f.brand, "accord": f.accord}.items() if v}

    try:
        from src.agent.agent import run_agent
        agent = get_agent()
        result = run_agent(agent, request.query, filters=filters)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    perfumes = _extract_perfume_cards(result.get("intermediate_steps", []))
    intent = _extract_intent(result.get("intermediate_steps", []))

    return ChatResponse(
        response=result["output"],
        perfumes=perfumes,
        hyde_doc=result.get("hyde_doc"),
        intent=intent,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Log a user preference rating for a recommendation.
    Scores accumulate in data/feedback_log.jsonl as RLHF seed data.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": request.query,
        "response": request.response,
        "score": request.score,
        "comment": request.comment,
    }
    try:
        _log_feedback(entry)
        return FeedbackResponse(logged=True, message="Feedback recorded. Thank you!")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {exc}") from exc
