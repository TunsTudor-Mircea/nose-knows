from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage

from src.agent.state import AgentState
from src.tools.guard import validate_response
from src.tools.hyde import generate_hyde_document
from src.tools.intent import classify_intent
from src.tools.recommender import generate_recommendation
from src.tools.retriever import retrieve_fragrances

_SAFE_FALLBACK = (
    "I wasn't able to generate a confident recommendation for that request. "
    "Please try describing the mood, occasion, or specific notes you enjoy."
)

_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Nodes — each returns only the fields it updates
# ---------------------------------------------------------------------------

def node_classify_intent(state: AgentState) -> dict:
    intent = classify_intent(state["query"])
    return {"intent": intent}


def node_generate_hyde(state: AgentState) -> dict:
    hyde_doc = generate_hyde_document(state["query"])
    return {"hyde_doc": hyde_doc}


def node_retrieve(state: AgentState) -> dict:
    # Use the HyDE doc as the retrieval query when available
    retrieval_query = state.get("hyde_doc") or state["query"]
    filters = state.get("filters") or {}
    filters_json = json.dumps({k: v for k, v in filters.items() if k != "top_k"})
    retrieved = retrieve_fragrances(retrieval_query, filters_json)
    return {"retrieved": retrieved}


def node_recommend(state: AgentState) -> dict:
    query = state["query"]
    if state.get("intent") == "follow_up":
        from langchain_core.messages import AIMessage as _AI
        prev = next(
            (m.content for m in reversed(state.get("messages", [])) if isinstance(m, _AI)),
            None,
        )
        if prev:
            query = f"[Previous response]\n{prev}\n\n[Follow-up question]\n{query}"
    recommendation = generate_recommendation(query, state.get("retrieved", ""))
    return {"recommendation": recommendation}


def node_validate(state: AgentState) -> dict:
    validation = validate_response(
        state.get("recommendation", ""),
        state.get("retrieved", ""),
    )
    return {"validation": validation}


def node_regenerate(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0) + 1
    # Nudge the recommender to be more grounded on retry
    nudged_query = state["query"] + " (focus strictly on the retrieved perfume notes)"
    recommendation = generate_recommendation(nudged_query, state.get("retrieved", ""))
    return {"recommendation": recommendation, "retry_count": retry_count}


def node_finalize(state: AgentState) -> dict:
    validation = state.get("validation", "")
    if re.match(r"^PASS[:\s]", validation, re.IGNORECASE):
        answer = re.sub(r"^PASS:\s*", "", validation, flags=re.IGNORECASE).strip()
    else:
        # Max retries hit — return recommendation or safe fallback
        answer = state.get("recommendation") or _SAFE_FALLBACK

    return {
        "final_answer": answer,
        "messages": [AIMessage(content=answer)],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """Skip HyDE for note_based queries — the query itself is already specific."""
    if state.get("intent") == "note_based":
        return "retrieve"
    return "hyde"


def route_after_validate(state: AgentState) -> str:
    validation = state.get("validation", "")
    if re.match(r"^PASS[:\s]", validation, re.IGNORECASE):
        return "finalize"
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        return "finalize"
    return "regenerate"
