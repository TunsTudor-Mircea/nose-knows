"""
NoseKnows LangGraph agent.

Graph flow:
  START → classify_intent
        → (note_based) → retrieve
        → (mood/occasion) → generate_hyde → retrieve
  retrieve → recommend → validate
           → (PASS) → finalize → END
           → (FAIL, retries left) → regenerate → validate (loop)
           → (FAIL, max retries) → finalize → END (safe fallback)
"""
from __future__ import annotations

import os

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    node_classify_intent,
    node_finalize,
    node_generate_hyde,
    node_recommend,
    node_regenerate,
    node_refuse,
    node_retrieve,
    node_validate,
    route_after_intent,
    route_after_validate,
)
from src.agent.state import AgentState

_graph = None


def _make_checkpointer():
    """
    Try PostgresSaver (persists state across server restarts).
    Falls back to MemorySaver if the package or DB isn't available.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver

        url = os.getenv(
            "POSTGRES_URL",
            "postgresql://nosknows:nosknows@localhost:5432/nosknows",
        )
        # psycopg2 (sync) — strip asyncpg prefix if present
        url = url.replace("postgresql+asyncpg://", "postgresql://")
        saver = PostgresSaver.from_conn_string(url)
        saver.setup()
        return saver
    except Exception as exc:  # noqa: BLE001
        print(f"[graph] PostgresSaver unavailable ({exc}), falling back to MemorySaver.")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", node_classify_intent)
    builder.add_node("refuse", node_refuse)
    builder.add_node("generate_hyde", node_generate_hyde)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("recommend", node_recommend)
    builder.add_node("validate", node_validate)
    builder.add_node("regenerate", node_regenerate)
    builder.add_node("finalize", node_finalize)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"refuse": "refuse", "hyde": "generate_hyde", "retrieve": "retrieve"},
    )
    builder.add_edge("refuse", END)
    builder.add_edge("generate_hyde", "retrieve")
    builder.add_edge("retrieve", "recommend")
    builder.add_edge("recommend", "validate")
    builder.add_conditional_edges(
        "validate",
        route_after_validate,
        {"finalize": "finalize", "regenerate": "regenerate"},
    )
    builder.add_edge("regenerate", "validate")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=_make_checkpointer())


def get_graph():
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def run_graph(session_id: str, query: str, filters: dict | None = None) -> dict:
    """Synchronous entry point called from FastAPI via asyncio.to_thread."""
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = {
        "session_id": session_id,
        "query": query,
        "filters": filters or {},
        "intent": None,
        "hyde_doc": None,
        "retrieved": None,
        # previous_retrieved is intentionally omitted here so LangGraph carries
        # it forward from the checkpoint (set by node_finalize on each turn).
        "recommendation": None,
        "validation": None,
        "final_answer": None,
        "retry_count": 0,
        "messages": [HumanMessage(content=query)],
    }
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(state, config=config)

    return {
        "output": result.get("final_answer") or result.get("recommendation", ""),
        "hyde_doc": result.get("hyde_doc"),
        "intent": result.get("intent"),
        "retrieved": result.get("retrieved", ""),
    }
