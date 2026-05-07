"""
Tool: FragranceRetriever

Wraps HyDERetriever and exposes it as a LangChain tool.  The agent calls
this tool with a query (and optional JSON-encoded filters) to get back a
formatted list of candidate perfumes from ChromaDB.
"""
from __future__ import annotations

import json
import os


def _format_results(results) -> str:
    """Render retrieved perfumes as a readable numbered list."""
    if not results:
        return "No matching perfumes found."
    lines = []
    for i, p in enumerate(results, 1):
        m = p.metadata
        lines.append(
            f"{i}. {m.get('perfume', '?')} — {m.get('brand', '?')}\n"
            f"   Top: {m.get('top_notes', '')}\n"
            f"   Heart: {m.get('heart_notes', '')}\n"
            f"   Base: {m.get('base_notes', '')}\n"
            f"   Accords: {m.get('accords', '')}\n"
            f"   Rating: {m.get('rating', 'N/A')}  Distance: {p.distance:.3f}"
        )
    return "\n\n".join(lines)


def retrieve_fragrances(query: str, filters_json: str = "{}") -> str:
    """Search the fragrance database for perfumes matching the query."""
    from src.model import load_llm
    from src.rag.retriever import HyDERetriever

    try:
        filters = json.loads(filters_json) if filters_json.strip() else {}
    except json.JSONDecodeError:
        filters = {}

    db_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
    top_k = int(os.getenv("RETRIEVER_TOP_K", "5"))

    retriever = HyDERetriever(llm=load_llm(), db_path=db_path, top_k=top_k)
    results, _hyde_doc = retriever.retrieve(query, filters=filters or None)
    return _format_results(results)
