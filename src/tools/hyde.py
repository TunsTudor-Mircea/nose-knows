"""
Tool: HyDEGenerator

Converts a natural-language user query into a hypothetical structured
fragrance profile (top / heart / base notes + accords). This hypothetical
document lives in the same embedding space as the real perfume chunks in
ChromaDB, allowing more precise semantic retrieval.
"""
from __future__ import annotations

_HYDE_PROMPT = """\
You are a master perfumer and fragrance writer.

Given the description below, write a concise hypothetical perfume profile \
with the following structure (fill in actual note names, do not leave blanks):

Top notes: <comma-separated notes>
Heart notes: <comma-separated notes>
Base notes: <comma-separated notes>
Accords: <comma-separated accords>
Mood: <one sentence describing the overall feeling>

Description: {query}

Hypothetical perfume profile:"""


def generate_hyde_document(query: str) -> str:
    """Generate a hypothetical structured fragrance profile for the given query."""
    from src.model import load_llm

    llm = load_llm()
    return llm.invoke(_HYDE_PROMPT.format(query=query)).strip()
