"""
Tool: RecommendationGenerator

Takes the original user query and a block of retrieved perfume context
(produced by FragranceRetriever) and generates a final, human-friendly
recommendation with per-note rationale.
"""
from __future__ import annotations

_RECOMMEND_PROMPT = """\
You are NoseKnows, an expert AI fragrance consultant. Your job is to recommend \
perfumes based on a user's description and a set of candidate fragrances.

User request:
{query}

Fragrance context:
{context}

Instructions:
1. If the context contains previously recommended perfumes, answer specifically \
   about those — do not invent new ones.
2. Recommend 2-3 perfumes that best match the user's request and explain WHY \
   using the specific notes and accords.
3. Connect the olfactory properties to the user's described mood, occasion, or style.
4. Keep the tone warm, knowledgeable, and accessible — not overly technical.
5. If none of the candidates are a great match, say so honestly.

Your recommendation:"""


def generate_recommendation(query: str, retrieved_context: str) -> str:
    """Generate a final fragrance recommendation with note-level rationale."""
    from src.model import load_llm

    llm = load_llm()
    return llm.invoke(_RECOMMEND_PROMPT.format(query=query, context=retrieved_context)).strip()
