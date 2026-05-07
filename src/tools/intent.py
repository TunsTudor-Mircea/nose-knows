"""
Tool: IntentClassifier

Classifies the user's query into one of four intent categories:
  - mood_based     : subjective / emotional description ("I want to feel powerful")
  - occasion_based : tied to an event or setting ("for a beach wedding")
  - note_based     : already contains specific note names ("something with oud and rose")
  - follow_up      : references a previous recommendation ("what about something lighter?")
"""
from __future__ import annotations

_INTENT_PROMPT = """\
You are a perfume-intent classifier. Given the user query below, output EXACTLY one of these \
labels (no other text):
  mood_based | occasion_based | note_based | follow_up

Rules:
- mood_based    : emotional/lifestyle descriptors, no specific notes mentioned
- occasion_based: references a specific setting, event, season, or time of day
- note_based    : mentions one or more actual perfume ingredients (rose, oud, musk, etc.)
- follow_up     : references a previous recommendation or asks for a variation

User query: {query}
Label:"""


def classify_intent(query: str) -> str:
    """
    Classify the user's fragrance query into one of four categories:
    mood_based, occasion_based, note_based, or follow_up.
    """
    from src.model import load_llm

    llm = load_llm()
    raw = llm.invoke(_INTENT_PROMPT.format(query=query)).strip().lower()

    for label in ("mood_based", "occasion_based", "note_based", "follow_up"):
        if label in raw:
            return label
    return "mood_based"
