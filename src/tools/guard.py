"""
Tool: SafetyGuard

Three-layer output validation:
  1. Toxicity — fast regex blocklist pre-filter, then detoxify model.
  2. Hallucination guard — spaCy NER (PRODUCT/ORG entities) + context grounding.
     Falls back to capitalised-phrase heuristic if spaCy is unavailable.
  3. Output length — truncate at last complete sentence if response exceeds 600 tokens.

Domain-relevance check removed: the guard validates the *response*, not the query.
Indirect queries ("a perfume for Paris") are valid input handled by HyDE upstream.
"""
from __future__ import annotations

import re

_TOXICITY_THRESHOLD = 0.5
_HALLUCINATION_THRESHOLD = 0.5
_MAX_TOKENS = 600

_SAFE_FALLBACK = (
    "I'm sorry, I wasn't able to generate a suitable recommendation for that request. "
    "Please try rephrasing your query or describe the mood or occasion you have in mind."
)

# Fast pre-filter for obvious prohibited language before invoking the heavy model.
# No trailing \b so variants like "fucking", "shitty", "bitching" are also caught.
_BLOCKLIST_RE = re.compile(
    r"\b(fuck|shit|bitch|asshole|bastard|cunt|nigger|nigga|faggot|retard|whore)",
    re.IGNORECASE,
)


# ── Toxicity ──────────────────────────────────────────────────────────────────

def _blocklist_check(text: str) -> bool:
    return bool(_BLOCKLIST_RE.search(text))


def _is_toxic(text: str) -> tuple[bool, dict]:
    """Return (is_toxic, scores_dict) using detoxify."""
    try:
        from detoxify import Detoxify  # lazy import — heavy model
        scores: dict = Detoxify("original").predict(text)
        flagged = any(v > _TOXICITY_THRESHOLD for v in scores.values())
        return flagged, scores
    except ImportError:
        return False, {}
    except Exception as exc:
        print(f"[guard] Toxicity check failed: {exc}")
        return False, {}


# ── Hallucination ─────────────────────────────────────────────────────────────

def _extract_entities(text: str) -> list[str]:
    """
    Extract PRODUCT and ORG entities via spaCy NER.
    Falls back to capitalised noun-phrase regex when spaCy/model is unavailable.
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded yet — use heuristic fallback
            return re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", text)
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ("PRODUCT", "ORG")]
    except ImportError:
        return re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", text)


def _hallucination_score(response: str, context: str) -> float:
    """
    Grounding score: fraction of extracted entities present in the retrieved context.
    Returns 1.0 when no entities are found (nothing to ground).
    """
    entities = _extract_entities(response)
    if not entities:
        return 1.0
    context_lower = context.lower()
    grounded = sum(1 for e in entities if e.lower() in context_lower)
    return grounded / len(entities)


# ── Output length ─────────────────────────────────────────────────────────────

def _truncate_to_sentence(text: str, max_tokens: int = _MAX_TOKENS) -> str:
    """Truncate to max_tokens (approx words) at the last complete sentence boundary."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    truncated = " ".join(words[:max_tokens])
    last_boundary = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
    if last_boundary > 0:
        return truncated[:last_boundary + 1].strip()
    return truncated.strip()


# ── Public API ────────────────────────────────────────────────────────────────

def validate_response(response: str, retrieved_context: str = "") -> str:
    """Validate a recommendation response. Returns 'PASS: <text>' or 'FAIL: <reason>'."""
    # 1a. Fast blocklist pre-filter
    if _blocklist_check(response):
        return f"FAIL: Response contains prohibited language. {_SAFE_FALLBACK}"

    # 1b. Detoxify model
    toxic, scores = _is_toxic(response)
    if toxic:
        flagged = [k for k, v in scores.items() if v > _TOXICITY_THRESHOLD]
        return f"FAIL: Response flagged for toxicity ({', '.join(flagged)}). {_SAFE_FALLBACK}"

    # 2. Hallucination guard
    if retrieved_context:
        score = _hallucination_score(response, retrieved_context)
        if score < _HALLUCINATION_THRESHOLD:
            return (
                f"FAIL: High hallucination risk (grounding score={score:.2f}). "
                "Response mentions names not present in the retrieved context. "
                "Please base the recommendation strictly on the provided candidates."
            )

    # 3. Output length guard
    response = _truncate_to_sentence(response)

    return f"PASS: {response}"
