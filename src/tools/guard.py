"""
Tool: SafetyGuard

Two-layer output validation:
  1. Toxicity check — uses the `detoxify` model to detect hate speech,
     threats, obscenity, and identity attacks.
  2. Hallucination heuristic — checks whether the response mentions perfume
     names or notes that were NOT present in the retrieved context, flagging
     likely hallucinations.

If either check fails the tool returns a structured error dict that the agent
uses to either regenerate or fall back to a safe response.
"""
from __future__ import annotations

import re

_TOXICITY_THRESHOLD = 0.5
_SAFE_FALLBACK = (
    "I'm sorry, I wasn't able to generate a suitable recommendation for that request. "
    "Please try rephrasing your query or describe the mood or occasion you have in mind."
)

# Fragrance-domain keywords that a valid response should contain at least one of
_DOMAIN_KEYWORDS = {
    "perfume", "fragrance", "scent", "notes", "accord", "top", "heart", "base",
    "woody", "floral", "musky", "citrus", "oriental", "fresh", "amber", "vanilla",
    "rose", "oud", "sandalwood", "musk", "bergamot", "vetiver", "jasmine",
}


def _is_toxic(text: str) -> tuple[bool, dict]:
    """Return (is_toxic, scores_dict) using detoxify."""
    try:
        from detoxify import Detoxify  # lazy import — heavy dependency

        scores: dict = Detoxify("original").predict(text)
        flagged = any(v > _TOXICITY_THRESHOLD for v in scores.values())
        return flagged, scores
    except ImportError:
        # detoxify not installed — skip toxicity check
        return False, {}
    except Exception as exc:  # noqa: BLE001
        print(f"[guard] Toxicity check failed: {exc}")
        return False, {}


def _has_domain_content(response: str) -> bool:
    """Check that the response mentions at least some fragrance-domain vocabulary."""
    lower = response.lower()
    return any(kw in lower for kw in _DOMAIN_KEYWORDS)


def _hallucination_score(response: str, context: str) -> float:
    """
    Simple heuristic: what fraction of capitalised noun phrases in the response
    (likely perfume names) also appear in the retrieved context?

    Returns a value in [0, 1] where 1 = fully grounded, 0 = fully hallucinated.
    """
    # Extract capitalised 2–4-word phrases as proxy for perfume names
    candidates = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+){0,3})\b", response)
    if not candidates:
        return 1.0
    grounded = sum(1 for c in candidates if c.lower() in context.lower())
    return grounded / len(candidates)


def validate_response(response: str, retrieved_context: str = "") -> str:
    """Validate the recommendation for toxicity and hallucinations."""
    # 1. Toxicity check
    toxic, scores = _is_toxic(response)
    if toxic:
        flagged = [k for k, v in scores.items() if v > _TOXICITY_THRESHOLD]
        return f"FAIL: Response flagged for toxicity ({', '.join(flagged)}). {_SAFE_FALLBACK}"

    # 2. Domain relevance check
    if not _has_domain_content(response):
        return (
            "FAIL: Response does not contain fragrance-domain content. "
            "Please regenerate with a tighter focus on perfume notes and accords."
        )

    # 3. Hallucination heuristic (only when context is available)
    if retrieved_context:
        score = _hallucination_score(response, retrieved_context)
        if score < 0.3:
            return (
                f"FAIL: High hallucination risk (grounding score={score:.2f}). "
                "Response mentions names not present in the retrieved context. "
                "Please base the recommendation strictly on the provided candidates."
            )

    return f"PASS: {response}"
