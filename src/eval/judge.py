"""
LLM-judge — sends (query, response, retrieved) to OpenAI and parses scores.

Blind and strict: no rubric hints, no intent labels. Explicit score anchors
are designed so that a generic baseline model scores ~3 and a fine-tuned
expert-voice model scores 4-5, making the delta meaningful.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass


_JUDGE_PROMPT = """\
You are an evaluator of an AI fragrance recommendation system. Be critical — reserve 5 for excellent responses and use the full range of scores.

User query: {query}

Agent response:
{response}

Retrieved perfume candidates the agent had access to:
{retrieved}

Rate on three dimensions (integer 1-5):

RELEVANCE — does the recommendation match the specific nuances of the query?
  5 = addresses every nuance, nothing irrelevant
  4 = good match, minor gap
  3 = broadly correct but generic — could be a response to any similar query
  2 = misses key aspects of the query
  1 = wrong direction entirely

GROUNDEDNESS — is the response grounded in the actual retrieved candidates with no hallucination?
  5 = names specific perfumes from the retrieved list AND cites their actual notes/accords
  4 = names perfumes from the list, minimal note detail
  3 = mentions perfumes but notes are vague or unverifiable from the retrieved list
  2 = some hallucinated perfume names or notes not present in the retrieved list
  1 = ignores retrieved candidates, mostly fabricated

HELPFULNESS — does the response read like a real fragrance expert, not a generic chatbot?
  5 = specific perfume names + brands, explains WHY each fits using actual notes, expert confident tone, 3-5 focused sentences
  4 = names perfumes, gives some note-based reasoning
  3 = generic advice, catalogue-style, no real expertise shown — reads like it could have been written without knowing anything about perfume
  2 = vague, surface-level, unhelpful to someone who actually wants to buy a perfume
  1 = confusing, off-topic, or refuses to recommend without good reason

Respond with ONLY valid JSON, no markdown, no prose:
{{"relevance": <int>, "groundedness": <int>, "helpfulness": <int>, "reasoning": "<one sentence identifying the main strength or weakness>"}}
"""


@dataclass
class JudgeScore:
    relevance: int
    groundedness: int
    helpfulness: int
    reasoning: str
    raw: str  # raw judge output, for debugging


def _extract_json(text: str) -> dict:
    """Tolerate small judge slip-ups: strip code fences, grab first {...} blob."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in judge output: {text!r}")
    return json.loads(m.group(0))


def judge(
    *,
    query: str,
    response: str,
    retrieved: str,
    model: str = "gpt-4o-mini",
) -> JudgeScore:
    """Single blind judge call. Raises on missing API key or unparseable output."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or export it before running eval."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    prompt = _JUDGE_PROMPT.format(
        query=query or "(empty)",
        response=response or "(empty)",
        retrieved=retrieved or "(no candidates retrieved)",
    )

    last_err: Exception | None = None
    raw_out = ""
    for attempt in range(2):
        suffix = "" if attempt == 0 else "\n\nReminder: output ONLY the JSON object, nothing else."
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt + suffix}],
            temperature=0.0,
        )
        raw_out = completion.choices[0].message.content or ""
        try:
            data = _extract_json(raw_out)
            return JudgeScore(
                relevance=int(data["relevance"]),
                groundedness=int(data["groundedness"]),
                helpfulness=int(data["helpfulness"]),
                reasoning=str(data.get("reasoning", "")).strip(),
                raw=raw_out,
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            last_err = e

    raise ValueError(f"Judge produced unparseable output after retry: {raw_out!r}") from last_err
