"""
NoseKnows Agent — custom ReAct loop with structured fallback.

Small models (2-3B) often skip tool calls entirely. This agent handles that
with a two-tier strategy:
  1. ReAct loop — ask the LLM to reason step-by-step using tools.
  2. Structured fallback — if the LLM skips all tools, run the essential
     pipeline directly (intent → HyDE → retrieve → recommend → validate).

Usage:
    from src.agent.agent import build_agent, run_agent

    agent = build_agent()
    result = run_agent(agent, "I want something cozy for a winter date night")
    print(result["output"])
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from src.tools.guard import validate_response
from src.tools.hyde import generate_hyde_document
from src.tools.intent import classify_intent
from src.tools.recommender import generate_recommendation
from src.tools.retriever import retrieve_fragrances

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS: dict[str, Callable] = {
    "classify_intent": classify_intent,
    "generate_hyde_document": generate_hyde_document,
    "retrieve_fragrances": retrieve_fragrances,
    "generate_recommendation": generate_recommendation,
    "validate_response": validate_response,
}

_TOOL_DESCRIPTIONS = """\
classify_intent(query) -> "mood_based" | "occasion_based" | "note_based" | "follow_up"
generate_hyde_document(query) -> hypothetical perfume profile (notes + accords)
retrieve_fragrances(query | filters_json) -> numbered list of matching perfumes
generate_recommendation(original_query | retrieved_context) -> warm recommendation text
validate_response(recommendation | retrieved_context) -> "PASS: ..." or "FAIL: ..."
"""

_FEW_SHOT = """\
Example:
Question: I want something fresh and green for a morning walk in the park.
Thought: I should classify the intent first.
Action: classify_intent
Action Input: I want something fresh and green for a morning walk in the park.
Observation: occasion_based
Thought: This is occasion_based, so I should generate a HyDE document.
Action: generate_hyde_document
Action Input: I want something fresh and green for a morning walk in the park.
Observation: Top notes: bergamot, green tea, lemon\\nHeart notes: violet leaf, muguet\\nBase notes: white musk, cedar\\nAccords: fresh, green, aromatic
Thought: Now I retrieve matching fragrances using the HyDE document.
Action: retrieve_fragrances
Action Input: Top notes: bergamot, green tea, lemon\\nHeart notes: violet leaf, muguet\\nBase notes: white musk, cedar\\nAccords: fresh, green, aromatic
Observation: 1. L'Eau d'Issey — Issey Miyake\\n   Top: yuzu, freesia\\n   Heart: water lily\\n   Base: cedar, musk
Thought: I have candidates. Now I generate the recommendation.
Action: generate_recommendation
Action Input: I want something fresh and green for a morning walk in the park. | 1. L'Eau d'Issey — Issey Miyake\\n   Top: yuzu, freesia\\n   Heart: water lily\\n   Base: cedar, musk
Observation: For a morning walk, L'Eau d'Issey is perfect — its yuzu top note...
Thought: I should validate this recommendation.
Action: validate_response
Action Input: For a morning walk, L'Eau d'Issey is perfect — its yuzu top note... | 1. L'Eau d'Issey — Issey Miyake
Observation: PASS: For a morning walk, L'Eau d'Issey is perfect — its yuzu top note...
Thought: I now have the validated recommendation.
Final Answer: For a morning walk, L'Eau d'Issey is perfect — its yuzu top note...
"""

_SYSTEM_PROMPT = f"""\
You are NoseKnows, an expert AI fragrance consultant.
You MUST use tools to answer every question. Never answer directly without calling tools first.

Tools available:
{_TOOL_DESCRIPTIONS}

IMPORTANT FORMAT RULES — follow EXACTLY:
1. Every step must start with "Thought:", then "Action:", then "Action Input:".
2. Wait for the "Observation:" before writing the next Thought.
3. Only write "Final Answer:" after calling validate_response.
4. For retrieve_fragrances and generate_recommendation, separate two arguments with " | ".

{_FEW_SHOT}
Now answer the following question using the same step-by-step format:
"""


# ---------------------------------------------------------------------------
# Agent dataclass
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    llm: Any
    max_iterations: int = 8
    verbose: bool = True


def build_agent(verbose: bool = True) -> Agent:
    from src.model import load_llm
    return Agent(
        llm=load_llm(),
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _call_tool(action: str, action_input: str) -> str:
    tool_fn = _TOOLS.get(action.strip())
    if tool_fn is None:
        return f"Unknown tool '{action}'. Available: {', '.join(_TOOLS)}"

    parts = action_input.split(" | ", 1)
    arg1 = parts[0].strip()
    arg2 = parts[1].strip() if len(parts) > 1 else ""

    name = action.strip()
    if name == "retrieve_fragrances":
        return tool_fn(arg1, arg2 or "{}")
    if name == "generate_recommendation":
        return tool_fn(arg1, arg2)
    if name == "validate_response":
        return tool_fn(arg1, arg2)
    return tool_fn(action_input.strip())


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse(text: str) -> tuple[str, str | None, str | None, str | None]:
    thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    thought = thought_m.group(1).strip() if thought_m else text.strip()

    final_m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if final_m:
        return thought, None, None, final_m.group(1).strip()

    action_m = re.search(r"Action:\s*(\w+)", text)
    input_m = re.search(r"Action Input:\s*(.+?)(?=\nThought:|\nAction:|\nFinal Answer:|$)", text, re.DOTALL)

    return (
        thought,
        action_m.group(1).strip() if action_m else None,
        input_m.group(1).strip() if input_m else "",
        None,
    )


# ---------------------------------------------------------------------------
# Structured fallback pipeline (runs when the LLM skips all tools)
# ---------------------------------------------------------------------------

_GUARD_REFUSAL = (
    "I couldn't understand that as a fragrance request. Could you tell me a bit more — "
    "the mood you want, the occasion, or any notes you love (or want to avoid)?"
)

_HALLUCINATION_REFUSAL = (
    "I couldn't find good matches in our catalogue for that one — and I'd rather not "
    "guess. Could you rephrase with a mood, occasion, or specific notes you'd like?"
)


def _run_structured_pipeline(query: str, filters: dict | None = None) -> dict[str, Any]:
    """Deterministic pipeline: intent → HyDE → retrieve → recommend → validate.

    Routes to a refusal message in two cases:
      - intent classifies as `guard` (gibberish/off-topic/injection)
      - validate_response returns FAIL (hallucination detected)
    """
    steps = []
    filters_json = ""
    if filters:
        import json
        filters_json = json.dumps({k: v for k, v in filters.items() if k != "top_k"})

    class _Step:
        def __init__(self, tool, tool_input):
            self.tool = tool
            self.tool_input = tool_input

    # 1. Intent
    intent = classify_intent(query)
    steps.append(("classify_intent", intent))

    # Short-circuit on guard intent — no retrieval, no LLM recommendation
    if intent == "guard":
        steps.append(("validate_response", "REFUSE: guard intent classified — query not answerable as fragrance request"))
        return {
            "output": _GUARD_REFUSAL,
            "intermediate_steps": [(_Step(t, i), i) for t, i in steps],
            "hyde_doc": None,
        }

    # 2. HyDE (skip for note_based)
    hyde_doc = None
    retrieval_query = query
    if intent in ("mood_based", "occasion_based"):
        hyde_doc = generate_hyde_document(query)
        retrieval_query = hyde_doc
        steps.append(("generate_hyde_document", hyde_doc))

    # 3. Retrieve
    retrieved = retrieve_fragrances(retrieval_query, filters_json)
    steps.append(("retrieve_fragrances", retrieved))

    # 4. Recommend
    recommendation = generate_recommendation(query, retrieved)
    steps.append(("generate_recommendation", recommendation))

    # 5. Validate — on FAIL, refuse instead of returning the unvalidated response
    validation = validate_response(recommendation, retrieved)
    steps.append(("validate_response", validation))

    if validation.strip().upper().startswith("FAIL"):
        output = _HALLUCINATION_REFUSAL
    else:
        output = re.sub(r"^PASS:\s*", "", validation, flags=re.IGNORECASE).strip()

    return {
        "output": output,
        "intermediate_steps": [(_Step(t, i), i) for t, i in steps],
        "hyde_doc": hyde_doc,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agent(
    agent: Agent,
    query: str,
    filters: dict | None = None,
) -> dict[str, Any]:
    augmented = query
    if filters:
        augmented += " [filters: " + ", ".join(f"{k}={v}" for k, v in filters.items()) + "]"

    scratchpad = ""
    steps = []
    hyde_doc: str | None = None
    tools_used: set[str] = set()

    prompt = _SYSTEM_PROMPT + f"\nQuestion: {augmented}\n"

    for iteration in range(agent.max_iterations):
        full_prompt = prompt + scratchpad
        raw = agent.llm.invoke(full_prompt)

        if agent.verbose:
            print(f"\n── iter {iteration + 1} ──\n{raw}\n")

        thought, action, action_input, final_answer = _parse(raw)

        if final_answer and "retrieve_fragrances" in tools_used:
            output = re.sub(r"^PASS:\s*", "", final_answer, flags=re.IGNORECASE).strip()
            return {"output": output, "intermediate_steps": steps, "hyde_doc": hyde_doc}

        if not action:
            break  # no parseable action — fall through to structured pipeline

        observation = _call_tool(action, action_input)
        tools_used.add(action)

        if agent.verbose:
            print(f"[{action}] → {observation[:300]}")

        if action == "generate_hyde_document":
            hyde_doc = observation

        class _Step:
            def __init__(self, t, i):
                self.tool = t
                self.tool_input = i

        steps.append((_Step(action, action_input), observation))
        scratchpad += (
            f"Thought: {thought}\nAction: {action}\n"
            f"Action Input: {action_input}\nObservation: {observation}\n"
        )

    # LLM didn't use essential tools — run structured pipeline instead
    if agent.verbose:
        print("[agent] Falling back to structured pipeline.")
    return _run_structured_pipeline(query, filters)
