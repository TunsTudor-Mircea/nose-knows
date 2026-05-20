"""
CLI eval runner — feeds each query in data/eval_queries.jsonl through the live
agent, sends (query, response, retrieved) to the LLM judge, writes per-query
results to data/eval_results.jsonl, and prints an aggregate summary.

Usage:
    python -m src.eval.run_eval \\
        --queries data/eval_queries.jsonl \\
        --output data/eval_results.jsonl \\
        --tag baseline-gemma2-2b \\
        --model gpt-4o-mini

    # Smoke test on first 5 queries:
    python -m src.eval.run_eval --limit 5 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.agent.agent import build_agent, run_agent
from src.eval.judge import judge
from src.eval.metrics import aggregate, format_summary


def _extract_retrieved(intermediate_steps: list) -> str:
    """Pull the retrieve_fragrances observation out of the agent's step trace."""
    for step, observation in intermediate_steps:
        if getattr(step, "tool", None) == "retrieve_fragrances":
            return observation
    return ""


def _serialize_trace(intermediate_steps: list) -> list[dict]:
    """Convert intermediate_steps into a JSON-serializable list for the frontend."""
    trace = []
    for step, observation in intermediate_steps:
        trace.append({
            "tool": getattr(step, "tool", "unknown"),
            "tool_input": getattr(step, "tool_input", ""),
            "observation": observation,
        })
    return trace


def _run_one(agent, row: dict, judge_model: str) -> dict:
    query = row["query"]

    record: dict = {
        "id": row["id"],
        "query": query,
        "intent": row["intent"],
    }

    t0 = time.time()
    try:
        agent_result = run_agent(agent, query)
        response = agent_result.get("output", "") or ""
        intermediate_steps = agent_result.get("intermediate_steps", [])
        retrieved = _extract_retrieved(intermediate_steps)
        record["response"] = response
        record["retrieved"] = retrieved
        record["hyde_doc"] = agent_result.get("hyde_doc")
        record["trace"] = _serialize_trace(intermediate_steps)
        record["agent_latency_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        record["error"] = f"agent_failed: {e}"
        record["traceback"] = traceback.format_exc()
        return record

    try:
        score = judge(
            query=query,
            response=response,
            retrieved=retrieved,
            model=judge_model,
        )
        record["scores"] = {
            "relevance": score.relevance,
            "groundedness": score.groundedness,
            "helpfulness": score.helpfulness,
        }
        record["judge_reasoning"] = score.reasoning
    except Exception as e:
        record["error"] = f"judge_failed: {e}"
        record["traceback"] = traceback.format_exc()

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM-judge eval over the static query set.")
    parser.add_argument("--queries", default="data/eval_queries.jsonl",
                        help="Path to eval queries JSONL.")
    parser.add_argument("--output", default="data/eval_results.jsonl",
                        help="Path to append per-query result rows.")
    parser.add_argument("--tag", required=True,
                        help="Run label, e.g. 'baseline-gemma2-2b' or 'finetuned-lora-v1'.")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="Judge model id.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of queries (smoke testing).")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress agent verbose ReAct trace.")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"error: queries file not found: {queries_path}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with queries_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} queries from {queries_path}")
    print(f"Tag: {args.tag}  Judge: {args.model}")

    agent = build_agent(verbose=not args.quiet)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    with output_path.open("a") as out:
        for i, row in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}] {row['id']}  intent={row['intent']}")
            rec = _run_one(agent, row, args.model)
            rec["tag"] = args.tag
            rec["run_id"] = run_id
            rec["judge_model"] = args.model
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            results.append(rec)

            if "scores" in rec:
                s = rec["scores"]
                print(f"  → rel={s['relevance']} grd={s['groundedness']} hlp={s['helpfulness']}"
                      f"   {rec.get('judge_reasoning','')}")
            else:
                print(f"  ! {rec.get('error','unknown error')}")

    agg = aggregate(results)
    print(format_summary(agg, args.tag))
    print(f"Per-row results appended to: {output_path}")
    print(f"Run id: {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
