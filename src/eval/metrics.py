"""Aggregation helpers for eval results — mean/std overall and per-intent."""
from __future__ import annotations

import math
from collections import defaultdict


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def aggregate(results: list[dict]) -> dict:
    """
    results: list of rows written to eval_results.jsonl. Each row has:
      intent, scores: {relevance, groundedness, helpfulness}, error (optional)
    """
    ok = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    dims = ("relevance", "groundedness", "helpfulness")

    overall = {}
    for d in dims:
        vals = [r["scores"][d] for r in ok]
        m, s = _mean_std(vals)
        overall[d] = {"mean": round(m, 2), "std": round(s, 2), "n": len(vals)}

    per_intent: dict[str, dict] = defaultdict(dict)
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_intent[r["intent"]].append(r)
    for intent, rows in by_intent.items():
        for d in dims:
            vals = [r["scores"][d] for r in rows]
            m, s = _mean_std(vals)
            per_intent[intent][d] = {"mean": round(m, 2), "std": round(s, 2), "n": len(vals)}

    return {
        "total": len(results),
        "scored": len(ok),
        "failed": len(failed),
        "overall": overall,
        "per_intent": dict(per_intent),
    }


def format_summary(agg: dict, tag: str) -> str:
    lines = [
        "",
        f"=== Eval summary — tag: {tag} ===",
        f"Total: {agg['total']}  scored: {agg['scored']}  failed: {agg['failed']}",
        "",
        "Overall (mean ± std, n):",
    ]
    for d, v in agg["overall"].items():
        lines.append(f"  {d:14s} {v['mean']:.2f} ± {v['std']:.2f}  (n={v['n']})")
    lines.append("")
    lines.append("Per intent:")
    for intent, dims in sorted(agg["per_intent"].items()):
        parts = [f"{d}={dims[d]['mean']:.2f}" for d in ("relevance", "groundedness", "helpfulness")]
        n = next(iter(dims.values()))["n"]
        lines.append(f"  {intent:16s} " + "  ".join(parts) + f"   (n={n})")
    lines.append("")
    return "\n".join(lines)
