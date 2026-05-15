"""
Export DPO (Direct Preference Optimisation) training pairs from Postgres feedback.

Usage:
    python -m src.rlhf.export_dpo_pairs
    python -m src.rlhf.export_dpo_pairs --output data/dpo_pairs.jsonl --min-pairs 5

For each user query that received at least one positive (score=1) AND one negative
(score=-1) rated assistant response (across any session), this script emits a JSONL
line in the format expected by trl.DPOTrainer:

    {"prompt": "<user query>", "chosen": "<good response>", "rejected": "<bad response>"}

Strategy when a query has multiple rated responses:
  - chosen  → the response with the highest score (ties: most recent)
  - rejected → the response with the lowest score (ties: most recent)

If a query only has all-positive or all-negative ratings, it is skipped (DPO needs
contrast). The script prints a summary at the end.

Cross-session matching: two responses are considered "for the same query" when their
user-message text is identical (exact match, case-insensitive strip). This means
repeated queries across sessions contribute to the pairing pool.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def _fetch_rated_responses() -> list[dict]:
    """Return all feedback rows joined with their assistant message and preceding user query."""
    from sqlalchemy import select, text
    from src.db.session import AsyncSessionLocal
    from src.db.models import Feedback as DBFeedback, Message as DBMessage

    results = []
    async with AsyncSessionLocal() as db:
        fb_rows = (
            await db.execute(
                select(DBFeedback).order_by(DBFeedback.created_at.asc())
            )
        ).scalars().all()

        for fb in fb_rows:
            # Load the assistant message that was rated
            asst_msg = (
                await db.execute(
                    select(DBMessage).where(DBMessage.id == fb.message_id)
                )
            ).scalar_one_or_none()

            if asst_msg is None or asst_msg.role != "assistant":
                continue

            # Find the user message that prompted this assistant response
            user_msg = (
                await db.execute(
                    select(DBMessage)
                    .where(
                        DBMessage.session_id == asst_msg.session_id,
                        DBMessage.role == "user",
                        DBMessage.created_at < asst_msg.created_at,
                    )
                    .order_by(DBMessage.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

            if user_msg is None:
                continue

            results.append(
                {
                    "prompt": user_msg.content.strip(),
                    "response": asst_msg.content.strip(),
                    "score": fb.score,          # -1 or 1
                    "feedback_at": fb.created_at.isoformat(),
                    "message_id": str(asst_msg.id),
                    "session_id": str(asst_msg.session_id),
                }
            )

    return results


def _build_dpo_pairs(rows: list[dict]) -> list[dict]:
    """
    Group rows by normalised prompt text and emit (chosen, rejected) pairs.
    Returns list of DPO pair dicts.
    """
    # Group by lowercased-stripped prompt
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = row["prompt"].lower().strip()
        by_prompt[key].append(row)

    pairs: list[dict] = []
    skipped_no_contrast = 0

    for prompt_key, responses in by_prompt.items():
        positives = [r for r in responses if r["score"] > 0]
        negatives = [r for r in responses if r["score"] < 0]

        if not positives or not negatives:
            skipped_no_contrast += 1
            continue

        # Pick best positive (highest score, break ties by recency)
        chosen = sorted(positives, key=lambda r: (r["score"], r["feedback_at"]), reverse=True)[0]
        # Pick worst negative (lowest score, break ties by recency)
        rejected = sorted(negatives, key=lambda r: (r["score"], r["feedback_at"]))[0]

        # Use the original-case prompt from the chosen row
        pairs.append(
            {
                "prompt": chosen["prompt"],
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                # Metadata (not used by DPOTrainer but useful for debugging)
                "_chosen_message_id": chosen["message_id"],
                "_rejected_message_id": rejected["message_id"],
            }
        )

    return pairs, skipped_no_contrast  # noqa: return type is tuple[list, int]


def _write_jsonl(pairs: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


async def main(output: str, min_pairs: int, dry_run: bool) -> None:
    print("Fetching feedback from Postgres…")
    rows = await _fetch_rated_responses()
    print(f"  Found {len(rows)} rated responses.")

    if not rows:
        print("No feedback found. Run the app and rate some responses first.")
        sys.exit(0)

    pairs, skipped = _build_dpo_pairs(rows)  # type: ignore[misc]

    print(f"  Built {len(pairs)} DPO pairs  ({skipped} prompts skipped — no contrast).")

    if len(pairs) < min_pairs:
        print(
            f"WARNING: only {len(pairs)} pairs found (minimum requested: {min_pairs}).\n"
            "  Collect more feedback (mix of thumbs-up and thumbs-down) before training."
        )

    if dry_run:
        print("\n--- DRY RUN: first 3 pairs ---")
        for p in pairs[:3]:
            print(json.dumps(p, indent=2, ensure_ascii=False))
        print("(not writing file)")
        return

    _write_jsonl(pairs, output)
    print(f"  Written to: {output}")

    # Summary stats
    scores = [r["score"] for r in rows]
    pos = sum(1 for s in scores if s > 0)
    neg = sum(1 for s in scores if s < 0)
    print(
        f"\nSummary:\n"
        f"  Total rated responses : {len(rows)}\n"
        f"  Positive (thumbs up)  : {pos}\n"
        f"  Negative (thumbs down): {neg}\n"
        f"  DPO pairs exported    : {len(pairs)}\n"
        f"  Output file           : {output}"
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Export DPO training pairs from NoseKnows Postgres feedback."
    )
    parser.add_argument(
        "--output",
        default=os.getenv("DPO_PAIRS_PATH", "data/dpo_pairs.jsonl"),
        help="Output JSONL file path (default: data/dpo_pairs.jsonl)",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=10,
        help="Warn if fewer than N pairs are produced (default: 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print first 3 pairs without writing any file",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output, args.min_pairs, args.dry_run))


if __name__ == "__main__":
    cli()
