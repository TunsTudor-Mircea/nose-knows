"""
Synthetic fine-tuning data generator.

For each perfume in data/chunks.jsonl, prompts the SLM to produce an
(input, reasoning, output) triple of the form:

  Input:    A subjective, lifestyle-oriented user request
  Reasoning: Step-by-step olfactory reasoning
  Output:   Structured note recommendation with rationale

Results are written to data/synthetic_triples.jsonl.

Usage:
    python -m src.finetune.generate_synthetic
    python -m src.finetune.generate_synthetic --sample 500 --output data/synthetic_triples.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

_GENERATION_PROMPT = """\
You are a perfumery creative writer and olfactory expert.

Below is a real perfume's note profile. Your task is to invent a realistic user request \
that this perfume would perfectly answer, then write the reasoning and recommendation.

Perfume profile:
{perfume_profile}

Generate a training example in EXACTLY this JSON format (no extra text):
{{
  "input": "<a subjective user request using lifestyle/mood language, not technical note names>",
  "reasoning": "<step-by-step reasoning connecting the user's language to the specific notes>",
  "output": "<the final recommendation explaining which notes match and why>"
}}

JSON:"""

_INSTRUCTION_SYSTEM = (
    "You are NoseKnows, an expert AI fragrance consultant. "
    "Given the user's description, recommend a perfume and explain your reasoning."
)


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _extract_json(raw: str) -> dict | None:
    """Try to extract a JSON object from the model's raw output."""
    import re

    # Find the first {...} block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def generate_triples(
    chunks: list[dict],
    sample_size: int,
    output_path: str,
    seed: int = 42,
) -> None:
    from src.model import load_llm

    llm = load_llm()

    random.seed(seed)
    selected = random.sample(chunks, min(sample_size, len(chunks)))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for chunk in tqdm(selected, desc="Generating triples"):
            prompt = _GENERATION_PROMPT.format(perfume_profile=chunk["text"])
            try:
                raw = llm.invoke(prompt)
                triple = _extract_json(raw)
                if not triple or not all(k in triple for k in ("input", "reasoning", "output")):
                    skipped += 1
                    continue

                # Convert to instruction-following format for SFTTrainer
                record = {
                    "perfume_id": chunk["id"],
                    "perfume_name": chunk["metadata"].get("perfume", ""),
                    "brand": chunk["metadata"].get("brand", ""),
                    "raw_triple": triple,
                    # Formatted as a chat-style prompt for fine-tuning
                    "prompt": (
                        f"<s>[INST] <<SYS>>\n{_INSTRUCTION_SYSTEM}\n<</SYS>>\n\n"
                        f"{triple['input']} [/INST]"
                    ),
                    "completion": (
                        f"Reasoning: {triple['reasoning']}\n\n"
                        f"Recommendation: {triple['output']}</s>"
                    ),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[synthetic] Error on chunk {chunk['id']}: {exc}")
                skipped += 1

    print(f"Done. Written: {written:,}  Skipped: {skipped:,}  → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fine-tuning triples")
    parser.add_argument("--chunks", default="data/chunks.jsonl")
    parser.add_argument("--output", default="data/synthetic_triples.jsonl")
    parser.add_argument("--sample", type=int, default=5000, help="Number of perfumes to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks):,} chunks. Sampling {args.sample}.")
    generate_triples(chunks, args.sample, args.output, args.seed)
