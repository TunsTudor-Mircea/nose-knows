# Dataset Generation — NoseKnows

This document describes the synthetic dataset generation phase for NoseKnows. The goal is to take the raw Fragrantica CSV and produce a structured JSONL file of training conversations for fine-tuning Gemma 4. Everything runs inside a single Kaggle notebook, split across two phases: preprocessing and generation.

---

## Why synthetic data

The Fragrantica dataset contains perfume records, not conversations. A model trained directly on raw CSV rows would not learn anything useful about talking to a user. What we need is thousands of examples of a specific shape: a user asks something about fragrance, and a knowledgeable assistant answers by grounding the recommendation in actual notes and accords.

No such conversational dataset exists for perfumery at this scale, so we generate it synthetically using a larger model, Qwen3-8B, to write the conversations. This is a well-established technique in LLM fine-tuning: use a strong model to produce high-quality training data for a smaller one. The quality of this synthetic data is the single biggest factor in how well the fine-tuned model performs.

---

## Hardware

The notebook runs on Kaggle's dual T4 GPU configuration. Each T4 has 16 GB VRAM; `device_map="auto"` splits the model across both cards automatically. The P100 is not suitable because PyTorch 2.10+ does not support its CUDA capability (sm_60), causing all generation calls to fail silently.

---

## Phase 1 — Preprocessing

Before generation starts, the raw CSV goes through a preprocessing pipeline that runs on CPU and takes about five minutes. It is fully deterministic, so re-running it on session resume costs nothing.

### Loading

The CSV is semicolon-separated and latin-1 encoded, with 24,063 rows. Rows missing perfume names, brands, note fields in all tiers, or rating counts are dropped immediately.

### Normalization

Several fields need cleaning before use. Note fields (top, middle, base) are parsed into Python lists: parenthetical qualifiers like "bergamot (Italian)" are stripped to just "bergamot", and "and" separators are unified to commas. The rating value column uses comma-decimal format ("1,42") which is converted to standard floats. Gender values are unified to men, women, or unisex. Implausible year values outside 1900 to 2024 are set to NaN rather than dropping the row, since year is useful context but not mandatory.

### Quality filters

Four filters run in sequence. Perfumes with fewer than 50 ratings are removed — below this threshold the rating is unreliable and the perfume is too niche to recommend. Each row must have at least two main accords or at least two top notes, otherwise there is not enough information for a grounded assistant turn. The total distinct notes across all three tiers must be at least four. Finally, placeholder profiles are dropped: if every note in the record is just an accord name repeated, the note fields add nothing beyond what the accord columns already say.

### Deduplication

Within each brand group, every pair of perfumes is compared by the symmetric difference of their full note fingerprints (the union of top, middle and base notes as a frozenset). If two perfumes from the same brand differ by two notes or fewer, one is dropped. The higher-rated one is kept. This removes flanker releases that would otherwise fill the dataset with near-identical examples. The comparison is O(N²) within each brand group, which is fine because no single brand has thousands of entries.

### Actual scale after preprocessing

Running on the real CSV produced these numbers:

- Input rows: 24,063
- After structural drop: 24,063 (none dropped at this stage)
- After quality filters: 18,267
- After deduplication: 17,804
- Gold perfumes (rating >= 200): 8,717
- Silver perfumes (50 <= rating < 200): 9,087
- Total training examples: 26,521

### Tier assignment

Perfumes with 200 or more ratings are gold and get two training examples generated. Perfumes between 50 and 199 ratings are silver and get one. Gold perfumes are more widely known, have more real-world reception behind their ratings, and are more likely to be asked about by actual users.

### Question type assignment

Training examples are assigned question types using largest-remainder allocation, which guarantees the distribution is exact rather than approximate. The five types and their proportions are: occasion_based at 25%, mood_based at 25%, note_based at 20%, comparison at 15%, and structured_preference at 15%. After allocation, types are interleaved so no single type clusters at one end of the dataset.

---

## Phase 2 — Generation

### Model loading

Qwen3-8B is loaded from HuggingFace using 4-bit NF4 quantization via bitsandbytes 0.46.1. This version is the minimum required by the installed transformers build. The correct loading configuration does not pass a top-level `dtype` argument alongside `quantization_config` — doing so causes the new transformers loading pipeline to cast weights to bfloat16 before quantizing, which spikes VRAM usage and causes OOM. The dtype is controlled entirely through `bnb_4bit_compute_dtype` inside `BitsAndBytesConfig`.

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` before loading reduces memory fragmentation on T4.

The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle — secrets require the `kaggle_secrets` API.

After loading, VRAM is verified across both GPUs. If total allocated is under 3 GB, quantization silently failed and the notebook raises immediately rather than letting thousands of batches fail. The correct allocation is approximately 6 GB total across both cards.

### Thinking mode

Qwen3-8B has a built-in reasoning mode that produces an internal chain of thought before generating output. This is activated by passing `enable_thinking=True` to `apply_chat_template`. The thinking trace is stripped from the output before JSON parsing — it is used internally to reason about which notes to highlight and how to connect them to the query.

The thinking budget is capped at 512 tokens. Without a cap, reasoning traces can reach 1,500 tokens or more per call, which would make the full run far too slow.

The `apply_chat_template` call returns a `BatchEncoding` object in newer transformers versions, not a plain tensor. Passing a `BatchEncoding` directly to `model.generate()` raises `AttributeError` because the generate function tries to access `.shape[0]` on it. The fix is to extract `input_ids` explicitly after the template call:

```python
if hasattr(template_output, "input_ids"):
    input_ids = template_output.input_ids.to(model.device)
elif isinstance(template_output, dict):
    input_ids = template_output["input_ids"].to(model.device)
else:
    input_ids = template_output.to(model.device)
```

The `except` clause that catches fallback cases catches both `TypeError` and `AttributeError`, since different transformers versions raise different exception types when thinking parameters are unsupported.

### Batching

Each generation call processes five perfumes. The call includes name, brand, gender, top notes, middle notes, base notes, accords, year where available, and perfumer where available. Instructions specify how many examples to generate per perfume and which question types to use. Five perfumes per call balances throughput against the risk of the model confusing which notes belong to which perfume inside a single call.

### The generator prompt

Qwen3-8B receives a fixed system prompt explaining its role as a data factory and a per-call user prompt containing the actual records and instructions. The system prompt includes three few-shot examples covering mood_based, note_based and structured_preference. Occasion_based and comparison are left out intentionally so the model generalises to those types rather than copying a template.

### The NoseKnows system prompt

Every training example in the final dataset includes a fixed system prompt that Gemma 4 will see at inference time. This is never generated by Qwen — it is injected programmatically and is byte-for-byte identical across all 26,521 examples. This matters for head-only fine-tuning: the trainable layers always see the same persona context and learn to condition their output on it consistently.

### Defensive parsing

Qwen3-8B's raw output starts with a `<think>...</think>` block followed by the actual JSON. The parsing pipeline strips the thinking block first, then strips known Qwen3 special tokens (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) that survive `skip_special_tokens=False` decoding and would otherwise corrupt the regex fallback, then strips markdown fences, then attempts `json.loads()`, then falls back to a regex extraction of the outermost `[...]` block. If all strategies fail, the batch is logged to `failures.jsonl` with the full raw output (capped at 3,000 chars) and generation continues.

Each parsed example is validated: both `user` and `assistant` must be non-empty strings. Malformed entries are dropped silently; valid ones from a partial batch are saved.

### Checkpointing

After every batch, valid examples are appended to `dataset.jsonl` in append mode, and `checkpoint.json` is updated with the next batch index and current counts. On session resume, preprocessing re-runs in five minutes, then the generation loop reads the checkpoint and skips already-processed batches. Losing a session costs at most one batch of five perfumes.

Progress is printed every ten batches with a rolling ETA. A tqdm bar shows batch-level progress continuously.

---

## Dataset item structure

Each line of `dataset.jsonl` is a self-contained JSON object. The `messages` array follows the standard chat template format that `trl.SFTTrainer` with `apply_chat_template` expects out of the box. The `_meta` field sits outside the messages array and is ignored by the trainer automatically, but is useful for auditing the dataset after generation.

A complete item looks like this:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are NoseKnows, a fragrance consultant who knows perfumery inside out. When someone describes what they are after, whether a mood, an occasion, or notes they love or cannot stand, you recommend real perfumes by name and brand and explain exactly why they fit, grounding your answer in the actual notes and accords. Warm, confident, specific. Never vague, never a catalogue. 3 to 5 sentences."
    },
    {
      "role": "user",
      "content": "I want something that feels like a quiet evening by the fireplace, maybe a little mysterious."
    },
    {
      "role": "assistant",
      "content": "Shalimar by Guerlain was basically made for that mood. It opens with bergamot, then softens into iris and jasmine before settling into a long base of vanilla, opoponax and smoky incense. There is something almost cinematic about how it develops on skin, intimate and unhurried. A timeless choice for exactly that kind of evening."
    }
  ],
  "_meta": {
    "perfume": "shalimar",
    "brand": "guerlain",
    "question_type": "mood_based",
    "tier": "gold"
  }
}
```

The system field is identical across every line in the file. The user field is generated by Qwen3-8B and varies in phrasing, formality and length. The assistant field is also generated by Qwen3-8B and always contains the perfume name, the brand, at least two or three specific notes from the actual record, and a reason tied to the user's query. The `_meta` field records which perfume the example came from, which question type was assigned, and whether it was a gold or silver row.

---

## Files produced

`dataset.jsonl` is the main output and the only file needed for fine-tuning. `failures.jsonl` contains raw outputs of batches where JSON parsing failed completely; these can be retried at batch size 1 if needed. `checkpoint.json` tracks generation progress and must not be deleted between sessions. `generation_report.json` is written at the end of a complete run and summarises all preprocessing and generation statistics.