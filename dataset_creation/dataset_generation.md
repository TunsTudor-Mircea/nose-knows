# Dataset Generation — NoseKnows

<<<<<<< Updated upstream
This document describes the synthetic dataset generation phase for NoseKnows in full detail. The goal is to take the raw Fragrantica CSV and turn it into a structured JSONL file of training conversations that will be used to fine-tune Gemma 2 2B. The entire process runs inside a single Kaggle notebook, split across two phases: preprocessing and generation.
=======
This document describes the synthetic dataset generation phase for NoseKnows. The goal is to take the raw Fragrantica CSV and produce a structured JSONL file of training conversations used to fine-tune Gemma 2 2B. Everything runs inside a single Kaggle notebook split across two phases: preprocessing and generation.
>>>>>>> Stashed changes

---

## Why synthetic data

<<<<<<< Updated upstream
The Fragrantica dataset contains perfume records, not conversations. A model trained directly on those rows would learn nothing about how to actually talk to a user. What we need instead is thousands of examples shaped like real interactions: a user describes what they are looking for, and a knowledgeable assistant responds by naming a specific perfume, grounding the recommendation in actual notes and accords, and writing in a tone that feels like a person rather than a database.

No such conversational dataset exists for perfumery at this scale. The approach here is to generate it synthetically using a larger, more capable model, Qwen3-8B, to write the conversations. This is a well-established technique in LLM fine-tuning: use a strong model to produce high-quality training data for a smaller one. The quality of this synthetic data matters more than the quantity, which is why so much of the pipeline is devoted to getting it right before a single generation call is made.
=======
The Fragrantica dataset contains perfume records, not conversations. A model trained directly on those rows would learn nothing about how to talk to a user. What we need instead is thousands of examples shaped like real interactions: a user describes what they are looking for, and a knowledgeable assistant responds by naming a specific perfume, grounding the recommendation in actual notes and accords, and writing in a tone that feels like a person rather than a database entry.

No such conversational dataset exists for perfumery at this scale, so we generate it synthetically using a larger model, Qwen3-8B, to write the conversations. This is standard practice in LLM fine-tuning: use a strong model to produce high-quality training data for a smaller one. The quality of this data matters more than the quantity, which is why so much of the pipeline is devoted to getting it right before any generation call is made.
>>>>>>> Stashed changes

---

## Hardware

<<<<<<< Updated upstream
The notebook runs on Kaggle's dual T4 GPU configuration. Each T4 has 16 GB VRAM; `device_map="auto"` splits the model across both cards automatically. The P100 is not usable here because PyTorch 2.10+ does not support its CUDA capability (sm_60), which causes all generation calls to fail silently with empty exception messages. The T4 at sm_75 is fully supported.
=======
The notebook runs on Kaggle's **dual T4** GPU configuration. Each T4 has 16 GB VRAM; `device_map="auto"` splits the model across both cards automatically.

The P100 was tried first and does not work for this task. PyTorch 2.10+ requires a minimum CUDA capability of sm_70. The P100 has sm_60, which means PyTorch cannot dispatch CUDA kernels to it. When the generation loop was run on P100, every single batch failed with an empty exception message in under a second — not a timeout, but an immediate CUDA kernel launch failure. The model appeared to load (the VRAM counter showed 1.54 GB instead of the expected ~6 GB, confirming the weights were not actually placed on GPU) and then every `model.generate()` call crashed silently. Switching to T4 resolved all of this.
>>>>>>> Stashed changes

---

## Phase 1 — Preprocessing

<<<<<<< Updated upstream
Before any generation starts, the raw CSV goes through a preprocessing pipeline running entirely on CPU. It takes about five minutes and is fully deterministic, so re-running it on session resume costs nothing. The pipeline exists to ensure Qwen3-8B never receives a perfume record it cannot write a grounded, specific response about.

### Loading

The CSV (`fra_cleaned.csv`) is semicolon-separated and latin-1 encoded, with 24,063 rows across 18 columns. Rows missing a perfume name, a brand, note fields in all three tiers, or a rating count are dropped immediately since they are unrecoverable regardless of other filters. In practice none of the 24,063 rows failed this check.

### Normalization

Several fields need cleaning before they are useful. Note fields (Top, Middle, Base) are parsed into Python lists: parenthetical qualifiers like "bergamot (Italian)" are stripped to "bergamot", and "and" separators are unified to commas. The Rating Value column uses comma-decimal format ("1,42") which is converted to standard floats. Gender values are unified to one of three categories: men, women, or unisex. Year values outside the range 1900 to 2024 are set to NaN rather than dropping the row, since year is useful generation context but not required.

### Quality filters

Four filters run in sequence. Perfumes with fewer than 50 ratings are removed — below this threshold the rating is too unreliable to be meaningful and the perfume is too niche to be worth recommending. Each row must then have at least two main accords or at least two top notes; without this minimum the record carries too little information for Qwen to write a specific response. The total distinct notes across all three tiers must be at least four for the same reason. Finally, placeholder profiles are dropped: if every note in a record is just an accord name repeated, the note fields add nothing beyond what the accord columns already say, and the generator would have nothing specific to reason about.

### Deduplication

The Fragrantica dataset contains many flanker releases, variations of the same fragrance that differ by only one or two middle notes. Training on all of them would fill the dataset with near-identical examples and hurt generalization. Within each brand group, every pair of perfumes is compared by the symmetric difference of their full note fingerprints, which is the union of top, middle and base notes as a frozenset. If two perfumes from the same brand differ by two notes or fewer, the lower-rated one is dropped. The comparison runs in O(N²) per brand group, which is acceptable since no single brand has thousands of entries.
=======
Before any generation starts, the raw CSV goes through a preprocessing pipeline that runs entirely on CPU and takes about five minutes. It is fully deterministic, so re-running it on session resume costs nothing.

The pipeline exists to ensure Qwen3-8B never receives a perfume record it cannot write a grounded, specific response about. Feeding low-quality rows — sparse notes, near-duplicate flankers, placeholder accords — wastes thinking budget on perfumes that will produce weak training examples.

### Loading

The CSV (`fra_cleaned.csv`) is semicolon-separated and latin-1 encoded, with 24,063 rows across 18 columns. Rows missing a perfume name, a brand, note fields in all three tiers, or a rating count are dropped immediately. In practice none of the 24,063 rows failed this check, so the structural drop stage passes all rows through.

### Normalization

Several fields need cleaning before they are useful. Note fields (Top, Middle, Base) are parsed into Python lists: parenthetical qualifiers like "bergamot (Italian)" are stripped to just "bergamot", and "and" separators are unified to commas. The Rating Value column uses comma-decimal format ("1,42") which is converted to standard floats. Gender values are unified to men, women, or unisex. Year values outside the range 1900 to 2024 are set to NaN rather than dropping the row, since year is useful context for generation but not required.

### Quality filters

Four filters run in sequence and together reduce 24,063 rows to 18,267.

Perfumes with fewer than 50 ratings are removed first. Below this threshold the rating is unreliable and the perfume is too niche to be worth recommending. Each remaining row must have at least two main accords or at least two top notes; without this minimum the record carries too little information for Qwen to write a specific response. The total distinct notes across all three tiers must be at least four for the same reason. Finally, placeholder profiles are dropped: if every note in a record is just an accord name repeated, the note fields add nothing beyond what the accord columns already say.

### Deduplication

The Fragrantica dataset contains many flanker releases — variations of the same fragrance that differ by only one or two middle notes. Training on all of them would fill the dataset with near-identical examples and hurt generalization.
>>>>>>> Stashed changes

Within each brand group, every pair of perfumes is compared by the symmetric difference of their full note fingerprints (the union of top, middle and base notes as a frozenset). If two perfumes from the same brand differ by two notes or fewer, the lower-rated one is dropped. This brings the total from 18,267 down to **17,804 perfumes**. These are the unique, information-rich perfumes that generation draws from.

### Why the example count goes up from 17,804 to 26,521

<<<<<<< Updated upstream
- Input rows: 24,063
- After structural drop: 24,063 (none dropped at this stage)
- After quality filters: 18,267
- After deduplication: 17,804
- Gold perfumes (rating count >= 200): 8,717
- Silver perfumes (50 <= rating count < 200): 9,087
- Total training examples targeted: 26,521
=======
This is worth explaining clearly because it looks contradictory at first.
>>>>>>> Stashed changes

After deduplication we have 17,804 perfumes. These are divided into two tiers based on rating count. Perfumes with 200 or more ratings are **gold** (8,717 of them) and are assigned two training examples each, using two different question types. Perfumes between 50 and 199 ratings are **silver** (9,087 of them) and get one training example each.

<<<<<<< Updated upstream
Perfumes with 200 or more ratings are classified as gold and get two training examples generated from them. Perfumes between 50 and 199 ratings are silver and get one. Gold perfumes are more widely known, have more real-world reception behind their ratings, and are more likely to come up in actual user queries.

### Question type assignment

Training examples are assigned question types before generation using a largest-remainder allocation, which guarantees the distribution across the full dataset is exact rather than approximate. The five types and their proportions are: occasion_based at 25%, mood_based at 25%, note_based at 20%, comparison at 15%, and structured_preference at 15%. After allocation, types are interleaved across the sequence so no single type clusters at one end.
=======
The logic is that popular, well-reviewed perfumes are more likely to come up in real user queries and deserve more coverage in the training data. The multiplication gives:

- 8,717 gold × 2 examples = 17,434
- 9,087 silver × 1 example = 9,087
- **Total: 26,521 examples planned**

This 26,521 is a theoretical target assuming the full generation run completes. It never did — see the actual results section below.

### Question type assignment

Training examples are assigned question types before generation using a largest-remainder allocation, which guarantees the distribution is exact rather than approximate. The five types and their target proportions are: occasion_based at 25%, mood_based at 25%, note_based at 20%, comparison at 15%, and structured_preference at 15%. After allocation, types are interleaved so no single type clusters at one end of the sequence.
>>>>>>> Stashed changes

---

## Phase 2 — Generation

### Model loading

<<<<<<< Updated upstream
Qwen3-8B is loaded from HuggingFace using 4-bit NF4 quantization via bitsandbytes 0.46.1. This specific version is the minimum required by the installed transformers build on Kaggle. The loading configuration does not pass a top-level `dtype` argument alongside `quantization_config` — doing so causes the new transformers loading pipeline to cast weights to bfloat16 before quantizing, which spikes VRAM usage and causes OOM. The dtype is controlled entirely through `bnb_4bit_compute_dtype` inside `BitsAndBytesConfig`.

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` before loading reduces fragmentation on T4 during weight loading. The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle — secrets require the `kaggle_secrets` API.

After loading, total VRAM is checked across both GPUs. If total allocated is under 3 GB, quantization silently failed and the notebook raises immediately rather than letting thousands of batches fail silently. The correct allocation is approximately 6 GB total across both cards.

### Thinking mode

Qwen3-8B has a built-in reasoning mode that produces an internal chain of thought before generating the final output. This is activated by passing `enable_thinking=True` to `apply_chat_template`. The thinking trace is stripped from the output during parsing and never appears in any training example. Its role is to improve the quality and grounding of the generated assistant turns by forcing the model to reason about which notes are distinctive before it starts writing.

The thinking budget is capped at 512 tokens. Without a cap, reasoning traces can reach 1,500 tokens or more per call, which would make the full run infeasible within Kaggle's time limits.

One implementation detail worth noting: `apply_chat_template` with `return_tensors="pt"` returns a `BatchEncoding` object in newer transformers versions, not a plain tensor. Passing a `BatchEncoding` directly to `model.generate()` raises `AttributeError` because the generate function expects a plain tensor and tries to access `.shape[0]` on it. The fix is explicit extraction:
=======
Qwen3-8B is loaded from HuggingFace using 4-bit NF4 quantization via bitsandbytes 0.46.1. This specific version is the minimum required by the installed transformers build on Kaggle. The loading configuration does not pass a top-level `dtype` argument alongside `quantization_config`. Doing so causes the new transformers loading pipeline to cast weights to bfloat16 before quantizing, which spikes VRAM usage and causes OOM. The dtype is controlled entirely through `bnb_4bit_compute_dtype` inside `BitsAndBytesConfig`.

Setting `PYTORCH_ALLOC_CONF=expandable_segments:True` before loading reduces fragmentation on T4. The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle — secrets require the `kaggle_secrets` API.

After loading, total VRAM is checked across both GPUs. If total allocated is under 3 GB, quantization silently failed and the notebook raises immediately rather than letting thousands of batches fail. The correct allocation is approximately 6 GB total across both T4 cards.

### Thinking mode

Qwen3-8B has a built-in reasoning mode that produces an internal chain of thought before generating output. This is activated by passing `enable_thinking=True` to `apply_chat_template`. The thinking trace is stripped during parsing and never appears in any training example. Its role is to improve the grounding of generated assistant turns — the model reasons about which notes are distinctive before it starts writing.

The thinking budget is capped at 512 tokens. Without a cap, reasoning traces can reach 1,500 tokens or more per call, which would make the full run infeasible within Kaggle's session limits.

One implementation detail: `apply_chat_template` with `return_tensors="pt"` returns a `BatchEncoding` object in newer transformers versions, not a plain tensor. Passing it directly to `model.generate()` raises `AttributeError`. The fix is explicit extraction after the template call:
>>>>>>> Stashed changes

```python
if hasattr(template_output, "input_ids"):
    input_ids = template_output.input_ids.to(model.device)
elif isinstance(template_output, dict):
    input_ids = template_output["input_ids"].to(model.device)
else:
    input_ids = template_output.to(model.device)
```

The fallback clause catches both `TypeError` and `AttributeError` since different transformers versions raise different exception types when thinking parameters are unsupported.

### Batching

<<<<<<< Updated upstream
Each generation call processes five perfumes. Each call includes perfume name, brand, gender, top notes, middle notes, base notes, accords, year where available, and perfumer where available. The instructions specify exactly how many examples to generate per perfume and which question types to use, since those were already assigned deterministically during preprocessing. Five perfumes per call is a deliberate balance: fewer means more overhead and slower throughput; more than five and the model starts mixing up which notes belong to which perfume within a single call.
=======
Each generation call processes five perfumes. The call includes perfume name, brand, gender, top notes, middle notes, base notes, accords, year where available, and perfumer where available. The instructions specify how many examples to generate per perfume and which question types to use, since these were already assigned deterministically during preprocessing. Five perfumes per call balances throughput against the risk of the model confusing which notes belong to which perfume within a single call.
>>>>>>> Stashed changes

### The generator prompt

Qwen3-8B receives a fixed system prompt on every call explaining its role as a training data factory, and a per-call user prompt containing the actual perfume records and generation instructions. The system prompt includes three few-shot examples covering mood_based, note_based and structured_preference queries. Occasion_based and comparison are left out of the examples intentionally, so the model has to generalise to those types from the description rather than copy a template directly.

<<<<<<< Updated upstream
The NoseKnows system prompt that appears in every training example is never generated by Qwen. It is injected programmatically after generation and is byte-for-byte identical across all examples. This consistency is important for fine-tuning: the trainable layers always see exactly the same persona context.

### Defensive parsing

Qwen3-8B's raw output always starts with a `<think>...</think>` block followed by the actual JSON. The parsing pipeline strips the thinking block first, then strips known Qwen3 special tokens (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) that survive `skip_special_tokens=False` decoding, then strips any markdown fences, then attempts `json.loads()` on the cleaned string, then falls back to a regex extraction of the outermost `[...]` block if the direct parse fails. If all strategies fail, the full raw output (capped at 3,000 chars) is written to `failures.jsonl` and generation continues. The loop never crashes on a single bad output.

Each parsed example is also validated: both `user` and `assistant` must be non-empty strings. Partially malformed batches are handled by saving the valid examples and logging the shortfall.

### Actual generation results

The generation notebook processed 1,095 out of 17,804 perfumes before the Kaggle session expired (approximately 219 batches). The run produced:

- 1,104 valid training examples written to `dataset.jsonl`
- 14 failed batches out of 219 attempted (6.4% failure rate)
- 0 JSON parse errors in the final output

The question type distribution across the 1,104 examples:

| Question type | Count | Proportion |
|---|---|---|
| occasion_based | 225 | 20.4% |
| mood_based | 218 | 19.7% |
| note_based | 222 | 20.1% |
| comparison | 218 | 19.7% |
| structured_preference | 221 | 20.0% |

Tier distribution: 565 gold examples (51.2%), 539 silver examples (48.8%).

The average user turn is 90 characters. The average assistant turn is 337 characters, consistently 3 to 5 sentences grounded in specific notes from the actual record.
=======
The NoseKnows system prompt that appears in every training example is never generated by Qwen. It is injected programmatically after generation and is byte-for-byte identical across all examples. This consistency matters for fine-tuning: the trainable layers always see exactly the same persona context.

### Defensive parsing

Qwen3-8B's raw output always starts with a `<think>...</think>` block followed by the actual JSON. The parsing pipeline strips the thinking block first, then strips known Qwen3 special tokens (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) that survive `skip_special_tokens=False` decoding, then strips any markdown fences, then attempts `json.loads()`, then falls back to a regex extraction of the outermost `[...]` block. If all strategies fail, the full raw output (capped at 3,000 chars) is written to `failures.jsonl` and generation continues. The loop never crashes on a single bad output.

### Checkpointing and session management
>>>>>>> Stashed changes

After every batch, valid examples are appended to `dataset.jsonl` in append mode and `checkpoint.json` is updated with the next batch index and counts. On session resume, preprocessing re-runs in about five minutes on CPU, and the generation loop reads the checkpoint and skips all already-processed batches.

<<<<<<< Updated upstream
After every batch, valid examples are appended to `dataset.jsonl` in append mode and `checkpoint.json` is updated with the next batch index and current counts. On session resume, preprocessing re-runs in about five minutes on CPU (free, deterministic), and the generation loop reads the checkpoint and skips all already-processed batches. Losing a Kaggle session costs at most one batch of five perfumes.

A critical detail learned during the project: notebooks must be run via Save Version, not interactively. In interactive sessions, Kaggle does not persist `/kaggle/working/` outputs to the Output tab when the session ends. Only committed runs (Save Version) guarantee the files survive session expiry.

An additional safety guard is present in Cell 8: if the first 10 batches all produce zero examples, the loop raises immediately rather than burning the entire session on a silent failure. This was added after a session where `MAX_NEW_TOKENS=1024` caused every batch to be cut off mid-JSON, producing 3,561 failures and zero output.
=======
Notebooks must be run via **Save Version**, not interactively. In interactive sessions, Kaggle does not persist `/kaggle/working/` outputs to the Output tab when the session ends. Only committed runs guarantee the files survive session expiry.

A safety guard in the generation loop aborts the run if the first 10 batches all produce zero examples. This was added after a session where `MAX_NEW_TOKENS=1024` (reduced to save time) caused every batch to be cut off mid-JSON, producing 3,561 logged failures and zero written examples across an entire 12-hour session.

---

## Actual generation results

The generation notebook ran for one Kaggle session before the GPU quota was exhausted. It processed 219 out of 3,560 planned batches, covering 1,095 of the 17,804 preprocessed perfumes.

**What the current dataset.jsonl actually contains:**

- Total examples written: **1,104**
- Failed batches: 14 out of 219 (6.4% failure rate)
- JSON parse errors in final output: 0
- All 1,104 records have valid structure (3 messages, correct role order, no empty fields)

Question type distribution across the 1,104 examples:

| Question type | Count | Proportion |
|---|---|---|
| occasion_based | 225 | 20.4% |
| mood_based | 218 | 19.7% |
| note_based | 222 | 20.1% |
| comparison | 218 | 19.7% |
| structured_preference | 221 | 20.0% |

Tier distribution: 565 gold examples (51.2%), 539 silver examples (48.8%).

The average user turn is 90 characters. The average assistant turn is 337 characters, consistently 3 to 5 sentences grounded in specific notes from the actual record.

**These 1,104 examples are what the fine-tuning notebook trains on.** Split 90/10, this gives 993 training examples and 111 validation examples. The theoretical 26,521-example target remains achievable by resuming generation in future sessions using the saved `checkpoint.json`, but the fine-tuning run does not wait for it.
>>>>>>> Stashed changes

---

## Dataset item structure

<<<<<<< Updated upstream
Each line of `dataset.jsonl` is a self-contained JSON object. The `messages` array follows the standard chat template format that `trl.SFTTrainer` with `apply_chat_template` expects. The `_meta` field sits outside the messages array and is ignored by the trainer automatically; it is kept for auditing purposes.
=======
Each line of `dataset.jsonl` is a self-contained JSON object. The `messages` array follows the standard chat template format that `trl.SFTTrainer` with `apply_chat_template` expects. The `_meta` field sits outside the messages array and is ignored by the trainer automatically.
>>>>>>> Stashed changes

A real example from the dataset:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are NoseKnows, a fragrance consultant who knows perfumery inside out. When someone describes what they are after, whether a mood, an occasion, or notes they love or cannot stand, you recommend real perfumes by name and brand and explain exactly why they fit, grounding your answer in the actual notes and accords. Warm, confident, specific. Never vague, never a catalogue. 3 to 5 sentences."
    },
    {
      "role": "user",
      "content": "I need a scent for a casual weekend brunch with friends — something bright and approachable."
    },
    {
      "role": "assistant",
      "content": "Pride-Edition-Man by Bruno Banani is the kind of fragrance that feels like a summer morning. It starts with the sharp, juicy bite of grapefruit and red apple, then softens into the creamy sweetness of guava and walnut. The base of vetiver and amber keeps it grounded without getting heavy. Bright, sociable, and easy to wear."
    }
  ],
  "_meta": {
    "perfume": "pride-edition-man",
    "brand": "bruno-banani",
    "question_type": "occasion_based",
    "tier": "silver"
  }
}
```

The system field is identical across every line. The user field varies in phrasing, formality and length. The assistant field always names the perfume and brand, references at least two or three notes from the actual record, and gives a reason tied to the user's query. The `_meta` field records which perfume the example came from, which question type was assigned, and whether it was gold or silver tier.

---

## Files produced

<<<<<<< Updated upstream
`dataset.jsonl` is the main output and the only file needed for fine-tuning. `failures.jsonl` contains raw outputs of batches where JSON parsing failed completely, capped at 3,000 chars per entry; these can be retried at batch size 1 if needed. `checkpoint.json` tracks generation progress and must not be deleted between sessions. `generation_report.json` is written at the end of a complete run and summarises all preprocessing and generation statistics.
=======
`dataset.jsonl` is the main output and the only file needed for fine-tuning. `failures.jsonl` contains raw outputs of batches where JSON parsing failed completely; these can be retried at batch size 1 if needed. `checkpoint.json` tracks generation progress across sessions and must not be deleted between runs. `generation_report.json` is written at the end of a complete run and summarises all preprocessing and generation statistics.
>>>>>>> Stashed changes
