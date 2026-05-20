# Dataset Generation — NoseKnows

This document describes the synthetic dataset generation phase for NoseKnows. The goal is to take the raw Fragrantica CSV and produce a structured JSONL file of training conversations used to fine-tune Gemma 2 2B IT. Everything runs inside a single Kaggle notebook split across two phases: preprocessing and generation.

---

## Why synthetic data

The Fragrantica dataset contains perfume records, not conversations. A model trained directly on those rows would learn nothing about how to talk to a user. What we need instead is examples shaped like real interactions: a user describes what they are looking for, and a knowledgeable assistant responds by naming a specific perfume, grounding the recommendation in actual notes and accords, and writing in a tone that feels like a person rather than a database entry.

No such conversational dataset exists for perfumery at this scale, so we generate it synthetically using Qwen3-8B. This is standard practice in LLM fine-tuning: use a strong model to produce high-quality training data for a smaller one. The quality of this data matters more than the quantity, which is why the preprocessing pipeline filters aggressively before a single generation call is made.

---

## Hardware

The notebook runs on Kaggle's dual T4 GPU configuration. Each T4 has 16 GB VRAM. `device_map="auto"` splits Qwen3-8B across both cards automatically, with the model occupying approximately 6 GB total across both GPUs at 4-bit NF4 quantization.

The P100 is not usable for this task. PyTorch 2.10+ requires a minimum CUDA capability of sm_70. The P100 has sm_60, which means PyTorch cannot dispatch CUDA kernels to it. Generation calls fail silently with empty exception messages rather than a clear error, and the VRAM counter shows only 1.54 GB instead of the expected 6 GB, confirming the weights were never placed on GPU.

---

## Phase 1 — Preprocessing

Before any generation starts, the raw CSV goes through a preprocessing pipeline that runs entirely on CPU and takes about five minutes. It is fully deterministic, so re-running it on session resume costs nothing. The pipeline ensures Qwen3-8B receives only records rich enough to write a grounded, specific response about.

### Loading

The CSV (`fra_cleaned.csv`) is semicolon-separated and latin-1 encoded with 24,063 rows across 18 columns. Rows missing a perfume name, a brand, note fields in all three tiers, or a rating count are dropped immediately. None of the 24,063 rows fail this check, so all pass through to normalization.

### Normalization

Note fields (Top, Middle, Base) are parsed into Python lists. Parenthetical qualifiers like "bergamot (Italian)" are stripped to just "bergamot", and "and" separators are unified to commas. The Rating Value column uses comma-decimal format ("1,42") which is converted to standard floats. Gender values are unified to men, women, or unisex. Year values outside 1900 to 2024 are set to NaN rather than dropping the row, since year is useful generation context but not mandatory.

### Quality filters

Four filters run in sequence, reducing the dataset from 24,063 to 18,267 rows.

Perfumes with fewer than 50 ratings are removed. Below this threshold the rating is unreliable and the perfume is too niche to recommend. Each remaining row must have at least two main accords or at least two top notes, otherwise the record carries too little information for a grounded response. The total distinct notes across all three tiers must be at least four. Finally, placeholder profiles are dropped: if every note in a record is just an accord name repeated, the note fields add nothing beyond what the accord columns already say.

### Deduplication

The Fragrantica dataset contains many flanker releases — variations of the same fragrance differing by only one or two middle notes. Training on all of them produces near-identical examples and hurts generalization.

Within each brand group, every pair of perfumes is compared by the symmetric difference of their full note fingerprints (the union of top, middle and base notes as a frozenset). If two perfumes from the same brand differ by two notes or fewer, the lower-rated one is dropped. This brings the total from 18,267 to 17,804 perfumes. The comparison runs in O(N²) per brand group, which is acceptable since no brand has thousands of entries.

### Tier assignment and example count

The 17,804 surviving perfumes are divided into two tiers based on rating count. Perfumes with 200 or more ratings are **gold** (8,717 perfumes) and produce two training examples each, using two different question types. Perfumes between 50 and 199 ratings are **silver** (9,087 perfumes) and produce one example each.

This gives a total generation target of 26,521 examples: 8,717 × 2 = 17,434 from gold, plus 9,087 × 1 = 9,087 from silver. The increase from 17,804 perfumes to 26,521 examples comes from gold perfumes producing two examples rather than one. Gold perfumes are more widely known and more likely to come up in real user queries, so the extra coverage is intentional.

### Question type assignment

Training examples are assigned question types before generation using a largest-remainder allocation, which guarantees the distribution is exact rather than approximate. The five types and their target proportions are: occasion_based at 25%, mood_based at 25%, note_based at 20%, comparison at 15%, and structured_preference at 15%. After allocation, types are interleaved across the sequence so no single type clusters at one end.

---

## Phase 2 — Generation

### Model loading

Qwen3-8B is loaded from HuggingFace using 4-bit NF4 quantization via bitsandbytes 0.46.1. This version is the minimum required by the installed transformers build on Kaggle. The loading configuration does not pass a top-level `dtype` argument alongside `quantization_config` — doing so causes the new transformers loading pipeline to cast weights to bfloat16 before quantizing, which spikes VRAM and causes OOM. The dtype is controlled entirely through `bnb_4bit_compute_dtype` inside `BitsAndBytesConfig`.

The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle — secrets require the `kaggle_secrets` API. After loading, total VRAM is checked across both GPUs. If total allocated is under 3 GB, quantization silently failed and the notebook raises immediately.

### Thinking mode

Qwen3-8B has a built-in reasoning mode that produces an internal chain of thought before generating output, activated by passing `enable_thinking=True` to `apply_chat_template`. The thinking trace is stripped during parsing and never appears in any training example. Its role is to improve the grounding of generated assistant turns — the model reasons about which notes are distinctive before it starts writing. The thinking budget is capped at 512 tokens to keep generation time manageable within Kaggle's session limits.

`apply_chat_template` returns a `BatchEncoding` object in newer transformers versions rather than a plain tensor. Passing it directly to `model.generate()` raises `AttributeError` because the generate function expects `.shape[0]`. The fix extracts `input_ids` explicitly after the template call. The fallback clause catches both `TypeError` and `AttributeError` since different transformers versions raise different exception types when thinking parameters are unsupported.

### Batching

Each generation call processes five perfumes. The call includes perfume name, brand, gender, top notes, middle notes, base notes, accords, year where available, and perfumer where available. Instructions specify how many examples to generate per perfume and which question types to use, since these were assigned deterministically during preprocessing. Five perfumes per call balances throughput against the risk of the model mixing up which notes belong to which perfume within a single call.

### The generator prompt

Qwen3-8B receives a fixed system prompt explaining its role as a training data factory, and a per-call user prompt containing the actual perfume records and generation instructions. The system prompt includes three few-shot examples covering mood_based, note_based and structured_preference queries. Occasion_based and comparison are left out intentionally so the model generalises to those types rather than copying a template directly.

The NoseKnows system prompt that appears in every training example is never generated by Qwen. It is injected programmatically after generation and is byte-for-byte identical across all examples.

### Defensive parsing

Qwen3-8B's raw output starts with a `<think>...</think>` block followed by the actual JSON. The parsing pipeline strips the thinking block first, then strips known Qwen3 special tokens (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) that survive `skip_special_tokens=False` decoding, then strips any markdown fences, then attempts `json.loads()`, then falls back to a regex extraction of the outermost `[...]` block. If all strategies fail, the full raw output (capped at 3,000 chars) is written to `failures.jsonl` and generation continues. Each parsed example is validated: both `user` and `assistant` must be non-empty strings.

### Checkpointing and session management

After every batch, valid examples are appended to `dataset.jsonl` in append mode and `checkpoint.json` is updated with the next batch index and counts. On session resume, preprocessing re-runs in about five minutes on CPU, and the generation loop reads the checkpoint and skips already-processed batches.

Notebooks must be run via **Save Version**, not interactively. Kaggle does not persist `/kaggle/working/` outputs to the Output tab for interactive sessions. Only committed runs guarantee files survive session expiry.

---

## Dataset

The generation run produced 1,104 training examples from 219 processed batches, covering 1,095 of the 17,804 preprocessed perfumes. The batch failure rate was 6.4% (14 failed batches out of 219). All 1,104 records in the output have valid structure with no JSON parse errors.

Question type distribution:

| Question type | Count | Proportion |
|---|---|---|
| occasion_based | 225 | 20.4% |
| mood_based | 218 | 19.7% |
| note_based | 222 | 20.1% |
| comparison | 218 | 19.7% |
| structured_preference | 221 | 20.0% |

Tier distribution: 565 gold (51.2%), 539 silver (48.8%). Average user turn: 90 characters. Average assistant turn: 337 characters. The dataset is split 90/10 for fine-tuning: 993 training examples and 111 validation examples.

Generation can be resumed in future sessions using the saved `checkpoint.json` to extend the dataset toward the 26,521-example target.

---

## Dataset item structure

Each line of `dataset.jsonl` is a self-contained JSON object. The `messages` array follows the standard chat template format that `trl.SFTTrainer` expects. The `_meta` field sits outside the messages array and is ignored by the trainer automatically.

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

`dataset.jsonl` is the main output and the only file needed for fine-tuning. `failures.jsonl` contains raw outputs of batches where JSON parsing failed completely; these can be retried at batch size 1 if needed. `checkpoint.json` tracks generation progress across sessions and must not be deleted between runs. `generation_report.json` is written at the end of a complete run and summarises all preprocessing and generation statistics.