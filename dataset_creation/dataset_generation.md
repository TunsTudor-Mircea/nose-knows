# Dataset Generation — NoseKnows

This document describes the synthetic dataset generation phase for NoseKnows in detail. The goal of this phase is to take the raw Fragrantica CSV and produce a structured JSONL file of training conversations that will be used to fine-tune Gemma 4. Everything happens inside a single Kaggle notebook, split across two logical phases: preprocessing and generation.

---

## Why synthetic data

The Fragrantica dataset contains perfume records, not conversations. A language model trained directly on raw CSV rows would not learn anything useful about how to talk to a user. What we actually need is thousands of examples of the following shape: a user asks something about fragrance, and a knowledgeable assistant answers by grounding the recommendation in specific notes and accords.

Since no such conversational dataset exists for perfumery at this scale, we generate it synthetically using a larger, more capable model, Qwen3-8B, to write the conversations for us. This is a well-established technique in LLM fine-tuning: use a strong model to produce high-quality training data for a smaller one. The quality of this synthetic data is the single biggest factor in how well the fine-tuned model performs, which is why so much of this pipeline is devoted to getting it right.

---

## Phase 1 — Preprocessing

Before any generation happens, the raw CSV goes through a preprocessing pipeline that runs entirely on CPU and takes about five minutes. Its job is to remove everything that would produce weak or redundant training examples.

### Loading

The CSV is semicolon-separated and latin-1 encoded, with 24,063 rows. Rows with missing perfume names, missing brands, no notes in any tier, or no rating count are dropped immediately since they cannot produce a meaningful training example.

### Normalization

Several fields need cleaning before they can be used. Note fields (top, middle, base) are parsed into proper Python lists: parenthetical qualifiers like "bergamot (Italian)" are stripped to just "bergamot", and separators like "and" are unified to commas. The rating value column uses comma-decimal format ("1,42") which is converted to standard floats. Gender values are unified to one of three categories: men, women, or unisex. Implausible year values outside the range 1900 to 2024 are set to NaN rather than dropping the row, since year is useful context but not required for generation.

### Quality filters

Four filters are applied in sequence. First, perfumes with fewer than 50 ratings are removed. Below this threshold the rating is too unreliable to be meaningful, and the perfume is too niche to be worth recommending. Second, each row must have either at least two main accords or at least two top notes, otherwise there is not enough information to build a grounded assistant turn. Third, the total count of distinct notes across all three tiers must be at least four. Fourth, placeholder profiles are dropped: if every note in the record is just an accord name repeated (for example, top is "floral", middle is "woody", base is "musk"), the note fields add nothing beyond what the accord columns already say, and the generator would have nothing specific to reason about.

### Deduplication

This is the most important preprocessing step. The Fragrantica dataset contains many flanker releases: variations of the same fragrance that differ by only one or two middle notes. Training on all of them would fill the dataset with near-identical examples and hurt generalization.

Within each brand group, every pair of perfumes is compared by the symmetric difference of their full note fingerprints (the union of top, middle and base notes as a set). If two perfumes from the same brand differ by two notes or fewer, one of them is dropped. The one with the higher rating count is kept, since it has more real-world validation. The comparison is O(N²) within each brand group, which is acceptable because no single brand has thousands of entries.

### Tier assignment

After deduplication, each remaining perfume is assigned a tier based on its rating count. Perfumes with 200 or more ratings are classified as gold and will have two training examples generated from them. Perfumes between 50 and 199 ratings are silver and get one example. Gold perfumes are more widely known and have more real-world user context behind their ratings, so generating more examples from them is intentional.

### Question type assignment

The full pool of training examples is assigned question types using a largest-remainder allocation, which guarantees the distribution is exact rather than approximate. The five types and their target proportions are: occasion_based at 25%, mood_based at 25%, note_based at 20%, comparison at 15%, and structured_preference at 15%. The first three cover straightforward queries; the last two are harder and require the model to reason about relationships between fragrances or between user preferences and note profiles. After allocation, types are interleaved across the sequence so no single type clusters at the beginning or end of any processing run.

---

## Phase 2 — Generation

Once preprocessing is done, the notebook loads Qwen3-8B and runs the generation loop.

### Model loading

Qwen3-8B is loaded from HuggingFace using 4-bit NF4 quantization via bitsandbytes. This keeps peak VRAM usage under 8 GB, leaving headroom for the KV cache during long thinking traces. The model is placed on the single P100 GPU using `device_map="auto"`. The HuggingFace token is read from Kaggle secrets so it is never hardcoded.

The model is set to eval mode immediately after loading. No gradients are needed during generation, and keeping the model in train mode would waste memory and slow things down.

### Why P100 and not dual T4

The P100 has 16 GB of VRAM, the same as a single T4, but significantly higher memory bandwidth. More relevant here: dual T4 on Kaggle uses NVLink to connect two GPUs, but this setup is designed for training models that do not fit on one GPU, not for inference throughput. For a single-batch generation task, the inter-GPU communication overhead actually makes dual T4 slower than a single card. P100 is the right choice.

### Thinking mode

Qwen3-8B has a built-in reasoning mode that produces an internal chain of thought before generating the final output. This is activated by passing `enable_thinking=True` to `apply_chat_template`. The thinking trace is not included in the output the model produces for the user turn; it is used internally to reason about which notes to highlight, how to connect them to the query, and what kind of person would ask the question.

The thinking budget is capped at 512 tokens. Without a cap, Qwen can produce reasoning traces of 1,500 tokens or more per call, which would make the full generation run take far longer than the available Kaggle GPU quota. At 512 tokens the model still reasons meaningfully but is forced to be concise. If the installed version of transformers does not support the `thinking_budget` parameter yet, the notebook falls back gracefully and runs without the cap.

### Batching

Each generation call processes five perfumes at once. The call includes the preprocessed fields for each perfume (name, brand, gender, top notes, middle notes, base notes, accords, year where available, perfumer where available) and a set of instructions specifying how many examples to generate for each perfume and which question types to use. The question types were already assigned deterministically during preprocessing, so the distribution across the full dataset is guaranteed regardless of how many sessions the generation takes.

Five perfumes per call is a deliberate balance. Fewer perfumes per call means more overhead and slower throughput. More than five and the model starts confusing which notes belong to which perfume inside the same call, which produces hallucinated recommendations. Five is the practical upper limit for reliable grounding.

### The generator prompt

Qwen3-8B receives two prompts. The system prompt explains its role as a training data factory: it is not answering a user, it is writing conversations for a future model to learn from. It describes the constraints (no invented notes, 3 to 5 sentences, plain prose in the assistant turn) and the output format (a raw JSON array, no markdown, no preamble).

Three few-shot examples are included covering mood_based, note_based and structured_preference queries. Occasion_based and comparison are left uncovered intentionally: the model must generalize to those types from the system prompt description rather than copying a template. This produces more varied outputs for those two types.

The user prompt for each call contains the actual perfume records and the generation instructions. It changes every call; the system prompt stays constant.

### The NoseKnows system prompt

Every training example in the final dataset includes a fixed system prompt that Gemma 4 will see at inference time. This prompt is never generated by Qwen; it is injected programmatically by the Python script and is byte-for-byte identical across all examples. This matters for head-only fine-tuning: the trainable layers see the same system context every time and learn to condition their output on it consistently.

The system prompt reads: "You are NoseKnows, a fragrance consultant who knows perfumery inside out. When someone describes what they are after, whether a mood, an occasion, or notes they love or cannot stand, you recommend real perfumes by name and brand and explain exactly why they fit, grounding your answer in the actual notes and accords. Warm, confident, specific. Never vague, never a catalogue. 3 to 5 sentences."

### Defensive parsing

Qwen3-8B's raw output is not clean JSON. It starts with a `<think>...</think>` block containing the internal reasoning trace, followed by the actual output. The parsing pipeline strips the thinking block first, then removes any markdown fences the model may have added despite being told not to, then attempts `json.loads()` on the cleaned string. If that fails, a regex extraction looks for the outermost `[...]` block and tries again. If both strategies fail, the batch is logged to `failures.jsonl` and the loop continues. The generation never crashes on a single bad output.

Each parsed example is also validated: both the `user` and `assistant` fields must be non-empty strings. Partially malformed outputs where some examples are valid are saved; only the invalid ones are discarded.

### Checkpointing

After every batch of five perfumes, three things happen: valid examples are appended to `dataset.jsonl`, a failure record is appended to `failures.jsonl` if the batch failed, and `checkpoint.json` is updated with the index of the next batch to process and the current counts of written examples and failures.

When the notebook is re-run after a Kaggle session ends, the preprocessing phase re-runs in about five minutes on CPU (it is fast and deterministic). The generation loop then reads `checkpoint.json`, calculates which batch to start from, and skips everything already processed. The `dataset.jsonl` file is opened in append mode, so nothing is overwritten. This means losing a session costs at most one batch of five perfumes, not the entire run.

The ETA is printed every ten batches so it is easy to track how many sessions the full run will need.

---

## Output format

Each line of `dataset.jsonl` is a JSON object with two keys. The `messages` key contains a list of three role-content pairs (system, user, assistant) in the standard chat template format that `trl.SFTTrainer` expects. The `_meta` key contains traceability information: the perfume name, brand, question type and tier. This field is not part of the messages array and is ignored by the trainer, but it is useful for auditing the dataset after generation if something looks wrong.

A typical training example looks like this:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are NoseKnows, a fragrance consultant..."
    },
    {
      "role": "user",
      "content": "I want something dark and a little indulgent, the kind of scent that makes a room notice when you walk in."
    },
    {
      "role": "assistant",
      "content": "Tom Ford Black Orchid is probably what you are describing. It opens with black truffle and ylang ylang, which sounds unusual but lands as deeply opulent on skin, and the heart is built around orchid and blackcurrant before a base of dark chocolate, patchouli and sandalwood takes over. The whole thing has this dense, almost edible quality that somehow never tips into sweetness. Very much a statement fragrance."
    }
  ],
  "_meta": {
    "perfume": "black orchid",
    "brand": "tom ford",
    "question_type": "mood_based",
    "tier": "gold"
  }
}
```

---

## Expected scale

Starting from 24,063 raw rows, the preprocessing pipeline is expected to produce somewhere between 12,000 and 14,000 perfumes after all filters and deduplication. With gold rows producing two examples and silver rows producing one, the total training set should land around 18,000 to 21,000 examples. At five perfumes per batch with thinking mode on a P100, the full generation run takes roughly 25 to 35 hours across three Kaggle sessions.

---

## Files produced

`dataset.jsonl` is the main output and the only file needed for fine-tuning. `failures.jsonl` contains the raw outputs of batches Qwen failed to produce valid JSON for; these can be retried individually at a smaller batch size if needed. `checkpoint.json` tracks generation progress and should not be deleted between sessions. `generation_report.json` is written at the end of a complete run and summarizes all preprocessing and generation statistics.