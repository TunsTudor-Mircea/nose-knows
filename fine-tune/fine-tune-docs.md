# Fine-tuning — NoseKnows

This document describes the fine-tuning phase for NoseKnows in full detail. The goal is to take the 1,104-example synthetic dataset produced in the previous phase and use it to adapt `google/gemma-2-2b-it` to generate fragrance recommendations in the NoseKnows style. Everything runs in a single Kaggle notebook across 14 cells.

---

## Model choice

The base model is `google/gemma-2-2b-it` — the instruction-tuned variant of Gemma 2 2B. The `-it` suffix matters: the base `gemma-2-2b` does not have a chat template set on the tokenizer and raises `ValueError: Cannot use chat template functions` when SFTTrainer tries to format the dataset. The instruction-tuned variant has the chat template built in and is the correct starting point for a conversational assistant.

The model is a pure text generation model with no multimodal components. It loads cleanly with `AutoModelForCausalLM` and `AutoTokenizer`, and the `gemma2` architecture is supported in the standard PyPI transformers release — no source install needed. Actual parameter count as reported at runtime is 1.60B. At 4-bit NF4 quantization the model occupies 2.22 GB of VRAM.

The model is gated on HuggingFace. The license agreement must be accepted at `https://huggingface.co/google/gemma-2-2b-it` before the HF token grants download access. A 403 Forbidden error at tokenizer loading almost always means this step was skipped. The 401 Unauthorized error that sometimes follows is a secondary fallback attempt by transformers, not a separate problem.

---

## Hardware

The notebook runs on Kaggle's dual T4 configuration but uses only one GPU. `CUDA_VISIBLE_DEVICES="1"` is set at the very top of Cell 2 before any CUDA initialization, restricting PyTorch to GPU 1 only. This GPU is then remapped internally as `cuda:0`.

The reason for this restriction is a conflict between `device_map="auto"` and the trainer's DataParallel wrapping. When both GPUs are visible, `device_map="auto"` places the model on GPU 1 (where more memory was free at load time), but the trainer wraps the model in DataParallel expecting parameters on GPU 0 (device_ids[0]). This produces `RuntimeError: module must have its parameters and buffers on device cuda:0 but found one of them on device cuda:1`. Restricting visibility to one GPU eliminates the conflict entirely.

`CUDA_VISIBLE_DEVICES` must be set before `import torch` runs, which is why it sits at the top of Cell 2 rather than Cell 5.

bf16 is used for training rather than fp16. The initial configuration used fp16, which caused `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`. This happens because gemma-2-2b-it uses bfloat16 as its native weight dtype, and the fp16 gradient scaler cannot unscale bfloat16 gradients. Switching to bf16 resolves this — the T4 supports bf16 at the trainer level even though it is less stable for large model training.

---

## Setup and seeding

All random number generators are seeded with 42 across Python's `random` module, NumPy, PyTorch, and CUDA before any training code runs. This makes the train/val split and training process reproducible across runs on the same hardware.

---

## Dataset preparation

The input is `dataset.jsonl`, generated in the previous phase and uploaded as a Kaggle dataset. Each line has a `messages` array with three roles (system, user, assistant) and a `_meta` field.

Two transformations happen during loading.

First, `_meta` is stripped. SFTTrainer does not understand it and having it present can cause silent issues with some versions of the datasets library.

Second, the system role is merged into the user turn. gemma-2-2b-it's chat template only accepts `user` and `assistant` (internally `model`) roles. Passing a system message raises `TemplateError: System role not supported`. The fix is to prepend the system prompt content to the user content, separated by a double newline, and drop the system role entirely. This preserves all information while staying within the supported format. The same merging is applied at inference time in Cell 13.

After these transformations, records are validated: each must produce exactly two messages in user/assistant order with no empty content. All 1,104 records from the current dataset pass this check.

Records are then shuffled with the fixed seed and split 90/10, giving 993 training examples and 111 validation examples.

SFTTrainer applies the model's built-in chat template to format each conversation into the token sequence the model trains on. Only the assistant turn tokens contribute to the loss — the user turn is masked out. This means the model learns to generate fragrance recommendations given a user query, not to memorise the full conversation.

---

## Model loading

The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle — secrets require the `kaggle_secrets` API.

The tokenizer is loaded first. gemma-2-2b-it does not set a pad token by default. The notebook sets it to the EOS token after loading. Without this, SFTTrainer fails when padding sequences in a batch to the same length.

Two model config flags are set after loading: `use_cache=False` disables the KV cache during training (only needed for autoregressive inference, wastes VRAM during forward pass), and `pretraining_tp=1` is required for gradient checkpointing compatibility with quantized models in some peft versions.

The VRAM guard after loading checks that total allocated exceeds 1.0 GB. For gemma-2-2b-it at 4-bit, the actual allocation is 2.22 GB — if it falls below 1.0 GB, quantization silently failed.

---

## Architecture inspection

Cell 6 confirms the architecture before LoRA configuration runs. From the actual run:

- Top-level modules: `model` (Gemma2Model), `lm_head` (Linear)
- Config type: Gemma2Config
- `num_hidden_layers`: 26
- `num_hidden_layers` sits directly on `model.config`, not nested under `text_config`

---

## LoRA configuration

The fine-tuning strategy is QLoRA: the base model stays frozen at 4-bit NF4, and small trainable adapter matrices are injected into the last 4 transformer decoder layers. Only the adapter weights are updated during training.

The choice of last 4 layers is deliberate. Layers 0 to 21 handle general language understanding, world knowledge about fragrance notes, sentence structure, and reasoning. The last 4 layers (22 to 25 in a 26-layer model) are where the model decides how to phrase and structure output. Training only those layers focuses the update on output style, tone, and the habit of grounding recommendations in specific notes from the record.

From the actual run:

- Total decoder layers: 26
- Layers targeted: 22, 23, 24, 25
- Trainable modules per layer: q_proj, k_proj, v_proj, o_proj (full attention, not grouped query), gate_proj, up_proj, down_proj
- Trainable parameters: 3,194,880
- Total parameters: 2,617,536,768
- Trainable percentage: 0.12%

Layer indices are resolved at runtime from `model.config.num_hidden_layers` rather than hardcoded, with fallbacks for nested configs and a regex scan of named modules as a last resort.

LoRA rank is 16, alpha is 32 (standard 2× scaling). Dropout is 0.05. `task_type=TaskType.CAUSAL_LM` is correct for `AutoModelForCausalLM`.

---

## Training callbacks

Two callbacks run alongside the trainer.

`EpochCheckpointCallback` saves a permanent named adapter checkpoint at the end of each epoch to `checkpoints/epoch_N/`. These sit outside the standard checkpoint rotation managed by `save_total_limit` and are never automatically deleted. If the session dies partway through epoch 3, the epoch 2 checkpoint is a clean usable adapter.

`LossHistoryCallback` collects training loss at every logging step and validation loss at every epoch end as (step, loss) and (epoch, loss) tuples. These feed the visualisation plots in Cell 12 and are written to the training report JSON.

---

## Training configuration

With one GPU visible, the effective batch size is 16: 2 examples per device, 8 gradient accumulation steps, 1 GPU. With 993 training examples this gives 62 optimizer steps per epoch and 186 total steps across 3 epochs. Warmup covers 9 steps (approximately 5% of 186).

The learning rate is 2e-4, the standard starting point for LoRA SFT. The cosine scheduler decays it smoothly back toward zero over the run. `paged_adamw_8bit` pages optimizer states to CPU when not needed, the standard choice for QLoRA.

`max_length` is set to 512. The dataset has an average user turn of 90 characters and an average assistant turn of 337 characters. With the system prompt prepended to the user turn (adding roughly 350 characters), sequences stay comfortably within 512 tokens for most examples.

Two version-safety fixes are applied at runtime. The sequence length parameter was renamed from `max_seq_length` to `max_length` in newer trl releases — the correct name is detected via `inspect.signature(SFTConfig.__init__).parameters`. The tokenizer parameter was renamed from `tokenizer` to `processing_class` in newer trl releases — detected the same way. Both are passed via `**{kwarg: value}` to prevent `TypeError` across trl versions.

---

## Checkpointing strategy

`save_strategy="epoch"` and `eval_strategy="epoch"` are both set to epoch. This is required: `load_best_model_at_end=True` raises `ValueError` if save and eval strategies do not match. Epoch-level saving means one checkpoint is written per epoch, the last 3 are kept by `save_total_limit=3`.

On top of this, `EpochCheckpointCallback` writes permanent per-epoch adapters to `checkpoints/epoch_1/`, `checkpoints/epoch_2/`, `checkpoints/epoch_3/`. These are never rotated. If the step-level checkpoints are lost, the last completed epoch adapter is always recoverable.

`resume_from_checkpoint=True` tells the trainer to find the latest checkpoint automatically on re-run. With 186 total steps completing in roughly 1 to 1.5 hours, session expiry is not a realistic concern for this run. The checkpoint layers are present regardless.

The notebook must be run via Save Version, not interactively. Kaggle only persists `/kaggle/working/` outputs to the Output tab for committed runs.

---

## Output adapter

Cell 11 saves only the LoRA adapter weights, not the full base model. For r=16 on 4 layers of a 1.6B model the adapter is roughly 30 to 60 MB. `load_best_model_at_end=True` means the saved adapter is the one with the lowest validation loss across the 3 epochs, not necessarily the last one.

To load the adapter at inference time:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", ...)
model = PeftModel.from_pretrained(base_model, "/path/to/final_adapter")
```

The tokenizer is saved alongside the adapter. At inference time, the system prompt must be merged into the user turn in the same way as during training — the model never sees a system role.

---

## Visualisations

Three plots are produced after training and saved to `plots/`. All three are guarded against empty loss history lists so they skip cleanly if the session is resumed after training already completed.

The training loss curve shows loss against optimizer step across all 186 steps. A well-configured run should drop steeply in the first 10 to 20 steps and flatten out by epoch 2.

The validation loss per epoch shows three data points annotated with exact values. If validation loss rises between epoch 2 and epoch 3 while training loss is still falling, the model is overfitting and the epoch 2 checkpoint in `checkpoints/epoch_2/` is the better adapter to use.

The third plot overlays train and validation loss on the same axis with evaluation points mapped to approximate step positions. This is the most useful of the three for diagnosing convergence.

---

## Inference sanity check

Cell 13 runs three test queries through the fine-tuned model before the notebook finishes. The system prompt is merged into the user turn at inference time, matching the training format exactly. A working fine-tuned model names a specific perfume and brand, references actual notes, and writes 3 to 5 warm direct sentences. Generic output without specific perfume names or notes indicates the training did not converge or the dataset was insufficient.

---

## Files produced

`final_adapter/` contains the LoRA adapter weights and tokenizer — the only file needed to deploy NoseKnows. `checkpoints/epoch_N/` contains permanent per-epoch adapter snapshots. `checkpoints/checkpoint-N/` contains the rotating epoch-level trainer checkpoints. `plots/` contains the three loss curve PNGs. `training_report.json` records all hyperparameters, dataset sizes, loss histories, runtime and step counts.

---

## Parameters reference

| Parameter | Value | Reason |
|---|---|---|
| Base model | google/gemma-2-2b-it | Instruction-tuned variant — has chat template, required for SFT |
| Loading class | AutoModelForCausalLM | Correct for gemma2 text-only architecture |
| Actual parameters | 1.60B | Confirmed at runtime |
| Quantization | 4-bit NF4, double quant | 2.22 GB VRAM at runtime, fits on single T4 |
| Compute dtype | bfloat16 | fp16 raises NotImplementedError with gemma-2-2b-it native bf16 weights |
| Training dtype | bf16=True, fp16=False | Required to match model's native weight dtype |
| GPU setup | Single T4 via CUDA_VISIBLE_DEVICES="1" | Prevents DataParallel conflict with device_map="auto" |
| LoRA rank (r) | 16 | Higher than default 8, compensates for targeting only 4 layers |
| LoRA alpha | 32 | Standard 2× r scaling |
| LoRA dropout | 0.05 | Light regularisation for ~1,000-example dataset |
| LoRA layers | Layers 22-25 (last 4 of 26) | Targets output style, preserves general knowledge |
| Trainable params | 3,194,880 (0.12%) | Confirmed at runtime |
| task_type | CAUSAL_LM | Correct for AutoModelForCausalLM |
| System role handling | Merged into user turn | gemma-2-2b-it chat template does not support system role |
| Train/val split | 90/10 | 993 train, 111 val from 1,104 total |
| Max sequence length | 512 tokens | Covers merged system+user+assistant comfortably |
| Batch size per device | 2 | Conservative for T4 |
| Gradient accumulation | 8 steps | Effective batch size 16 on single GPU |
| Learning rate | 2e-4 | Standard for LoRA SFT |
| Warmup steps | 9 | ~5% of 186 total steps |
| LR scheduler | Cosine | Smooth decay, standard for SFT |
| Optimizer | paged_adamw_8bit | Memory-efficient for QLoRA |
| Epochs | 3 | 62 steps/epoch, 186 total |
| Save strategy | Epoch | Required to match eval_strategy for load_best_model_at_end |
| Save total limit | 3 | Keeps last 3 epoch checkpoints |
| Eval strategy | Epoch | Val loss at end of each of 3 epochs |
| Seed | 42 | Full reproducibility |
| Expected runtime | 1-1.5 hours | 186 steps on single T4 with 1.6B model |