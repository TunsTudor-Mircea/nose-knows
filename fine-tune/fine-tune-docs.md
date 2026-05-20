# Fine-tuning — NoseKnows

This document describes the fine-tuning phase for NoseKnows. The goal is to take the 1,104-example synthetic dataset from the generation phase and use it to adapt `google/gemma-2-2b-it` to produce fragrance recommendations in the NoseKnows style. Everything runs in a single Kaggle notebook.

---

## Model

The base model is `google/gemma-2-2b-it`, the instruction-tuned variant of Gemma 2 2B. The instruction-tuned variant is necessary because its tokenizer has a chat template built in, which SFTTrainer needs to format the dataset. The model has 1.60B parameters, is a pure text generation architecture with no multimodal components, and loads with `AutoModelForCausalLM`. The `gemma2` architecture is in the standard PyPI transformers release — no source install is needed.

The model is gated on HuggingFace. The license at `https://huggingface.co/google/gemma-2-2b-it` must be accepted before the HF token grants download access.

---

## Hardware

The notebook runs on Kaggle's dual T4 configuration but uses only one of the two cards. `CUDA_VISIBLE_DEVICES="1"` is set at the very top of the imports cell, before any CUDA initialization, restricting PyTorch to GPU 1 which is remapped internally as `cuda:0`. This is necessary because `device_map="auto"` places the model on GPU 1 at load time, while the trainer's DataParallel wrapping expects all parameters on GPU 0. Restricting to a single visible GPU removes the conflict entirely. The env var must be set before `import torch` runs, which is why it appears in the imports cell rather than the model loading cell.

Training uses bf16 precision. gemma-2-2b-it stores its weights natively in bfloat16, and the fp16 gradient scaler cannot unscale bfloat16 gradients, so bf16 is the correct training dtype for this model. The T4 supports it at the trainer level without issues.

---

## Dataset preparation

The input is `dataset.jsonl`, uploaded as a Kaggle dataset. Each line has a `messages` array with system, user and assistant roles, plus a `_meta` field.

Two transformations happen during loading. First, `_meta` is stripped since SFTTrainer does not understand it. Second, the system role is merged into the user turn. gemma-2-2b-it's chat template only accepts user and assistant roles, so the system prompt content is prepended to the user content with a double newline separator, and the system role is dropped. This preserves all information while staying within the supported format. The same merging is applied at inference time.

After transformation, records are validated for exactly two messages in user/assistant order with no empty content. All 1,104 records pass. The records are shuffled with seed 42 and split 90/10, giving 993 training examples and 111 validation examples. Only the assistant turn tokens contribute to the training loss — the user turn is masked out.

---

## Model loading

The HF token is read via `UserSecretsClient().get_secret("HF_TOKEN")` from Kaggle secrets. The tokenizer pad token is set to EOS after loading since gemma-2-2b-it does not set one by default. `use_cache=False` disables the KV cache during training, and `pretraining_tp=1` is set for gradient checkpointing compatibility with quantized models.

The model loads at 4-bit NF4 with double quantization, occupying 2.22 GB of VRAM on the single visible T4. A guard after loading raises immediately if total VRAM allocation is under 1.0 GB, which would indicate the quantization silently failed.

---

## Architecture

The model has 26 transformer decoder layers. The top-level named modules are `model` (Gemma2Model) and `lm_head` (Linear). `num_hidden_layers` sits directly on `model.config` without nesting.

---

## LoRA configuration

The fine-tuning strategy is QLoRA: the base model stays frozen at 4-bit NF4, and small trainable adapter matrices are injected into the last 4 decoder layers. Only these adapter weights are updated during training.

Layers 0 to 21 handle general language understanding, world knowledge about fragrance notes, and text coherence. Layers 22 to 25 are where the model shapes its output — phrasing, structure, tone. Targeting only these last four layers focuses training on exactly what needs to change: the NoseKnows voice, the habit of naming specific perfumes and notes, the warm direct register. General knowledge stays intact.

Each targeted layer has adapters on all seven projection modules: q_proj, k_proj, v_proj, o_proj (all four attention projections — full attention, not grouped query), gate_proj, up_proj, and down_proj. With LoRA rank 16 and alpha 32, this gives 3,194,880 trainable parameters out of 2,617,536,768 total — 0.12%. The layer indices are resolved at runtime from `model.config.num_hidden_layers` rather than hardcoded, with fallbacks for nested configs and a regex scan of named modules as a last resort.

---

## Callbacks

Two callbacks run alongside the trainer.

`EpochCheckpointCallback` saves a permanent adapter checkpoint at the end of each epoch to `checkpoints/epoch_N/`. These sit outside the trainer's checkpoint rotation and are never deleted automatically. They are the safest recovery point if a session ends unexpectedly.

`LossHistoryCallback` collects training loss at every logging step and validation loss at every epoch end. The collected values feed the visualisation plots and the training report JSON.

---

## Training

With one GPU visible, the effective batch size is 16: 2 examples per device and 8 gradient accumulation steps. With 993 training examples, this gives 62 optimizer steps per epoch and 186 total steps across 3 epochs. The learning rate is 2e-4 with a cosine decay schedule and 9 warmup steps (roughly 5% of 186). The optimizer is `paged_adamw_8bit`, which pages optimizer states to CPU when not needed — the standard choice for QLoRA on memory-constrained hardware.

The sequence length is 512. With the system prompt prepended to the user turn (the system prompt alone is around 350 characters) and an average assistant turn of 337 characters, most sequences fit comfortably within this limit.

Two API compatibility fixes run at startup. The sequence length parameter was renamed from `max_seq_length` to `max_length` in newer trl releases, and the tokenizer parameter was renamed from `tokenizer` to `processing_class`. Both are resolved at runtime via `inspect.signature` and passed dynamically, preventing `TypeError` across trl versions.

The training run is expected to complete in 1 to 1.5 hours on a single T4.

---

## Checkpointing

`save_strategy` and `eval_strategy` are both set to epoch, which is required by `load_best_model_at_end=True` — mismatched strategies raise a `ValueError`. One checkpoint is written per epoch, with the last 3 kept by `save_total_limit`. On top of this, `EpochCheckpointCallback` writes permanent epoch adapters that are never rotated. `resume_from_checkpoint=True` picks up the latest checkpoint automatically if the notebook is re-run.

The notebook must be run via Save Version, not interactively. Kaggle only persists `/kaggle/working/` outputs to the Output tab for committed runs.

---

## Output

The final adapter contains only the LoRA delta weights, not the full base model. For r=16 on 4 layers of a 1.6B model the adapter is roughly 30 to 60 MB. `load_best_model_at_end=True` means the saved adapter is the one with the lowest validation loss across the three epochs, not necessarily the last one.

To load the adapter at inference time:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", ...)
model = PeftModel.from_pretrained(base_model, "/path/to/final_adapter")
```

The tokenizer is saved alongside the adapter. At inference time the system prompt is merged into the user turn in the same way as during training.

---

## Visualisations

Three plots are saved to `plots/` after training. The training loss curve shows loss against optimizer step across all 186 steps — it should drop steeply in the first 10 to 20 steps and flatten by epoch 2. The validation loss per epoch shows three annotated data points; rising validation loss while training loss falls indicates overfitting, in which case the epoch 2 checkpoint is the better adapter. The third plot overlays both curves on the same axis with evaluation points mapped to approximate step positions.

---

## Inference check

Three test queries run through the fine-tuned model at the end of the notebook. The system prompt is merged into the user turn, matching the training format exactly. A working fine-tuned model names a specific perfume and brand, references actual notes, and writes 3 to 5 sentences in the NoseKnows register. Generic output without specific names or notes is a signal that the training did not converge or the dataset needs to be larger.

---

## Files produced

`final_adapter/` contains the LoRA weights and tokenizer — the only file needed to deploy NoseKnows. `checkpoints/epoch_N/` contains permanent per-epoch snapshots. `checkpoints/checkpoint-N/` contains the rotating trainer checkpoints used for resumption. `plots/` has the three loss curve PNGs. `training_report.json` records all hyperparameters, loss histories, dataset sizes, and runtime.

---

## Parameters reference

| Parameter | Value | Reason |
|---|---|---|
| Base model | google/gemma-2-2b-it | Instruction-tuned — chat template required for SFT |
| Loading class | AutoModelForCausalLM | Correct for gemma2 text-only architecture |
| Parameters | 1.60B | Confirmed at runtime |
| Quantization | 4-bit NF4, double quant | 2.22 GB VRAM at runtime |
| Compute dtype | bfloat16 | Native weight dtype of gemma-2-2b-it |
| Training dtype | bf16=True, fp16=False | Must match model's native weight dtype |
| GPU | Single T4 via CUDA_VISIBLE_DEVICES="1" | Prevents DataParallel conflict |
| LoRA rank | 16 | Higher than default 8, compensates for 4-layer targeting |
| LoRA alpha | 32 | Standard 2× r scaling |
| LoRA dropout | 0.05 | Light regularisation for ~1,000-example dataset |
| LoRA layers | 22, 23, 24, 25 (last 4 of 26) | Targets output style, preserves general knowledge |
| Trainable params | 3,194,880 (0.12%) | Confirmed at runtime |
| task_type | CAUSAL_LM | Correct for AutoModelForCausalLM |
| System role | Merged into user turn | gemma-2-2b-it does not support system role |
| Train/val split | 90/10 | 993 train, 111 val |
| Max sequence length | 512 tokens | Covers merged prompt + assistant turn |
| Batch size per device | 2 | Conservative for single T4 |
| Gradient accumulation | 8 steps | Effective batch size 16 |
| Learning rate | 2e-4 | Standard for LoRA SFT |
| Warmup steps | 9 | ~5% of 186 total steps |
| LR scheduler | Cosine | Standard for SFT |
| Optimizer | paged_adamw_8bit | Memory-efficient for QLoRA |
| Epochs | 3 | 62 steps/epoch, 186 total |
| Save strategy | Epoch | Required by load_best_model_at_end |
| Save total limit | 3 | Last 3 epoch checkpoints kept |
| Eval strategy | Epoch | Val loss at end of each epoch |
| Seed | 42 | Full reproducibility |
| Expected runtime | 1 to 1.5 hours | 186 steps on single T4 |