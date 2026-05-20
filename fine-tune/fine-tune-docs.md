# Fine-tuning — NoseKnows

This document describes the fine-tuning phase for NoseKnows in full detail. The goal is to take the 1,104-example synthetic dataset produced in the previous phase and use it to adapt `google/gemma-2-2b` to generate fragrance recommendations in the NoseKnows style. Everything runs in a single Kaggle notebook across 14 cells.

---

## Model choice

The base model is `google/gemma-2-2b`. This is a pure text generation model with no multimodal components, which means no vision encoders loading unused weights into VRAM, no nested config structures, and no special loading class. It loads cleanly with `AutoModelForCausalLM` and `AutoTokenizer`, and the `gemma2` architecture has been in the standard PyPI transformers release for long enough that no source install is needed.

The model has 2 billion parameters. At 4-bit NF4 quantization it occupies roughly 1.5 GB of VRAM, which is trivially small on a T4. This leaves almost the entire 16 GB of each card free for activations, optimizer states, and gradient computation during training, which is a significant advantage over larger models.

The model is gated on HuggingFace, meaning the user must accept Google's license agreement at `https://huggingface.co/google/gemma-2-2b` before the HF token grants download access. A 403 Forbidden error at tokenizer loading almost always means this step was skipped.

---

## Hardware

The notebook runs on Kaggle's dual T4 configuration. `device_map="auto"` distributes model layers across both cards, though at 2B parameters at 4-bit the model fits entirely on a single T4. fp16 is used rather than bf16 because the T4 has good fp16 tensor core support, while bf16 is less numerically stable on T4 than on newer architectures.

---

## Setup and seeding

All random number generators are seeded with 42 across Python's `random` module, NumPy, PyTorch, and CUDA before any training code runs. This makes the train/val split reproducible across runs on the same hardware.

---

## Dataset preparation

The input is `dataset.jsonl`, generated in the previous phase and uploaded as a Kaggle dataset. Each line has a `messages` array and a `_meta` field. Cell 4 strips `_meta` from every record before building the HuggingFace Dataset objects. SFTTrainer does not understand `_meta` and its presence can cause silent issues with some versions of the datasets library.

Validation runs while loading: each record must have exactly three messages in the order system, user, assistant, and no empty content fields. Anything that fails is counted and skipped rather than crashing the cell. From the current 1,104-example dataset, all 1,104 records pass this check.

After stripping and validation, the records are shuffled with the fixed seed and split 90/10. This gives 993 training examples and 111 validation examples. The shuffle happens before the split so the validation set is a random sample across all five question types and both tiers, not just the tail end of the file.

SFTTrainer reads the `messages` column and applies the model's built-in chat template to format each conversation into the token sequence the model trains on. Only the assistant turn tokens contribute to the loss; the system and user turns are masked out. This is standard SFT practice and means the model learns to generate the assistant turn given the context, not to memorise the full sequence.

---

## Model loading

`PYTORCH_ALLOC_CONF=expandable_segments:True` is set before loading to reduce VRAM fragmentation on T4. The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")`. Using `os.environ.get("HF_TOKEN")` does not work on Kaggle; secrets require the `kaggle_secrets` API.

The tokenizer is loaded first. Gemma 2 does not set a pad token by default, so the notebook sets it to the EOS token after loading. Without this, SFTTrainer fails when it tries to pad sequences in a batch to the same length.

Two model config flags are set after loading: `use_cache=False` disables the KV cache during training (it is only needed for autoregressive inference and wastes VRAM during the forward pass), and `pretraining_tp=1` is required for gradient checkpointing compatibility with quantized models in some peft versions.

After loading, the notebook checks total VRAM allocation across both GPUs. For gemma-2-2b at 4-bit the guard threshold is 1.0 GB — if total allocated is below this, quantization silently failed and the run should stop immediately rather than training on a wrongly loaded model.

---

## Architecture inspection

Cell 6 prints the top-level named modules and resolves `num_hidden_layers` from the model config. For gemma-2-2b this sits directly on `model.config` rather than nested under a `text_config` (unlike multimodal models). The output confirms the layer count before LoRA configuration runs.

---

## LoRA configuration

The fine-tuning strategy is QLoRA: the base model stays frozen at 4-bit NF4, and small trainable adapter matrices are injected into specific layers. Only the adapter weights are updated during training.

The adapters target the last 4 transformer decoder layers. This is a deliberate choice: the early and middle layers handle general language understanding, world knowledge about fragrance notes, and text coherence, all of which are already good in the base model. The last few layers are where the model decides how to phrase and structure output. Training only those layers focuses the update on output style, tone, and the habit of grounding recommendations in specific notes, which is exactly what the NoseKnows dataset teaches.

Layer indices are resolved at runtime from `model.config.num_hidden_layers` rather than hardcoded. The fallback chain handles flat configs, nested `text_config`, and a last-resort regex scan of named modules. This makes the script work across different model variants without manual adjustment.

The `target_modules` list covers all attention projections (q, k, v, o) and all MLP projections (gate, up, down) within the targeted layers. Whether k and v projections exist as separate modules depends on the attention implementation; if they are absent from the trainable parameter list, grouped query attention is in use, which is expected and fine.

LoRA rank is 16, alpha is 32. The 2x ratio is the standard scaling convention. Dropout is 0.05, light regularisation appropriate for a dataset of around a thousand examples. `task_type=TaskType.CAUSAL_LM` is correct for `AutoModelForCausalLM` and is what allows peft to configure the adapter without guessing.

After applying the config, Cell 7 prints every trainable module name and the trainable parameter percentage. For last-4-layers LoRA on a 2B model with r=16, the trainable percentage should be well under 1% of total parameters.

---

## Training callbacks

Two callbacks run alongside the trainer.

`EpochCheckpointCallback` saves a permanent named adapter checkpoint at the end of each epoch to `checkpoints/epoch_N/`. These sit entirely outside the standard checkpoint rotation managed by `save_total_limit` and are never automatically deleted. If the session dies partway through epoch 3, the epoch 2 checkpoint is a clean usable adapter.

`LossHistoryCallback` collects training loss at every logging step and validation loss at every epoch end as (step, loss) and (epoch, loss) tuples. These feed the visualisation plots in Cell 12 and are also written to the training report JSON.

---

## Training configuration

The effective batch size is 32: 2 examples per device, 8 gradient accumulation steps, 2 GPUs. With 993 training examples this gives approximately 31 optimizer steps per epoch and 93 total steps across 3 epochs.

The learning rate is 2e-4, the standard starting point for LoRA SFT. Warmup covers the first 5% of steps, roughly 5 steps, giving the optimizer time to settle before the adapters receive large updates. The cosine scheduler then decays the learning rate smoothly back toward zero. `paged_adamw_8bit` is used as the optimizer, which pages optimizer states to CPU when not needed and is the standard choice for QLoRA on memory-constrained hardware.

`max_seq_length` (or `max_length` depending on the trl version installed) is set to 512. The average assistant turn in the dataset is 337 characters, and the average user turn is 90 characters. Even with the system prompt prepended, sequences comfortably fit within 512 tokens.

One version-safety detail: `max_seq_length` was renamed to `max_length` in newer trl releases. The notebook resolves the correct parameter name at runtime using `inspect.signature(SFTConfig.__init__).parameters` and passes it via `**{seq_len_kwarg: MAX_SEQ_LENGTH}`. This prevents `TypeError: SFTConfig.__init__() got an unexpected keyword argument` across trl versions.

---

## Checkpointing strategy

There are two independent layers of checkpoint protection.

The first is step-level checkpoints written by the trainer every 10 optimizer steps. With 31 steps per epoch, this gives roughly 3 checkpoints per epoch. Only the 3 most recent are kept by `save_total_limit=3`. When the notebook is re-run, `resume_from_checkpoint=True` tells the trainer to find the latest of these and resume from it automatically, with no manual intervention. At 10-step granularity, the worst-case data loss on session kill is 10 steps of training.

The second is per-epoch permanent checkpoints from `EpochCheckpointCallback`, saved to `checkpoints/epoch_1/`, `checkpoints/epoch_2/`, `checkpoints/epoch_3/`. These are never touched by `save_total_limit` rotation. If the step-level checkpoints are somehow lost, the last completed epoch is always recoverable.

With only 93 total steps across 3 epochs, the entire fine-tuning run is expected to complete in 1 to 2 hours on dual T4. Session expiry is not a realistic concern here, but the checkpoint layers are present regardless.

As with generation, the notebook must be run via Save Version rather than interactively. Kaggle only persists `/kaggle/working/` outputs to the Output tab for committed runs.

---

## Output adapter

Cell 11 saves only the LoRA adapter weights, not the full base model. For r=16 on 4 layers of a 2B model, the adapter is roughly 30 to 60 MB. `load_best_model_at_end=True` means the saved adapter corresponds to the epoch with the lowest validation loss, not necessarily the last epoch.

To load the adapter at inference time:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", ...)
model = PeftModel.from_pretrained(base_model, "/path/to/final_adapter")
```

The tokenizer is saved alongside the adapter so the inference environment does not need to re-download it separately.

---

## Visualisations

Three plots are produced after training and saved to `plots/`. All three are guarded: if the loss history lists are empty (which can happen if the session is resumed after training already completed), the cell skips cleanly rather than crashing.

The training loss curve shows loss against optimizer step across all 93 steps. On a well-configured run this should drop steeply in the first 10 to 20 steps and flatten out by epoch 2.

The validation loss per epoch shows three data points annotated with exact values. What to look for: if validation loss rises between epoch 2 and epoch 3 while training loss is still falling, the model is overfitting and the epoch 2 checkpoint (available in `checkpoints/epoch_2/`) is the better adapter to use.

The third plot overlays train and validation loss on the same axis, with evaluation points mapped to approximate step positions. This is the most informative of the three for diagnosing whether the training converged, whether it overfit, or whether more data would help.

---

## Inference sanity check

Cell 13 runs three test queries through the fine-tuned model before the notebook finishes. The queries cover different question types: a mood/occasion query, a note preference, and a structured preference with a note to avoid. Reading the outputs tells you immediately whether the NoseKnows voice is present. A working fine-tuned model names a specific perfume and brand, references actual notes, and writes 3 to 5 warm, direct sentences. A model that failed to learn the style produces generic text without specific names or notes.

---

## Files produced

`final_adapter/` contains the LoRA adapter weights and tokenizer, and is the only file needed to deploy NoseKnows. `checkpoints/epoch_N/` contains permanent per-epoch adapter snapshots. `checkpoints/checkpoint-N/` contains the rotating step-level checkpoints for resumption. `plots/` contains the three training visualisation PNGs. `training_report.json` records all hyperparameters, dataset sizes, loss histories, runtime and step counts in one place.

---

## Parameters reference

| Parameter | Value | Reason |
|---|---|---|
| Base model | google/gemma-2-2b | Pure text causal LM, 2B params, no multimodal overhead |
| Loading class | AutoModelForCausalLM | Correct for gemma2 architecture |
| Quantization | 4-bit NF4, double quant | ~1.5 GB VRAM, fits on single T4 |
| Compute dtype | float16 | T4 fp16 stable; bf16 is not |
| LoRA rank (r) | 16 | Higher than default 8 to compensate for targeting only 4 layers |
| LoRA alpha | 32 | Standard 2x r scaling |
| LoRA dropout | 0.05 | Light regularisation for ~1,000-example dataset |
| LoRA layers | Last 4 transformer layers | Preserves general knowledge, targets output style |
| task_type | CAUSAL_LM | Correct for AutoModelForCausalLM |
| Train/val split | 90/10 | 993 train, 111 val from 1,104 total |
| Max sequence length | 512 tokens | Avg assistant: 337 chars, avg user: 90 chars |
| Batch size per device | 2 | Conservative for T4 |
| Gradient accumulation | 8 steps | Effective batch size 32 across 2 GPUs |
| Learning rate | 2e-4 | Standard for LoRA SFT |
| Warmup ratio | 0.05 | ~5 warmup steps out of 93 total |
| LR scheduler | Cosine | Smooth decay, standard for SFT |
| Optimizer | paged_adamw_8bit | Memory-efficient for QLoRA |
| Epochs | 3 | ~31 steps/epoch, 93 total |
| Save steps | 10 | ~3 intermediate checkpoints per epoch |
| Save total limit | 3 | Keeps last 3 step checkpoints |
| Eval strategy | Per epoch | Val loss at end of each of 3 epochs |
| Seed | 42 | Full reproducibility across Python, NumPy, PyTorch, CUDA |
| Expected runtime | 1-2 hours | 93 steps on dual T4 with 2B model |