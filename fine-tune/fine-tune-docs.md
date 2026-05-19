# Fine-tuning — NoseKnows

This document describes the fine-tuning phase for NoseKnows in detail. The goal is to take the synthetic dataset produced in the previous phase and use it to adapt `google/gemma-4-E4B-it` to generate fragrance recommendations in the NoseKnows style. Everything runs in a single Kaggle notebook across 13 cells.

---

## Model choice and the multimodal complication

The base model is `google/gemma-4-E4B-it`. Gemma 4 is a multimodal architecture, meaning every checkpoint includes a vision encoder and an audio encoder alongside the language decoder. There is no official text-only variant from Google. The model is loaded with `AutoModelForCausalLM` rather than `AutoModelForImageTextToText` — this tells HuggingFace Transformers to instantiate only the language model component, but the vision and audio encoder weights are still present in the checkpoint and load into VRAM. They are never called during text-only fine-tuning, so they contribute roughly 1.5-2 GB of overhead without affecting training at all.

The practical consequence is that peak VRAM during loading is higher than a truly text-only model of equivalent parameter count. On dual T4 (32 GB total) this is not a problem, but it is worth knowing when interpreting the VRAM readout after Cell 5.

---

## Hardware

The notebook runs on Kaggle's dual T4 configuration. `device_map="auto"` distributes model layers across both cards. Effective VRAM is roughly 32 GB total, split unevenly between the two cards depending on layer sizes.

fp16 is used rather than bf16. The T4 has good fp16 tensor core support, while bf16 is less numerically stable on T4 than on newer architectures like A100. The `bnb_4bit_compute_dtype` is set to `torch.float16` accordingly.

---

## Setup and seeding

Before any training code runs, all random number generators are seeded with the same value (42) across Python's `random` module, NumPy, PyTorch, and CUDA. This makes the train/val split and the training process reproducible across runs on the same hardware.

---

## Dataset preparation

The input is `dataset.jsonl`, generated in the previous phase and uploaded as a Kaggle dataset. Each line has a `messages` array (the actual training data) and a `_meta` field (traceability info). The first thing Cell 4 does is strip `_meta` from every record. SFTTrainer does not understand it and having it present can cause silent issues with some versions of the datasets library.

After stripping, the records are shuffled with the fixed seed and split 90/10 into train and validation sets. The shuffle happens before the split so the validation set is a random sample across all question types and perfume tiers, not just the tail end of the dataset. Both sets are wrapped in HuggingFace `Dataset` objects.

The `SFTTrainer` reads the `messages` column and applies the model's built-in chat template to format each conversation into the token sequence the model trains on. The system, user, and assistant turns are concatenated with the appropriate special tokens, and only the assistant turn tokens contribute to the loss — the system and user turns are masked out. This is the standard SFT approach and means the model learns to generate the assistant turn given the context, not to memorize the whole sequence.

---

## Model loading

`PYTORCH_ALLOC_CONF=expandable_segments:True` is set before loading to reduce VRAM fragmentation on T4. The HuggingFace token is read from Kaggle secrets via `UserSecretsClient().get_secret("HF_TOKEN")` rather than `os.environ.get()`, which does not work on Kaggle.

Two model config flags are set after loading. `use_cache=False` disables the KV cache during training — it is only needed for autoregressive inference, and keeping it on wastes VRAM during the forward pass. `pretraining_tp=1` is required for gradient checkpointing to work correctly with quantized models in some peft versions.

The tokenizer may not have a pad token set by default on Gemma models. If it is missing, it is set to the EOS token. This is necessary because SFTTrainer pads sequences in a batch to the same length, and without a pad token that step fails.

---

## LoRA configuration

The fine-tuning strategy is QLoRA (quantized LoRA): the base model stays frozen at 4-bit NF4, and small trainable adapter matrices are injected into specific layers. Only the adapter weights are updated during training, which keeps memory and compute requirements low.

The adapters are placed on the last 4 transformer decoder layers plus the `lm_head` projection. This is a deliberate choice. The early and middle layers of the model handle general language understanding, world knowledge about fragrance notes, and reasoning — all of which are already good in the base model. The last few layers and the output projection are where the model decides how to phrase and structure its output. Training only those layers focuses the update on output style, tone, and grounding behaviour, which is exactly what the NoseKnows dataset is designed to teach.

The layer indices are resolved at runtime from `model.config.num_hidden_layers` rather than hardcoded. This makes the script work regardless of the exact Gemma 4 variant. If the config attribute is nested under `text_config` (which can happen when the multimodal config wraps the language model config), the script falls back to reading from there.

The LoRA rank is 16 and alpha is 32. The alpha/rank ratio of 2 is the standard scaling convention — it controls how strongly the adapter updates are scaled before being added to the frozen weights. A dropout of 0.05 provides light regularisation appropriate for a dataset of a few thousand examples.

The `target_modules` list covers all attention projections (q, k, v, o) and all MLP projections (gate, up, down) within the targeted layers. After applying the config, Cell 6 prints every trainable module name so you can verify that nothing outside the last 4 layers is being trained.

---

## Training callbacks

Two custom callbacks run alongside the standard trainer.

`EpochCheckpointCallback` saves a named checkpoint at the end of each epoch to `checkpoints/epoch_N/`. These checkpoints are written by the callback directly and sit outside the standard checkpoint rotation managed by `save_total_limit`. They are never automatically deleted. If the session dies partway through epoch 3, the epoch 2 checkpoint is a clean, usable adapter.

`LossHistoryCallback` collects training loss at every logging step and validation loss at every epoch end. It stores them as lists of (step, loss) and (epoch, loss) tuples respectively. These are used by Cell 11 to produce the visualisations and are also written to the training report JSON.

---

## Training configuration

The key hyperparameters and the reasoning behind each:

`per_device_train_batch_size=2` with `gradient_accumulation_steps=8` on 2 GPUs gives an effective batch size of 32. This is large enough for stable gradient estimates on a dataset of a few thousand examples without exhausting VRAM during the backward pass.

`learning_rate=2e-4` is the standard starting point for LoRA SFT. Higher values risk destabilising the last-layer adapters; lower values converge too slowly for 3 epochs.

`warmup_ratio=0.05` warms up the learning rate over the first 5% of steps. With ~760 total steps across 3 epochs, that is roughly 38 warmup steps, which gives the optimizer time to settle before the adapters receive large gradient updates.

`lr_scheduler_type="cosine"` decays the learning rate smoothly from the peak back toward zero over the training run. Cosine decay is the standard choice for SFT and tends to produce lower final loss than linear decay on short runs.

`optim="paged_adamw_8bit"` is the memory-efficient optimizer for QLoRA. It pages optimizer states to CPU when they are not needed, reducing peak VRAM usage compared to standard AdamW. This is important when training on quantized models where VRAM is already partially consumed by the frozen weights.

`eval_strategy="epoch"` evaluates on the validation set after every epoch and prints the validation loss. `load_best_model_at_end=True` means the trainer keeps track of which epoch produced the lowest validation loss and loads that checkpoint at the end of training. The final adapter saved in Cell 10 is therefore always the best checkpoint, not necessarily the last one.

---

## Checkpointing strategy

There are two independent layers of protection against session failure.

The first is the step-level checkpoints written by the trainer itself every 50 optimizer steps. At roughly 253 steps per epoch, this gives about 5 checkpoints per epoch. Only the 3 most recent are kept to avoid filling disk. These are saved inside `checkpoints/` with standard HuggingFace checkpoint naming (`checkpoint-N`). When the notebook is re-run, `resume_from_checkpoint=True` tells the trainer to find the latest of these and resume from it automatically, no manual intervention needed.

The second is the per-epoch named checkpoints from `EpochCheckpointCallback`, saved to `checkpoints/epoch_1/`, `checkpoints/epoch_2/`, `checkpoints/epoch_3/`. These are permanent. A step-level checkpoint might be overwritten by the rotation, but an epoch checkpoint from `EpochCheckpointCallback` is never touched. If you lose the step-level checkpoints for some reason, you can always restart from the last completed epoch.

---

## Output adapter

Cell 10 saves only the LoRA adapter weights, not the full base model. The adapter is small, roughly 50-100 MB depending on the rank and number of targeted layers. It contains the delta matrices for each targeted projection in the last 4 layers plus lm_head.

To use the adapter at inference time in the NoseKnows backend:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", ...)
model = PeftModel.from_pretrained(base_model, "/path/to/final_adapter")
```

The tokenizer is also saved alongside the adapter so the inference environment does not need to re-download it.

---

## Visualisations

Three plots are produced after training and saved to `plots/`.

The first is the training loss curve plotted against optimizer step. This shows how the loss decreased over the full training run and whether it was still declining at the end of epoch 3, which would suggest more training could help.

The second is the validation loss plotted against epoch number, with the exact loss value annotated on each point. Three data points is not a lot, but on a dataset of this size it is enough to see whether the model is overfitting (validation loss rising while training loss falls) or still learning.

The third overlays train and validation loss on the same axis, with evaluation points mapped to approximate step positions. This is the most informative plot for diagnosing training dynamics.

---

## Inference sanity check

Cell 12 runs three test queries through the fine-tuned model before the notebook finishes. The queries cover different question types: a mood/occasion query, a straightforward note preference, and a structured preference with a note to avoid. Reading the outputs tells you immediately whether the NoseKnows voice is present — whether the model names specific perfumes and brands, references actual notes, and writes in the warm direct register the training data was designed to teach.

If the outputs look generic or do not mention specific notes, it is a signal that either the training did not converge, the learning rate was too low, or the dataset quality was not sufficient for the number of trainable parameters.

---

## Files produced

`final_adapter/` contains the LoRA adapter weights and tokenizer, and is the only file needed to deploy NoseKnows. `checkpoints/epoch_N/` contains permanent per-epoch adapter snapshots. `checkpoints/checkpoint-N/` contains the rotating step-level checkpoints used for resumption. `plots/` contains the three training visualisation PNGs. `training_report.json` records all hyperparameters, dataset sizes, loss histories, runtime, and step counts in one place for reproducibility.

---

## Parameters reference

| Parameter | Value | Reason |
|---|---|---|
| Base model | google/gemma-4-E4B-it | Official Google Gemma 4 instruction-tuned, 4B effective params |
| Loading class | AutoModelForCausalLM | Text-only fine-tuning, vision/audio encoders unused |
| Quantization | 4-bit NF4, double quant | Fits on dual T4 with headroom for gradients |
| Compute dtype | float16 | T4 fp16 tensor cores, more stable than bf16 on T4 |
| LoRA rank (r) | 16 | Higher than default 8 to compensate for targeting only 4 layers |
| LoRA alpha | 32 | Standard 2x r scaling |
| LoRA dropout | 0.05 | Light regularisation for small dataset |
| LoRA layers | Last 4 transformer layers + lm_head | Preserves general knowledge, targets output style |
| Train/val split | 90/10 | Standard split for datasets of this size |
| Max sequence length | 512 tokens | Covers system + user + assistant comfortably |
| Batch size per device | 2 | Conservative for T4 with LoRA activations |
| Gradient accumulation | 8 steps | Effective batch size of 32 across 2 GPUs |
| Learning rate | 2e-4 | Standard starting point for LoRA SFT |
| Warmup ratio | 0.05 | ~38 warmup steps out of 760 total |
| LR scheduler | Cosine | Smooth decay, standard for SFT |
| Optimizer | paged_adamw_8bit | Memory-efficient for QLoRA |
| Epochs | 3 | Avoids overfitting at this dataset scale |
| Save steps | 50 | ~5 intermediate checkpoints per epoch |
| Save total limit | 3 | Keeps only 3 most recent step checkpoints |
| Eval strategy | Per epoch | Validation loss at end of each epoch |
| Seed | 42 | Full reproducibility across Python, NumPy, PyTorch, CUDA |
