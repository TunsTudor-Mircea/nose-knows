"""
LoRA fine-tuning script for NoseKnows.

Fine-tunes the base SLM on synthetic (input → output) triples using:
  - PEFT LoRA adapters
  - trl SFTTrainer (supervised fine-tuning)
  - 4-bit quantisation (when GPU is available)

Adapter weights are saved to models/nosknows-lora/.

Usage:
    # Full run (overnight)
    python -m src.finetune.train

    # Quick smoke-test on 50 examples
    python -m src.finetune.train --max-samples 50 --epochs 1

    # Resume / continue from a checkpoint
    python -m src.finetune.train --resume-from models/nosknows-lora/checkpoint-500
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

load_dotenv()

# ---------------------------------------------------------------------------
# Defaults (all overridable via CLI args or env vars)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_ID = os.getenv("SLM_MODEL_ID", "google/gemma-2-2b-it")
_DEFAULT_DATA_PATH = "data/synthetic_triples.jsonl"
_DEFAULT_OUTPUT_DIR = "models/nosknows-lora"

_LORA_R = 16
_LORA_ALPHA = 32
_LORA_DROPOUT = 0.05
_LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # standard for Gemma / LLaMA

_MAX_SEQ_LEN = 512
_BATCH_SIZE = 4
_GRAD_ACCUM = 4
_LR = 2e-4
_EPOCHS = 3
_WARMUP_RATIO = 0.03
_WEIGHT_DECAY = 0.001


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, max_samples: int | None = None) -> Dataset:
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records.append({
                    "text": rec["prompt"] + " " + rec["completion"],
                })
    if max_samples:
        records = records[:max_samples]
    print(f"[train] Loaded {len(records):,} training examples from {data_path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Model & tokeniser
# ---------------------------------------------------------------------------

def load_base_model(model_id: str):
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    bnb_config = None
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        token=hf_token,
    )
    model.config.use_cache = False  # required for gradient checkpointing

    return model, tokenizer


def apply_lora(model) -> object:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=_LORA_R,
        lora_alpha=_LORA_ALPHA,
        lora_dropout=_LORA_DROPOUT,
        target_modules=_LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model_id: str = _DEFAULT_MODEL_ID,
    data_path: str = _DEFAULT_DATA_PATH,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    epochs: int = _EPOCHS,
    max_samples: int | None = None,
    resume_from: str | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_path, max_samples)

    print(f"[train] Loading base model {model_id} …")
    model, tokenizer = load_base_model(model_id)
    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=_BATCH_SIZE,
        gradient_accumulation_steps=_GRAD_ACCUM,
        learning_rate=_LR,
        warmup_ratio=_WARMUP_RATIO,
        weight_decay=_WEIGHT_DECAY,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=_MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    print("[train] Starting training …")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] Adapter weights saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoseKnows LoRA fine-tuning")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    parser.add_argument("--data", default=_DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=_EPOCHS)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (useful for smoke testing)")
    parser.add_argument("--resume-from", default=None,
                        help="Path to a checkpoint directory to resume from")
    args = parser.parse_args()

    train(
        model_id=args.model_id,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_samples=args.max_samples,
        resume_from=args.resume_from,
    )
