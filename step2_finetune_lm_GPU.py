#!/usr/bin/env python3
"""Step 2 — Fine-tune GPT-2 (small) with LoRA on human and AI translations.

Produces:
  - ``outputs/train_test_split.csv``
  - ``outputs/models/gpt2/{style}_lora/``   (one adapter per style)
  - ``outputs/finetune_summary.csv``
  - ``outputs/finetune_logs/``

SEED POLICY: RANDOM_SEED = 1024 is used ONLY for the train/test split.
Fine-tuning and inference are intentionally left non-deterministic.
"""

import os
import sys
import time
import math

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

import config
from utils import ensure_dir


def _tokenize_dataset(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize a list of texts and return a HuggingFace ``Dataset``."""

    def _tok(examples: dict) -> dict:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def _finetune_one_style(
    style: str,
    train_texts: list[str],
    val_texts: list[str],
    base_model_name: str,
    save_dir: str,
    log_dir: str,
    tokenizer: AutoTokenizer,
) -> dict:
    """Fine-tune a single LoRA adapter and return summary metrics."""

    print(f"    Fine-tuning style='{style}' ({len(train_texts)} train, {len(val_texts)} val) …")

    # Tokenize
    train_ds = _tokenize_dataset(train_texts, tokenizer, config.MAX_SEQ_LENGTH)
    val_ds = _tokenize_dataset(val_texts, tokenizer, config.MAX_SEQ_LENGTH)

    # Load fresh base model each time to avoid adapter contamination
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    style_log_dir = os.path.join(log_dir, f"{base_model_name.replace('/', '_')}_{style}")
    ensure_dir(style_log_dir)

    training_args = TrainingArguments(
        output_dir=style_log_dir,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=max(1, len(train_ds) // (config.BATCH_SIZE * 5)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
        # NOTE: No seed= parameter — training is intentionally non-deterministic
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    val_loss = eval_results["eval_loss"]
    val_ppl = math.exp(val_loss) if val_loss < 20 else float("inf")

    # Train loss from the last log entry
    train_loss = float("nan")
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "loss" in entry and "eval_loss" not in entry:
                train_loss = entry["loss"]
                break

    # Save adapter only
    adapter_path = os.path.join(save_dir, f"{style}_lora")
    ensure_dir(adapter_path)
    model.save_pretrained(adapter_path)
    print(f"    Adapter saved → {adapter_path}")

    return {
        "base_model": base_model_name,
        "style": style,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "final_val_ppl": val_ppl,
        "num_train_samples": len(train_texts),
        "num_epochs": config.NUM_EPOCHS,
    }


def main() -> None:
    start = time.time()
    print(f"[Step 2] {time.strftime('%Y-%m-%d %H:%M:%S')} — Fine-tuning GPT-2 with LoRA …")

    ai_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    if not os.path.isfile(ai_path):
        sys.exit(f"ERROR: {ai_path} not found. Run step1 first.")

    df = pd.read_csv(ai_path, encoding="utf-8")
    print(f"  Loaded {len(df)} rows from ai_translations.csv")

    # ── 1. Build paired train/test split on sentence_id ──────────────────────
    unique_ids = df["sentence_id"].unique().tolist()
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_SEED,
    )
    split_df = pd.DataFrame(
        [{"sentence_id": sid, "split": "train"} for sid in train_ids]
        + [{"sentence_id": sid, "split": "test"} for sid in test_ids]
    )
    split_path = os.path.join(config.OUTPUT_DIR, "train_test_split.csv")
    split_df.to_csv(split_path, index=False, encoding="utf-8")
    print(f"  Train/test split: {len(train_ids)} train, {len(test_ids)} test → {split_path}")

    train_set = set(train_ids)
    test_set = set(test_ids)

    # ── 2. Prepare texts per style ───────────────────────────────────────────
    # Human texts (one per sentence_id)
    human_df = df.drop_duplicates(subset=["sentence_id"])[["sentence_id", "human_translation_en"]]
    human_train = human_df[human_df["sentence_id"].isin(train_set)]["human_translation_en"].tolist()
    human_val = human_df[human_df["sentence_id"].isin(test_set)]["human_translation_en"].tolist()

    # AI system texts
    ai_systems = df["ai_system_id"].unique().tolist()
    style_texts: dict[str, tuple[list[str], list[str]]] = {"human": (human_train, human_val)}

    for ai_sys in ai_systems:
        sub = df[df["ai_system_id"] == ai_sys]
        tr = sub[sub["sentence_id"].isin(train_set)]["ai_translation_en"].tolist()
        va = sub[sub["sentence_id"].isin(test_set)]["ai_translation_en"].tolist()
        style_texts[ai_sys] = (tr, va)

    # pooled_ai: all AI translations combined
    all_ai_train = df[df["sentence_id"].isin(train_set)]["ai_translation_en"].tolist()
    all_ai_val = df[df["sentence_id"].isin(test_set)]["ai_translation_en"].tolist()
    style_texts["pooled_ai"] = (all_ai_train, all_ai_val)

    # ── 3. Load tokenizer once ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(config.PRIMARY_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 4. Fine-tune each style ──────────────────────────────────────────────
    model_dir = os.path.join(config.OUTPUT_DIR, "models", config.PRIMARY_MODEL_NAME)
    log_dir = os.path.join(config.OUTPUT_DIR, "finetune_logs")
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    summaries: list[dict] = []
    for style, (train_texts, val_texts) in style_texts.items():
        summary = _finetune_one_style(
            style=style,
            train_texts=train_texts,
            val_texts=val_texts,
            base_model_name=config.PRIMARY_MODEL_NAME,
            save_dir=model_dir,
            log_dir=log_dir,
            tokenizer=tokenizer,
        )
        summaries.append(summary)

    # ── 5. Save summary ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(config.OUTPUT_DIR, "finetune_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\n  Finetune summary → {summary_path}")
    print(summary_df.to_string(index=False))

    elapsed = time.time() - start
    print(f"\n[Step 2] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
