#!/usr/bin/env python3
"""Step 3 — Compute the cross-perplexity matrix.

For every test-set text, compute perplexity under every fine-tuned GPT-2
style model and save the result as ``outputs/perplexity_matrix.csv``.

SEED POLICY: No fixed seed is used in this step.
"""

import os
import sys
import time

import pandas as pd
from tqdm import tqdm

import config
from utils import load_finetuned_model, compute_perplexity, ensure_dir


def main() -> None:
    start = time.time()
    print(f"[Step 3] {time.strftime('%Y-%m-%d %H:%M:%S')} — Computing cross-perplexity matrix …")

    # ── Load prerequisite files ──────────────────────────────────────────────
    split_path = os.path.join(config.OUTPUT_DIR, "train_test_split.csv")
    ai_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    for p in (split_path, ai_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: {p} not found. Run previous steps first.")

    split_df = pd.read_csv(split_path, encoding="utf-8")
    ai_df = pd.read_csv(ai_path, encoding="utf-8")

    test_ids = set(split_df[split_df["split"] == "test"]["sentence_id"].tolist())
    print(f"  Test sentences: {len(test_ids)}")

    # ── Build list of (sentence_id, text_source, text) tuples ────────────────
    records: list[dict] = []
    # Human texts
    human_df = ai_df.drop_duplicates(subset=["sentence_id"])[["sentence_id", "human_translation_en"]]
    for _, row in human_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "text_source": "human",
                    "text": row["human_translation_en"],
                }
            )
    # AI texts
    for _, row in ai_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append(
                {
                    "sentence_id": row["sentence_id"],
                    "text_source": row["ai_system_id"],
                    "text": row["ai_translation_en"],
                }
            )
    print(f"  Total test texts to evaluate: {len(records)}")

    # ── Discover style adapters ──────────────────────────────────────────────
    model_base_dir = os.path.join(config.OUTPUT_DIR, "models", config.PRIMARY_MODEL_NAME)
    style_dirs: dict[str, str] = {}
    if os.path.isdir(model_base_dir):
        for name in os.listdir(model_base_dir):
            full = os.path.join(model_base_dir, name)
            if os.path.isdir(full) and name.endswith("_lora"):
                style_key = name.replace("_lora", "")
                style_dirs[style_key] = full

    if not style_dirs:
        sys.exit("ERROR: No fine-tuned adapters found. Run step2 first.")

    print(f"  Style models found: {list(style_dirs.keys())}")

    # ── Load models ──────────────────────────────────────────────────────────
    style_models: dict = {}
    for style_key, adapter_path in style_dirs.items():
        print(f"  Loading model for style '{style_key}' …")
        model, tokenizer = load_finetuned_model(
            config.PRIMARY_MODEL_NAME, adapter_path, config.DEVICE
        )
        style_models[style_key] = (model, tokenizer)

    # ── Compute cross-perplexity ─────────────────────────────────────────────
    output_rows: list[dict] = []
    for rec in tqdm(records, desc="  Perplexity"):
        row: dict = {
            "sentence_id": rec["sentence_id"],
            "text_source": rec["text_source"],
            "text": rec["text"],
        }
        for style_key, (model, tokenizer) in style_models.items():
            ppl, avg_nll = compute_perplexity(
                model,
                tokenizer,
                rec["text"],
                max_length=1024,
                stride=config.STRIDE,
                device=config.DEVICE,
            )
            row[f"ppl_{style_key}_model"] = ppl
            row[f"nll_{style_key}_model"] = avg_nll
        output_rows.append(row)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(output_rows)
    ensure_dir(config.OUTPUT_DIR)
    out_path = os.path.join(config.OUTPUT_DIR, "perplexity_matrix.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(out_df)} rows → {out_path}")

    # Quick sanity check
    ppl_cols = [c for c in out_df.columns if c.startswith("ppl_")]
    for c in ppl_cols:
        vals = out_df[c]
        print(f"    {c}: mean={vals.mean():.2f}, median={vals.median():.2f}, "
              f"min={vals.min():.2f}, max={vals.max():.2f}")

    elapsed = time.time() - start
    print(f"\n[Step 3] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
