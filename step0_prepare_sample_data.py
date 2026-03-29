#!/usr/bin/env python3
"""Step 0 — Prepare sample data.

Load ``unpc_parallel.csv``, clean it, and produce ``outputs/corpus_prepared.csv``.
"""

import os
import sys
import time
import pandas as pd

import config
from utils import set_seed, ensure_dir


def main() -> None:
    start = time.time()
    print(f"[Step 0] {time.strftime('%Y-%m-%d %H:%M:%S')} — Preparing sample data …")

    # ── 1. Load raw CSV ──────────────────────────────────────────────────────
    raw_path = os.path.join(config.DATA_DIR, config.SAMPLE_DATA_FILE)
    if not os.path.isfile(raw_path):
        sys.exit(f"ERROR: Input file not found: {raw_path}")

    df = pd.read_csv(raw_path, encoding="utf-8")
    total_loaded = len(df)
    print(f"  Loaded {total_loaded} rows from {raw_path}")

    # Validate expected columns
    expected_cols = {"text_id", "english_text", "chinese_text"}
    if not expected_cols.issubset(set(df.columns)):
        sys.exit(
            f"ERROR: Expected columns {expected_cols}; "
            f"found {set(df.columns)}"
        )

    # ── 2. Rename columns ───────────────────────────────────────────────────
    df = df.rename(
        columns={
            "text_id": "sentence_id",
            "chinese_text": "source_zh",
            "english_text": "human_translation_en",
        }
    )

    # ── 3. Strip whitespace ──────────────────────────────────────────────────
    for col in ["source_zh", "human_translation_en"]:
        df[col] = df[col].astype(str).str.strip()

    # ── 4. Filter short English texts ────────────────────────────────────────
    df["_en_tok_count"] = df["human_translation_en"].str.split().str.len()
    df = df[df["_en_tok_count"] >= config.MIN_ENGLISH_TOKENS].copy()
    rows_after_filter = len(df)
    print(f"  Rows after token-length filter (>= {config.MIN_ENGLISH_TOKENS} tokens): {rows_after_filter}")

    # ── 5. Remove exact duplicates ───────────────────────────────────────────
    df = df.drop_duplicates(subset=["source_zh", "human_translation_en"]).copy()
    rows_after_dedup = len(df)
    print(f"  Rows after deduplication: {rows_after_dedup}")

    # ── 6. Optional sampling ─────────────────────────────────────────────────
    set_seed(config.RANDOM_SEED)

    if config.SAMPLE_SIZE and config.SAMPLE_SIZE > 0:
        n = min(config.SAMPLE_SIZE, len(df))
        df = df.sample(n=n, random_state=config.RANDOM_SEED).copy()
        print(f"  Sampled {n} rows (RANDOM_SEED={config.RANDOM_SEED})")
    else:
        print("  Using all available rows (no sampling)")

    rows_after_sample = len(df)

    # ── 7. Reset index and save ──────────────────────────────────────────────
    df = df[["sentence_id", "source_zh", "human_translation_en"]].reset_index(drop=True)
    ensure_dir(config.OUTPUT_DIR)
    out_path = os.path.join(config.OUTPUT_DIR, "corpus_prepared.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved → {out_path}")

    # ── 8. Summary statistics ────────────────────────────────────────────────
    en_tok_lens = df["human_translation_en"].str.split().str.len()
    zh_char_lens = df["source_zh"].str.len()

    print("\n  === Summary Statistics ===")
    print(f"  Total rows loaded:          {total_loaded}")
    print(f"  Rows after filtering:       {rows_after_filter}")
    print(f"  Rows after deduplication:   {rows_after_dedup}")
    print(f"  Rows after sampling:        {rows_after_sample}")
    print(f"  Mean English token length:  {en_tok_lens.mean():.1f}")
    print(f"  Median English token length:{en_tok_lens.median():.1f}")
    print(f"  Mean Chinese char length:   {zh_char_lens.mean():.1f}")
    print(f"  Median Chinese char length: {zh_char_lens.median():.1f}")

    elapsed = time.time() - start
    print(f"\n[Step 0] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
