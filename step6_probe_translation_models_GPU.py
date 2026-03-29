#!/usr/bin/env python3
"""Step 6 — Probe translation models and unfinetuned GPT-2 baselines.

Computes sentence-level perplexity using each unfinetuned GPT-2 variant and
performs statistical tests comparing human vs. AI texts.

Produces ``outputs/probe_results.csv``.

SEED POLICY: No fixed seed is used for inference.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import config
from utils import (
    ensure_dir,
    load_base_model,
    load_finetuned_model,
    compute_perplexity,
)


def _probe_one_model(
    model,
    tokenizer,
    records: list[dict],
    device: str,
) -> pd.DataFrame:
    """Compute perplexity for every record and return a DataFrame."""
    rows: list[dict] = []
    for rec in tqdm(records, desc="    Probing", leave=False):
        ppl, avg_nll = compute_perplexity(
            model, tokenizer, rec["text"],
            max_length=1024, stride=config.STRIDE, device=device,
        )
        rows.append({
            "sentence_id": rec["sentence_id"],
            "text_source": rec["text_source"],
            "ppl": ppl,
            "avg_nll": avg_nll,
        })
    return pd.DataFrame(rows)


def _compute_stats(
    probe_df: pd.DataFrame,
    probe_model: str,
    finetuned: bool,
) -> list[dict]:
    """Compute per-source statistics and Wilcoxon test (human vs each AI)."""
    results: list[dict] = []
    human_ppls = probe_df[probe_df["text_source"] == "human"]["ppl"].values

    for src in probe_df["text_source"].unique():
        sub = probe_df[probe_df["text_source"] == src]
        mean_ppl = sub["ppl"].mean()
        std_ppl = sub["ppl"].std()

        # Wilcoxon test: human vs. this source
        w_stat, w_p, effect = float("nan"), float("nan"), float("nan")
        auroc = float("nan")
        if src != "human" and len(human_ppls) > 0 and len(sub) > 0:
            ai_ppls = sub["ppl"].values
            n = min(len(human_ppls), len(ai_ppls))
            if n > 10:
                try:
                    w_stat, w_p = wilcoxon(human_ppls[:n], ai_ppls[:n])
                    # rank-biserial effect size approximation
                    effect = 1 - (2 * w_stat) / (n * (n + 1))
                except Exception:
                    pass
            # AUROC: can perplexity from this model separate human from this AI?
            labels = np.array([0] * len(human_ppls) + [1] * len(ai_ppls))
            scores = np.concatenate([human_ppls, ai_ppls])
            if len(set(labels)) > 1:
                try:
                    auroc = roc_auc_score(labels, scores)
                except Exception:
                    pass

        results.append({
            "probe_model": probe_model,
            "finetuned": finetuned,
            "text_source": src,
            "mean_ppl": mean_ppl,
            "std_ppl": std_ppl,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "effect_size": effect,
            "detection_auroc": auroc,
        })

    return results


def main() -> None:
    start = time.time()
    print(f"[Step 6] {time.strftime('%Y-%m-%d %H:%M:%S')} — Probing models …")

    # ── Load test texts ──────────────────────────────────────────────────────
    split_path = os.path.join(config.OUTPUT_DIR, "train_test_split.csv")
    ai_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    for p in (split_path, ai_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: {p} not found. Run previous steps first.")

    split_df = pd.read_csv(split_path, encoding="utf-8")
    ai_df = pd.read_csv(ai_path, encoding="utf-8")
    test_ids = set(split_df[split_df["split"] == "test"]["sentence_id"])

    records: list[dict] = []
    human_df = ai_df.drop_duplicates(subset=["sentence_id"])
    for _, row in human_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append({"sentence_id": row["sentence_id"], "text_source": "human", "text": row["human_translation_en"]})
    for _, row in ai_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append({"sentence_id": row["sentence_id"], "text_source": row["ai_system_id"], "text": row["ai_translation_en"]})

    print(f"  Test texts: {len(records)}")

    all_results: list[dict] = []

    # ── 1. Probe unfinetuned GPT-2 variants ──────────────────────────────────
    for variant in config.GPT2_VARIANTS:
        print(f"\n  Probing unfinetuned {variant} …")
        try:
            model, tokenizer = load_base_model(variant, config.DEVICE)
            probe_df = _probe_one_model(model, tokenizer, records, config.DEVICE)
            results = _compute_stats(probe_df, variant, finetuned=False)
            all_results.extend(results)
            del model
            import torch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  WARNING: Failed to probe {variant}: {e}")

    # ── 2. Probe fine-tuned adapters (for comparison) ────────────────────────
    model_base_dir = os.path.join(config.OUTPUT_DIR, "models", config.PRIMARY_MODEL_NAME)
    if os.path.isdir(model_base_dir):
        for name in sorted(os.listdir(model_base_dir)):
            adapter_path = os.path.join(model_base_dir, name)
            if os.path.isdir(adapter_path) and name.endswith("_lora"):
                style = name.replace("_lora", "")
                probe_name = f"{config.PRIMARY_MODEL_NAME}+{style}_lora"
                print(f"\n  Probing fine-tuned {probe_name} …")
                try:
                    model, tokenizer = load_finetuned_model(
                        config.PRIMARY_MODEL_NAME, adapter_path, config.DEVICE
                    )
                    probe_df = _probe_one_model(model, tokenizer, records, config.DEVICE)
                    results = _compute_stats(probe_df, probe_name, finetuned=True)
                    all_results.extend(results)
                    del model
                    import torch
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except Exception as e:
                    print(f"  WARNING: Failed to probe {probe_name}: {e}")

    # ── 3. Probe local translation models (placeholder) ─────────────────────
    print("\n  NOTE: No local translation models detected — skipping translation model probes.")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(all_results)
    ensure_dir(config.OUTPUT_DIR)
    out_path = os.path.join(config.OUTPUT_DIR, "probe_results.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Probe results ({len(out_df)} rows) → {out_path}")

    elapsed = time.time() - start
    print(f"\n[Step 6] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
