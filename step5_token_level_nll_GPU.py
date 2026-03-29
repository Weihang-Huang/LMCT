#!/usr/bin/env python3
"""Step 5 — Token-level NLL annotation.

Computes per-token NLL under the human and pooled-AI GPT-2 models, derives
style annotations, and generates:

  - ``outputs/token_nll_annotations.csv``
  - ``outputs/sentence_nll_summary.csv``
  - ``outputs/token_nll_heatmap_samples.html``

SEED POLICY: No fixed seed is used for NLL computation.
RANDOM_SEED = 1024 is used ONLY for selecting the 20-sentence heatmap sample.
"""

import os
import sys
import time
import html as html_mod

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

import config
from utils import (
    set_seed,
    ensure_dir,
    load_finetuned_model,
    compute_token_nll,
)


def _style_label(nll_diff: float) -> str:
    """Assign a style label based on nll_diff = nll_human − nll_ai."""
    if nll_diff >= 1.5:
        return "strongly_ai_style"
    elif nll_diff >= 0.5:
        return "weakly_ai_style"
    elif nll_diff <= -1.5:
        return "strongly_human_style"
    elif nll_diff <= -0.5:
        return "weakly_human_style"
    else:
        return "neutral"


def _nll_diff_to_color(nll_diff: float) -> str:
    """Map nll_diff to a background colour for the HTML heatmap."""
    # Blue = human style (negative diff), Red = AI style (positive diff)
    clamp = max(-3.0, min(3.0, nll_diff))
    if clamp >= 0:
        intensity = int(min(255, clamp / 3.0 * 200))
        return f"rgba(255,{255 - intensity},{255 - intensity},0.7)"
    else:
        intensity = int(min(255, abs(clamp) / 3.0 * 200))
        return f"rgba({255 - intensity},{255 - intensity},255,0.7)"


def main() -> None:
    start = time.time()
    print(f"[Step 5] {time.strftime('%Y-%m-%d %H:%M:%S')} — Token-level NLL annotation …")

    # ── Load prerequisites ───────────────────────────────────────────────────
    split_path = os.path.join(config.OUTPUT_DIR, "train_test_split.csv")
    ai_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    for p in (split_path, ai_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: {p} not found. Run previous steps first.")

    split_df = pd.read_csv(split_path, encoding="utf-8")
    ai_df = pd.read_csv(ai_path, encoding="utf-8")
    test_ids = set(split_df[split_df["split"] == "test"]["sentence_id"])

    # Build test texts
    records: list[dict] = []
    human_df = ai_df.drop_duplicates(subset=["sentence_id"])
    for _, row in human_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append({"sentence_id": row["sentence_id"], "text_source": "human", "text": row["human_translation_en"]})
    for _, row in ai_df.iterrows():
        if row["sentence_id"] in test_ids:
            records.append({"sentence_id": row["sentence_id"], "text_source": row["ai_system_id"], "text": row["ai_translation_en"]})

    print(f"  Test texts: {len(records)}")

    # ── Load models ──────────────────────────────────────────────────────────
    human_adapter = os.path.join(config.OUTPUT_DIR, "models", config.PRIMARY_MODEL_NAME, "human_lora")
    pooled_adapter = os.path.join(config.OUTPUT_DIR, "models", config.PRIMARY_MODEL_NAME, "pooled_ai_lora")
    for ap in (human_adapter, pooled_adapter):
        if not os.path.isdir(ap):
            sys.exit(f"ERROR: Adapter not found: {ap}. Run step2 first.")

    model_h, tok_h = load_finetuned_model(config.PRIMARY_MODEL_NAME, human_adapter, config.DEVICE)
    model_a, tok_a = load_finetuned_model(config.PRIMARY_MODEL_NAME, pooled_adapter, config.DEVICE)

    # ── Compute token-level NLL ──────────────────────────────────────────────
    all_token_rows: list[dict] = []
    sentence_summaries: list[dict] = []

    for rec in tqdm(records, desc="  Token NLL"):
        tokens_h = compute_token_nll(model_h, tok_h, rec["text"], config.DEVICE)
        tokens_a = compute_token_nll(model_a, tok_a, rec["text"], config.DEVICE)

        # Align by position — both should have the same tokenization (same base tokenizer)
        n = min(len(tokens_h), len(tokens_a))
        if n == 0:
            continue

        sent_nll_h: list[float] = []
        sent_nll_a: list[float] = []

        for i in range(n):
            nll_h = tokens_h[i]["nll"]
            nll_a = tokens_a[i]["nll"]
            nll_diff = nll_h - nll_a
            nll_ratio = nll_h / nll_a if nll_a != 0 else float("inf")

            sent_nll_h.append(nll_h)
            sent_nll_a.append(nll_a)

            all_token_rows.append(
                {
                    "sentence_id": rec["sentence_id"],
                    "text_source": rec["text_source"],
                    "token_position": tokens_h[i]["token_position"],
                    "token": tokens_h[i]["token"],
                    "nll_human": nll_h,
                    "nll_ai": nll_a,
                    "nll_diff": nll_diff,
                    "nll_ratio": nll_ratio,
                    "style_label": _style_label(nll_diff),
                }
            )

        # sentence-level summary
        arr_h = np.array(sent_nll_h)
        arr_a = np.array(sent_nll_a)
        sentence_summaries.append(
            {
                "sentence_id": rec["sentence_id"],
                "text_source": rec["text_source"],
                "mean_nll_human": arr_h.mean(),
                "mean_nll_ai": arr_a.mean(),
                "mean_nll_diff": (arr_h - arr_a).mean(),
                "n_tokens": n,
                "pct_ai_style": sum(1 for r in all_token_rows[-n:] if "ai_style" in r["style_label"]) / n,
                "pct_human_style": sum(1 for r in all_token_rows[-n:] if "human_style" in r["style_label"]) / n,
                "pct_neutral": sum(1 for r in all_token_rows[-n:] if r["style_label"] == "neutral") / n,
            }
        )

    # ── z-scores across the full token set ───────────────────────────────────
    token_df = pd.DataFrame(all_token_rows)
    if len(token_df) > 0:
        token_df["z_nll_human"] = stats.zscore(token_df["nll_human"].values)
        token_df["z_nll_ai"] = stats.zscore(token_df["nll_ai"].values)
    else:
        token_df["z_nll_human"] = []
        token_df["z_nll_ai"] = []

    # ── Save ─────────────────────────────────────────────────────────────────
    ensure_dir(config.OUTPUT_DIR)

    token_path = os.path.join(config.OUTPUT_DIR, "token_nll_annotations.csv")
    token_df.to_csv(token_path, index=False, encoding="utf-8")
    print(f"  Token annotations ({len(token_df)} rows) → {token_path}")

    sent_df = pd.DataFrame(sentence_summaries)
    sent_path = os.path.join(config.OUTPUT_DIR, "sentence_nll_summary.csv")
    sent_df.to_csv(sent_path, index=False, encoding="utf-8")
    print(f"  Sentence summaries ({len(sent_df)} rows) → {sent_path}")

    # ── HTML heatmaps for 20 random sentences ────────────────────────────────
    set_seed(config.RANDOM_SEED)
    unique_keys = token_df[["sentence_id", "text_source"]].drop_duplicates()
    n_sample = min(20, len(unique_keys))
    sample_keys = unique_keys.sample(n=n_sample, random_state=config.RANDOM_SEED)

    html_parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Token NLL Heatmaps</title>",
        "<style>body{font-family:monospace;margin:20px} .sent{margin-bottom:20px;} "
        ".token{display:inline-block;padding:2px 1px;margin:1px;border-radius:3px;} "
        "h3{margin-bottom:4px;} .legend{margin-bottom:20px;}</style></head><body>",
        "<h1>Token NLL Heatmap Samples</h1>",
        "<div class='legend'><b>Color key:</b> "
        "<span style='background:rgba(100,100,255,0.7);padding:2px 6px;'>human-style</span> "
        "<span style='background:rgba(240,240,240,0.7);padding:2px 6px;'>neutral</span> "
        "<span style='background:rgba(255,100,100,0.7);padding:2px 6px;'>AI-style</span></div>",
    ]

    for _, key_row in sample_keys.iterrows():
        sid = key_row["sentence_id"]
        src = key_row["text_source"]
        subset = token_df[(token_df["sentence_id"] == sid) & (token_df["text_source"] == src)]
        html_parts.append(f"<div class='sent'><h3>{html_mod.escape(str(sid))} ({html_mod.escape(str(src))})</h3>")
        for _, tok_row in subset.iterrows():
            color = _nll_diff_to_color(tok_row["nll_diff"])
            escaped = html_mod.escape(tok_row["token"])
            title = f"nll_h={tok_row['nll_human']:.2f} nll_a={tok_row['nll_ai']:.2f} diff={tok_row['nll_diff']:.2f}"
            html_parts.append(f"<span class='token' style='background:{color}' title='{title}'>{escaped}</span>")
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    html_path = os.path.join(config.OUTPUT_DIR, "token_nll_heatmap_samples.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"  Heatmaps → {html_path}")

    elapsed = time.time() - start
    print(f"\n[Step 5] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
