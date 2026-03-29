#!/usr/bin/env python3
"""Step 7 — Generate summary report.

Compiles all experiment outputs into ``outputs/summary_report.csv`` and
prints a human-readable summary to stdout.
"""

import os
import sys
import time

import pandas as pd

import config
from utils import ensure_dir


def _try_load(name: str) -> pd.DataFrame | None:
    """Attempt to load a CSV from OUTPUT_DIR, return None on failure."""
    path = os.path.join(config.OUTPUT_DIR, name)
    if not os.path.isfile(path):
        print(f"  WARNING: {name} not found — skipping")
        return None
    return pd.read_csv(path, encoding="utf-8")


def main() -> None:
    start = time.time()
    print(f"[Step 7] {time.strftime('%Y-%m-%d %H:%M:%S')} — Generating summary report …")

    rows: list[dict] = []

    # ── Corpus statistics ────────────────────────────────────────────────────
    corpus = _try_load("corpus_prepared.csv")
    if corpus is not None:
        en_lens = corpus["human_translation_en"].str.split().str.len()
        zh_lens = corpus["source_zh"].str.len()
        rows.append({"section": "corpus_stats", "metric": "total_sentences", "value": str(len(corpus))})
        rows.append({"section": "corpus_stats", "metric": "mean_en_tokens", "value": f"{en_lens.mean():.1f}"})
        rows.append({"section": "corpus_stats", "metric": "median_en_tokens", "value": f"{en_lens.median():.1f}"})
        rows.append({"section": "corpus_stats", "metric": "mean_zh_chars", "value": f"{zh_lens.mean():.1f}"})
        rows.append({"section": "corpus_stats", "metric": "median_zh_chars", "value": f"{zh_lens.median():.1f}"})

    # ── Finetune summary ─────────────────────────────────────────────────────
    ft = _try_load("finetune_summary.csv")
    if ft is not None:
        for _, r in ft.iterrows():
            rows.append({
                "section": "finetune_summary",
                "metric": f"{r['base_model']}_{r['style']}_val_ppl",
                "value": f"{r['final_val_ppl']:.2f}",
            })
            rows.append({
                "section": "finetune_summary",
                "metric": f"{r['base_model']}_{r['style']}_val_loss",
                "value": f"{r['final_val_loss']:.4f}",
            })

    # ── Classification best results ──────────────────────────────────────────
    clf = _try_load("classification_results.csv")
    if clf is not None:
        # Best per experiment
        for exp in clf["experiment"].unique():
            sub = clf[clf["experiment"] == exp]
            best = sub.loc[sub["f1"].idxmax()]
            rows.append({
                "section": "classification_best",
                "metric": f"{exp}_best_classifier",
                "value": best["classifier"],
            })
            for m in ["accuracy", "precision", "recall", "f1", "auroc"]:
                rows.append({
                    "section": "classification_best",
                    "metric": f"{exp}_best_{m}",
                    "value": f"{best[m]:.4f}",
                })

    # ── Token annotation statistics ──────────────────────────────────────────
    tok = _try_load("token_nll_annotations.csv")
    if tok is not None:
        rows.append({"section": "token_annotation_stats", "metric": "total_tokens", "value": str(len(tok))})
        for label in tok["style_label"].unique():
            count = (tok["style_label"] == label).sum()
            pct = count / len(tok) * 100
            rows.append({
                "section": "token_annotation_stats",
                "metric": f"pct_{label}",
                "value": f"{pct:.1f}%",
            })

    # ── Sentence NLL summary ─────────────────────────────────────────────────
    sent = _try_load("sentence_nll_summary.csv")
    if sent is not None:
        rows.append({"section": "token_annotation_stats", "metric": "total_sentences_annotated", "value": str(len(sent))})
        rows.append({"section": "token_annotation_stats", "metric": "mean_nll_diff",
                      "value": f"{sent['mean_nll_diff'].mean():.4f}"})

    # ── Probe summary ────────────────────────────────────────────────────────
    probe = _try_load("probe_results.csv")
    if probe is not None:
        for _, r in probe.iterrows():
            rows.append({
                "section": "probe_summary",
                "metric": f"{r['probe_model']}_{'ft' if r['finetuned'] else 'base'}_{r['text_source']}_mean_ppl",
                "value": f"{r['mean_ppl']:.2f}",
            })
            if pd.notna(r.get("detection_auroc")) and r["text_source"] != "human":
                rows.append({
                    "section": "probe_summary",
                    "metric": f"{r['probe_model']}_{'ft' if r['finetuned'] else 'base'}_{r['text_source']}_auroc",
                    "value": f"{r['detection_auroc']:.4f}",
                })

    # ── Save ─────────────────────────────────────────────────────────────────
    report_df = pd.DataFrame(rows)
    ensure_dir(config.OUTPUT_DIR)
    out_path = os.path.join(config.OUTPUT_DIR, "summary_report.csv")
    report_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Summary report ({len(report_df)} rows) → {out_path}")

    # ── Human-readable stdout ────────────────────────────────────────────────
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              AI Translation Style Detection Report          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝\n")

    for section in report_df["section"].unique():
        print(f"  [{section}]")
        sub = report_df[report_df["section"] == section]
        for _, r in sub.iterrows():
            print(f"    {r['metric']}: {r['value']}")
        print()

    elapsed = time.time() - start
    print(f"[Step 7] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
