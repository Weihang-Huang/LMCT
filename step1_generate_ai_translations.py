#!/usr/bin/env python3
"""Step 1 — Generate AI translations via LLM API gateway.

Reads ``outputs/corpus_prepared.csv`` and produces ``outputs/ai_translations.csv``.

Each Chinese source sentence is translated into English by every model listed
in ``config.AI_SYSTEMS`` using a unified prompt.  The translations are
collected together with the original human (reference) translation.

API KEY
-------
Set the environment variable ``AI_GATEWAY_API_KEY`` before running, or pass it
directly when invoking the script::

    export AI_GATEWAY_API_KEY="your-key-here"
    python step1_generate_ai_translations.py

SEED POLICY: RANDOM_SEED is NOT used in this step.  LLM inference is
inherently non-deterministic and must NOT be seeded.
"""

import os
import sys
import time

import pandas as pd
import requests

import config
from utils import ensure_dir

# Inline import of the gateway client so the project stays self-contained.
from ai_gateway import Client

# ── Translation prompt ────────────────────────────────────────────────────────
# A single, unified prompt is used for every model so that the ONLY variation
# across AI systems is the model's internal style.  The prompt follows the
# best-practice guidelines from recent LLM translation research (Chandra et al.
# 2024; Zhang et al. 2026, arXiv:2603.09998) — it instructs the model to
# produce a faithful, natural English translation and to output nothing but the
# translation itself.

SYSTEM_PROMPT = (
    "You are a professional Chinese-to-English translator specialising in "
    "United Nations and international-affairs texts. "
    "Translate the following Chinese text into fluent, natural English that "
    "faithfully preserves the original meaning and register. "
    "Output ONLY the English translation — no explanations, preamble, "
    "annotations, or commentary."
)

# ── Retry / rate-limit parameters ─────────────────────────────────────────────
MAX_RETRIES: int = 3
INITIAL_BACKOFF_S: float = 2.0          # seconds; doubled on each retry


def _translate_one(
    client: Client,
    source_zh: str,
) -> str:
    """Call the LLM gateway and return the English translation string.

    Retries up to ``MAX_RETRIES`` times with exponential back-off on transient
    HTTP errors (429, 500, 502, 503, 504).

    Returns:
        The translated English text.  On persistent failure the string
        ``"[TRANSLATION_FAILED]"`` is returned so downstream steps can
        filter or flag it.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_zh},
    ]
    backoff = INITIAL_BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client(messages)
            # Strip stray whitespace / quotes the model may add
            return result.strip().strip('"').strip("'")
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                print(f"      HTTP {status} — retrying in {backoff:.0f}s "
                      f"(attempt {attempt}/{MAX_RETRIES})")
                time.sleep(backoff)
                backoff *= 2
            else:
                print(f"      ERROR (HTTP {status}) on attempt {attempt}: {exc}")
                return "[TRANSLATION_FAILED]"
        except Exception as exc:
            print(f"      Unexpected error on attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
            else:
                return "[TRANSLATION_FAILED]"
    return "[TRANSLATION_FAILED]"


def main() -> None:
    start = time.time()
    print(f"[Step 1] {time.strftime('%Y-%m-%d %H:%M:%S')} — Generating AI translations …")

    # ── Load prepared corpus ─────────────────────────────────────────────────
    corpus_path = os.path.join(config.OUTPUT_DIR, "corpus_prepared.csv")
    if not os.path.isfile(corpus_path):
        sys.exit(f"ERROR: Prerequisite missing — {corpus_path}. Run step0 first.")

    df = pd.read_csv(corpus_path, encoding="utf-8").iloc[0:1000]
    print(f"  Loaded {len(df)} sentences from corpus_prepared.csv")

    # ── Resolve API key ──────────────────────────────────────────────────────
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "")
    api_key = "46e28427137e489bad85014aa535a290"
    if not api_key:
        sys.exit(
            "ERROR: AI_GATEWAY_API_KEY environment variable is not set.\n"
            "       Export it before running:\n"
            "         export AI_GATEWAY_API_KEY=\"your-key-here\""
        )

    # ── Translate with each AI system ────────────────────────────────────────
    rows: list[dict] = []
    total_models = len(config.AI_SYSTEMS)

    for model_idx, ai_sys in enumerate(config.AI_SYSTEMS, 1):
        print(f"\n  [{model_idx}/{total_models}] Translating with model: {ai_sys}")
        client = Client(ai_sys, api_key)

        success_count = 0
        fail_count = 0
        for row_idx, row in df.iterrows():
            ai_text = _translate_one(client, row["source_zh"])
            if ai_text == "[TRANSLATION_FAILED]":
                fail_count += 1
            else:
                success_count += 1

            rows.append(
                {
                    "sentence_id": row["sentence_id"],
                    "source_zh": row["source_zh"],
                    "human_translation_en": row["human_translation_en"],
                    "ai_system_id": ai_sys,
                    "ai_translation_en": ai_text,
                }
            )

            # Progress indicator every 50 sentences
            done = row_idx + 1
            if done % 5 == 0 or done == len(df):
                print(f"    {done}/{len(df)} sentences done", end="\r")

        print(f"    {ai_sys}: {success_count} OK, {fail_count} failed"
              f" out of {len(df)}")

    # ── Save output ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows)
    ensure_dir(config.OUTPUT_DIR)
    out_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n  Saved {len(out_df)} rows → {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    fail_total = (out_df["ai_translation_en"] == "[TRANSLATION_FAILED]").sum()
    if fail_total > 0:
        print(f"\n  WARNING: {fail_total} translations failed across all models.")
        print("  Failed rows have ai_translation_en = '[TRANSLATION_FAILED]'.")
        print("  Downstream steps will treat them as missing data.")
    else:
        print("\n  All translations succeeded.")

    elapsed = time.time() - start
    print(f"\n[Step 1] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
