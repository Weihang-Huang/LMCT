#!/usr/bin/env python3
"""Step 4 — Classification experiments and model-scale ablation.

Uses perplexity features to classify human vs. AI translations.
Also fine-tunes gpt2-medium / gpt2-large / gpt2-xl for the scale ablation,
computing perplexity and running the two-feature classifier for each.

Produces:
  - ``outputs/classification_results.csv``
  - ``outputs/roc_curves.png``
  - ``outputs/models/{variant}/{style}_lora/``   (ablation adapters)

SEED POLICY: RANDOM_SEED = 1024 is used for CV fold generation and data
splitting only.  Fine-tuning and inference remain non-deterministic.
"""

import os
import sys
import time
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

import config
from utils import (
    set_seed,
    ensure_dir,
    load_finetuned_model,
    compute_perplexity,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_classifiers() -> list[tuple[str, Pipeline]]:
    """Return a list of (name, pipeline) classifier objects."""
    return [
        (
            "LogisticRegression",
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
        ),
        (
            "SVM_RBF",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
        ),
        (
            "RandomForest",
            Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200))]),
        ),
    ]


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    experiment_name: str,
    base_model: str,
    feature_desc: str,
) -> list[dict]:
    """Run 5-fold CV for all classifiers and return result rows."""
    set_seed(config.RANDOM_SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    results: list[dict] = []

    for clf_name, pipeline in _get_classifiers():
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
        cv_res = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
        )
        results.append(
            {
                "experiment": experiment_name,
                "base_model": base_model,
                "classifier": clf_name,
                "features": feature_desc,
                "accuracy": cv_res["test_accuracy"].mean(),
                "precision": cv_res["test_precision"].mean(),
                "recall": cv_res["test_recall"].mean(),
                "f1": cv_res["test_f1"].mean(),
                "auroc": cv_res["test_roc_auc"].mean(),
            }
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_texts(texts: list[str], tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tok(examples: dict) -> dict:
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    ds.set_format("torch")
    return ds


def _finetune_ablation(
    variant: str,
    style: str,
    train_texts: list[str],
    val_texts: list[str],
) -> str:
    """Fine-tune one LoRA adapter for the ablation variant. Returns adapter path."""
    print(f"    Ablation fine-tune: model={variant}, style={style} "
          f"({len(train_texts)} train, {len(val_texts)} val)")

    tokenizer = AutoTokenizer.from_pretrained(variant)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = _tokenize_texts(train_texts, tokenizer, config.MAX_SEQ_LENGTH)
    val_ds = _tokenize_texts(val_texts, tokenizer, config.MAX_SEQ_LENGTH)

    model = AutoModelForCausalLM.from_pretrained(variant)
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=["c_attn", "c_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    log_dir = os.path.join(config.OUTPUT_DIR, "finetune_logs", f"{variant.replace('/', '_')}_{style}")
    ensure_dir(log_dir)

    training_args = TrainingArguments(
        output_dir=log_dir,
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
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer.train()

    adapter_path = os.path.join(config.OUTPUT_DIR, "models", variant, f"{style}_lora")
    ensure_dir(adapter_path)
    model.save_pretrained(adapter_path)
    print(f"    Adapter saved → {adapter_path}")
    return adapter_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    start = time.time()
    print(f"[Step 4] {time.strftime('%Y-%m-%d %H:%M:%S')} — Classification & ablation …")

    ppl_path = os.path.join(config.OUTPUT_DIR, "perplexity_matrix.csv")
    if not os.path.isfile(ppl_path):
        sys.exit(f"ERROR: {ppl_path} not found. Run step3 first.")

    ppl_df = pd.read_csv(ppl_path, encoding="utf-8")
    print(f"  Loaded perplexity matrix: {len(ppl_df)} rows")

    # ── Labels ───────────────────────────────────────────────────────────────
    ppl_df["label"] = (ppl_df["text_source"] != "human").astype(int)

    all_results: list[dict] = []

    # ── Experiment 1: two-feature classification (primary model) ─────────────
    print("\n  Experiment 1 — Two-feature (ppl_human_model, ppl_pooled_ai_model)")
    two_feat_cols = ["ppl_human_model", "ppl_pooled_ai_model"]
    missing = [c for c in two_feat_cols if c not in ppl_df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing}, skipping Experiment 1")
    else:
        X = ppl_df[two_feat_cols].values
        y = ppl_df["label"].values
        res = _run_cv(X, y, "two_feature", config.PRIMARY_MODEL_NAME, "ppl_human+ppl_pooled_ai")
        all_results.extend(res)
        for r in res:
            print(f"    {r['classifier']}: F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}")

    # ── Experiment 2: full-feature classification ────────────────────────────
    print("\n  Experiment 2 — Full-feature (all ppl columns)")
    ppl_cols = sorted([c for c in ppl_df.columns if c.startswith("ppl_")])
    X_full = ppl_df[ppl_cols].values
    y = ppl_df["label"].values
    res = _run_cv(X_full, y, "full_feature", config.PRIMARY_MODEL_NAME, "all_ppl_columns")
    all_results.extend(res)
    for r in res:
        print(f"    {r['classifier']}: F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}")

    # ── Experiment 3: cross-system generalisation ────────────────────────────
    print("\n  Experiment 3 — Cross-system generalisation (leave-one-system-out)")
    ai_systems = [s for s in ppl_df["text_source"].unique() if s != "human"]
    for held_out in ai_systems:
        train_mask = (ppl_df["text_source"] == "human") | (
            (ppl_df["text_source"] != held_out) & (ppl_df["text_source"] != "human")
        )
        # Actually: train on all except the held-out AI system; test on held-out + human
        train_sub = ppl_df[
            (ppl_df["text_source"] == "human") |
            (~ppl_df["text_source"].isin([held_out]))
        ].copy()
        # But we also need test set: the held-out system texts + equal number of human texts
        test_ai = ppl_df[ppl_df["text_source"] == held_out]
        test_human = ppl_df[ppl_df["text_source"] == "human"].sample(
            n=min(len(test_ai), len(ppl_df[ppl_df["text_source"] == "human"])),
            random_state=config.RANDOM_SEED,
        )
        test_sub = pd.concat([test_ai, test_human])

        if len(two_feat_cols) > 0 and all(c in ppl_df.columns for c in two_feat_cols):
            X_train = train_sub[two_feat_cols].values
            y_train = train_sub["label"].values
            X_test = test_sub[two_feat_cols].values
            y_test = test_sub["label"].values

            for clf_name, pipeline in _get_classifiers():
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred.astype(float)
                all_results.append(
                    {
                        "experiment": f"cross_system_holdout_{held_out}",
                        "base_model": config.PRIMARY_MODEL_NAME,
                        "classifier": clf_name,
                        "features": "ppl_human+ppl_pooled_ai",
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, zero_division=0),
                        "recall": recall_score(y_test, y_pred, zero_division=0),
                        "f1": f1_score(y_test, y_pred, zero_division=0),
                        "auroc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0,
                    }
                )
        print(f"    Held out '{held_out}': done")

    # ── Experiment 4: Model scale ablation ───────────────────────────────────
    print("\n  Experiment 4 — Model scale ablation")

    # We need train/test texts for ablation fine-tuning
    split_path = os.path.join(config.OUTPUT_DIR, "train_test_split.csv")
    ai_trans_path = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")
    split_df = pd.read_csv(split_path, encoding="utf-8")
    ai_df = pd.read_csv(ai_trans_path, encoding="utf-8")

    train_ids = set(split_df[split_df["split"] == "train"]["sentence_id"])
    test_ids = set(split_df[split_df["split"] == "test"]["sentence_id"])

    human_unique = ai_df.drop_duplicates(subset=["sentence_id"])
    human_train = human_unique[human_unique["sentence_id"].isin(train_ids)]["human_translation_en"].tolist()
    human_val = human_unique[human_unique["sentence_id"].isin(test_ids)]["human_translation_en"].tolist()
    pooled_ai_train = ai_df[ai_df["sentence_id"].isin(train_ids)]["ai_translation_en"].tolist()
    pooled_ai_val = ai_df[ai_df["sentence_id"].isin(test_ids)]["ai_translation_en"].tolist()

    # Build the test texts list (same as used in step3)
    test_records: list[dict] = []
    for _, row in human_unique.iterrows():
        if row["sentence_id"] in test_ids:
            test_records.append({"sentence_id": row["sentence_id"], "text_source": "human", "text": row["human_translation_en"]})
    for _, row in ai_df.iterrows():
        if row["sentence_id"] in test_ids:
            test_records.append({"sentence_id": row["sentence_id"], "text_source": row["ai_system_id"], "text": row["ai_translation_en"]})

    roc_data: dict[str, tuple] = {}  # variant -> (fpr, tpr, auc_score)

    # Also compute ROC for the primary model
    if "ppl_human_model" in ppl_df.columns and "ppl_pooled_ai_model" in ppl_df.columns:
        X_primary = ppl_df[["ppl_human_model", "ppl_pooled_ai_model"]].values
        y_primary = ppl_df["label"].values
        lr = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
        lr.fit(X_primary, y_primary)
        y_prob_primary = lr.predict_proba(X_primary)[:, 1]
        fpr_p, tpr_p, _ = roc_curve(y_primary, y_prob_primary)
        roc_data[config.PRIMARY_MODEL_NAME] = (fpr_p, tpr_p, roc_auc_score(y_primary, y_prob_primary))

    for variant in config.ABLATION_MODEL_NAMES:
        print(f"\n  --- Ablation: {variant} ---")
        try:
            # Fine-tune human + pooled_ai
            _finetune_ablation(variant, "human", human_train, human_val)
            _finetune_ablation(variant, "pooled_ai", pooled_ai_train, pooled_ai_val)

            # Compute perplexity
            human_adapter = os.path.join(config.OUTPUT_DIR, "models", variant, "human_lora")
            pooled_adapter = os.path.join(config.OUTPUT_DIR, "models", variant, "pooled_ai_lora")

            model_h, tok_h = load_finetuned_model(variant, human_adapter, config.DEVICE)
            model_a, tok_a = load_finetuned_model(variant, pooled_adapter, config.DEVICE)

            ablation_rows: list[dict] = []
            for rec in test_records:
                ppl_h, _ = compute_perplexity(model_h, tok_h, rec["text"], 1024, config.STRIDE, config.DEVICE)
                ppl_a, _ = compute_perplexity(model_a, tok_a, rec["text"], 1024, config.STRIDE, config.DEVICE)
                ablation_rows.append({
                    "ppl_human_model": ppl_h,
                    "ppl_pooled_ai_model": ppl_a,
                    "label": 0 if rec["text_source"] == "human" else 1,
                })

            # Free GPU memory
            del model_h, model_a
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            abl_df = pd.DataFrame(ablation_rows)
            X_abl = abl_df[["ppl_human_model", "ppl_pooled_ai_model"]].values
            y_abl = abl_df["label"].values

            res = _run_cv(X_abl, y_abl, "model_scale_ablation", variant, "ppl_human+ppl_pooled_ai")
            all_results.extend(res)
            for r in res:
                print(f"    {r['classifier']}: F1={r['f1']:.3f}  AUROC={r['auroc']:.3f}")

            # ROC for overlay
            lr_abl = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
            lr_abl.fit(X_abl, y_abl)
            y_prob_abl = lr_abl.predict_proba(X_abl)[:, 1]
            fpr_v, tpr_v, _ = roc_curve(y_abl, y_prob_abl)
            roc_data[variant] = (fpr_v, tpr_v, roc_auc_score(y_abl, y_prob_abl))

        except Exception as e:
            print(f"  WARNING: Ablation for {variant} failed: {e}")
            continue

    # ── Save classification results ──────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    ensure_dir(config.OUTPUT_DIR)
    results_path = os.path.join(config.OUTPUT_DIR, "classification_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    print(f"\n  Classification results → {results_path}")

    # ── ROC curves ───────────────────────────────────────────────────────────
    if roc_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        for variant, (fpr, tpr, auc_val) in sorted(roc_data.items()):
            ax.plot(fpr, tpr, label=f"{variant} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — GPT-2 Model Scale Ablation")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        roc_path = os.path.join(config.OUTPUT_DIR, "roc_curves.png")
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        print(f"  ROC curves → {roc_path}")

    elapsed = time.time() - start
    print(f"\n[Step 4] Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
