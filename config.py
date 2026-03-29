"""
Central configuration for the AI Translation Style Detection experiment suite.

SEED POLICY
-----------
RANDOM_SEED = 1024 is used ONLY for data-level randomness:
  - Corpus sampling, row sub-sampling
  - Train / test split
  - Cross-validation fold generation
  - Dummy translation perturbation choices
  - Selection of random examples for visualisation

It must NEVER be applied to:
  - Fine-tuning (transformers.Trainer seed, TrainingArguments seed)
  - Model training loops
  - Forward-pass inference
  - Perplexity computation
  - Token-level NLL computation
  - torch.manual_seed() or torch.cuda.manual_seed_all() before training/inference
"""

import os
import torch

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "outputs")

# ── Input data ───────────────────────────────────────────────────────────────
SAMPLE_DATA_FILE: str = "unpc_parallel.csv"

# Set to a positive integer for a fast debug run (e.g. 200).
# Set to None or 0 to use all available rows.
SAMPLE_SIZE: int | None = 1000

# Minimum number of whitespace-delimited English tokens a row must have to be
# kept.  This filters out single-letter section markers (A, B, C …).
MIN_ENGLISH_TOKENS: int = 500

# ── GPT-2 model family ──────────────────────────────────────────────────────
GPT2_VARIANTS: list[str] = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
PRIMARY_MODEL_NAME: str = "gpt2"
ABLATION_MODEL_NAMES: list[str] = ["gpt2-medium", "gpt2-large", "gpt2-xl"]

# ── AI translation systems ──────────────────────────────────────────────────
AI_SYSTEMS: list[str] = [
    "grok-3",
    "llama-3.3-70b-instruct",
]

# ── LoRA hyper-parameters ────────────────────────────────────────────────────
LORA_R: int = 8
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.1

# ── Training hyper-parameters ────────────────────────────────────────────────
LEARNING_RATE: float = 2e-5
NUM_EPOCHS: int = 3
BATCH_SIZE: int = 4
MAX_SEQ_LENGTH: int = 512          # Must be <= 1024
STRIDE: int = 256

# ── Evaluation ───────────────────────────────────────────────────────────────
TEST_SPLIT_RATIO: float = 0.2

# ── Reproducibility (DATA-LEVEL ONLY) ───────────────────────────────────────
# See the seed policy docstring at the top of this file.
RANDOM_SEED: int = 1024

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
