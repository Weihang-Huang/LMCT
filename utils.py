"""
Shared utility functions for the AI Translation Style Detection experiment suite.

SEED POLICY
-----------
set_seed() is restricted to DATA-LEVEL reproducibility only.
It sets seeds for Python random and NumPy — NEVER for PyTorch.
Do NOT call set_seed() before model training or inference.
"""

import os
import random
import math
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ─────────────────────────────────────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set the random seed for DATA-LEVEL reproducibility only.

    Sets:
        - Python built-in ``random``
        - NumPy

    Does NOT set:
        - ``torch.manual_seed``
        - ``torch.cuda.manual_seed_all``

    .. warning::
        This function must NEVER be called before model training or inference.
        It is intended solely for data sampling, splitting, and CV fold
        generation.
    """
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not already exist.

    Returns:
        The same *path* for convenience.
    """
    os.makedirs(path, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_base_model(
    model_name: str,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a base GPT-2 model and tokenizer **without** any LoRA adapter.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"gpt2"``).
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(model, tokenizer)`` tuple. The tokenizer will have its
        ``pad_token`` set to ``eos_token``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device)
    model.eval()
    return model, tokenizer


def load_finetuned_model(
    base_model_name: str,
    adapter_path: str,
    device: str,
) -> tuple[PeftModel, AutoTokenizer]:
    """Load a base GPT-2 model with a LoRA adapter merged on top.

    Args:
        base_model_name: HuggingFace model identifier (e.g. ``"gpt2"``).
        adapter_path: Local path to the saved LoRA adapter directory.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.config.pad_token_id = tokenizer.eos_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity computation (sliding-window approach)
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int = 1024,
    stride: int = 256,
    device: str = "cpu",
) -> tuple[float, float]:
    """Compute sentence-level perplexity and average NLL for *text*.

    Uses the HuggingFace sliding-window method for sequences that exceed
    *max_length*.

    Args:
        model: A causal LM (GPT-2 base or with LoRA adapter).
        tokenizer: Corresponding tokenizer.
        text: Input text string.
        max_length: Context window size (must be <= 1024 for GPT-2).
        stride: Stride for the sliding window.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(perplexity, average_nll)`` — both are positive floats.
        If the text is empty or produces no evaluable tokens, returns
        ``(float('inf'), float('inf'))``.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    if seq_len == 0:
        return float("inf"), float("inf")

    nlls: list[float] = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end  # tokens whose loss we actually record

        input_slice = input_ids[:, begin:end]
        target_slice = input_slice.clone()
        # mask everything before the new tokens so they're not double-counted
        target_slice[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(input_slice, labels=target_slice)
            neg_log_likelihood = outputs.loss  # mean over target tokens

        # outputs.loss is average NLL over the non-masked target tokens
        nlls.append(neg_log_likelihood.item() * target_len)

        prev_end = end
        if end == seq_len:
            break

    total_nll = sum(nlls)
    total_tokens = prev_end  # number of tokens evaluated
    if total_tokens == 0:
        return float("inf"), float("inf")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


# ─────────────────────────────────────────────────────────────────────────────
# Token-level NLL computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_nll(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu",
) -> list[dict]:
    """Compute per-token negative log-likelihood for *text*.

    Args:
        model: A causal LM.
        tokenizer: Corresponding tokenizer.
        text: Input text string.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        A list of dicts, one per token (starting from the second token,
        since the first has no preceding context), each containing:

        - ``token_position`` (int)
        - ``token`` (str, decoded)
        - ``nll`` (float)
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    if seq_len <= 1:
        return []

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits

    # shift: predict token i from position i-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )  # shape: (seq_len - 1,)

    results: list[dict] = []
    for i, nll_val in enumerate(token_losses):
        token_id = shift_labels[0, i].item()
        token_str = tokenizer.decode([token_id])
        results.append(
            {
                "token_position": i + 1,  # 1-based, offset since we skip pos 0
                "token": token_str,
                "nll": nll_val.item(),
            }
        )
    return results
