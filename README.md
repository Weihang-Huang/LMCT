# AI Translation Style Detection: Experiment Suite

## Overview

This repository contains a complete, reproducible experiment pipeline for **AI translation style detection** using perplexity and negative log-likelihood (NLL) as computed by fine-tuned language models.

The study uses the **GPT-2 family** (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`) as the base models for all fine-tuning and perplexity experiments. No other model architecture is used.

AI translations are generated via six large language models accessed through a unified API gateway:
`gpt-5.1`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini`, `grok-3`, `llama-3.3-70b-instruct`.

The pipeline:

1. Prepares a Chinese–English parallel corpus (UN Parallel Corpus v1.0 subset)
2. Generates AI translations (Chinese → English) via LLM API calls
3. Fine-tunes GPT-2 with LoRA adapters on human- and AI-translated texts
4. Computes a cross-perplexity matrix across all style models
5. Trains classifiers to distinguish human from AI translations using perplexity features
6. Runs a model-scale ablation across all four GPT-2 variants
7. Annotates token-level AI translation style using per-token NLL
8. Probes unfinetuned models as baselines
9. Compiles a summary report

## GPU-Required Steps

Files with `_GPU` in the filename contain operations that require NVIDIA GPU
acceleration (LoRA fine-tuning, perplexity computation, model inference). Steps
without this tag are CPU-only and can run on any machine:

| Step | Filename | GPU? |
|---|---|---|
| 0 | `step0_prepare_sample_data.py` | No |
| 1 | `step1_generate_ai_translations.py` | No |
| 2 | `step2_finetune_lm_GPU.py` | **Yes** |
| 3 | `step3_compute_perplexity_GPU.py` | **Yes** |
| 4 | `step4_classify_style_GPU.py` | **Yes** |
| 5 | `step5_token_level_nll_GPU.py` | **Yes** |
| 6 | `step6_probe_translation_models_GPU.py` | **Yes** |
| 7 | `step7_generate_report.py` | No |

## Requirements

### Hardware

| GPT-2 Variant | Minimum VRAM |
|---|---|
| `gpt2` / `gpt2-medium` | >= 8 GB |
| `gpt2-large` | >= 16 GB |
| `gpt2-xl` | >= 24 GB |

CPU-only mode is supported but substantially slower.

### Software

- Python 3.10+
- CUDA 12.x recommended

### Install Dependencies

```bash
pip install -r requirements.txt
```

## API Gateway Configuration

Step 1 calls an LLM API gateway for Chinese-to-English translation. You must
set the API subscription key before running:

```bash
export AI_GATEWAY_API_KEY="your-subscription-key-here"
```

The gateway endpoint and model catalogue are configured in `ai_gateway.py`.
The current models used for translation are listed in `config.py` under
`AI_SYSTEMS`.

## Input Data

Place a file named `unpc_parallel.csv` in the `data/` directory. It must contain exactly these columns:

| CSV Column | Experiment Variable |
|---|---|
| `text_id` | `sentence_id` |
| `english_text` | `human_translation_en` |
| `chinese_text` | `source_zh` |

There is **no `doc_id` column**. All rows are treated as independent aligned sentence pairs.

## Quick Start (Sample Data)

```bash
# 1. Place the data file
cp unpc_parallel.csv data/

# 2. Set API key
export AI_GATEWAY_API_KEY="your-key"

# 3. Run the pipeline
python step0_prepare_sample_data.py
python step1_generate_ai_translations.py
python step2_finetune_lm_GPU.py
python step3_compute_perplexity_GPU.py
python step4_classify_style_GPU.py
python step5_token_level_nll_GPU.py
python step6_probe_translation_models_GPU.py
python step7_generate_report.py
```

By default, `config.py` uses `SAMPLE_SIZE = 200` for a fast debug run.

## Full Run (UN Parallel Corpus)

1. Obtain the full UN Parallel Corpus v1.0 (Chinese–English).
2. Convert it to the CSV format described above with columns `text_id`, `english_text`, `chinese_text`.
3. Place the file in `data/` as `unpc_parallel.csv`.
4. In `config.py`, set `SAMPLE_SIZE = None` (or `0`) to use all rows.
5. Set the `AI_GATEWAY_API_KEY` environment variable.
6. Run the pipeline as above.

## GPT-2 Model Variants

| Variant | HuggingFace ID | Parameters | Layers | Hidden |
|---------|----------------|------------|--------|--------|
| Small   | `gpt2`         | 117M       | 12     | 768    |
| Medium  | `gpt2-medium`  | 345M       | 24     | 1024   |
| Large   | `gpt2-large`   | 774M       | 36     | 1280   |
| XL      | `gpt2-xl`      | 1.5B       | 48     | 1600   |

The **primary model** for the main experiments is `gpt2` (Small). The other three variants are used in the model-scale ablation (Step 4).

## AI Translation Systems

| Model ID | Provider | Description |
|---|---|---|
| `gpt-5.1` | OpenAI | Flagship GPT-5.1 |
| `gpt-4o-mini` | OpenAI | Lightweight GPT-4o variant |
| `gpt-5` | OpenAI | GPT-5 |
| `gpt-5-mini` | OpenAI | Lightweight GPT-5 variant |
| `grok-3` | xAI | Grok-3 |
| `llama-3.3-70b-instruct` | Meta | Llama 3.3 70B instruction-tuned |

All systems use the same translation prompt to ensure the only variation is model-internal style.

## Configuration

All adjustable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SAMPLE_SIZE` | 200 | Rows to sample (set `None` for all) |
| `MIN_ENGLISH_TOKENS` | 5 | Minimum English tokens per sentence |
| `AI_SYSTEMS` | 6 LLMs (see above) | AI translation system IDs |
| `LORA_R` | 8 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA alpha |
| `LEARNING_RATE` | 2e-5 | Training learning rate |
| `NUM_EPOCHS` | 3 | Training epochs |
| `BATCH_SIZE` | 4 | Batch size |
| `MAX_SEQ_LENGTH` | 512 | Max sequence length (<= 1024) |
| `STRIDE` | 256 | Sliding window stride for perplexity |
| `TEST_SPLIT_RATIO` | 0.2 | Fraction held out for testing |
| `RANDOM_SEED` | 1024 | Data-level seed (see below) |

## Output Files

| File | Description |
|---|---|
| `corpus_prepared.csv` | Cleaned, filtered parallel corpus |
| `ai_translations.csv` | AI translations from 6 LLMs |
| `train_test_split.csv` | Sentence-level train/test split |
| `finetune_summary.csv` | Fine-tuning metrics per style model |
| `perplexity_matrix.csv` | Cross-perplexity: every text x every style model |
| `classification_results.csv` | Classifier accuracy, F1, AUROC by experiment |
| `roc_curves.png` | Overlaid ROC curves for the GPT-2 scale ablation |
| `token_nll_annotations.csv` | Per-token NLL with style labels |
| `sentence_nll_summary.csv` | Sentence-level NLL aggregates |
| `token_nll_heatmap_samples.html` | Interactive HTML heatmaps for 20 sample sentences |
| `probe_results.csv` | Probing results for unfinetuned models |
| `summary_report.csv` | Compiled summary of all experiments |

## Reproducing Paper Results

1. Use `RANDOM_SEED = 1024` for all data-level random processes.
2. Do **NOT** set seeds during fine-tuning or inference.
3. Use `PRIMARY_MODEL_NAME = "gpt2"` for the main experiments.
4. Use a large `SAMPLE_SIZE` (or the full corpus) for the actual experiment.
5. Ensure CUDA is available for practical runtime.
6. Set the `AI_GATEWAY_API_KEY` environment variable for Step 1.

**Note:** Training results may vary slightly across runs because model computation is intentionally left non-deterministic. This is by design — enforcing determinism can degrade model performance.

## Troubleshooting

### CUDA out of memory

- Reduce `BATCH_SIZE` in `config.py` (e.g., to 1 or 2).
- Reduce `MAX_SEQ_LENGTH` (e.g., to 256).
- For `gpt2-xl`, you may need gradient checkpointing or a GPU with >= 24 GB VRAM.
- Set `DEVICE = "cpu"` as a last resort.

### API gateway errors

Step 1 calls the LLM gateway for each sentence and model. If a model returns an HTTP error, it is retried up to 3 times with exponential back-off. Persistent failures for a model are logged and skipped. Ensure your `AI_GATEWAY_API_KEY` is valid and has quota remaining.

### Model download failure

HuggingFace Hub downloads may fail due to network issues. The models are cached after the first download. Set `HF_HOME` to control the cache directory.

### Short sentences filtered as expected

Section markers like `A`, `B`, `C` are single-token entries and are intentionally removed by the `MIN_ENGLISH_TOKENS` filter in Step 0.
