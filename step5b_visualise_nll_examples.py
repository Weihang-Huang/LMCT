#!/usr/bin/env python3
"""Step 5b — Publication-quality colour-coded NLL visualisation.

Automatically selects 2 maximally contrastive sentence pairs from the
token-level NLL annotation data produced by step5, then renders each pair
as a two-row colour-coded figure (human translation / AI translation) for
inclusion in the manuscript.

Outputs
-------
outputs/figures/nll_example_pair_1.pdf
outputs/figures/nll_example_pair_1.png
outputs/figures/nll_example_pair_2.pdf
outputs/figures/nll_example_pair_2.png
outputs/selected_examples.csv   <- audit log of selection criteria

Selection logic (Option B — automated)
---------------------------------------
For every sentence_id in the test set that has BOTH a human and at least one
AI translation in token_nll_annotations.csv:

  contrast_score = mean_delta_nll(best_AI_row) − mean_delta_nll(human_row)

where mean_delta_nll is the mean of nll_diff = nll_human_model − nll_ai_model
across all tokens in that sentence/source combination.

NOTE on nll_diff semantics
--------------------------
nll_diff is always scored as:
    nll_human_model(token) − nll_pooled_ai_model(token)
for EVERY row regardless of text_source (human or AI translation).
text_source describes whose translation the text IS, not which model scored it.
A positive nll_diff means the human-tuned model found that token more
surprising — i.e. the token is AI-characteristic in style.

Colour scheme
-------------
  delta_nll > 0  ->  red   (AI-characteristic)
  delta_nll ~ 0  ->  white (neutral)
  delta_nll < 0  ->  blue  (human-characteristic)

Saturation scales linearly with |delta_nll|, clamped at +-NLL_CLAMP.

SEED POLICY: RANDOM_SEED (1024) is used only for tie-breaking in selection.
No model inference is performed in this script.
"""

import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import config
from utils import set_seed, ensure_dir


# ── Tunable constants ──────────────────────────────────────────────────────────
MIN_TOKENS: int = 8        # discard sentences shorter than this
MAX_TOKENS: int = 35       # discard sentences longer than this (readability)
NLL_CLAMP: float = 3.0     # delta_nll magnitude beyond which colour saturates
N_PAIRS: int = 2           # number of example pairs to generate

# Figure layout (all in inches unless noted)
FONT_SIZE: float = 9.0
TOKEN_H: float = 0.36      # rendered height of one token row (inches)
LINE_SPACING: float = 1.4  # multiplier on TOKEN_H for inter-line gap
ROW_GAP: float = 0.50      # vertical gap between human and AI rows (inches)
TOP_PAD: float = 0.55      # padding above first row (for source label)
BOT_PAD: float = 0.65      # padding below last row (for colorbar)
FIG_WIDTH: float = 7.0     # total figure width (inches)
DPI: int = 300

# Approximate monospace character width as fraction of axes width.
# At FONT_SIZE=9 in a 7-inch figure this is empirically ~0.013.
# Increase if tokens overflow right edge; decrease if figure is too sparse.
CHAR_WIDTH_FRAC: float = 0.013

# Wrap tokens at this many characters per display line.
# Keep consistent with CHAR_WIDTH_FRAC: max_chars ~ 1 / CHAR_WIDTH_FRAC
MAX_CHARS_PER_LINE: int = 72

# Known candidate column names for the Chinese source text in ai_translations.csv.
# The first match found will be used; if none match, the label is omitted.
ZH_SOURCE_COLUMN_CANDIDATES: list[str] = [
    "chinese_source", "source_zh", "zh_text", "zh", "source",
    "chinese_zh", "source_text_zh",
]


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _delta_to_rgb(delta: float) -> tuple[float, float, float]:
    """Map nll_diff to RGB using a red-white-blue diverging scheme.

    Positive delta (AI-characteristic)    -> red
    Zero                                  -> white
    Negative delta (human-characteristic) -> blue
    """
    t = max(-NLL_CLAMP, min(NLL_CLAMP, delta)) / NLL_CLAMP  # in [-1, 1]
    if t >= 0:
        r, g, b = 1.0, 1.0 - 0.78 * t, 1.0 - 0.78 * t
    else:
        t = -t
        r, g, b = 1.0 - 0.78 * t, 1.0 - 0.78 * t, 1.0
    return r, g, b


# ── Selection logic ────────────────────────────────────────────────────────────

def select_pairs(token_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame describing the top N_PAIRS sentence pairs,
    ranked by contrast_score = mean_delta_nll(best_AI) - mean_delta_nll(human).

    Applies MIN_TOKENS / MAX_TOKENS filter for readability.
    Uses RANDOM_SEED only for tie-breaking via pre-shuffle.
    """
    set_seed(config.RANDOM_SEED)

    # Sentence-level mean delta_nll and token count per (sentence_id, text_source)
    sent_means = (
        token_df
        .groupby(["sentence_id", "text_source"])["nll_diff"]
        .agg(mean_delta="mean", n_tokens="count")
        .reset_index()
    )

    # Apply readability filter on token count
    sent_means = sent_means[
        sent_means["n_tokens"].between(MIN_TOKENS, MAX_TOKENS)
    ]

    human_rows = sent_means[sent_means["text_source"] == "human"].copy()
    ai_rows    = sent_means[sent_means["text_source"] != "human"].copy()

    # For each sentence_id pick the AI system with the highest mean_delta
    best_ai = (
        ai_rows
        .sort_values("mean_delta", ascending=False)
        .drop_duplicates(subset=["sentence_id"], keep="first")
        .rename(columns={
            "mean_delta":  "ai_mean_delta",
            "text_source": "ai_system",
            "n_tokens":    "ai_n_tokens",
        })
    )

    merged = human_rows.merge(
        best_ai[["sentence_id", "ai_mean_delta", "ai_system", "ai_n_tokens"]],
        on="sentence_id",
        how="inner",
    ).rename(columns={
        "mean_delta": "human_mean_delta",
        "n_tokens":   "human_n_tokens",
    })

    merged["contrast_score"] = (
        merged["ai_mean_delta"] - merged["human_mean_delta"]
    )

    # Shuffle first to break ties randomly, then sort descending
    merged = merged.sample(frac=1, random_state=config.RANDOM_SEED)
    merged = merged.sort_values("contrast_score", ascending=False)

    return merged.head(N_PAIRS).reset_index(drop=True)


# ── Token wrapping ─────────────────────────────────────────────────────────────

def _wrap_tokens(
    tokens: list[str],
    nll_diffs: list[float],
) -> list[tuple[list[str], list[float]]]:
    """Group tokens into display lines respecting MAX_CHARS_PER_LINE.

    Tokens are decoded by tokenizer.decode() which produces a leading space
    for word-initial tokens — this is used directly for display.

    Returns a list of (token_list, nll_diff_list) tuples, one per line.
    """
    lines: list[tuple[list[str], list[float]]] = []
    cur_tokens: list[str] = []
    cur_diffs:  list[float] = []
    cur_len = 0

    for tok, diff in zip(tokens, nll_diffs):
        # tokenizer.decode() already produces human-readable strings with
        # leading spaces where appropriate; no Ġ/Ċ substitution needed.
        display = tok
        tok_len = max(1, len(display))

        if cur_tokens and cur_len + tok_len > MAX_CHARS_PER_LINE:
            lines.append((cur_tokens, cur_diffs))
            cur_tokens, cur_diffs, cur_len = [], [], 0

        cur_tokens.append(display)
        cur_diffs.append(diff)
        cur_len += tok_len

    if cur_tokens:
        lines.append((cur_tokens, cur_diffs))

    return lines


# ── Token row rendering ────────────────────────────────────────────────────────

def _draw_token_row(
    ax: plt.Axes,
    tokens: list[str],
    nll_diffs: list[float],
    y_start: float,
    label: str,
    fig_height: float,
) -> float:
    """Draw colour-coded token boxes onto *ax*.

    Parameters
    ----------
    ax         : target axes (xlim/ylim both [0, 1], axis off)
    tokens     : list of decoded token strings
    nll_diffs  : list of nll_diff values, one per token
    y_start    : top of the drawing area in axes-fraction coordinates (0-1)
    label      : row label shown in italic above the tokens
    fig_height : total figure height in inches (used for consistent geometry)

    Returns
    -------
    float : y-coordinate (axes fraction) immediately below the last drawn line
    """
    # Row label
    ax.text(
        0.0, y_start + 0.02,
        label,
        transform=ax.transAxes,
        fontsize=FONT_SIZE - 1,
        fontstyle="italic",
        color="#444444",
        va="bottom",
    )
    y_cursor = y_start - 0.03  # small gap below label

    # line_height as a fraction of axes height, derived from known fig geometry.
    # ax occupies almost the full figure height, so axes_height ≈ fig_height.
    line_height_frac = TOKEN_H / fig_height
    line_step_frac   = line_height_frac * LINE_SPACING

    lines = _wrap_tokens(tokens, nll_diffs)

    for line_tokens, line_diffs in lines:
        x_cursor = 0.0
        for tok_display, diff in zip(line_tokens, line_diffs):
            r, g, b = _delta_to_rgb(diff)
            n_chars = max(1, len(tok_display))
            w = n_chars * CHAR_WIDTH_FRAC

            # Coloured background rectangle
            rect = mpatches.FancyBboxPatch(
                (x_cursor, y_cursor - line_height_frac),
                w,
                line_height_frac,
                boxstyle="square,pad=0",
                facecolor=(r, g, b),
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(rect)

            # Token text centred in its box
            ax.text(
                x_cursor + w / 2,
                y_cursor - line_height_frac / 2,
                tok_display,
                transform=ax.transAxes,
                fontsize=FONT_SIZE,
                fontfamily="monospace",
                ha="center",
                va="center",
                color="black",
                clip_on=False,
            )
            x_cursor += w

        y_cursor -= line_step_frac

    return y_cursor  # bottom of last line


# ── Colorbar ───────────────────────────────────────────────────────────────────

def _add_colorbar(fig: plt.Figure, ax: plt.Axes, cb_bottom_frac: float) -> None:
    """Add a horizontal diverging colorbar at *cb_bottom_frac* (figure fraction).

    The colorbar spans the middle 70% of the axes width to avoid crowding.
    bbox_inches='tight' at save time will include it even if it extends below
    the axes bounding box.
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "nll_diverge",
        [
            (0.22, 0.22, 1.00),   # blue  — strongly human-characteristic
            (1.00, 1.00, 1.00),   # white — neutral
            (1.00, 0.22, 0.22),   # red   — strongly AI-characteristic
        ],
        N=256,
    )
    norm = Normalize(vmin=-NLL_CLAMP, vmax=NLL_CLAMP)

    ax_pos = ax.get_position()
    cb_ax = fig.add_axes([
        ax_pos.x0 + ax_pos.width * 0.15,   # left
        cb_bottom_frac,                      # bottom (figure fraction)
        ax_pos.width * 0.70,                 # width
        0.012,                               # height
    ])

    cb = ColorbarBase(cb_ax, cmap=cmap, norm=norm, orientation="horizontal")
    cb.set_label(
        r"$\Delta_{\mathrm{NLL}}$ = NLL$_{\mathrm{human\,model}}$"
        r" $-$ NLL$_{\mathrm{AI\,model}}$"
        r"    (red = AI-characteristic,  blue = human-characteristic)",
        fontsize=FONT_SIZE - 1.5,
    )
    cb.set_ticks([-NLL_CLAMP, -1.5, -0.5, 0.0, 0.5, 1.5, NLL_CLAMP])
    cb.ax.tick_params(labelsize=FONT_SIZE - 2)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[Step 5b] Loading token annotations …")

    token_path = os.path.join(config.OUTPUT_DIR, "token_nll_annotations.csv")
    ai_path    = os.path.join(config.OUTPUT_DIR, "ai_translations.csv")

    if not os.path.isfile(token_path):
        raise FileNotFoundError(
            f"Missing: {token_path}\n"
            "Run step5_token_level_nll_GPU.py first."
        )

    token_df = pd.read_csv(token_path, encoding="utf-8")

    # ── Resolve Chinese source column (graceful fallback) ─────────────────────
    source_map: dict = {}
    if os.path.isfile(ai_path):
        ai_df = pd.read_csv(ai_path, encoding="utf-8")
        zh_col = next(
            (c for c in ZH_SOURCE_COLUMN_CANDIDATES if c in ai_df.columns),
            None,
        )
        if zh_col:
            print(f"  Found Chinese source column: '{zh_col}'")
            src = ai_df.drop_duplicates("sentence_id")[["sentence_id", zh_col]]
            source_map = dict(zip(src["sentence_id"], src[zh_col]))
        else:
            print(
                "  WARNING: No Chinese source column found in ai_translations.csv.\n"
                f"  Searched for: {ZH_SOURCE_COLUMN_CANDIDATES}\n"
                "  Source labels will be omitted from figures."
            )

    # ── Select pairs ──────────────────────────────────────────────────────────
    print("[Step 5b] Selecting example pairs …")
    pairs_df = select_pairs(token_df)

    out_dir  = ensure_dir(os.path.join(config.OUTPUT_DIR, "figures"))
    log_path = os.path.join(config.OUTPUT_DIR, "selected_examples.csv")
    pairs_df.to_csv(log_path, index=False)
    print(f"  Selection log -> {log_path}")
    print(
        pairs_df[[
            "sentence_id", "ai_system",
            "human_mean_delta", "ai_mean_delta", "contrast_score",
        ]].to_string(index=False)
    )

    # ── Render each pair ───────────────────────────────────────────────────────
    for pair_idx, pair_row in pairs_df.iterrows():
        sid     = pair_row["sentence_id"]
        ai_sys  = pair_row["ai_system"]
        fig_num = pair_idx + 1

        print(f"\n[Step 5b] Rendering pair {fig_num}: "
              f"sentence_id={sid}, AI system={ai_sys} …")

        human_toks = (
            token_df[
                (token_df["sentence_id"] == sid) &
                (token_df["text_source"] == "human")
            ]
            .sort_values("token_position")
        )
        ai_toks = (
            token_df[
                (token_df["sentence_id"] == sid) &
                (token_df["text_source"] == ai_sys)
            ]
            .sort_values("token_position")
        )

        if human_toks.empty or ai_toks.empty:
            print(f"  WARNING: missing token data for pair {fig_num}, skipping.")
            continue

        h_tokens = human_toks["token"].tolist()
        h_diffs  = human_toks["nll_diff"].tolist()
        a_tokens = ai_toks["token"].tolist()
        a_diffs  = ai_toks["nll_diff"].tolist()

        # ── Compute figure height from content ────────────────────────────────
        h_lines = len(_wrap_tokens(h_tokens, h_diffs))
        a_lines = len(_wrap_tokens(a_tokens, a_diffs))
        total_line_inches = (h_lines + a_lines) * TOKEN_H * LINE_SPACING

        # Extra height for source label if present
        zh_src   = source_map.get(sid, "")
        src_pad  = 0.25 if zh_src else 0.0

        fig_height = (
            TOP_PAD
            + src_pad
            + total_line_inches
            + ROW_GAP
            + BOT_PAD
        )
        fig_height = max(fig_height, 2.5)  # enforce minimum readable height

        # ── Build figure ──────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, fig_height))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        y_cursor = 0.97  # start near top of axes (axes fraction)

        # Optional Chinese source label
        if zh_src:
            short_src = textwrap.shorten(zh_src, width=90, placeholder="…")
            ax.text(
                0.0, y_cursor,
                f"Source (ZH): {short_src}",
                transform=ax.transAxes,
                fontsize=FONT_SIZE - 1.5,
                color="#888888",
                fontstyle="italic",
                va="top",
            )
            y_cursor -= (src_pad / fig_height) + 0.01

        # Human row
        y_cursor = _draw_token_row(
            ax, h_tokens, h_diffs,
            y_start=y_cursor,
            label="Human translation:",
            fig_height=fig_height,
        )

        # Gap between human and AI rows
        y_cursor -= ROW_GAP / fig_height

        # AI row
        _draw_token_row(
            ax, a_tokens, a_diffs,
            y_start=y_cursor,
            label=f"AI translation ({ai_sys}):",
            fig_height=fig_height,
        )

        # Colorbar — placed at a fixed fraction from the figure bottom
        cb_bottom = 0.02
        _add_colorbar(fig, ax, cb_bottom)

        # ── Save ──────────────────────────────────────────────────────────────
        base = os.path.join(out_dir, f"nll_example_pair_{fig_num}")
        fig.savefig(base + ".pdf", bbox_inches="tight", dpi=DPI)
        fig.savefig(base + ".png", bbox_inches="tight", dpi=DPI)
        plt.close(fig)
        print(f"  Saved -> {base}.pdf / .png")

    print("\n[Step 5b] Done.")
    print(
        "\nNext steps:\n"
        "  1. Inspect outputs/figures/nll_example_pair_*.png visually.\n"
        "  2. Check outputs/selected_examples.csv to confirm sentence choices.\n"
        "  3. If token boxes overflow or spacing looks off, adjust\n"
        "     CHAR_WIDTH_FRAC and LINE_SPACING at the top of this file.\n"
        "  4. Once happy, copy PDFs to LMCT_main/figures/ and update main.tex."
    )


if __name__ == "__main__":
    main()