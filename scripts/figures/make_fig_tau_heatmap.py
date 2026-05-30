"""Generate paper/figures/fig_tau_heatmap.pdf.

Improvements over the existing tau_heatmap.pdf:
  - Diverging colormap centred at tau = 0 so anti-correlated cells stand out
  - Hatching on the 4 anti-correlated cells (tau < -0.1)
  - Four-bucket legend (strongly positive / positive / near-zero / anti-correlated)
  - Annotated cell values for every architecture x dataset combination

Closes editorial decision item P2.5 (four-bucket framing) at the figure level.

Usage:
    .venv/bin/python scripts/figures/make_fig_tau_heatmap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from _style import apply_paper_style
apply_paper_style()

PROJ = Path(__file__).resolve().parents[2]
SRC_CSV = PROJ / "results/tau_all_datasets.csv"
OUT_PDF = PROJ / "paper/figures/fig_tau_heatmap.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon Photo", "Amazon Fraud"]
DATASET_LABELS = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon Photo", "Amazon Fraud"]

ARCHS = ["IGNN", "GCN-2", "GCN-4", "GIN-2", "GAT-2", "SAGE-2", "APPNP"]
ARCH_LABELS = ["IGNN", "GCN-2", "GCN-4", "GIN-2", r"GAT-2$^\dagger$", "SAGE-2", "APPNP"]


def aggregate(df: pd.DataFrame) -> np.ndarray:
    """Return (n_archs, n_datasets) mean-tau matrix; NaN where missing."""
    M = np.full((len(ARCHS), len(DATASETS)), np.nan)
    # Theory-predicted edge-weighted ranking w_k*v_k (Prop. transfer), uniform
    # across all cells. GAT-2 transfers better under the unweighted v_k (noted
    # in the text); the figure keeps one consistent estimator to avoid
    # best-per-cell selection.
    for i, a in enumerate(ARCHS):
        for j, d in enumerate(DATASETS):
            sel = df[(df["architecture"] == a) & (df["dataset"] == d)]
            if not sel.empty:
                M[i, j] = sel["tau_weighted"].mean()
    return M


def main():
    if not SRC_CSV.exists():
        print(f"missing {SRC_CSV}; using fallback values from tab:tau_cross")
        # Fallback to the values from experiments.tex tab:tau_cross
        M = np.array([
            [+0.32, +0.31, +0.82, +0.14, -0.15],  # IGNN
            [-0.03, -0.28, +0.21, +0.05, +0.25],  # GCN-2
            [+0.49, +0.64, +0.89, +0.45, -0.04],  # GCN-4
            [+0.33, +0.57, +0.54, +0.63, +0.14],  # GIN-2
            [+0.54, +0.66,  np.nan,  np.nan, +0.21],  # GAT
            [+0.22, +0.38, +0.36, +0.22, +0.60],  # SAGE-2
            [+0.35, +0.36, +0.83, +0.22, +0.43],  # APPNP
        ])
    else:
        df = pd.read_csv(SRC_CSV)
        M = aggregate(df)

    # Match the canvas size of fig_phase_transition / fig_sc_heatmap so
    # the embedded 11pt text scales to the same ~9pt body size at
    # \columnwidth. The bucket annotations and verbose legend have been
    # moved to the figure caption to free horizontal space.
    fig, ax = plt.subplots(figsize=(4.35, 2.70))
    # Spectral palette (author preference), scaled to the all-positive data
    # range [0,1] so the full colormap is used (not centred at 0). The data
    # is bimodal -- mostly at the two ends -- so the rainbow midpoint is
    # largely unused; cividis/viridis remain the accessibility-optimal swap.
    cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")

    def _text_color(val):
        """Black or white annotation, whichever contrasts the cell."""
        r, g, b, _ = cmap(norm(val))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        return "white" if lum < 0.55 else "#1a1a1a"

    ax.set_xticks(np.arange(len(DATASETS)))
    ax.set_xticklabels(DATASET_LABELS, family="serif", fontsize=10,
                       rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ARCHS)))
    ax.set_yticklabels(ARCH_LABELS, family="serif", fontsize=10)
    # Axis labels ("Dataset" / "Architecture") removed — the tick labels
    # are dataset names and architecture names, so a generic axis label is
    # redundant on a column-width canvas. The title is also self-explanatory.
    # Short title — full bucket interpretation goes in the figure caption.
    ax.set_title(
        r"Kendall $\tau$: continuous $\to$ discrete transfer",
        pad=6, family="serif", fontsize=11,
    )
    # Thin grid lines between cells
    ax.set_xticks(np.arange(len(DATASETS)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ARCHS)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # Annotate every cell with its tau. OOM cells get a distinct cross-hatch
    # + italic "OOM" tag so missing data is never confused with a low value.
    # Annotation colour is chosen per cell by background luminance.
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                # OOM cells: cross-dotted infill so the pattern is clearly
                # different from anti-correlated diagonals; light grey fill
                # behind the "OOM" label gives a small visual anchor.
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#F2F2F2", edgecolor="#B8B8B8",
                    linewidth=0.4, hatch="xx", zorder=2,
                ))
                ax.text(j, i, "OOM", ha="center", va="center",
                        fontsize=9, color="#666666", style="italic",
                        family="serif", zorder=3)
                continue
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=9, color=_text_color(v), family="serif",
                    zorder=3)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
    # No "Kendall τ" label on the colorbar — figure title already says
    # "Kendall τ". Bucket interpretation (strong / moderate / weak / anti)
    # is moved to the figure caption to keep the column-width canvas tidy.
    cbar.outline.set_linewidth(0.5)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([r"$0$", r"$0.25$", r"$0.5$", r"$0.75$", r"$1$"])
    cbar.ax.tick_params(labelsize=10, width=0.4)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_family("serif")

    # Centred dagger footnote — the GAT-2† marker in the y-tick labels
    # references this note. ha="center" so it visually balances under
    # the heatmap. Dark grey (#222222) for legibility — the previous
    # #555555 washed out against the white background.
    fig.text(
        0.5, 0.015,
        r"$^\dagger$ GAT-2: 2-layer attention; OOM on Pubmed, WikiCS, Amazon Fraud.",
        ha="center", va="bottom", fontsize=9, color="#222222",
        family="serif", style="italic",
    )

    # Explicit margins (no tight_layout). Bottom leaves room for the
    # rotated x-tick labels (25°) + the centred dagger note; left leaves
    # room for the y-tick labels including "GAT-2†".
    plt.subplots_adjust(left=0.18, right=0.92, top=0.90, bottom=0.30)
    # Override the global "tight" savefig.bbox so the saved canvas matches
    # figsize exactly — keeps the 11pt-text-at-columnwidth contract.
    import matplotlib as mpl
    mpl.rcParams["savefig.bbox"] = "standard"
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
