"""Generate paper/figures/fig_greedy_curves.pdf.

Cumulative damage vs k for {Greedy upper-bound proxy, AEGIS-static,
Degree, Random} across the three subgraph datasets used in
tab:greedy_topk (Cora, Citeseer, WikiCS).

AEGIS-iterative was removed: it is not the headline result (the
closed-form static ranker is), it is never referenced in the paper text,
and including a weaker variant of our own method confuses the
single-shot-SVD narrative.

Usage:
    .venv/bin/python scripts/figures/make_fig_greedy_curves.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import apply_paper_style
apply_paper_style()

PROJ = Path(__file__).resolve().parents[2]
SRC_CSV = PROJ / "results/greedy_topk_attack.csv"
OUT_PDF = PROJ / "paper/figures/fig_greedy_curves.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "WikiCS"]
METHODS_ORDER = ["Greedy", "AEGIS", "Degree", "Random"]
METHOD_STYLE = {
    "Greedy":           dict(color="#222222", marker="^", linestyle="-",
                             linewidth=0.9, markersize=3.5,
                             label="Greedy upper-bound proxy"),
    "AEGIS":            dict(color="#0072B2", marker="o", linestyle="-",
                             linewidth=0.9, markersize=3.5,
                             label="AEGIS-ranked (static)"),
    "Degree":           dict(color="#D55E00", marker="d", linestyle=":",
                             linewidth=0.9, markersize=3.0,
                             label="Degree-ranked"),
    "Random":           dict(color="#CC79A7", marker="x", linestyle="-.",
                             linewidth=0.9, markersize=3.5,
                             label="Random"),
}


def aggregate_static(df: pd.DataFrame, dataset: str, method: str):
    """Return arrays (k, mean_damage, std_damage)."""
    sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])
    grp = sub.groupby("k")["cumulative_damage"]
    ks = sorted(grp.groups.keys())
    means = np.array([grp.get_group(k).mean() for k in ks])
    stds = np.array([grp.get_group(k).std(ddof=1) for k in ks])
    return np.array(ks), means, stds


def main():
    df = pd.read_csv(SRC_CSV)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), sharey=False)

    for ax, dname in zip(axes, DATASETS):
        for method in METHODS_ORDER:
            ks, m, s = aggregate_static(df, dname, method)
            if len(ks) == 0:
                continue
            sty = METHOD_STYLE[method]
            ax.plot(ks, m, **sty)
            ax.fill_between(ks, m - s, m + s, color=sty["color"], alpha=0.10,
                            linewidth=0)
        ax.set_title(dname)
        ax.set_xlabel(r"Edges removed $k$")
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.grid(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel(r"Cumulative $\ell_2$ damage")

    # Stack legend in two rows: 4 method entries (Greedy, AEGIS, Degree,
    # Random) -> ncol=2 gives a 2x2 grid that halves the horizontal
    # footprint vs a single 4-column row.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
        handlelength=2.6,
        columnspacing=2.4,
        handletextpad=0.6,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.32)
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
