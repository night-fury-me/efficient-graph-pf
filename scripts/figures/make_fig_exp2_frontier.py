"""Generate paper/figures/fig_exp2_frontier.pdf.

Independent-attacker frontier: damage of a faithful GR-BCD attacker (that never
sees sigma_1) vs clean accuracy, comparing the sigma_1(S_c) penalty against a
generic spectral-norm cap on ||W||. At matched accuracy the sigma_1 penalty blunts
the independent attacker more (lower-left is better). Cora + Citeseer, 10 seeds.
Source: results/exp2/exp2_merged.csv.

Usage:
    .venv/bin/python scripts/figures/make_fig_exp2_frontier.py
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
SRC_CSV = PROJ / "results/exp2/exp2_merged.csv"
OUT_PDF = PROJ / "paper/figures/fig_exp2_frontier.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer"]
SERIES = {
    "sc_penalty":    dict(color="#0072B2", marker="o", linestyle="-",
                          label=r"$\sigma_1(S_c)$ penalty"),
    "lipschitz_cap": dict(color="#D55E00", marker="d", linestyle="--",
                          label=r"Spectral cap on $\|W\|$"),
}


def _curve(d, defense):
    """(acc, grbcd_damage) sweep points, mean over seeds, ordered by accuracy."""
    sub = d[d["defense"] == defense]
    key = "lambda" if defense == "sc_penalty" else "c"
    pts = []
    for v in sorted(sub[key].unique()):
        g = sub[sub[key] == v]
        pts.append((g["acc"].mean(), g["grbcd_dmg_sub"].mean()))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main():
    df = pd.read_csv(SRC_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=False)

    for ax, dname in zip(axes, DATASETS):
        d = df[df["dataset"] == dname]
        for defense, sty in SERIES.items():
            ax.plot(*_curve(d, defense), color=sty["color"], marker=sty["marker"],
                    linestyle=sty["linestyle"], linewidth=0.9, markersize=3.8,
                    label=sty["label"])
        b = d[d["defense"] == "baseline"]
        ax.plot(b["acc"].mean(), b["grbcd_dmg_sub"].mean(), marker="*",
                color="#222222", markersize=7, linestyle="none", label="Baseline")
        ax.set_title(dname)
        ax.set_xlabel("Clean accuracy")
        ax.grid(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("GR-BCD damage")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               handlelength=2.2, columnspacing=1.6, handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
