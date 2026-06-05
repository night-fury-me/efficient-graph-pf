"""Generate paper/figures/fig_exp3_budget_sweep.pdf.

Damage-equivalence across the budget sweep: absolute equilibrium-shift damage
||Delta Z*|| of AEGIS's single closed-form query vs a faithful 125-epoch GR-BCD /
PR-BCD attacker (Geisler 2021), per dataset, vs deletion budget k. AEGIS is an
explicit line; its near-coincidence with the two iterative attackers is the
damage-equivalence. Mean over 10 seeds. Source: results/exp3/exp3_merged.csv.

Usage:
    .venv/bin/python scripts/figures/make_fig_exp3_budget_sweep.py
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
SRC_CSV = PROJ / "results/exp3/exp3_merged.csv"
OUT_PDF = PROJ / "paper/figures/fig_exp3_budget_sweep.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon", "AmazonFraud"]
TITLES = {"AmazonFraud": "Amazon Fraud"}
KS = [1, 2, 5, 10, 20, 50]
METHODS = {
    "damage_aegis_w": dict(color="#0072B2", marker="o", linestyle="-",
                           label="AEGIS (one query)"),
    "damage_grbcd":   dict(color="#D55E00", marker="^", linestyle="--",
                           label="GR-BCD (125 ep)"),
    "damage_prbcd":   dict(color="#222222", marker="s", linestyle=":",
                           label="PR-BCD (125 ep)"),
}


def main():
    df = pd.read_csv(SRC_CSV)
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.7), sharex=True)

    for ax, dname in zip(axes.ravel(), DATASETS):
        d = df[df["dataset"] == dname]
        for col, sty in METHODS.items():
            y = np.array([d[d["k"] == k][col].mean() for k in KS])
            ax.plot(KS, y, color=sty["color"], marker=sty["marker"],
                    linestyle=sty["linestyle"], linewidth=0.9, markersize=3.5,
                    label=sty["label"])
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels(KS)
        ax.minorticks_off()
        ax.set_title(TITLES.get(dname, dname))
        ax.grid(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in axes[1, :]:
        ax.set_xlabel(r"Deletion budget $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Damage $\|\Delta Z^*\|$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               handlelength=2.4, columnspacing=1.8, handletextpad=0.5)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
