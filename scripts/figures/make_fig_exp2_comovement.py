"""Generate paper/figures/fig_exp2_comovement.pdf.

One knob, coherent movement: over the sigma_1(S_c) penalty sweep, AEGIS attack
damage, sigma_1(S_c) itself, and an INDEPENDENT GR-BCD attacker's damage all fall
together (each shown as a fraction of its lambda=0 baseline). The independent
GR-BCD curve is the non-definitional part: penalizing sigma_1 also blunts an
attacker that never sees it. Cora + Citeseer, 10 seeds. Source: results/exp2/exp2_merged.csv.

Usage:
    .venv/bin/python scripts/figures/make_fig_exp2_comovement.py
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
OUT_PDF = PROJ / "paper/figures/fig_exp2_comovement.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer"]
LAMBDAS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
METRICS = {
    "attack_dmg":    dict(color="#0072B2", marker="o", linestyle="-",
                          label="AEGIS attack damage"),
    "sigma1":        dict(color="#D55E00", marker="s", linestyle="--",
                          label=r"$\sigma_1(S_c)$"),
    "grbcd_dmg_sub": dict(color="#222222", marker="^", linestyle=":",
                          label="Independent GR-BCD damage"),
}


def main():
    df = pd.read_csv(SRC_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)

    for ax, dname in zip(axes, DATASETS):
        d = df[df["dataset"] == dname]
        for col, sty in METRICS.items():
            base = d[d["defense"] == "baseline"][col].mean()
            sweep = d[d["defense"] == "sc_penalty"]
            y = np.array([sweep[np.isclose(sweep["lambda"], lam)][col].mean() / base
                          for lam in LAMBDAS])
            ax.plot(LAMBDAS, y, color=sty["color"], marker=sty["marker"],
                    linestyle=sty["linestyle"], linewidth=0.9, markersize=3.5,
                    label=sty["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dname)
        ax.set_xlabel(r"$\sigma_1$ penalty $\lambda$")
        ax.grid(True, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Fraction of baseline")

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
