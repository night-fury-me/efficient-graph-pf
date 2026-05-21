import os
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path so `train.viz` resolves when run as
# `python scripts/plot_fewshot_band.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.viz import apply_publication_style, apply_paper_margins, new_paper_figure, save_figure

apply_publication_style()

color_map = {
    "Full-FT": "tab:red",
    "LoRA (r=2, α=8)": "tab:green",
}
marker_map = {
    "Full-FT": "s",
    "LoRA (r=2, α=8)": "o",
}

def main():
    # ===== Hard-coded data (from results/fewshot/fewshot_summary.csv) =====
    data = {
        "Full-FT": {
            "budget": [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75],
            "mean": [
                0.00914449067889441,
                0.005200689519449933,
                0.004896397558573979,
                0.0037373582215187314,
                0.002279166549206287,
                0.002172608043351724,
                0.001728140497172399,
            ],
            "std": [
                0.00011428661666533153,
                0.0011597384305534661,
                0.0005770510425047578,
                0.00013101828267471884,
                0.00017374449569174878,
                0.0002803348531778711,
                8.451449882265652e-05,
            ],
        },
        "LoRA (r=2, α=8)": {
            "budget": [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75],
            "mean": [
                0.009115307698918117,
                0.00900616902326446,
                0.008557405557625561,
                0.00754711134694318,
                0.005356145992113131,
                0.004475824915180352,
                0.004158512859772734,
            ],
            "std": [
                0.00011387733028816054,
                0.002663052980451266,
                0.00042352378872684407,
                0.0005782867662538587,
                0.0003166024942925993,
                0.0005050660552740931,
                0.00025986534974955746,
            ],
        },
    }

    # Prepare data
    methods = ["Full-FT", "LoRA (r=2, α=8)"]
    fig, ax = new_paper_figure()
    for method in methods:
        budgets = np.array(data[method]["budget"], dtype=float)
        means = np.array(data[method]["mean"], dtype=float)
        stds = np.array(data[method]["std"], dtype=float)

        # Plot line
        ax.plot(
            budgets,
            means,
            color=color_map[method],
            marker=marker_map[method],
            markeredgecolor="black",
            markeredgewidth=0.8,
            linewidth=1.2,
            markersize=6,
            label=method,
            zorder=3,
        )
        # Plot error band
        ax.fill_between(
            budgets,
            means - stds,
            means + stds,
            color=color_map[method],
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )

    # Axes labels
    ax.set_xlabel("Budget fraction ($\\beta$, %)")
    ax.set_ylabel("Test RMSE$_V$ (log scale)")

    # Grid and limits
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)
    # Symlog keeps log spacing while allowing budget=0 (zeroshot)
    ax.set_xscale("symlog", linthresh=0.01, linscale=1.0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75])
    ax.set_xticklabels(["0", "1", "5", "10", "30", "50", "75"])
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2e-2)

    # Legend
    ax.legend(
        frameon=True,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        borderaxespad=0.0,
    )

    # Prevent axis-label cropping in vector outputs
    apply_paper_margins(fig)

    save_figure(fig, os.path.join("results", "fewshot", "fewshot_band"), formats=("pdf", "svg"))

if __name__ == "__main__":
    main()
