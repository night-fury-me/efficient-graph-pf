import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Ensure repo root is on sys.path so `train.viz` resolves when run as
# `python scripts/plot_pareto_tradeoff.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.viz import apply_publication_style, apply_paper_margins, new_paper_figure, save_figure


@dataclass(frozen=True)
class MethodPoint:
    name: str
    trainable_pct: float
    rmse_v: float


def compute_pareto_frontier(points: List[MethodPoint]) -> List[MethodPoint]:
    """Return non-dominated points (lower trainable_pct, lower rmse_v)."""
    points_sorted = sorted(points, key=lambda p: p.trainable_pct)
    frontier: List[MethodPoint] = []
    best_rmse = float("inf")
    for p in points_sorted:
        if p.rmse_v < best_rmse - 1e-12:
            frontier.append(p)
            best_rmse = p.rmse_v
    return frontier


def main() -> None:
    # ===== Data from docs/efficiency-accuracy-trade-off.md =====
    # trainable_pct = 100 - P_reduced
    points = [
        MethodPoint("Full FT", 100.00, 9.35e-4),
        MethodPoint("Head Only", 10.00, 1.58e-3),
        MethodPoint("LoRA Only", 3.44, 3.61e-3),
        MethodPoint("LoRA + PHead", 14.54, 1.20e-3),
    ]

    apply_publication_style()

    color_map = {
        "Full FT": "tab:blue",
        "Head Only": "tab:orange",
        "LoRA Only": "tab:green",
        "LoRA + PHead": "tab:red",
    }
    marker_map = {
        "Full FT": "s",
        "Head Only": "^",
        "LoRA Only": "o",
        "LoRA + PHead": "D",
    }

    # Use explicit margins (more reliable than constrained_layout for PDF/SVG crops)
    fig, ax = new_paper_figure()

    # Plot points (no point annotations; legend is sufficient)
    for p in points:
        size = 70 if p.name != "LoRA + Head" else 90
        edge = "black" if p.name == "LoRA + Head" else "black"
        ax.scatter(
            p.trainable_pct,
            p.rmse_v,
            s=size,
            marker=marker_map[p.name],
            color=color_map[p.name],
            edgecolor=edge,
            linewidth=0.8,
            label=p.name,
            zorder=3,
        )

    # Pareto frontier
    frontier = compute_pareto_frontier(points)
    fx = [p.trainable_pct for p in frontier]
    fy = [p.rmse_v for p in frontier]
    ax.plot(
        fx,
        fy,
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="Pareto frontier",  # Feedback (B)
        zorder=2,
    )

    # Axes labels
    ax.set_xlabel("Trainable parameters used (\u03c1, %, log scale)")
    ax.set_ylabel("Target RMSE$_V$ (\u2193)")

    # Grid and limits
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(3, 120)
    ax.set_xticks([3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["3", "5", "10", "20", "50", "100"])
    # Feedback (A): invert x-axis so "better" is bottom-left
    ax.invert_xaxis()

    # Legend (move left to avoid overlap with Pareto line)
    ax.legend(
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        borderaxespad=0.0,
    )

    # Prevent axis-label cropping in vector outputs
    apply_paper_margins(fig)

    save_figure(fig, os.path.join("results", "pareto", "pareto_tradeoff"), formats=("pdf", "svg"))


if __name__ == "__main__":
    main()
