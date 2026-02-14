import os
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt


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

    # ===== Styling (paper-friendly) =====
    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

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
    fig, ax = plt.subplots(figsize=(4.2, 2.7), constrained_layout=False)

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
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.97)

    # Output
    out_dir = os.path.join("results", "pareto")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "pareto_tradeoff.pdf")
    out_svg = os.path.join(out_dir, "pareto_tradeoff.svg")

    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)

    print(f"[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_svg}")


if __name__ == "__main__":
    main()
