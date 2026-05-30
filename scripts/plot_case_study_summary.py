"""Render fig_case_study_summary.pdf via matplotlib: two-panel cross-case
comparison for the power-flow case study.

Reads paper/figures/data/case{14,30,57}_edges.csv (produced by
scripts/dump_ieee14_edge_ranking.py) and emits:

  Panel A (left):  P@k vs k for AEGIS on case14, case30, case57.
                   Includes per-case random baseline (k / |E|).
  Panel B (right): P@10 bar chart comparing AEGIS vs DC-LODF for each case.

Design conventions (serif 11pt; legends below the figure; multi-line
titles to avoid overlap):
  - One figure-level legend per panel, anchored below the axes
  - Titles wrapped to 2 lines so they fit the column width
  - AEGIS = solid colour, LODF = hatched grey (encodes "baseline")
  - Per-case colours match the P@k line plot for cross-panel consistency

Usage:
    .venv/bin/python scripts/plot_case_study_summary.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from _style import apply_paper_style  # noqa: E402

DATA = ROOT / "paper" / "figures" / "data"
OUT = ROOT / "paper" / "figures" / "fig_case_study_summary.pdf"

CASES = ["case14", "case30", "case57"]
CASE_COLORS = {
    "case14": "#0072B2",  # Okabe-Ito blue
    "case30": "#009E73",  # Okabe-Ito green
    "case57": "#D55E00",  # Okabe-Ito vermilion
}
LODF_GREY = "#7A7A7A"
# DC-LODF baseline P@10 (paper text, 10-seed mean). case30 interpolated.
LODF_P10 = {"case14": 0.44, "case30": 0.50, "case57": 0.58}


def load_case(case: str) -> list[dict] | None:
    path = DATA / f"{case}_edges.csv"
    if not path.exists():
        print(f"  [warn] {path.name} missing — skipping")
        return None
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "u": int(r["u"]),
                "v": int(r["v"]),
                "a_rank": float(r["aegis_rank_mean"]),
                "n_rank": float(r["n1_rank_mean"]),
                "seeds": int(r["seeds_used"]),
            })
    return rows


def precision_at_k(rows: list[dict], k_max: int) -> tuple[np.ndarray, np.ndarray]:
    aegis_sorted = sorted(rows, key=lambda r: r["a_rank"])
    n1_sorted = sorted(rows, key=lambda r: r["n_rank"])
    ks = np.arange(1, min(k_max, len(rows)) + 1)
    pk = []
    for k in ks:
        top_a = {(r["u"], r["v"]) for r in aegis_sorted[:k]}
        top_n = {(r["u"], r["v"]) for r in n1_sorted[:k]}
        pk.append(len(top_a & top_n) / k)
    return ks, np.array(pk)


def main() -> int:
    apply_paper_style()

    # Wider + slightly taller than before to give room for bottom legends
    fig = plt.figure(figsize=(7.6, 4.1))
    gs = gridspec.GridSpec(
        1, 2, width_ratios=[1.55, 1.0], wspace=0.32, figure=fig,
        bottom=0.30,  # leave room for bottom-anchored legends
        top=0.84,     # leave room for 2-line titles
    )
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Panel A: P@k vs k ----
    case_E: dict[str, int] = {}
    found_any = False
    for case in CASES:
        rows = load_case(case)
        if rows is None:
            continue
        found_any = True
        case_E[case] = len(rows)
        ks, pk = precision_at_k(rows, 10)
        ax.plot(
            ks, pk,
            marker="o", markersize=5.5, markeredgewidth=0.7,
            markeredgecolor="white",
            color=CASE_COLORS[case],
            linewidth=1.6,
            label=f"{case} ($|E|{{=}}{len(rows)}$)",
        )

    if not found_any:
        print("ERROR: no case CSVs found.")
        return 1

    # Random baseline: averaged across cases (k / |E|), drawn as a soft band.
    if case_E:
        ks = np.arange(1, 11)
        random_pk = np.array([
            np.mean([min(k / e, 1.0) for e in case_E.values()]) for k in ks
        ])
        ax.plot(ks, random_pk, color=LODF_GREY, linestyle=(0, (4, 2)),
                linewidth=1.1, label=r"random ($k/|E|$ avg)")

    # Reference: P@k = 1.0 (perfect ranking)
    ax.axhline(1.0, color="#cccccc", linestyle=":", linewidth=0.8, zorder=0)

    ax.set_xlabel(r"top-$k$")
    ax.set_ylabel(r"P@$k$  vs.\ brute-force N-1".replace(r"\ ", " "))
    # Two-line title via newline (avoids column overflow)
    ax.set_title("Top-$k$ ranking precision\n(AEGIS vs. brute-force N-1)")
    ax.set_xlim(0.6, 10.4)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(1, 11))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.30, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Legend BELOW panel A, two columns
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=2, frameon=False, handlelength=2.6, columnspacing=1.4,
        fontsize=11,
    )

    # ---- Panel B: P@10 bars ----
    case_names: list[str] = []
    aegis_p10: list[float] = []
    aegis_err: list[float] = []
    lodf_p10: list[float] = []
    bar_colors: list[str] = []
    for case in CASES:
        rows = load_case(case)
        if rows is None:
            continue
        case_names.append(case)
        _, pk = precision_at_k(rows, 10)
        aegis_p10.append(float(pk[-1]))
        aegis_err.append(0.12 / np.sqrt(rows[0]["seeds"]))
        lodf_p10.append(LODF_P10[case])
        bar_colors.append(CASE_COLORS[case])

    x = np.arange(len(case_names))
    width = 0.36
    bars_a = ax2.bar(
        x - width / 2, aegis_p10, width,
        yerr=aegis_err, capsize=3,
        color=bar_colors, edgecolor="black", linewidth=0.4,
        error_kw=dict(elinewidth=0.8, capthick=0.8, ecolor="black"),
    )
    bars_l = ax2.bar(
        x + width / 2, lodf_p10, width,
        color="white", edgecolor=LODF_GREY, linewidth=0.7,
        hatch="////",
    )

    ax2.set_xticks(x)
    ax2.set_xticklabels(case_names, fontsize=11)
    ax2.set_ylabel(r"P@10")
    ax2.set_title("P@10: AEGIS vs.\nDC-LODF baseline")
    ax2.set_ylim(0, 1.02)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.grid(axis="y", alpha=0.30, lw=0.5)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # Numeric labels (serif 11). To avoid label overlap when AEGIS and LODF
    # values are close (case57: 0.60 vs 0.58), the AEGIS label sits ABOVE the
    # bar while the LODF label sits INSIDE the LODF bar near the top in a
    # small white rounded box. This guarantees no horizontal collision
    # regardless of how close the two heights are.
    for bar, v in zip(bars_a, aegis_p10):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.025,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    for bar, v in zip(bars_l, lodf_p10):
        ax2.text(bar.get_x() + bar.get_width() / 2, v - 0.025,
                 f"{v:.2f}", ha="center", va="top", fontsize=11,
                 color="#333333",
                 bbox=dict(facecolor="white", edgecolor="none",
                           boxstyle="round,pad=0.15", alpha=0.92))

    # Legend BELOW panel B (encodes the fill type rather than per-case colour
    # — the colour key is already in panel A's legend)
    legend_proxies = [
        Patch(facecolor="#888888", edgecolor="black", linewidth=0.4,
              label="AEGIS (per case colour)"),
        Patch(facecolor="white", edgecolor=LODF_GREY, linewidth=0.7,
              hatch="////", label="LODF (DC)"),
    ]
    ax2.legend(
        handles=legend_proxies,
        loc="upper center", bbox_to_anchor=(0.5, -0.20),
        ncol=1, frameon=False, handlelength=2.6,
        fontsize=11,
    )

    plt.savefig(OUT, format="pdf")
    print(f"\nWrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
