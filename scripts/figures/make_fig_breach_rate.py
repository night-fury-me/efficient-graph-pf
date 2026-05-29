"""Generate paper/figures/fig_breach_rate.pdf.

Visual companion to tab:breach (experiments.tex). Plots the breach rate
(% of nodes whose prediction flips under the S_c-optimal constrained
perturbation) as a function of the perturbation budget epsilon, per
dataset, with 95% confidence-interval bands (Student t, 9 df) across
10 seeds.

Design conventions (shared with fig_tightness_eps):
  - Okabe-Ito colourblind-safe palette
  - Serif 11pt typography (apply_paper_style)
  - Citation graphs (Cora, Citeseer, Pubmed): solid lines, cool palette
  - Product graphs (WikiCS): dashed line, warm palette
  - Right-end value labels at epsilon = 0.20 with leader lines so the
    reader can match each line to its terminal breach rate without
    eye-tracing back to the legend
  - Soft horizontal band at breach rate >= 10% to flag the
    "high-failure" zone (operationally where the linearization stops
    being trustworthy as a safety guarantee)

Story the figure tells:
  - Pubmed is the dangerous outlier: 27.4% mean breach at eps=0.20,
    95% CI [12.3%, 42.5%], median 7.8% at eps=0.10 (vs. 10.3% mean). The
    wide CI band and the median marker communicate the right-skew.
  - Citeseer and WikiCS stay below 2% breach even at eps=0.20 --
    AEGIS-identified radii hold up under stress in these regimes.
  - Cora is intermediate: breach rises to 7.6% at eps=0.20.

Usage:
    .venv/bin/python scripts/figures/make_fig_breach_rate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats as sps

from _style import apply_paper_style
apply_paper_style()

PROJ = Path(__file__).resolve().parents[2]
SRC_CSV = PROJ / "results/exp_breach_rates.csv"
OUT_PDF = PROJ / "paper/figures/fig_breach_rate.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["Cora", "Citeseer", "Pubmed", "WikiCS"]

# Shared with fig_tightness_eps -- visual consistency across the paper.
STYLE = {
    "Cora":     dict(color="#0072B2", marker="o", linestyle="-",
                     label="Cora",     group="citation"),
    "Citeseer": dict(color="#D55E00", marker="s", linestyle="-",
                     label="Citeseer", group="citation"),
    "Pubmed":   dict(color="#009E73", marker="^", linestyle="-",
                     label="Pubmed",   group="citation"),
    "WikiCS":   dict(color="#CC79A7", marker="D", linestyle="--",
                     label="WikiCS",   group="product"),
}

# Only the four epsilons reported in tab:breach -- others may be present
# in the CSV but are not in the table, so we match the paper's narrative.
EPSILONS = [0.01, 0.05, 0.10, 0.20]
WARNING_ZONE_Y = 10.0   # >= 10% breach flagged as high-failure zone


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (dataset, eps) with mean / std breach rate.

    breach_rate in the CSV is stored as a fraction in [0, 1]; we
    convert to percent for the plot so the y-axis matches tab:breach.
    """
    df = df[df["epsilon"].round(2).isin([round(e, 2) for e in EPSILONS])]
    g = df.groupby(["dataset", "epsilon"])["breach_rate"]
    out = g.agg(["mean", "std", "count", "median"]).reset_index()
    out["mean_pct"] = out["mean"] * 100.0
    out["std_pct"] = out["std"] * 100.0
    out["median_pct"] = out["median"] * 100.0
    # 95% CI half-width on the mean (Student t, n-1 df), in percent --
    # matches tab:breach / R2_03 (results/revision_R2/stats_reanalysis.csv)
    # exactly, e.g. Pubmed eps=0.20: 2.262 * 21.10 / sqrt(10) = 15.10.
    tcrit = sps.t.ppf(0.975, out["count"] - 1)
    out["ci95_pct"] = tcrit * out["std_pct"] / np.sqrt(out["count"])
    return out


def main() -> int:
    if not SRC_CSV.exists():
        print(f"ERROR: {SRC_CSV} missing")
        return 1
    df = pd.read_csv(SRC_CSV)
    agg = aggregate(df)

    # Canvas size is matched to the embedded width ratio used in the
    # side-by-side figure* (breach at 0.32·\textwidth, scalability at
    # 0.65·\textwidth). For embedded text to render at the same visual
    # size in both panels, breach's source width must be
    #     scal_src_width · (0.32 / 0.65) ≈ 4.0 in,
    # otherwise breach's text would be ~12% smaller than scalability's
    # because it scales down further on the way into the paper.
    fig, ax = plt.subplots(figsize=(4.00, 2.75))

    # --- Warning band: breach rate >= 10% ---------------------------------
    # Soft red wash so the eye reads "danger zone" without overpowering
    # the data lines.
    ax.axhspan(WARNING_ZONE_Y, 60, facecolor="#D55E00", alpha=0.06,
               zorder=0)
    ax.axhline(WARNING_ZONE_Y, color="#D55E00", linewidth=0.5,
               linestyle=(0, (3, 2)), alpha=0.6, zorder=1)
    # Anchor the danger-zone label at the upper-left corner of the warning
    # band so it does not clash with the Pubmed curve near eps=0.10.
    ax.text(0.005, WARNING_ZONE_Y + 1.0,
            u"≥ 10% breach", family="serif",
            color="#D55E00", fontsize=11, style="italic",
            ha="left", va="bottom", zorder=2)

    # --- Per-dataset curves with mean + std band --------------------------
    # Right-end label nudges to prevent vertical collision (tuned for the
    # 3.55x2.4-in canvas: low-breach Citeseer/WikiCS get split symmetrically
    # below/above the 1-2% band; Pubmed sits at 27.4% so it needs no nudge).
    label_offsets = {
        "Cora":     +0.0,
        "Citeseer": -1.8,
        "Pubmed":   +0.0,
        "WikiCS":   +1.8,
    }
    for dname in DATASETS:
        sub = agg[agg["dataset"] == dname].sort_values("epsilon")
        if sub.empty:
            print(f"  [warn] no rows for {dname}")
            continue
        x = sub["epsilon"].to_numpy()
        m = sub["mean_pct"].to_numpy()
        ci = sub["ci95_pct"].to_numpy()
        sty = STYLE[dname]

        # 95% CI band (Student t, 9 df), clipped at 0 since breach rate is
        # non-negative. Half-widths match tab:breach / R2_03 exactly.
        lo = np.clip(m - ci, 0, None)
        hi = m + ci
        ax.fill_between(x, lo, hi, color=sty["color"], alpha=0.14,
                        linewidth=0, zorder=2)
        # Mean line + markers
        ax.plot(x, m, color=sty["color"], linestyle=sty["linestyle"],
                marker=sty["marker"], linewidth=1.4, markersize=4.8,
                markeredgewidth=0.5, markeredgecolor="white",
                label=sty["label"], zorder=4)

        # Right-end value label at eps=0.20 with thin leader
        x_end = x[-1]
        y_end = m[-1]
        y_label = y_end + label_offsets[dname]
        ax.plot([x_end, x_end + 0.015], [y_end, y_label],
                color=sty["color"], linewidth=0.4, alpha=0.7,
                zorder=3)
        ax.text(x_end + 0.018, y_label, f"{y_end:.1f}%",
                color=sty["color"], family="serif", fontsize=11,
                va="center", ha="left", zorder=5)

    # Pubmed median marker at eps=0.10: small white diamond on the curve
    # exposes the right-skew (median 7.8% vs. mean 10.3%) without the
    # textual annotation, which is now carried by the figure caption to
    # avoid overlapping the data on the 3.55-in canvas.
    pub_sub = agg[(agg["dataset"] == "Pubmed") &
                  (agg["epsilon"] == 0.10)]
    if not pub_sub.empty:
        med = float(pub_sub["median_pct"].iloc[0])
        ax.plot(0.10, med, marker="d", markersize=5.5,
                markerfacecolor="white", markeredgecolor="#009E73",
                markeredgewidth=0.9, zorder=6)

    # --- Axes -------------------------------------------------------------
    # Stretch the x-axis to use the full width available in the canvas:
    # right limit reaches past the last x-tick (0.20) to make room for the
    # right-end value labels (27.4%, 7.6%, 1.8%, 1.5%).
    ax.set_xlim(-0.003, 0.228)
    ax.set_ylim(-1.5, 55)
    ax.set_xticks(EPSILONS)
    ax.set_xticklabels([f"0.{int(e*100):02d}" for e in EPSILONS],
                       family="serif", fontsize=11)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_yticklabels([f"{t}%" for t in [0, 10, 20, 30, 40, 50]],
                       family="serif", fontsize=11)
    # Short axis labels — descriptive text moves to the caption to keep
    # the 3.55-in canvas readable at the embedded 11pt body size.
    # STIX mathtext renders ε in the same serif family as the body text.
    ax.set_xlabel(r"Budget $\varepsilon$", family="serif", fontsize=11)
    ax.set_ylabel("Breach rate", family="serif", fontsize=11)
    ax.grid(True, alpha=0.30, lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Legend pinned to the upper-left of the plot, inside the axes — the
    # bottom-of-figure legend was getting clipped on the narrow canvas.
    # Legend: the four dataset curves plus a proxy entry that defines the
    # white-diamond Pubmed-median marker drawn above, so the figure is
    # self-contained (no orphan glyph). Top-left is empty because all
    # curves start near 0% breach, leaving room for the extra row.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="d", linestyle="none",
                          markerfacecolor="white", markeredgecolor="#009E73",
                          markeredgewidth=0.9, markersize=5.5))
    labels.append("Pubmed median")
    leg = ax.legend(
        handles, labels,
        loc="upper left", bbox_to_anchor=(0.02, 0.98),
        ncol=2, frameon=False, fontsize=11,
        handlelength=1.8, columnspacing=1.0, handletextpad=0.4,
        labelspacing=0.25, borderaxespad=0.0,
    )
    for txt in leg.get_texts():
        txt.set_family("serif")

    # Margins tuned so the y-label ("Breach rate") and x-label ("Budget ε")
    # render fully inside the figsize box (savefig.bbox="standard" clips
    # anything outside, so the labels must be inside the canvas).
    plt.subplots_adjust(bottom=0.22, top=0.97, left=0.18, right=0.92)
    # Disable the global "tight" bbox so the saved canvas matches figsize
    # exactly. The "tight" default expands the bbox to include the legend /
    # right-end labels, which would defeat the column-width match and
    # cause text to be rendered smaller than 11pt in the final paper.
    import matplotlib as mpl
    mpl.rcParams["savefig.bbox"] = "standard"
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF.relative_to(PROJ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
