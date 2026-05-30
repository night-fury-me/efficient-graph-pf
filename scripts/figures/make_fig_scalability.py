"""Generate paper/figures/fig_scalability.pdf.

Two-panel log-log plot:
  (left) wall-clock time vs N for dense vs matrix-free paths
  (right) peak GPU memory (MB) vs N for dense vs matrix-free
With OOM markers at the dense-path failure boundary (~N=500) and a
horizontal 24 GB GPU-memory ceiling.

Closes editorial decision implicit ask: the "matrix-free up to N=7650
on a single GPU" claim is much stronger as a figure than as a table.

Usage:
    .venv/bin/python scripts/figures/make_fig_scalability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from _style import apply_paper_style
apply_paper_style()

PROJ = Path(__file__).resolve().parents[2]
SRC_CSV = PROJ / "results/exp_scalability_10seed.csv"
SRC_CSV_AMZ = PROJ / "results/exp_scalability_amazon_10seed.csv"
OUT_PDF = PROJ / "paper/figures/fig_scalability.pdf"
OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

GPU_MEM_CEILING_MB = 24 * 1024  # RTX 4090 24 GB
PUBMED_N = 19717


def _load_combined():
    rows = []
    if SRC_CSV.exists():
        df = pd.read_csv(SRC_CSV)
        rows.append(df)
    if SRC_CSV_AMZ.exists():
        df = pd.read_csv(SRC_CSV_AMZ)
        rows.append(df)
    if not rows:
        sys.exit("No scalability CSVs found.")
    return pd.concat(rows, ignore_index=True)


def aggregate_by_N(df: pd.DataFrame):
    """Mean/std time + memory per N for dense and matrix-free."""
    results = []
    for n, sub in df.groupby("N"):
        dense_ok = sub[sub["dense_status"] == "OK"]
        mf_ok = sub[sub["mf_status"] == "OK"]
        results.append({
            "N": n,
            "dense_time_mean": dense_ok["dense_time"].mean() if not dense_ok.empty else np.nan,
            "dense_time_std": dense_ok["dense_time"].std(ddof=1) if len(dense_ok) > 1 else 0.0,
            "dense_mem_mean": dense_ok["dense_mem_mb"].mean() if not dense_ok.empty else np.nan,
            "dense_mem_std": dense_ok["dense_mem_mb"].std(ddof=1) if len(dense_ok) > 1 else 0.0,
            "dense_status": ("OK" if not dense_ok.empty else
                             ("OOM" if "OOM" in sub["dense_status"].values else "N/A")),
            "mf_time_mean": mf_ok["mf_time"].mean() if not mf_ok.empty else np.nan,
            "mf_time_std": mf_ok["mf_time"].std(ddof=1) if len(mf_ok) > 1 else 0.0,
            "mf_mem_mean": mf_ok["mf_mem_mb"].mean() if not mf_ok.empty else np.nan,
            "mf_mem_std": mf_ok["mf_mem_mb"].std(ddof=1) if len(mf_ok) > 1 else 0.0,
        })
    return pd.DataFrame(results).sort_values("N")


def _band(ax, x, mean, std, color, label, marker, linestyle="-", zorder=3):
    """Plot mean line + ±1 std shadow, with three redundant visual cues so
    the variability stays visible on a log-y axis where bands are inherently
    hairline-thin:

      (1) Filled shadow at higher alpha (0.32) so the band itself registers.
      (2) Thin solid edge curves at the lo/hi boundary -- guarantees a
          minimum 1-pixel-wide visible feature even when std/mean << 1.
      (3) Asymmetric vertical capped error bars at each data point -- screen-
          pixel-sized caps that never collapse to invisibility on log axes,
          and also handle the std=0 case (caps just sit on the mean).
    """
    x = np.asarray(x); mean = np.asarray(mean); std = np.asarray(std)
    lo = np.maximum(mean - std, mean * 1e-6 + 1e-6)
    hi = mean + std
    # (1) filled band, opaque enough to see at hairline widths
    ax.fill_between(x, lo, hi, color=color, alpha=0.32, linewidth=0,
                    zorder=zorder - 1)
    # (2) thin solid edges at the band boundary
    ax.plot(x, lo, color=color, linewidth=0.5, alpha=0.7,
            zorder=zorder - 1)
    ax.plot(x, hi, color=color, linewidth=0.5, alpha=0.7,
            zorder=zorder - 1)
    # (3) capped error bars + markers (no line; line drawn separately)
    yerr = np.vstack([mean - lo, hi - mean])
    ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor=color,
                elinewidth=0.7, capsize=2.5, capthick=0.7,
                zorder=zorder)
    # mean line + markers on top
    ax.plot(x, mean, linestyle=linestyle, marker=marker,
            color=color, linewidth=1.0, markersize=4.0,
            markeredgewidth=0.5, markeredgecolor="white",
            label=label, zorder=zorder + 1)


def main():
    df = _load_combined()
    agg = aggregate_by_N(df)

    fig, (axt, axm) = plt.subplots(1, 2, figsize=(8.4, 3.1))

    # ---- TIME PANEL --------------------------------------------------------
    dense = agg.dropna(subset=["dense_time_mean"])
    mf = agg.dropna(subset=["mf_time_mean"])
    _band(axt, dense["N"], dense["dense_time_mean"], dense["dense_time_std"],
          color="#D55E00", label="Dense path", marker="o")
    _band(axt, mf["N"], mf["mf_time_mean"], mf["mf_time_std"],
          color="#0072B2", label="Matrix-free path", marker="s")
    # Set y-limits explicitly so OOM markers live in clear space at the top
    axt.set_xscale("log")
    axt.set_yscale("log")
    axt.set_ylim(0.08, 5e3)
    axt.set_xlim(35, 3e4)

    # OOM markers for dense at the top of the panel
    oom_rows = agg[agg["dense_status"] == "OOM"]
    if not oom_rows.empty:
        axt.scatter(oom_rows["N"], [3e3] * len(oom_rows), marker="x",
                    color="#D55E00", s=35, linewidth=1.0,
                    zorder=5, clip_on=False)

    # Pubmed-OOM vertical line + label tucked into the bottom-right
    axt.axvline(PUBMED_N, color="#888888", linestyle="--", linewidth=0.5)
    axt.text(PUBMED_N * 0.85, 0.13,
             "Pubmed OOM\n" + r"($N{=}19{,}717$)",
             fontsize=11, ha="right", va="bottom",
             color="#666666", style="italic")

    axt.set_xlabel(r"Graph size $N$")
    axt.set_ylabel("Wall-clock time (s)")
    axt.set_title("(a) Computation time")
    axt.grid(True, which="major")
    axt.grid(True, which="minor", linewidth=0.3, alpha=0.3)
    for spine in ("top", "right"):
        axt.spines[spine].set_visible(False)

    # ---- MEMORY PANEL ------------------------------------------------------
    dense_m = agg.dropna(subset=["dense_mem_mean"])
    mf_m = agg.dropna(subset=["mf_mem_mean"])
    axm.set_xscale("log")
    axm.set_yscale("log")
    axm.set_xlim(35, 3e4)
    axm.set_ylim(40, 1.2e5)

    _band(axm, dense_m["N"], dense_m["dense_mem_mean"],
          dense_m["dense_mem_std"],
          color="#D55E00", label="Dense path", marker="o")
    _band(axm, mf_m["N"], mf_m["mf_mem_mean"],
          mf_m["mf_mem_std"],
          color="#0072B2", label="Matrix-free path", marker="s")
    if not oom_rows.empty:
        axm.scatter(oom_rows["N"], [7e4] * len(oom_rows),
                    marker="x", color="#D55E00", s=35, linewidth=1.0,
                    label="Dense OOM (>24 GB)", zorder=5, clip_on=False)

    # 24 GB ceiling line — label below the ceiling, mid-right
    axm.axhline(GPU_MEM_CEILING_MB, color="#888888", linestyle="--",
                linewidth=0.5)
    axm.text(2.5e4, GPU_MEM_CEILING_MB * 0.55,
             "24 GB GPU ceiling", fontsize=11, ha="right",
             color="#444444")

    # Pubmed OOM vertical + label in the bottom of the panel
    axm.axvline(PUBMED_N, color="#888888", linestyle="--", linewidth=0.5)
    axm.text(PUBMED_N * 0.85, 55,
             "Pubmed OOM\n" + r"($N{=}19{,}717$)", fontsize=11,
             ha="right", va="bottom", color="#666666", style="italic")

    axm.set_xlabel(r"Graph size $N$")
    axm.set_ylabel("Peak GPU memory (MB)")
    axm.set_title("(b) Peak GPU memory")
    axm.grid(True, which="major")
    axm.grid(True, which="minor", linewidth=0.3, alpha=0.3)
    for spine in ("top", "right"):
        axm.spines[spine].set_visible(False)

    # Shared legend below both panels
    handles, labels = [], []
    for ax in (axt, axm):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(OUT_PDF, format="pdf")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
