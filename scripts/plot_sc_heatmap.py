"""Render fig_sc_heatmap.pdf via matplotlib for publication-quality output.

Layout (single figure, three panels):
   - Heatmap  (top, wide)   : |S_c|  --  rows = node-channels, cols = edges
   - v_1 strip (mid, wide)  : |v_1|  --  AEGIS attack direction, x-aligned
   - Spectrum (right, tall) : sigma_k(S_c)  --  shows the spectral gap

Why this layout: it tells the full AEGIS story in a single figure:
  (1) S_c has banded sensitivity structure  (heatmap)
  (2) The SVD-optimal direction v_1 picks up that structure  (v_1 strip)
  (3) The leading singular value is well-separated  (spectrum)

Data: produced by scripts/dump_sc_matrix_cora.py (.npy + .json in data/).

Usage:
    .venv/bin/python scripts/plot_sc_heatmap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'figures'))
from _style import apply_paper_style
DATA = ROOT / "paper" / "figures" / "data"
OUT = ROOT / "paper" / "figures" / "fig_sc_heatmap.pdf"

OK_BLUE = "#0072B2"
OK_GREEN = "#009E73"
OK_VERMILION = "#D55E00"
OK_GREY = "#888888"


def setup_style() -> None:
    apply_paper_style()


def main() -> int:
    setup_style()

    sc_path = DATA / "sc_matrix.npy"
    v1_path = DATA / "sc_v1.npy"
    meta_path = DATA / "sc_meta.json"

    if not (sc_path.exists() and v1_path.exists() and meta_path.exists()):
        print(f"ERROR: missing dumps in {DATA}. Run scripts/dump_sc_matrix_cora.py first.")
        return 1

    sc = np.load(sc_path)
    v1 = np.abs(np.load(v1_path))
    meta = json.loads(meta_path.read_text())

    D, E = sc.shape
    print(f"S_c shape ({D}, {E})  sigma_1={meta['sigma_1']:.2f}  gap={meta['gap_ratio']:.3f}")

    # Singular spectrum
    sigma = torch.linalg.svdvals(torch.from_numpy(sc.astype(np.float32))).numpy()
    k_show = min(12, len(sigma))
    sigma_show = sigma[:k_show]

    # Mean-pool rows down for display
    target_rows = 36
    if D > target_rows:
        grp = D // target_rows
        sc_disp = sc[: grp * target_rows].reshape(target_rows, grp, E).mean(axis=1)
    else:
        sc_disp = sc

    # 4.35-in width matches fig_phase_transition / fig_breach_rate so that
    # at \columnwidth the embedded text renders at the same ~8.9pt body
    # size. Height kept moderate so the figure does not consume too much
    # vertical space in a single-column slot — the original aspect (2.50)
    # was too wide for one column, but full square (3.00 tall) overflowed.
    #
    # Layout: 2 rows x 2 cols.
    #   col 0 : heatmap + colorbar (top) | v_1 strip (bottom)
    #   col 1 : spectrum (top only)
    fig = plt.figure(figsize=(4.35, 2.55))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[3.4, 1.15],
        height_ratios=[2.6, 1.0],
        # wspace must clear: colorbar tick labels (right side of colorbar)
        # + the spectrum's "σ_k" y-label (left side of spectrum). At a
        # 4.35-in canvas these two collide at wspace<=0.30; 0.42 leaves a
        # visible breathing gap.
        wspace=0.42, hspace=0.22,
        left=0.10, right=0.98, top=0.84, bottom=0.18,
        figure=fig,
    )
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_v1 = fig.add_subplot(gs[1, 0], sharex=ax_hm)
    ax_sp = fig.add_subplot(gs[0, 1])
    # Colorbar attached to heatmap with absolute pad (in inches via "size"
    # as a fraction of parent axes). Result: colorbar sits flush to the
    # heatmap, leaving the gridspec wspace as the only gap to the spectrum.
    divider = make_axes_locatable(ax_hm)
    cax = divider.append_axes("right", size="2.2%", pad=0.08)

    # ---- Heatmap ----
    im = ax_hm.imshow(
        sc_disp,
        aspect="auto", cmap="cividis", interpolation="nearest",
        vmin=0, vmax=sc_disp.max(),
        extent=[-0.5, E - 0.5, sc_disp.shape[0] - 0.5, -0.5],
    )
    ax_hm.set_ylabel(u"node–channel", family="serif", fontsize=11)
    # Short title — meta details (N, d, |E|, seed) belong in the figure
    # caption. Mathtext $|S_c|$ renders the subscript properly via the
    # STIX fontset configured in apply_paper_style().
    ax_hm.set_title(
        r"Structural sensitivity matrix $|S_c|$",
        pad=4, family="serif", fontsize=11,
    )
    plt.setp(ax_hm.get_xticklabels(), visible=False)
    ax_hm.tick_params(axis="y", length=2.5, width=0.4)

    # Highlight top-3 edges by |v_1| (vertical lines spanning heatmap)
    top3 = np.argsort(-v1)[:3]
    for k, idx in enumerate(top3):
        ax_hm.axvline(idx, color=OK_VERMILION,
                      lw=1.0, alpha=0.85 - 0.18 * k, zorder=3)

    # Colorbar: |S_c| label is mounted ABOVE the colorbar (as a unit title)
    # instead of beside it -- this keeps the gap between colorbar and the
    # spectrum panel reserved for the spectrum's "sigma_k(S_c)" y-label only.
    cb = fig.colorbar(im, cax=cax)
    # No colorbar label: the heatmap title already says
    # "Structural sensitivity matrix" (i.e. |S_c|). Adding a rotated label
    # on the colorbar collides with the spectrum's "σ_k" y-label in the
    # narrow gap on a 4.35-in canvas.
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.4, length=2, labelsize=9)

    # ---- v_1 strip ----
    bar_colors = np.array([OK_GREEN] * E, dtype=object)
    bar_colors[top3] = OK_VERMILION
    ax_v1.bar(
        np.arange(E), v1 / v1.max(),
        color=bar_colors, edgecolor="black", linewidth=0.25, width=0.85,
    )
    ax_v1.set_xlim(-0.5, E - 0.5)
    ax_v1.set_ylim(0, 1.18)
    ax_v1.set_xlabel(r"edge index $(i,j) \in E$", family="serif", fontsize=11)
    ax_v1.set_ylabel(r"$|v_1|/\max$", family="serif", fontsize=11)
    ax_v1.set_yticks([0, 0.5, 1.0])
    ax_v1.tick_params(axis="both", labelsize=10)
    ax_v1.grid(axis="y", alpha=0.25, lw=0.4)
    ax_v1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_v1.spines[sp].set_visible(False)
    # Top-3 marker labels (plain serif bold — no mathtext, so they render
    # in the same Nimbus Roman family as the rest of the figure). When two
    # top-ranked edges are adjacent in `idx`, stagger their y-positions so
    # the labels don't collide.
    for rank, idx in enumerate(top3):
        # If a previous label sits within 2 bar-widths, place this one to the
        # side of the bar so the labels do not stack on top of each other.
        adj_prev = [int(prev) for prev in top3[:rank]
                    if abs(int(idx) - int(prev)) <= 2]
        if adj_prev:
            # Shift horizontally away from the adjacent previous label.
            dx = +1.6 if int(idx) > adj_prev[0] else -1.6
        else:
            dx = 0.0
        ax_v1.text(idx + dx, v1[idx] / v1.max() + 0.06,
                   f"#{rank+1}",
                   ha="center", va="bottom", fontsize=10, color=OK_VERMILION,
                   family="serif", fontweight="bold")

    # ---- Spectrum sidebar ----
    # Highlight the leading singular value (sigma_1) by recolouring its bar
    # vermilion -- visually links the "spectral gap" annotation to the
    # specific bar it refers to, instead of leaving the eye to chase it.
    ks = np.arange(1, k_show + 1)
    bar_face = [OK_VERMILION if i == 0 else OK_BLUE for i in range(k_show)]
    bar_edge = ["#7A2E00" if i == 0 else "#003F66" for i in range(k_show)]
    ax_sp.bar(
        ks, sigma_show,
        color=bar_face, edgecolor=bar_edge, linewidth=0.4, width=0.78,
    )
    if len(sigma) >= 2:
        gap_pct = (sigma[0] - sigma[1]) / sigma[0] * 100
        # Place the double-headed arrow past BOTH σ₁ and σ₂ bars (σ₂'s
        # right edge is ~2.39) so it sits in clear background. The thin
        # dashed whiskers below carry the eye from each bar top out to the
        # arrow's vertical axis.
        x_arrow = 2.85
        ax_sp.annotate(
            "", xy=(x_arrow, sigma[1]),
            xytext=(x_arrow, sigma[0]),
            arrowprops=dict(arrowstyle="<->", color=OK_VERMILION, lw=1.1,
                            shrinkA=0, shrinkB=0),
        )
        # Thin reference whiskers at σ₁ / σ₂ levels (anchor the arrow back
        # to the actual bar tops it refers to).
        ax_sp.hlines(sigma[0], 1, x_arrow, color=OK_VERMILION,
                     lw=0.5, linestyles=(0, (2, 1.5)))
        ax_sp.hlines(sigma[1], 2, x_arrow, color=OK_VERMILION,
                     lw=0.5, linestyles=(0, (2, 1.5)))
        # Gap label in plain serif bold (Nimbus Roman, no mathtext) sits
        # just to the right of the arrow.
        ax_sp.text(
            x_arrow + 0.30, (sigma[0] + sigma[1]) / 2,
            f"{gap_pct:.0f}%\ngap",
            color=OK_VERMILION, fontsize=10, va="center", ha="left",
            family="serif", fontweight="bold",
        )
    ax_sp.set_xlabel(r"singular index $k$", labelpad=2, family="serif", fontsize=11)
    # Short y-label — full "$\sigma_k(S_c)$" would collide with the
    # colorbar tick labels on a narrow 4.35-in canvas. Title carries the
    # broader context. STIX mathtext renders the subscript properly.
    ax_sp.set_ylabel(r"$\sigma_k$", labelpad=2, family="serif", fontsize=11)
    ax_sp.set_title(r"Spectrum of $S_c$", pad=4, family="serif", fontsize=11)
    ax_sp.set_xlim(0.3, k_show + 0.7)
    ax_sp.set_ylim(0, sigma_show.max() * 1.12)
    ax_sp.set_xticks([1, max(1, k_show // 2), k_show])
    ax_sp.tick_params(axis="both", length=2.5, width=0.4, labelsize=10)
    ax_sp.grid(axis="y", alpha=0.25, lw=0.4)
    ax_sp.set_axisbelow(True)
    for sp in ("top", "right"):
        ax_sp.spines[sp].set_visible(False)

    # Override the global savefig.bbox="tight" so the saved canvas matches
    # figsize exactly — otherwise matplotlib expands the bbox to include
    # any artist outside the figure box, defeating the columnwidth match.
    import matplotlib as mpl
    mpl.rcParams["savefig.bbox"] = "standard"
    plt.savefig(OUT, format="pdf")
    print(f"Wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
