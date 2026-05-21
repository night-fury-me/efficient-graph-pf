"""Shared matplotlib helpers for publication-quality plots.

Centralizes the look-and-feel that was previously duplicated across
`scripts/plot_*.py`:
  * `apply_publication_style()` — serif fonts, sizes, colorblind palette.
  * `new_paper_figure()` — fig+ax sized for two-column papers.
  * `apply_paper_margins(fig)` — consistent subplot margins.
  * `save_figure(fig, base_path, formats=...)` — multi-format export.

In-training diagnostic plots (`train/plotting.py`) deliberately use plain
matplotlib defaults; they are not publication artifacts. Use this module for
plots that go into papers/reports.
"""

from __future__ import annotations

import os
from typing import Sequence, Tuple


PAPER_FIGSIZE: Tuple[float, float] = (4.2, 2.7)
PAPER_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.22, "top": 0.97}

_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,  # embed TrueType for editable PDF text
    "svg.fonttype": "none",  # leave SVG text as text, not paths
}


def apply_publication_style() -> None:
    """Apply seaborn-colorblind cycle + serif rcParams. Idempotent."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-colorblind")
    plt.rcParams.update(_PAPER_RC)


def new_paper_figure(figsize: Tuple[float, float] = PAPER_FIGSIZE):
    """Create a (fig, ax) sized for paper plots. Call `apply_paper_margins(fig)`
    after laying out content."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    return fig, ax


def apply_paper_margins(fig) -> None:
    """Apply the standard paper-plot subplot margins."""
    fig.subplots_adjust(**PAPER_MARGINS)


def save_figure(
    fig,
    base_path: str,
    *,
    formats: Sequence[str] = ("pdf", "svg"),
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.02,
    dpi: int = 240,
    verbose: bool = True,
) -> list[str]:
    """Save `fig` to multiple formats under a common base path.

    `base_path` may include or omit an extension; each format produces
    `{base_path_without_ext}.{format}`. PNG outputs include `dpi`; vector
    formats ignore it.
    """
    base, _ = os.path.splitext(base_path)
    out_paths: list[str] = []
    dirname = os.path.dirname(base) or "."
    os.makedirs(dirname, exist_ok=True)
    for fmt in formats:
        out = f"{base}.{fmt}"
        kwargs = {"bbox_inches": bbox_inches, "pad_inches": pad_inches}
        if fmt.lower() == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        out_paths.append(out)
        if verbose:
            print(f"[OK] Saved: {out}")
    return out_paths


__all__ = [
    "PAPER_FIGSIZE",
    "PAPER_MARGINS",
    "apply_publication_style",
    "new_paper_figure",
    "apply_paper_margins",
    "save_figure",
]
