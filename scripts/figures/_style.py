"""Shared matplotlib style for the AEGIS paper figures.

Calling apply_paper_style() once at the top of a figure script gives
all subsequent plots a consistent serif/11pt look with thin lines and
small markers, matching the IEEE conference body text.
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.font_manager as fm

# Register Nimbus Roman (a Times-equivalent free serif shipped with
# Ghostscript / urw-base35-fonts) so matplotlib's font manager picks it
# up even when its cache was built before the font was installed.
# mathptmx — used by the TikZ figures in this paper — falls back to
# Nimbus Roman if Times is unavailable, so matching it here keeps every
# figure visually consistent in the final paper.
_NIMBUS_FILES = [
    "/usr/share/fonts/gsfonts/NimbusRoman-Regular.otf",
    "/usr/share/fonts/gsfonts/NimbusRoman-Bold.otf",
    "/usr/share/fonts/gsfonts/NimbusRoman-Italic.otf",
    "/usr/share/fonts/gsfonts/NimbusRoman-BoldItalic.otf",
]
for _p in _NIMBUS_FILES:
    if os.path.exists(_p):
        try:
            fm.fontManager.addfont(_p)
        except Exception:
            pass


def apply_paper_style():
    mpl.rcParams.update({
        # --- typography ---
        "font.family":      "serif",
        # Prefer Nimbus Roman (matches mathptmx-rendered TikZ figures),
        # then Times equivalents, then DejaVu Serif as the universal
        # fallback. "serif" at the tail ensures something always resolves.
        "font.serif":       ["Nimbus Roman", "Times New Roman",
                              "Nimbus Roman No9 L", "Liberation Serif",
                              "DejaVu Serif", "serif"],
        "font.size":        11,
        "axes.titlesize":   11,
        "axes.labelsize":   11,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  11,
        # STIX is a Times-like serif math font shipped with matplotlib;
        # it visually matches Nimbus Roman text far better than the
        # default Computer Modern ("cm") fontset, so $\sigma_k$, $|S_c|$
        # etc. render in the same serif family as the surrounding labels.
        "mathtext.fontset": "stix",
        "text.usetex":      False,

        # --- thin lines + small markers ---
        "lines.linewidth":  0.9,
        "lines.markersize": 3.5,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth":  0.5,

        # --- thin axes + ticks ---
        "axes.linewidth":   0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size":  3.0,
        "ytick.major.size":  3.0,

        # --- grid ---
        "grid.linestyle":   ":",
        "grid.linewidth":   0.4,
        "grid.alpha":       0.55,

        # --- legend ---
        "legend.frameon":   False,
        "legend.handlelength": 2.4,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,

        # --- output ---
        "pdf.fonttype":     42,  # TrueType (editable in vector tools)
        "ps.fonttype":      42,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.04,
    })
