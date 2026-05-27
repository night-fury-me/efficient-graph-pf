"""Generate τ heatmap for cross-architecture transfer (Fig. tau_heatmap).

Rules: font=serif, fontsize=11, format=pdf.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "text.usetex": True,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

models = ["IGNN", "GCN-2", "GCN-4", "GIN-2", r"GAT$^\dagger$", "SAGE-2", "APPNP"]
datasets = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Am. Photo"]

tau = np.array([
    [+.32, +.31, +.82, +.14, -.15],
    [-.03, -.28, +.21, +.05, +.25],
    [+.49, +.64, +.89, +.45, -.04],
    [+.33, +.57, +.54, +.63, +.14],
    [+.54, +.66, np.nan, np.nan, +.21],
    [+.22, +.38, +.36, +.22, +.60],
    [+.35, +.36, +.83, +.22, +.43],
])

mask = np.isnan(tau)

cmap = sns.diverging_palette(10, 145, s=90, l=50, as_cmap=True)
cmap.set_bad(color="0.88")

fig, ax = plt.subplots(figsize=(4.2, 2.8))

sns.heatmap(
    tau,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    vmin=-0.30,
    vmax=0.90,
    linewidths=0.6,
    linecolor="white",
    xticklabels=datasets,
    yticklabels=models,
    cbar_kws={"label": r"Kendall $\tau$", "shrink": 0.85},
    annot_kws={"fontsize": 9},
    ax=ax,
)

for i, j in zip(*np.where(mask)):
    ax.text(j + 0.5, i + 0.5, "OOM", ha="center", va="center",
            fontsize=8, color="0.45", style="italic")

ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_title(r"Continuous-to-discrete transfer $\tau$", fontsize=11, pad=8)

fig.tight_layout()
fig.savefig("paper/figures/tau_heatmap.pdf", bbox_inches="tight", dpi=300)
print("Saved to paper/figures/tau_heatmap.pdf")
