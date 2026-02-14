import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Input CSVs
# =========================
FULL_FT_CSV = "results/mlruns/d8222b00a0ac4c7a84377d343ee89853/artifacts/run/artifacts/history.csv"
LORA_HEAD_CSV = "results/mlruns/47a86a38f1f54a3d9be987baaaf87534/artifacts/run/artifacts/history.csv"

# =========================
# Output
# =========================
OUT_DIR = "./results/pareto"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FULL_PNG = os.path.join(OUT_DIR, "physics_loss_full_ft.png")
OUT_FULL_SVG = os.path.join(OUT_DIR, "physics_loss_full_ft.svg")
OUT_FULL_PDF = os.path.join(OUT_DIR, "physics_loss_full_ft.pdf")
OUT_LORA_PNG = os.path.join(OUT_DIR, "physics_loss_lora_head.png")
OUT_LORA_SVG = os.path.join(OUT_DIR, "physics_loss_lora_head.svg")
OUT_LORA_PDF = os.path.join(OUT_DIR, "physics_loss_lora_head.pdf")

# Plot every Nth point to reduce noise
PLOT_EVERY = 5
SMOOTH_WINDOW = 5

# Zoom band around 1.0 to emphasize small jumps
ZOOM_AROUND_ONE = True
ZOOM_BAND = (0.9, 1.30)

# =========================
# Styling (match pareto plot)
# =========================
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

COLOR_TRAIN = "tab:blue"
COLOR_VAL = "tab:orange"


def _load_history(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"epoch", "split", "loss"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)} in {path}")
    return df


def _pivot_loss(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure consistent ordering per epoch
    df = df.copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"]).sort_values("epoch")
    df = df[df["epoch"] > 0]
    pivot = df.pivot_table(index="epoch", columns="split", values="loss", aggfunc="mean")
    return pivot


def _downsample(pivot: pd.DataFrame, every: int) -> pd.DataFrame:
    if every <= 1:
        return pivot
    return pivot.iloc[::every].copy()


def _plot_one(
    *,
    loss: pd.DataFrame,
    title: str,
    out_png: str,
    out_svg: str,
    out_pdf: str,
    marker_edge: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.7), constrained_layout=False)

    edge_kwargs = {}
    if marker_edge:
        edge_kwargs = {"markeredgecolor": "black", "markeredgewidth": 0.6}

    if "train" in loss.columns:
        # Raw curve
        ax.plot(
            loss.index,
            loss["train"],
            marker="s",
            linewidth=0.8,
            markersize=3.2,
            label="Training",
            color=COLOR_TRAIN,
            # alpha=0.75,
            zorder=1,
            **edge_kwargs,
        )
    if "val" in loss.columns:
        # Raw curve
        ax.plot(
            loss.index,
            loss["val"],
            marker="o",
            linewidth=0.8,
            markersize=3.2,
            label="Validation",
            color=COLOR_VAL,
            # alpha=0.75,
            zorder=1,
            **edge_kwargs,
        )

    y = loss["train"] if "train" in loss.columns else loss["val"]
    y = y.replace([float("inf"), float("-inf")], pd.NA).dropna()

    # Use symlog to emphasize changes around 1.0 while still compressing large values.
    use_symlog = False
    if not y.empty:
        y_pos = y[y > 0]
        if not y_pos.empty:
            ratio = float(y_pos.max() / y_pos.min())
            use_symlog = ratio >= 10.0 or (y_pos.min() <= 2.0 <= y_pos.max())

    if use_symlog:
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.8, base=10)

    # Zoom around 1.0 to make small jumps more visible.
    if ZOOM_AROUND_ONE:
        ax.set_ylim(*ZOOM_BAND)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Physics loss (symlog, linthresh=1.0)" if use_symlog else "Physics loss")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.8)
    ax.legend(frameon=True, loc="best")

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.22, top=0.97)

    fig.savefig(out_png, dpi=240, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"[OK] Saved: {out_png}")
    print(f"[OK] Saved: {out_svg}")
    print(f"[OK] Saved: {out_pdf}")
    plt.show()


def main() -> None:
    full_df = _load_history(FULL_FT_CSV, "Full-FT")
    lora_df = _load_history(LORA_HEAD_CSV, "LoRA+Head")

    full_loss = _pivot_loss(full_df)
    lora_loss = _pivot_loss(lora_df)

    full_loss = _downsample(full_loss, PLOT_EVERY)
    lora_loss = _downsample(lora_loss, PLOT_EVERY)

    # =========================
    # Plot (separate diagrams)
    # =========================
    _plot_one(
        loss=full_loss,
        title="Full FT Physics Loss",
        out_png=OUT_FULL_PNG,
        out_svg=OUT_FULL_SVG,
        out_pdf=OUT_FULL_PDF,
        marker_edge=True,
    )

    _plot_one(
        loss=lora_loss,
        title="LoRa+PHead FT Phyics Loss",
        out_png=OUT_LORA_PNG,
        out_svg=OUT_LORA_SVG,
        out_pdf=OUT_LORA_PDF,
        marker_edge=True,
    )


if __name__ == "__main__":
    main()
