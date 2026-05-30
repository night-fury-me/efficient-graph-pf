"""Post-process the existing matfree_error_bounds.csv to correct the
neumann_residual mislabel.

The 8-hour R2_04 run produced ``results/revision_R2/matfree_error_bounds.csv``
with a column ``neumann_residual`` whose values are actually sigma_1(J_z)
estimates from the power-iteration loop (the vector was renormalised every
step, so the recorded last_term converged to kappa, not to ||J^K b||).

This script derives the correct quantities from what is already stored:
  * ``kappa_estimate`` <-- renamed from ``neumann_residual``
  * ``neumann_residual_analytic_K200`` = kappa_estimate^200 (analytical residual
    at the Neumann depth the original script attempted)

The Halko bound cannot be back-filled without re-running rSVD (the original
``op.top_k_svd(k=6, ...)`` returned only 6 singular values, so sigma_{k+1}
was unavailable). ``halko_bound_estimate`` is left as NaN.

Output: ``results/revision_R2/matfree_error_bounds_corrected.csv``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("results/revision_R2/matfree_error_bounds.csv")
DST = Path("results/revision_R2/matfree_error_bounds_corrected.csv")

NEUMANN_K = 200  # depth the original power-iteration loop ran to


def main():
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}; nothing to post-process.")
    df = pd.read_csv(SRC)
    if "neumann_residual" not in df.columns:
        raise SystemExit(
            "Source CSV does not have a 'neumann_residual' column; "
            "either already post-processed or schema changed."
        )

    # 1) Rename the mislabeled column.
    df = df.rename(columns={
        "neumann_residual": "kappa_estimate",
        "K_used": "K_kappa_iter",
    })

    # 2) Derive analytical Neumann residual = kappa^K (K=200 was the loop's max).
    def _analytic(kappa):
        if pd.isna(kappa):
            return float("nan")
        if kappa >= 1.0:
            return float("inf")
        # Avoid underflow to exactly 0 in float64 by clamping at machine eps.
        return max(float(kappa) ** NEUMANN_K, np.finfo(float).tiny)

    df["neumann_residual_analytic_K200"] = df["kappa_estimate"].apply(_analytic)

    # 3) Document column changes inline.
    df.to_csv(DST, index=False)
    print(f"Wrote {len(df)} rows to {DST}")
    print()
    print("Column changes:")
    print("  neumann_residual -> kappa_estimate "
          "(power-iteration estimate of sigma_1(J_z), not the Neumann residual)")
    print("  K_used -> K_kappa_iter")
    print("  + new column neumann_residual_analytic_K200 = kappa^200")
    print()
    print("Note: halko_bound_estimate stays NaN for real-dataset rows because")
    print("      the original rSVD requested k=6 sigmas, leaving sigma_{k+1} ")
    print("      unavailable; the patched R2_04 script now requests k=7.")
    print()
    print("Per-dataset summary (corrected):")
    summary = df[df["dataset"] != "Synthetic_ER500"].groupby("dataset").agg(
        n=("seed", "count"),
        N=("N", "first"),
        kappa_mean=("kappa_estimate", "mean"),
        kappa_std=("kappa_estimate", "std"),
        neumann_analytic_mean=("neumann_residual_analytic_K200", "mean"),
    )
    # Format the tiny analytical residual in scientific notation
    summary["kappa_mean"] = summary["kappa_mean"].round(3)
    summary["kappa_std"] = summary["kappa_std"].round(3)
    summary["neumann_analytic_mean"] = summary["neumann_analytic_mean"].apply(
        lambda x: f"{x:.2e}"
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
