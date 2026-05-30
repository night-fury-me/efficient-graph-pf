"""Revision-R2 P2.2 — report kappa = ||J_z||_2 directly (not rho).

For each (dataset, seed), trains IGNN, computes J_z exactly, and reports the
triple (||A_hat||_2, ||W||_2, kappa = ||J_z||_2) and the kappa-based ecrit =
(1 - kappa) / ||W||_2. Replaces the spectral-radius rho with the operator
norm kappa across the cross-domain table, eliminating the 28% optimism.

Closes: P2.2 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_07_kappa_direct.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_R2._common import (
    SEEDS,
    forward_and_subgraph,
    full_graph_ctx_Z,
    load_dataset,
    reconverge,
    train_ignn,
)

from iem.adversarial import _compute_structural_jacobian, extract_ego_subgraph

SUBGRAPH_N = 50
OUT_CSV = Path("results/revision_R2/kappa_direct.csv")

DATASET_NAMES = ['Cora', 'Citeseer', 'Pubmed', 'WikiCS', 'Amazon']


def kappa_and_rho(model, X_sub, A_sub):
    """Return (||A_hat||_2, ||W||_2, kappa = ||J_z||_2, rho = spectral radius)."""
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, _, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
    # ||J_z||_2 via SVD on materialized J_z (works for subgraph N=50)
    sv = torch.linalg.svdvals(J_z)
    kappa = float(sv[0].item())
    # Spectral radius via eigenvalues
    eigs = torch.linalg.eigvals(J_z)
    rho = float(eigs.abs().max().item())
    # ||A_hat||_2 and ||W||_2 (use the spectral-normalized W in model)
    A_norm = float(torch.linalg.svdvals(A_sub)[0].item())
    W = None
    for name, p in model.named_parameters():
        if "W" in name or "weight" in name and p.dim() == 2 and p.shape[0] == p.shape[1]:
            if W is None or p.numel() > W.numel():
                W = p
    W_norm = float(torch.linalg.svdvals(W)[0].item()) if W is not None else float("nan")
    return A_norm, W_norm, kappa, rho


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dname in DATASET_NAMES:
        for seed in SEEDS:
            X, A_hat, y, train_mask, n_features, n_classes = load_dataset(dname)
            X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
            train_mask = train_mask.to(device)
            model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed)
            X_sub, A_sub, Z_sub, ctx_sub, _ctx_full, _Z_full, idx = forward_and_subgraph(model, X, A_hat, max_nodes=SUBGRAPH_N)
            A_n, W_n, kappa, rho = kappa_and_rho(model, X_sub, A_sub)
            ecrit_kappa = (1 - kappa) / W_n if W_n > 0 else float("nan")
            ecrit_rho = (1 - rho) / W_n if W_n > 0 else float("nan")
            optimism_pct = 100.0 * (ecrit_rho - ecrit_kappa) / max(ecrit_kappa, 1e-9)
            rows.append({
                "dataset": dname,
                "seed": seed,
                "A_hat_op_norm": A_n,
                "W_op_norm": W_n,
                "kappa_Jz_op_norm": kappa,
                "rho_Jz_spectral_radius": rho,
                "eta_pseudospec_index_proxy": kappa / max(rho, 1e-9),
                "ecrit_kappa": ecrit_kappa,
                "ecrit_rho": ecrit_rho,
                "rho_based_optimism_pct": optimism_pct,
            })
            print(f"  {dname:10s} seed={seed:5d} "
                  f"||A||={A_n:.3f} ||W||={W_n:.3f} "
                  f"kappa={kappa:.3f} rho={rho:.3f} "
                  f"ecrit_kappa={ecrit_kappa:.3f} "
                  f"rho_opt%={optimism_pct:+.1f}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    print("\nUpdate tab:cross_domain with the kappa column (replace rho)"
          " and the new (||A||_2, ||W||_2, kappa) triple.")


if __name__ == "__main__":
    main()
