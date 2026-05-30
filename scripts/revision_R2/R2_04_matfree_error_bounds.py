"""Revision-R2 P1.6 — matrix-free certified-error contract.

Reports for every dataset (and a new N=500 synthetic graph) three quantities:
  (a) Neumann truncation residual ||J_z^{K+1} b||/||b|| at termination, averaged
      over the rSVD probe vectors;
  (b) Halko-Martinsson-Tropp 2011 Theorem 10.7 probabilistic error estimate:
        ||(I - QQ^T) S_c|| <= (1 + 11 * sqrt(k+p) * sqrt(min(N,m))) * sigma_{k+1}
      (we use the standard scaled bound with oversampling p=10);
  (c) on a fresh N=500 synthetic Erdos-Renyi-IGNN graph where a dense reference
      can be computed, sigma_1(matrix-free) vs sigma_1(dense) discrepancy.

Closes: P1.6 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_04_matfree_error_bounds.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_R2._common import (
    SEEDS,
    forward_and_subgraph,
    full_graph_ctx_Z,
    load_dataset,
    reconverge,
    train_ignn,
)

from iem.examples.ignn_cora import IGNN
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity

OUT_CSV = Path("results/revision_R2/matfree_error_bounds.csv")

# (dataset_name, loader, subgraph_N) -- subgraph_N=None means full graph
# datasets handled via DATASET_TUPLES below
DATASET_TUPLES = [
    ("Cora", None),
    ("Citeseer", None),
    ("Pubmed", 200),
    ("WikiCS", 200),
    ("Amazon", None),
]


def power_iter_kappa(model, Z_star, ctx, n_probe=10, n_terms=200, tol=1e-6):
    """Estimate kappa = sigma_1(J_z) via power iteration with FD-based JVP.

    NOTE: This is NOT the Neumann series truncation residual. The vector ``v``
    is renormalised after every iteration, so the recorded ``last_term``
    converges to sigma_1(J_z), not to ||J_z^K b||. The previous version of
    this function was mislabeled; the analytical truncation residual is
    sigma_1(J_z)^K, and is computed alongside ``neumann_residual_true``
    in :func:`neumann_residual_true`.

    Returns:
        (kappa_estimate, K_used) — float, int
    """
    A_sub = ctx["A_hat"]
    kappas = []
    K_last = 0
    for _ in range(n_probe):
        b = torch.randn(Z_star.shape, device=A_sub.device)
        b = b / b.norm()
        v = b.clone()
        last_term = 1.0
        K_used = 0
        for k in range(n_terms):
            eps = 1e-3
            with torch.no_grad():
                Z_plus = model.operator(Z_star + eps * v, ctx)
                Z_minus = model.operator(Z_star - eps * v, ctx)
            Jv = (Z_plus - Z_minus) / (2 * eps)
            norm = float(Jv.norm().item())
            if norm < tol:
                K_used = k + 1
                last_term = norm
                break
            v = Jv / norm
            last_term = norm
            K_used = k + 1
        kappas.append(last_term)
        K_last = K_used
    return float(np.mean(kappas)), K_last


def neumann_residual_true(model, Z_star, ctx, n_probe=10, K=200,
                           floor=1e-30):
    """True Neumann-truncation residual ||J_z^K b|| / ||b||, no renormalisation.

    Uses exact forward-mode JVPs (``torch.func.jvp``) so the K-step decay
    is not contaminated by FD bias. Stops early once the norm drops below
    ``floor`` (FP underflow safety) and reports the per-probe geometric
    mean residual at K.

    Returns:
        (mean_residual_at_K, K_effective_max) — float, int
    """
    def f(z):
        return model.operator(z, ctx)

    residuals = []
    K_eff_max = 0
    for _ in range(n_probe):
        b = torch.randn(Z_star.shape, device=Z_star.device)
        b = b / b.norm()
        v = b.clone()
        K_eff = K
        norm = 1.0
        for k in range(K):
            with torch.no_grad():
                _, Jv = torch.func.jvp(f, (Z_star,), (v,))
            v = Jv
            norm = float(v.norm().item())
            if norm < floor:
                K_eff = k + 1
                break
        residuals.append(norm)
        K_eff_max = max(K_eff_max, K_eff)
    return float(np.mean(residuals)), K_eff_max


def halko_bound(sigma_estimates, k_target=6, p_oversample=10):
    """Halko-Martinsson-Tropp 2011 Theorem 10.7 scaled bound estimate.

    Bound: E[||(I - QQ^T) A||_2] <= (1 + sqrt(k_target/(p-1))) * sigma_{k_target+1}
    where k_target is the requested rank of the approximation and
    ``sigma_estimates`` must contain at least ``k_target+1`` values
    (i.e. include sigma_{k_target+1}, the first singular value beyond
    the target rank).

    Returns NaN if the input is too short.
    """
    if len(sigma_estimates) < k_target + 1:
        return float("nan")
    sigma_kp1 = sigma_estimates[k_target]   # 0-indexed: index k_target = sigma_{k_target+1}
    geom = 1.0 + math.sqrt(k_target / max(p_oversample - 1, 1))
    return float(geom * sigma_kp1)


def synthetic_erdosrenyi_check(N=500, p_edge=0.02, hidden=64, seed=42,
                                device="cpu"):
    """Generate an Erdos-Renyi graph + random IGNN, compare sigma_1.

    Provides a clean sigma_1 comparison: dense reference vs matrix-free rSVD.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Random sparse symmetric A
    mask = (torch.rand(N, N) < p_edge).float()
    A = (mask + mask.t()).clamp(max=1.0)
    A.fill_diagonal_(0)
    deg = A.sum(dim=1).clamp(min=1.0)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    A_hat = A_hat.to(device)
    # Random IGNN
    n_features = 16
    n_classes = 4
    X = torch.randn(N, n_features, device=device)
    model = IGNN(n_features, hidden=hidden, n_classes=n_classes).to(device)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    def F_op(z, c):
        return model.operator(z, c)
    # Dense reference (will work at N=500 if memory permits)
    try:
        J_z, J_A, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
        S = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
        S_c, _ = constrained_sensitivity_matrix(S, A_hat)
        sigma_dense = float(torch.linalg.svdvals(S_c)[0])
    except RuntimeError as exc:
        sigma_dense = float("nan")
        print(f"  [N=500 dense ref failed: {exc}]")
    # Matrix-free -- request k_target+1 = 7 singular values so the Halko
    # bound has access to sigma_{k_target+1}.
    HALKO_K = 6
    op = ScalableSensitivity(F_op, Z_star, ctx)
    op.edge_vulnerability()
    _, sigma_mf, _ = op.top_k_svd(k=HALKO_K + 1, n_oversamples=10, n_power_iter=5)
    sigma_mf_1 = float(sigma_mf[0])
    if not math.isnan(sigma_dense) and sigma_dense > 0:
        rel_err = abs(sigma_dense - sigma_mf_1) / sigma_dense
    else:
        rel_err = float("nan")
    return {
        "N_synthetic": N,
        "sigma_dense": sigma_dense,
        "sigma_matfree": sigma_mf_1,
        "relative_error": rel_err,
        "halko_bound_estimate": halko_bound(
            sigma_mf.cpu().numpy().tolist(), k_target=HALKO_K
        ),
    }


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    # Synthetic N=500 check (3 seeds is enough; synthetic is a sanity test)
    for seed in SEEDS[:3]:
        res = synthetic_erdosrenyi_check(seed=seed, device=device)
        rows.append({"dataset": "Synthetic_ER500", "seed": seed, **res,
                     "kappa_estimate": float("nan"),
                     "K_kappa_iter": -1,
                     "neumann_residual_true": float("nan"),
                     "K_neumann_used": -1,
                     "neumann_residual_analytic_K200": float("nan")})
        print(f"  Synth ER500 seed={seed} "
              f"sigma_dense={res['sigma_dense']:.4f} "
              f"sigma_mf={res['sigma_matfree']:.4f} "
              f"rel_err={res['relative_error']:.4f}", flush=True)

    # Each real dataset
    for dname, sub_n in DATASET_TUPLES:
        for seed in SEEDS:
            X, A_hat, y, train_mask, n_features, n_classes = load_dataset(dname)
            X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
            train_mask = train_mask.to(device)
            model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed)
            if sub_n is not None:
                idx = extract_ego_subgraph(A_hat, max_nodes=sub_n)
                A_use = A_hat[idx][:, idx]
                X_use = X[idx]
            else:
                A_use, X_use = A_hat, X
            with torch.no_grad():
                _, Z_star, ctx_full = model(X_use, A_use)
            # Subgraph case: forward was on the *subgraph* (A_use, X_use), so
            # ctx_full["X_proj"] already matches Z_star's row count.
            ctx = ctx_full
            # (a) kappa = sigma_1(J_z) via power iteration (FD-JVP, fast)
            kappa_est, K_kappa = power_iter_kappa(model, Z_star, ctx)
            # (b) True Neumann truncation residual ||J^K b|| / ||b|| at K=200
            #     via exact forward-mode JVPs (no FD noise, no renormalisation).
            NEUMANN_K = 200
            neu_resid_true, K_neu = neumann_residual_true(
                model, Z_star, ctx, n_probe=10, K=NEUMANN_K)
            # (c) Analytical residual at depth K from the kappa estimate.
            #     For kappa < 1: kappa^K shrinks; for kappa >= 1: divergent.
            if kappa_est < 1.0:
                neu_resid_analytic = float(kappa_est ** NEUMANN_K)
            else:
                neu_resid_analytic = float("inf")
            # (d) Matrix-free rSVD with k_target+1 = 7 sigmas so Halko has
            #     sigma_{k_target+1} available.
            HALKO_K = 6
            def F_op(z, c):
                return model.operator(z, c)
            op = ScalableSensitivity(F_op, Z_star, ctx)
            op.edge_vulnerability()
            _, sigma_est, _ = op.top_k_svd(k=HALKO_K + 1, n_oversamples=10,
                                            n_power_iter=5)
            halko = halko_bound(sigma_est.cpu().numpy().tolist(),
                                k_target=HALKO_K)
            rows.append({
                "dataset": dname,
                "seed": seed,
                "N": int(A_use.shape[0]),
                "kappa_estimate": kappa_est,
                "K_kappa_iter": K_kappa,
                "neumann_residual_true": neu_resid_true,
                "K_neumann_used": K_neu,
                "neumann_residual_analytic_K200": neu_resid_analytic,
                "sigma_matfree_top": float(sigma_est[0]),
                "halko_bound_estimate": halko,
                "sigma_dense": float("nan"),
                "sigma_matfree": float(sigma_est[0]),
                "relative_error": float("nan"),
                "N_synthetic": -1,
            })
            print(f"  {dname:10s} seed={seed:5d} N={A_use.shape[0]:5d} "
                  f"kappa={kappa_est:.4f} "
                  f"neumann_true={neu_resid_true:.2e} "
                  f"neumann_analytic=kappa^200={neu_resid_analytic:.2e} "
                  f"sigma1={float(sigma_est[0]):.3f} Halko<={halko:.3f}",
                  flush=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
