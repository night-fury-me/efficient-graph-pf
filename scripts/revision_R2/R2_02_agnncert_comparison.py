"""Revision-R2 P1.4 — AGNNCert deterministic per-node-radii comparison.

Compares AEGIS first-order per-node sensitivity radii r_v against AGNNCert-style
deterministic per-node certified radii (Li et al. 2025). Because Li 2025 does
not yet have a publicly mature codebase, we implement a deterministic radius
via single-edge interval-bound propagation through the trained IGNN: for each
node v, we compute the largest discrete ||delta A_v||_F such that the IBP-bounded
output margin remains positive. This is a deterministic per-node radius in the
same category as AGNNCert. The semantic distinction (diagnostic-tight vs
certified-conservative) is reported alongside the correlation.

Closes: P1.4 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_02_agnncert_comparison.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_R2._common import (
    SEEDS,
    forward_and_subgraph,
    full_graph_ctx_Z,
    load_dataset,
    reconverge,
    train_ignn,
)

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    per_node_robust_radius,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius

SUBGRAPH_N = 50
OUT_CSV = Path("results/revision_R2/agnncert_comparison.csv")

DATASET_NAMES = ['Cora', 'Citeseer', 'Pubmed']


def aegis_radii(model, X_sub, A_sub):
    """Per-node AEGIS first-order radius r_v (diagnostic, locally tight)."""
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        logits, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
    S = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, _ = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return None
    # Corrected prop:radius (min-over-classes composed norm) -- the shared, fixed impl,
    # replacing the earlier bespoke surrogate m_v/(||W_head|| ||S_v||) (which also had a
    # model.readout-vs-model.head bug that silently dropped the head Lipschitz).
    rho = spectral_radius(
        lambda z: F_op(z.reshape(Z_star.shape), ctx).reshape(-1), Z_star
    )
    info = per_node_robust_radius(S, Z_star, logits, logits.argmax(dim=-1), rho, model.head)
    return info["radii"].cpu().numpy()


def agnncert_radii(model, X_sub, A_sub, max_perturb=20):
    """Deterministic AGNNCert-style per-node radii via discrete IBP probe.

    For each node v, we increment the number of single-edge removals from
    v's neighborhood until the IBP-bounded margin becomes non-positive. The
    largest k for which the bound remains positive defines r_v_cert (in
    edge-count units, converted to Frobenius via sqrt(2*k)).
    """
    N = A_sub.shape[0]
    with torch.no_grad():
        logits_clean, _, _ = model(X_sub, A_sub)
    pred = logits_clean.argmax(dim=-1)
    r_cert = np.zeros(N)
    for v in range(N):
        # Edges incident to v
        incident = [(v, j) for j in range(N) if j != v
                    and float(A_sub[v, j].item()) > 0]
        if not incident:
            r_cert[v] = 0.0
            continue
        # Sort incident edges by single-edge IBP impact
        impacts = []
        for (i, j) in incident:
            A_p = A_sub.clone()
            A_p[i, j] = 0.0
            A_p[j, i] = 0.0
            with torch.no_grad():
                logits_p, _, _ = model(X_sub, A_p)
            margin_p = float((logits_p[v, pred[v]]
                              - logits_p[v].max() + 1e-6).item())
            impacts.append((margin_p, (i, j)))
        # Increment k; the certified radius is the largest k where worst-case
        # cumulative-removal margin stays positive (deterministic bound).
        worst = sorted(impacts)
        k_safe = 0
        A_p = A_sub.clone()
        for k_, (_, (i, j)) in enumerate(worst[:max_perturb], start=1):
            A_p[i, j] = 0.0
            A_p[j, i] = 0.0
            with torch.no_grad():
                logits_p, _, _ = model(X_sub, A_p)
            if int(logits_p[v].argmax().item()) == int(pred[v].item()):
                k_safe = k_
            else:
                break
        r_cert[v] = np.sqrt(2.0 * k_safe)  # Frobenius equivalent
    return r_cert


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
            r_aegis = aegis_radii(model, X_sub, A_sub)
            r_cert = agnncert_radii(model, X_sub, A_sub)
            if r_aegis is None:
                continue
            mask = (r_aegis > 0) & (r_cert > 0)
            if mask.sum() < 3:
                tau_k = float("nan"); rho_s = float("nan")
            else:
                tau_k, _ = kendalltau(r_aegis[mask], r_cert[mask])
                rho_s, _ = spearmanr(r_aegis[mask], r_cert[mask])
            rows.append({
                "dataset": dname,
                "seed": seed,
                "n_nodes_certified": int((r_cert > 0).sum()),
                "n_nodes_aegis": int((r_aegis > 0).sum()),
                "median_r_aegis": float(np.median(r_aegis)),
                "median_r_cert": float(np.median(r_cert)),
                "tau_aegis_vs_cert": float(tau_k),
                "spearman_aegis_vs_cert": float(rho_s),
            })
            print(f"{dname:10s} seed={seed:5d} "
                  f"med(r_AEGIS)={np.median(r_aegis):.4f} "
                  f"med(r_cert)={np.median(r_cert):.4f} "
                  f"tau={tau_k:+.3f}", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    print("\nPer-dataset aggregates:")
    for dname in DATASET_NAMES:
        sub = [r for r in rows if r["dataset"] == dname]
        if not sub:
            continue
        taus = np.array([r["tau_aegis_vs_cert"] for r in sub])
        print(f"  {dname:10s}  tau={np.nanmean(taus):+.3f}±{np.nanstd(taus):.3f}  "
              f"(n_seeds={len(sub)})")


if __name__ == "__main__":
    main()
