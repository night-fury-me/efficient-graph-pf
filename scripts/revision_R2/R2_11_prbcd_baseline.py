"""Revision-R2 G4 — PR-BCD baseline on Pubmed (50-node subgraph).

Compares PR-BCD's per-edge attack probability P_ij against AEGIS's v_ij,
plus discrete damage at k in {1,5,10} edges removed.

PR-BCD: continuous P_ij in [0,1] on candidate edges, sign-step PGD with
simplex projection (sum P <= budget) on the equilibrium-shift loss.

(Amazon Photo full-graph PR-BCD is omitted — pre-OGB-scale BCD runtime
is hours; see exp_amazon_fullgraph.py for the AEGIS-only full-graph τ.)

Usage: .venv/bin/python scripts/revision_R2/R2_11_prbcd_baseline.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.revision_R2._common import (
    SEEDS, load_dataset, train_ignn, reconverge,
)
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)

OUT_CSV = Path("results/revision_R2/prbcd_baseline.csv")
K_LIST = [1, 5, 10]
N_BUDGET = 50
N_PRBCD_ITERS = 100
LR_PRBCD = 0.05
SUBGRAPH_N = 50
DATASETS = ["Pubmed"]


def project_simplex(P, budget):
    P = torch.clamp(P, 0.0, 1.0)
    s = P.sum()
    if s.item() > budget:
        P = P * (budget / s.item())
    return P


def attack_loss(model, Z_star, ctx, A_sub, edge_list, P):
    dA = torch.zeros_like(A_sub)
    for ei, (i, j) in enumerate(edge_list):
        dA[i, j] = dA[i, j] - P[ei] * A_sub[i, j]
        dA[j, i] = dA[j, i] - P[ei] * A_sub[j, i]
    A_pert = A_sub + dA
    ctx_pert = {**ctx, "A_hat": A_pert}
    Z_new = model.operator(Z_star, ctx_pert)
    return (Z_new - Z_star).norm()


def prbcd(model, Z_star, ctx, A_sub, edge_list, budget, n_iters, lr):
    n_edges = len(edge_list)
    P = torch.full((n_edges,), budget / n_edges, device=A_sub.device,
                   dtype=A_sub.dtype, requires_grad=True)
    for it in range(n_iters):
        loss = attack_loss(model, Z_star, ctx, A_sub, edge_list, P)
        grad = torch.autograd.grad(loss, P)[0]
        with torch.no_grad():
            P_new = P + lr * grad.sign()
            P_new = project_simplex(P_new, budget)
        P = P_new.detach().requires_grad_(True)
    return P.detach()


def discrete_damage(model, Z_star, ctx, A_sub, edge_list, ranking, k):
    top = np.argsort(-ranking)[:k]
    A_p = A_sub.clone()
    for ei in top:
        i, j = edge_list[int(ei)]
        A_p[i, j] = 0.0; A_p[j, i] = 0.0
    ctx_p = {**ctx, "A_hat": A_p}
    Z_p = reconverge(model, Z_star.clone(), ctx_p)
    return float((Z_p - Z_star).norm())


def aegis_v(model, X_sub, A_sub):
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_star, ctx)
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return None, None, None, None
    return Z_star, ctx, S_c.norm(dim=0).cpu().numpy(), edge_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in DATASETS:
        try:
            X, A_hat, y, train_mask, n_features, n_classes = load_dataset(name)
        except Exception as exc:
            print(f"[skip {name}] {exc}")
            continue
        X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
        train_mask = train_mask.to(device)
        for seed in SEEDS:
            try:
                torch.manual_seed(seed); np.random.seed(seed)
                model = train_ignn(X, A_hat, y, train_mask,
                                    n_features, n_classes, device, seed)
                idx = extract_ego_subgraph(A_hat, max_nodes=SUBGRAPH_N)
                A_sub = A_hat[idx][:, idx]
                X_sub = X[idx]
                out = aegis_v(model, X_sub, A_sub)
                if out[0] is None:
                    print(f"  {name} seed={seed} empty S_c, skip"); continue
                Z_star, ctx, v_ij, edge_list = out
                n_edges = len(edge_list)
                budget = min(N_BUDGET, n_edges)
                t0 = time.time()
                P = prbcd(model, Z_star, ctx, A_sub, edge_list, budget,
                          n_iters=N_PRBCD_ITERS, lr=LR_PRBCD)
                t_p = time.time() - t0
                P_np = P.cpu().numpy()
                tau, _ = (kendalltau(v_ij, P_np) if n_edges >= 3
                          else (float("nan"), None))
                for k in K_LIST:
                    if k > n_edges: continue
                    d_a = discrete_damage(model, Z_star, ctx, A_sub,
                                           edge_list, v_ij, k)
                    d_p = discrete_damage(model, Z_star, ctx, A_sub,
                                           edge_list, P_np, k)
                    rows.append({
                        "dataset": name, "seed": seed, "k": k,
                        "n_edges": n_edges,
                        "damage_aegis": d_a, "damage_prbcd": d_p,
                        "aegis_over_prbcd": d_a / max(d_p, 1e-12),
                        "tau_aegis_vs_prbcd": tau, "t_prbcd_s": t_p,
                    })
                    print(f"  {name} seed={seed:5d} k={k:2d} "
                          f"AEGIS={d_a:.4f} PR-BCD={d_p:.4f} "
                          f"AEGIS/PR-BCD={d_a/max(d_p,1e-12):.3f} "
                          f"tau={tau:+.3f} t_PR-BCD={t_p:.1f}s", flush=True)
                del model
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[err {name} seed={seed}] {exc}")
    if not rows:
        sys.exit("No rows produced.")
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    for name in DATASETS:
        for k in K_LIST:
            sub = [r for r in rows if r["dataset"] == name and r["k"] == k]
            if not sub: continue
            ratios = np.array([r["aegis_over_prbcd"] for r in sub])
            taus = np.array([r["tau_aegis_vs_prbcd"] for r in sub
                              if np.isfinite(r["tau_aegis_vs_prbcd"])])
            print(f"  {name} k={k}: AEGIS/PR-BCD = "
                  f"{ratios.mean():.3f}±{ratios.std():.3f}, "
                  f"tau = {taus.mean():+.3f}±{taus.std():.3f}")


if __name__ == "__main__":
    main()
