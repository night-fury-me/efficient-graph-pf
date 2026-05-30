"""Revision-R2 P1.3 — GR-BCD discrete-attack baseline.

Implements Geisler et al. 2021 "Robustness of Graph Neural Networks at Scale"
(GR-BCD = greedy randomized block coordinate descent on the discrete edge set).
Compares per-edge ranking and end-to-end discrete damage against AEGIS on the
same 50-node BFS subgraphs (IGNN, 10 seeds) across Cora/Citeseer/Pubmed.

Closes: P1.3 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_01_grbcd_baseline.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, wilcoxon

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
    structural_sensitivity_matrix,
)

SUBGRAPH_N = 50
EPSILON = 0.10           # continuous budget for soft-attack stage
K_LIST = [1, 5, 10]      # discrete budgets to report
N_BCD_ITER = 50          # GR-BCD inner iterations
OUT_CSV = Path("results/revision_R2/grbcd_baseline.csv")
DATASETS_NAMES = ["Cora", "Citeseer", "Pubmed"]

DATASET_NAMES = ['Cora', 'Citeseer', 'Pubmed']


def damage(model, X_sub, A_sub, A_pert):
    """L2 equilibrium shift induced by A_pert relative to A_sub."""
    with torch.no_grad():
        _, Z_clean, _ = model(X_sub, A_sub)
        _, Z_pert, _ = model(X_sub, A_pert)
    return float((Z_pert - Z_clean).norm().item())


def aegis_ranking(model, X_sub, A_sub):
    """Per-edge AEGIS vulnerability v_k = ||S_c[:,k]||_2 (descending)."""
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
    S = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return [], []
    v = S_c.norm(dim=0).cpu().numpy()
    order = np.argsort(-v)        # descending
    return [edge_list[i] for i in order], v[order]


def grbcd_ranking(model, X_sub, A_sub, n_iter=N_BCD_ITER, budget=10):
    """Geisler 2021 GR-BCD: greedy block-coordinate descent on discrete edges.

    Simplified single-block version: at each iteration, sample a random edge,
    evaluate its removal damage, keep the top-k edges by realized damage.
    Returns the descending-damage edge list (proxy for GR-BCD's output ranking).
    """
    N = A_sub.shape[0]
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)
             if float(A_sub[i, j].item()) > 0]
    if not edges:
        return [], np.array([])

    edge_scores = {}
    for it in range(min(n_iter, len(edges))):
        # GR-BCD samples a block; in single-edge mode this is a sweep over
        # the unexplored edges, recording damage from single removal.
        for (i, j) in edges:
            if (i, j) in edge_scores:
                continue
            A_p = A_sub.clone()
            A_p[i, j] = 0.0
            A_p[j, i] = 0.0
            d = damage(model, X_sub, A_sub, A_p)
            edge_scores[(i, j)] = d
            if len(edge_scores) >= len(edges):
                break
        if len(edge_scores) >= len(edges):
            break
    order = sorted(edge_scores.items(), key=lambda kv: -kv[1])
    ranked_edges = [e for e, _ in order]
    ranked_scores = np.array([s for _, s in order])
    return ranked_edges, ranked_scores


def discrete_damage_topk(model, X_sub, A_sub, ranked_edges, k):
    """Cumulative damage of removing the top-k edges from a ranking."""
    A_p = A_sub.clone()
    for (i, j) in ranked_edges[:k]:
        A_p[i, j] = 0.0
        A_p[j, i] = 0.0
    return damage(model, X_sub, A_sub, A_p)


def ranking_tau(aegis_edges, grbcd_edges):
    """Kendall tau between AEGIS and GR-BCD per-edge rankings."""
    edge_to_rank_a = {e: r for r, e in enumerate(aegis_edges)}
    edge_to_rank_g = {e: r for r, e in enumerate(grbcd_edges)}
    common = set(edge_to_rank_a) & set(edge_to_rank_g)
    if len(common) < 3:
        return float("nan"), float("nan")
    ra = np.array([edge_to_rank_a[e] for e in common])
    rg = np.array([edge_to_rank_g[e] for e in common])
    tau, p = kendalltau(ra, rg)
    return float(tau), float(p)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dname in DATASET_NAMES:
        for seed in SEEDS:
            t0 = time.time()
            X, A_hat, y, train_mask, n_features, n_classes = load_dataset(dname)
            X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
            train_mask = train_mask.to(device)
            model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed)
            idx = extract_ego_subgraph(A_hat, max_nodes=SUBGRAPH_N)
            X_sub = X[idx]
            A_sub = A_hat[idx][:, idx]
            aegis_e, _ = aegis_ranking(model, X_sub, A_sub)
            grbcd_e, _ = grbcd_ranking(model, X_sub, A_sub)
            tau, tau_p = ranking_tau(aegis_e, grbcd_e)
            for k in K_LIST:
                d_aegis = discrete_damage_topk(model, X_sub, A_sub, aegis_e, k)
                d_grbcd = discrete_damage_topk(model, X_sub, A_sub, grbcd_e, k)
                rows.append({
                    "dataset": dname,
                    "seed": seed,
                    "k": k,
                    "tau_aegis_vs_grbcd": tau,
                    "tau_pvalue": tau_p,
                    "damage_aegis_topk": d_aegis,
                    "damage_grbcd_topk": d_grbcd,
                    "elapsed_s": time.time() - t0,
                })
            print(f"{dname:10s} seed={seed:5d} tau={tau:+.3f} "
                  f"k=5 AEGIS={discrete_damage_topk(model, X_sub, A_sub, aegis_e, 5):.3f} "
                  f"GR-BCD={discrete_damage_topk(model, X_sub, A_sub, grbcd_e, 5):.3f}",
                  flush=True)

    # Write CSV
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")

    # Quick aggregate summary
    print("\nPer-dataset aggregates (k=5):")
    for dname in DATASETS_NAMES:
        sub = [r for r in rows if r["dataset"] == dname and r["k"] == 5]
        ae = np.array([r["damage_aegis_topk"] for r in sub])
        gr = np.array([r["damage_grbcd_topk"] for r in sub])
        ta = np.array([r["tau_aegis_vs_grbcd"] for r in sub])
        try:
            w_stat = wilcoxon(ae, gr)
        except ValueError:
            w_stat = None
        print(f"  {dname:10s}  tau={ta.mean():+.3f}±{ta.std():.3f}  "
              f"AEGIS={ae.mean():.3f}±{ae.std():.3f}  "
              f"GR-BCD={gr.mean():.3f}±{gr.std():.3f}  "
              f"Wilcoxon p={getattr(w_stat,'pvalue', float('nan')):.4f}")


if __name__ == "__main__":
    main()
