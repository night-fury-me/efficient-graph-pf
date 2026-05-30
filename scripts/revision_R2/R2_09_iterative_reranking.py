"""Revision-R2 P2.4 — iterative AEGIS re-ranking proof-of-concept.

Static AEGIS rankings reach only 54% of the greedy proxy on Cora at k=5
because they cannot capture cascade effects. This script implements a simple
iterative variant: after each edge removal, recompute S_c on the perturbed
graph and re-rank, closing some of the gap to greedy at modest extra cost.

Closes: P2.4 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_09_iterative_reranking.py
"""
from __future__ import annotations

import csv
import sys
import time
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

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)

SUBGRAPH_N = 50
K_LIST = [5, 10]
RECOMPUTE_EVERY = 1   # 1 = recompute after every removal; >1 = block update
OUT_CSV = Path("results/revision_R2/iterative_reranking.csv")

DATASET_NAMES = ['Cora', 'Citeseer', 'Pubmed']


def aegis_ranking_dense(model, X_sub, A_sub):
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(F_op, Z_star, ctx)
    S = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edges = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return [], np.array([])
    v = S_c.norm(dim=0).cpu().numpy()
    order = np.argsort(-v)
    return [edges[i] for i in order], v[order]


def damage(model, X, A0, A_p):
    with torch.no_grad():
        _, Z_clean, _ = model(X, A0)
        _, Z_pert, _ = model(X, A_p)
    return float((Z_pert - Z_clean).norm().item())


def static_aegis_topk(model, X_sub, A_sub, k):
    edges, _ = aegis_ranking_dense(model, X_sub, A_sub)
    A_p = A_sub.clone()
    for (i, j) in edges[:k]:
        A_p[i, j] = 0.0; A_p[j, i] = 0.0
    return damage(model, X_sub, A_sub, A_p)


def iterative_aegis_topk(model, X_sub, A_sub, k):
    """Recompute S_c after each removal."""
    A_p = A_sub.clone()
    removed = set()
    for step in range(k):
        edges, _ = aegis_ranking_dense(model, X_sub, A_p)
        # First edge not yet removed
        for (i, j) in edges:
            if (i, j) in removed:
                continue
            A_p[i, j] = 0.0; A_p[j, i] = 0.0
            removed.add((i, j))
            break
        else:
            break
    return damage(model, X_sub, A_sub, A_p)


def greedy_topk(model, X_sub, A_sub, k):
    """Brute-force sequential greedy (most-damaging single edge at each step)."""
    N = A_sub.shape[0]
    A_p = A_sub.clone()
    removed = set()
    for step in range(k):
        best, best_dmg = None, -1.0
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in removed or float(A_p[i, j].item()) <= 0:
                    continue
                A_try = A_p.clone()
                A_try[i, j] = 0.0; A_try[j, i] = 0.0
                d = damage(model, X_sub, A_sub, A_try)
                if d > best_dmg:
                    best_dmg = d; best = (i, j)
        if best is None:
            break
        A_p[best[0], best[1]] = 0.0; A_p[best[1], best[0]] = 0.0
        removed.add(best)
    return damage(model, X_sub, A_sub, A_p)


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
            X_sub, A_sub, Z_sub, ctx_sub, _ctx_full, _Z_full, idx = forward_and_subgraph(model, X, A_hat, max_nodes=SUBGRAPH_N)
            for k in K_LIST:
                d_static = static_aegis_topk(model, X_sub, A_sub, k)
                d_iter = iterative_aegis_topk(model, X_sub, A_sub, k)
                d_greedy = greedy_topk(model, X_sub, A_sub, k)
                close_pct = ((d_iter - d_static) / max(d_greedy - d_static, 1e-9)
                             * 100.0)
                rows.append({
                    "dataset": dname,
                    "seed": seed,
                    "k": k,
                    "damage_static_aegis": d_static,
                    "damage_iterative_aegis": d_iter,
                    "damage_greedy": d_greedy,
                    "ratio_static_to_greedy": d_static / max(d_greedy, 1e-9),
                    "ratio_iter_to_greedy": d_iter / max(d_greedy, 1e-9),
                    "iter_gap_closure_pct": close_pct,
                    "elapsed_s": time.time() - t0,
                })
            print(f"  {dname:10s} seed={seed:5d} k=5  "
                  f"static={rows[-2]['damage_static_aegis']:.3f}  "
                  f"iter={rows[-2]['damage_iterative_aegis']:.3f}  "
                  f"greedy={rows[-2]['damage_greedy']:.3f}  "
                  f"closure={rows[-2]['iter_gap_closure_pct']:.0f}%",
                  flush=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
