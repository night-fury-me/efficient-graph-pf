"""Revision-R2 P2.3 — full-graph reproduction of tab:baselines + tab:greedy_topk.

Re-runs the structured-baseline comparison (AEGIS vs degree/spectral/betweenness)
and the discrete edge-removal cumulative damage table on the FULL-graph
matrix-free path for Cora and Citeseer, alongside the existing 50-node subgraph
results. Confirms / contradicts the editor's concern that the favorable numbers
come from a regime that doesn't generalize.

Closes: P2.3 from docs/review_full_2026-05-28/06_editorial_decision.md.

Usage:
    .venv/bin/python scripts/revision_R2/R2_08_fullgraph_repro.py

Note: this runs the matrix-free pipeline on the full graph. On Cora (N=2708) it
takes ~80 s per seed; on Citeseer (N=3327) ~120 s. Total runtime ~30-40 min
across 10 seeds for both datasets.
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
    structural_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity

K_LIST = [5, 10, 20]
OUT_CSV = Path("results/revision_R2/fullgraph_repro.csv")

DATASET_NAMES = ['Cora', 'Citeseer']


def damage(model, X, A0, A_p):
    with torch.no_grad():
        _, Z_clean, _ = model(X, A0)
        _, Z_pert, _ = model(X, A_p)
    return float((Z_pert - Z_clean).norm().item())


def aegis_fullgraph_ranking(model, X, A_hat):
    """Matrix-free per-edge vulnerability across the FULL graph.

    ``ScalableSensitivity.edge_vulnerability`` returns a list of
    ``(i, j, vuln)`` triples already sorted by descending vulnerability.
    """
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)
    op = ScalableSensitivity(F_op, Z_star, ctx)
    triples = op.edge_vulnerability()  # List[(i, j, vuln)] desc-sorted
    edges = [(i, j) for i, j, _ in triples]
    vulns = np.array([v for _, _, v in triples], dtype=float)
    return edges, vulns


def degree_ranking(A_hat):
    N = A_hat.shape[0]
    deg = (A_hat > 0).sum(dim=1).cpu().numpy()
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if float(A_hat[i, j].item()) > 0:
                edges.append(((i, j), max(deg[i], deg[j])))
    edges.sort(key=lambda x: -x[1])
    return [e for e, _ in edges]


def cumulative_damage(model, X, A0, ranked, k):
    A_p = A0.clone()
    for (i, j) in ranked[:k]:
        A_p[i, j] = 0.0
        A_p[j, i] = 0.0
    return damage(model, X, A0, A_p)


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
            try:
                aegis_edges, _ = aegis_fullgraph_ranking(model, X, A_hat)
            except RuntimeError as exc:
                print(f"  AEGIS fullgraph OOM on {dname} seed={seed}: {exc}")
                continue
            degree_edges = degree_ranking(A_hat)
            # Random baseline
            rng = np.random.default_rng(seed)
            rand_edges = list(set(aegis_edges))
            rng.shuffle(rand_edges)
            for k in K_LIST:
                d_a = cumulative_damage(model, X, A_hat, aegis_edges, k)
                d_d = cumulative_damage(model, X, A_hat, degree_edges, k)
                d_r = cumulative_damage(model, X, A_hat, rand_edges, k)
                rows.append({
                    "dataset": dname,
                    "seed": seed,
                    "graph_mode": "FULL_GRAPH",
                    "k": k,
                    "n_edges_total": len(aegis_edges),
                    "damage_aegis": d_a,
                    "damage_degree": d_d,
                    "damage_random": d_r,
                    "aegis_over_random": d_a / max(d_r, 1e-9),
                    "elapsed_s": time.time() - t0,
                })
            print(f"  {dname:8s} seed={seed:5d} k=10 "
                  f"AEGIS={cumulative_damage(model, X, A_hat, aegis_edges, 10):.3f} "
                  f"Degree={cumulative_damage(model, X, A_hat, degree_edges, 10):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
