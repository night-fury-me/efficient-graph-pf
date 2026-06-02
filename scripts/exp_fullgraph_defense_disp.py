"""Full-graph defense with R2_13's EXACT nonlinear-displacement metric.

Closes the metric caveat: exp_fullgraph_defense.py used sigma_1(S_c); R2_13
(subgraph, the 42%/61% headline) uses the reconverged displacement
||Z_pert - Z*|| under the v_1-direction perturbation at eps=0.10
(apply_edge_perturb). This re-runs the FULL Cora graph with R2_13's EXACT
metric to confirm the dilution is a graph-scale effect, not a
sigma_1-vs-displacement artifact.

v_1 / v_ij are computed matrix-free via the (fixed) ScalableSensitivity on the
full graph; the displacement itself is the dense reconverge under the
v_1-direction perturbation, byte-for-byte the R2_13 procedure.

Usage: .venv/bin/python scripts/exp_fullgraph_defense_disp.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import SEEDS, load_dataset, train_ignn, reconverge
from iem.scalable import ScalableSensitivity

OUT_CSV = Path("results/fullgraph_defense_disp.csv")
K_LIST = [5, 10]
EPSILON = 0.10
DATASET = "Cora"
SVD_SEED, SVD_K, SVD_POWER_ITER = 0, 6, 7


def build_op(model, X, A):
    with torch.no_grad():
        _, Z_star, ctx = model(X, A)
    op = ScalableSensitivity(lambda z, c: model.operator(z, c), Z_star, ctx)
    return op, Z_star, ctx


def v1_of(op):
    torch.manual_seed(SVD_SEED)
    _, _, Vh = op.top_k_svd(k=min(SVD_K, op.num_edges), n_power_iter=SVD_POWER_ITER)
    return Vh[0].detach()


def apply_edge_perturb(model, Z_star, ctx, A, edge_list, delta_vec, eps):
    """R2_13's exact metric: dA = eps*sym(delta_vec) in the |E| edge basis
    (b_k=(e_i e_j^T + e_j e_i^T)/sqrt2), reconverge, return ||Z_pert - Z*||."""
    dA = torch.zeros_like(A)
    sqrt2 = 2.0 ** 0.5
    dv = delta_vec.detach().cpu().numpy()
    for (i, j), d in zip(edge_list, dv):
        dA[i, j] += d / sqrt2
        dA[j, i] += d / sqrt2
    nrm = dA.norm()
    if nrm < 1e-12:
        return 0.0
    dA = eps * dA / nrm
    ctx_pert = {**ctx, "A_hat": A + dA}
    Z_pert = reconverge(model, Z_star.clone(), ctx_pert)
    return float((Z_pert - Z_star).norm())


def mask_adj(A, edges):
    Am = A.clone()
    for (i, j) in edges:
        Am[i, j] = 0.0
        Am[j, i] = 0.0
    return Am


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    X, A_hat, y, train_mask, n_features, n_classes = load_dataset(DATASET)
    X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
    train_mask = train_mask.to(device)

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes, device, seed)

        op, Z_star, ctx = build_op(model, X, A_hat)
        edge_list = op.edge_list
        n_edges = op.num_edges
        pos = {tuple(sorted(e)): idx for idx, e in enumerate(edge_list)}
        v1_clean = v1_of(op)
        ranked = [(i, j) for i, j, _ in op.edge_vulnerability()]  # desc by v_ij

        dmg_clean = apply_edge_perturb(model, Z_star, ctx, A_hat, edge_list, v1_clean, EPSILON)

        rng = np.random.RandomState(seed)
        for k in K_LIST:
            if k > n_edges:
                continue
            topk = ranked[:k]
            topk_idx = [pos[tuple(sorted(e))] for e in topk]
            rand_ids = rng.choice(n_edges, size=k, replace=False)
            rand_edges = [edge_list[int(t)] for t in rand_ids]

            # top-k masked graph + reconverged equilibrium
            A_v = mask_adj(A_hat, topk)
            ctx_v = {**ctx, "A_hat": A_v}
            Z_v = reconverge(model, Z_star.clone(), ctx_v)

            # non-adaptive: clean v_1 projected onto kept edges
            v1_na = v1_clean.clone()
            for ii in topk_idx:
                v1_na[ii] = 0.0
            if v1_na.norm() > 1e-12:
                v1_na = v1_na / v1_na.norm()
            dmg_na = apply_edge_perturb(model, Z_v, ctx_v, A_v, edge_list, v1_na, EPSILON)

            # adaptive: recompute v_1 on the masked graph, map to clean basis
            op_v, _, _ = build_op(model, X, A_v)
            v1_m = v1_of(op_v)
            v1_ad = torch.zeros_like(v1_clean)
            for ei_m, em in enumerate(op_v.edge_list):
                key = tuple(sorted(em))
                if key in pos:
                    v1_ad[pos[key]] = v1_m[ei_m]
            if v1_ad.norm() > 1e-12:
                v1_ad = v1_ad / v1_ad.norm()
            dmg_ad = apply_edge_perturb(model, Z_v, ctx_v, A_v, edge_list, v1_ad, EPSILON)

            # random-k masked graph (non-adaptive)
            A_r = mask_adj(A_hat, rand_edges)
            ctx_r = {**ctx, "A_hat": A_r}
            Z_r = reconverge(model, Z_star.clone(), ctx_r)
            v1_r = v1_clean.clone()
            for t in rand_ids:
                v1_r[int(t)] = 0.0
            if v1_r.norm() > 1e-12:
                v1_r = v1_r / v1_r.norm()
            dmg_r = apply_edge_perturb(model, Z_r, ctx_r, A_r, edge_list, v1_r, EPSILON)

            rd_na = 1.0 - dmg_na / max(dmg_clean, 1e-12)
            rd_ad = 1.0 - dmg_ad / max(dmg_clean, 1e-12)
            rd_r = 1.0 - dmg_r / max(dmg_clean, 1e-12)
            print(f"  seed={seed:5d} k={k:2d} clean={dmg_clean:.4f}  "
                  f"top-v NA={rd_na:+.1%}  adapt={rd_ad:+.1%}  random={rd_r:+.1%}", flush=True)
            rows.append({"seed": seed, "k": k, "n_edges": n_edges, "dmg_clean": dmg_clean,
                         "rd_topv_nonadapt": rd_na, "rd_topv_adapt": rd_ad, "rd_random": rd_r,
                         "gap_pp": (rd_ad - rd_na) * 100})

        del model, op
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [seed {seed} done {time.time()-t0:.0f}s, |E|={n_edges}]", flush=True)

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("\n=== FULL-GRAPH, R2_13 DISPLACEMENT metric (eps=0.10), Cora N=2708, 10 seeds ===")
    for k in K_LIST:
        sub = [r for r in rows if r["k"] == k]
        na = np.array([r["rd_topv_nonadapt"] for r in sub])
        ad = np.array([r["rd_topv_adapt"] for r in sub])
        ra = np.array([r["rd_random"] for r in sub])
        gp = np.array([r["gap_pp"] for r in sub])
        print(f"  k={k:2d}: top-v NA {na.mean():+.1%}+-{na.std():.1%} | "
              f"adapt {ad.mean():+.1%}+-{ad.std():.1%} | random {ra.mean():+.1%}+-{ra.std():.1%} | "
              f"adapt-NA gap {gp.mean():+.2f}pp")
    print(f"Wrote {OUT_CSV}  (subgraph R2_13 ref: 42%/61% top-v, 11%/18% random)")


if __name__ == "__main__":
    main()
