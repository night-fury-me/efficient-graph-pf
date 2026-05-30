"""Revision-R2 G6 — Adaptive-attacker defense ablation.

For Cora IGNN, 50-node BFS subgraphs, 10 seeds, budget eps=0.10:
  (i)  non-adaptive: mask top-k edges by v_{ij}, apply ORIGINAL S_c's v_1
       projected onto the kept-edge subspace; measure damage.
  (ii) adaptive: mask top-k edges, recompute S_c on the masked graph,
       use its NEW v_1; measure damage.
Report damage-reduction erosion adaptive-vs-nonadaptive.

Usage: .venv/bin/python scripts/revision_R2/R2_13_adaptive_defense.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

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

OUT_CSV = Path("results/revision_R2/adaptive_defense.csv")
K_LIST = [5, 10]
EPSILON = 0.10
DATASET = "Cora"
SUBGRAPH_N = 50


def aegis_sc(model, X_sub, A_sub):
    """Compute S_c, edge_list, v_ij scores, and v_1 SVD direction on subgraph."""
    with torch.no_grad():
        _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_star, ctx)
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if S_c.shape[1] == 0:
        return None, None, None, None, None, None
    v_ij = S_c.norm(dim=0).cpu().numpy()
    _, sig, Vt = torch.linalg.svd(S_c, full_matrices=False)
    v1 = Vt[0]
    return Z_star, ctx, S_c, edge_list, v_ij, v1


def apply_edge_perturb(model, Z_star, ctx, A_sub, edge_list, delta_vec, eps):
    """delta_vec is in |E| basis with constrained edge_basis b_k=(e_ie_j^T+e_je_i^T)/sqrt(2)."""
    dA = torch.zeros_like(A_sub)
    sqrt2 = (2.0) ** 0.5
    for (i, j), d in zip(edge_list, delta_vec.cpu().numpy()):
        dA[i, j] += d / sqrt2
        dA[j, i] += d / sqrt2
    nrm = dA.norm()
    if nrm < 1e-12:
        return float((Z_star - Z_star).norm())
    dA = eps * dA / nrm
    A_pert = A_sub + dA
    ctx_pert = {**ctx, "A_hat": A_pert}
    Z_pert = reconverge(model, Z_star.clone(), ctx_pert)
    return float((Z_pert - Z_star).norm())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    X, A_hat, y, train_mask, n_features, n_classes = load_dataset(DATASET)
    X, A_hat, y = X.to(device), A_hat.to(device), y.to(device)
    train_mask = train_mask.to(device)
    rows = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes,
                            device, seed)
        # Extract BFS subgraph
        idx = extract_ego_subgraph(A_hat, max_nodes=SUBGRAPH_N)
        A_sub = A_hat[idx][:, idx]
        X_sub = X[idx]
        out = aegis_sc(model, X_sub, A_sub)
        if out[0] is None:
            print(f"  skip {DATASET} seed={seed}: empty S_c")
            continue
        Z_star, ctx, S_c_clean, edge_list, v_ij, v1_clean = out
        n_edges = len(edge_list)
        # Clean (no mask) baseline using v_1 direction
        dmg_clean = apply_edge_perturb(model, Z_star, ctx, A_sub,
                                        edge_list, v1_clean, EPSILON)
        rng = np.random.RandomState(seed)
        for k in K_LIST:
            if k > n_edges: continue
            topk_v = np.argsort(-v_ij)[:k]
            topk_rand = rng.choice(n_edges, size=k, replace=False)
            # Mask top-k by zeroing in A_sub
            def mask(A, idxs):
                Am = A.clone()
                for ei in idxs:
                    i, j = edge_list[int(ei)]
                    Am[i, j] = 0.0; Am[j, i] = 0.0
                return Am
            A_v = mask(A_sub, topk_v)
            A_r = mask(A_sub, topk_rand)
            # --- non-adaptive: use ORIGINAL v1 projected to kept edges
            kept_v = np.setdiff1d(np.arange(n_edges), topk_v)
            v1_proj = torch.zeros_like(v1_clean)
            v1_proj[kept_v] = v1_clean[kept_v]
            if v1_proj.norm() > 1e-12:
                v1_proj = v1_proj / v1_proj.norm()
            ctx_v = {**ctx, "A_hat": A_v}
            Z_v = reconverge(model, Z_star.clone(), ctx_v)
            dmg_na = apply_edge_perturb(model, Z_v, ctx_v, A_v,
                                         edge_list, v1_proj, EPSILON)
            # --- adaptive: recompute S_c on masked graph, get new v_1
            out_adapt = aegis_sc(model, X_sub, A_v)
            if out_adapt[0] is None:
                dmg_adapt = float("nan")
            else:
                _, _, S_c_m, edge_list_m, _, v1_m_short = out_adapt
                v1_adapt = torch.zeros_like(v1_clean)
                pos = {tuple(sorted(e)): i for i, e in enumerate(edge_list)}
                for ei_m, em in enumerate(edge_list_m):
                    key = tuple(sorted(em))
                    if key in pos:
                        v1_adapt[pos[key]] = v1_m_short[ei_m]
                if v1_adapt.norm() > 1e-12:
                    v1_adapt = v1_adapt / v1_adapt.norm()
                dmg_adapt = apply_edge_perturb(model, Z_v, ctx_v, A_v,
                                                edge_list, v1_adapt, EPSILON)
            # Random-mask non-adaptive baseline
            kept_rand = np.setdiff1d(np.arange(n_edges), topk_rand)
            v1_proj_r = torch.zeros_like(v1_clean)
            v1_proj_r[kept_rand] = v1_clean[kept_rand]
            if v1_proj_r.norm() > 1e-12:
                v1_proj_r = v1_proj_r / v1_proj_r.norm()
            ctx_r = {**ctx, "A_hat": A_r}
            Z_r = reconverge(model, Z_star.clone(), ctx_r)
            dmg_rand_na = apply_edge_perturb(model, Z_r, ctx_r, A_r,
                                              edge_list, v1_proj_r, EPSILON)
            rd_na = 1.0 - dmg_na / max(dmg_clean, 1e-12)
            rd_adapt = 1.0 - dmg_adapt / max(dmg_clean, 1e-12)
            rd_rand = 1.0 - dmg_rand_na / max(dmg_clean, 1e-12)
            print(f"  seed={seed:5d} k={k:2d}  clean={dmg_clean:.4f}  "
                  f"top-v NA red={rd_na:+.1%}  adapt red={rd_adapt:+.1%}  "
                  f"random NA red={rd_rand:+.1%}", flush=True)
            rows.append({
                "dataset": DATASET, "seed": seed, "k": k, "epsilon": EPSILON,
                "n_edges": n_edges,
                "damage_clean_no_mask": dmg_clean,
                "damage_topv_nonadapt": dmg_na,
                "damage_topv_adapt": dmg_adapt,
                "damage_random_nonadapt": dmg_rand_na,
                "damage_red_topv_nonadapt": rd_na,
                "damage_red_topv_adapt": rd_adapt,
                "damage_red_random_nonadapt": rd_rand,
            })
        # cleanup
        del model
        torch.cuda.empty_cache()
    if not rows:
        sys.exit("No rows produced.")
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    for k in K_LIST:
        sub = [r for r in rows if r["k"] == k]
        na = np.array([r["damage_red_topv_nonadapt"] for r in sub])
        ad = np.array([r["damage_red_topv_adapt"] for r in sub
                        if not np.isnan(r["damage_red_topv_adapt"])])
        ra = np.array([r["damage_red_random_nonadapt"] for r in sub])
        print(f"  k={k}: non-adapt top-v {na.mean():+.1%}±{na.std():.1%}, "
              f"adapt top-v {ad.mean():+.1%}±{ad.std():.1%}, "
              f"random NA {ra.mean():+.1%}±{ra.std():.1%}")


if __name__ == "__main__":
    main()
