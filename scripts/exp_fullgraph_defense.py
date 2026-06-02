"""Full-graph AEGIS defense-informed edge protection (matrix-free).

FULL-GRAPH analogue of scripts/revision_R2/R2_13_adaptive_defense.py. Instead of
a 50-node BFS ego-subgraph, this runs on the ENTIRE Cora graph (N=2708) via the
matrix-free ScalableSensitivity pipeline (Neumann + JVP/VJP, no O((Nd)^2)
materialisation).

Metric (full-graph spec): worst-case first-order amplification sigma_1(S_c) =
top singular value of the constrained sensitivity matrix S_c (existing-edge
subspace), obtained matrix-free via ScalableSensitivity.top_k_svd. The paper
uses max_first_order_shift = sigma_1 * epsilon, so sigma_1 is the natural
full-graph "damage" proxy. (R2_13's subgraph metric is the post-perturbation
re-converged displacement ||Z_pert - Z*|| along v_1; on the full graph the
sigma_1 surrogate is the matrix-free equivalent and is what the revision asks
for. Damage-reduction is defined identically: 1 - damage_masked / damage_clean.)

Procedure, for Cora, each of 10 seeds:
  1. Train IGNN (same config as R2_13 via _common.train_ignn).
  2. Full-graph per-edge vulnerability v_ij = ||S_c[:, ij]|| via
     ScalableSensitivity.edge_vulnerability(). sigma_1_clean via top_k_svd.
  3. For k in {5, 10}:
       - mask top-k edges by v_ij (zero symmetric entries in A_hat), rebuild
         ScalableSensitivity on the masked graph, recompute sigma_1 ->
         damage-reduction (this IS the ADAPTIVE attacker: best new direction).
       - same for RANDOM-k edges.
       - NON-ADAPTIVE attacker: reuse the CLEAN top right-singular direction
         v_1, restricted to surviving edges + renormalised, pushed through the
         masked operator (op_masked.matvec); damage = ||S_c^masked v_1||.
       - adaptive-vs-non-adaptive gap = adaptive_red - nonadaptive_red.
  4. Report mean+-sd damage-reduction over seeds for top-k and random, and the
     adaptive-vs-non-adaptive gap. Write CSV under results/.

Usage:
    .venv/bin/python scripts/exp_fullgraph_defense.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import SEEDS, load_dataset, train_ignn
from iem.scalable import ScalableSensitivity

OUT_CSV = Path("results/fullgraph_defense.csv")
K_LIST = [5, 10]
DATASET = "Cora"

# Randomized-SVD settings. SVD_SEED is fixed and re-applied before EVERY
# top_k_svd call so the random sketch is identical across clean/masked graphs;
# this removes estimator noise from the (tiny) adaptive-vs-non-adaptive gap.
SVD_SEED = 0
SVD_K = 6           # retrieve a few singular triplets; sigma_1 is sigma[0]
SVD_POWER_ITER = 7  # power iterations for the randomized range finder


def _build_op(model, X, A):
    """Forward to fixed point on adjacency A, return a fresh ScalableSensitivity."""
    def F_op(z, c):
        return model.operator(z, c)
    with torch.no_grad():
        _, Z_star, ctx = model(X, A)
    op = ScalableSensitivity(F_op, Z_star, ctx)
    return op


def _sigma1(op):
    """sigma_1(S_c) via deterministic randomized SVD (matrix-free)."""
    if op.num_edges == 0:
        return 0.0, None
    torch.manual_seed(SVD_SEED)
    _, sigma, Vh = op.top_k_svd(k=min(SVD_K, op.num_edges),
                                n_power_iter=SVD_POWER_ITER)
    return float(sigma[0]), Vh[0].detach()


def _mask_adjacency(A, edges):
    """Zero the given undirected edges (symmetric) in a clone of A."""
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
        model = train_ignn(X, A_hat, y, train_mask, n_features, n_classes,
                            device, seed)

        # --- clean full-graph sensitivity ---
        op_clean = _build_op(model, X, A_hat)
        n_edges = op_clean.num_edges
        sigma1_clean, v1_clean = _sigma1(op_clean)
        triples = op_clean.edge_vulnerability()  # (i, j, vuln), desc-sorted
        ranked_edges = [(i, j) for i, j, _ in triples]
        # clean edge -> index in op_clean.edge_list (canonical i<j order)
        clean_pos = {tuple(sorted(e)): idx
                     for idx, e in enumerate(op_clean.edge_list)}

        rng = np.random.RandomState(seed)
        for k in K_LIST:
            if k > n_edges:
                continue
            topk_edges = ranked_edges[:k]
            rand_ids = rng.choice(n_edges, size=k, replace=False)
            rand_edges = [op_clean.edge_list[int(t)] for t in rand_ids]

            # ---- TOP-k masking: rebuild op, sigma_1 = ADAPTIVE attacker ----
            A_top = _mask_adjacency(A_hat, topk_edges)
            op_top = _build_op(model, X, A_top)
            sigma1_top_adapt, _ = _sigma1(op_top)
            red_top_adapt = 1.0 - sigma1_top_adapt / max(sigma1_clean, 1e-12)

            # ---- NON-ADAPTIVE attacker on the top-k-masked graph ----
            # Reuse CLEAN v_1 (edge-space right singular vector), restricted to
            # surviving edges of op_top and renormalised, pushed through the
            # masked operator: damage = ||S_c^masked v_1_restricted||.
            v1_restr = torch.zeros(op_top.num_edges, device=device,
                                   dtype=v1_clean.dtype)
            for new_idx, e in enumerate(op_top.edge_list):
                key = tuple(sorted(e))
                ci = clean_pos.get(key)
                if ci is not None:
                    v1_restr[new_idx] = v1_clean[ci]
            nrm = v1_restr.norm()
            if nrm > 1e-12:
                v1_restr = v1_restr / nrm
                sigma1_top_nonadapt = float(op_top.matvec(v1_restr).norm().item())
            else:
                sigma1_top_nonadapt = 0.0
            red_top_nonadapt = 1.0 - sigma1_top_nonadapt / max(sigma1_clean, 1e-12)
            adaptive_gap = red_top_adapt - red_top_nonadapt

            # ---- RANDOM-k masking: rebuild op, sigma_1 ----
            A_rand = _mask_adjacency(A_hat, rand_edges)
            op_rand = _build_op(model, X, A_rand)
            sigma1_rand, _ = _sigma1(op_rand)
            red_rand = 1.0 - sigma1_rand / max(sigma1_clean, 1e-12)

            print(f"  seed={seed:5d} k={k:2d}  sigma1_clean={sigma1_clean:.4f}  "
                  f"top-v red(adapt)={red_top_adapt:+.1%}  "
                  f"top-v red(non-adapt)={red_top_nonadapt:+.1%}  "
                  f"gap={adaptive_gap*100:+.2f}pp  "
                  f"random red={red_rand:+.1%}", flush=True)

            rows.append({
                "dataset": DATASET,
                "seed": seed,
                "graph_mode": "FULL_GRAPH",
                "k": k,
                "n_edges_total": n_edges,
                "sigma1_clean": sigma1_clean,
                "sigma1_topk_adapt": sigma1_top_adapt,
                "sigma1_topk_nonadapt": sigma1_top_nonadapt,
                "sigma1_random": sigma1_rand,
                "damage_red_topk_adapt": red_top_adapt,
                "damage_red_topk_nonadapt": red_top_nonadapt,
                "damage_red_random": red_rand,
                "adaptive_gap_pp": adaptive_gap * 100.0,
                "elapsed_s": time.time() - t0,
            })

        del model, op_clean
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [seed {seed} done in {time.time()-t0:.0f}s, "
              f"|E|={n_edges}]", flush=True)

    if not rows:
        sys.exit("No rows produced.")

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # --- compact summary table ---
    def ms(xs):
        a = np.asarray(xs, dtype=float)
        return a.mean() * 100.0, a.std() * 100.0

    print("\n" + "=" * 74)
    print(f"FULL-GRAPH AEGIS edge protection  |  {DATASET} N=2708  |  "
          f"{len(SEEDS)} seeds  |  metric: sigma_1(S_c)")
    print("=" * 74)
    print(f"{'k':>3} | {'top-k red (adapt)':>22} | {'random-k red':>16} | "
          f"{'adapt-vs-nonadapt gap':>22}")
    print("-" * 74)
    for k in K_LIST:
        sub = [r for r in rows if r["k"] == k]
        if not sub:
            continue
        ta_m, ta_s = ms([r["damage_red_topk_adapt"] for r in sub])
        ra_m, ra_s = ms([r["damage_red_random"] for r in sub])
        gaps = np.asarray([r["adaptive_gap_pp"] for r in sub], dtype=float)
        print(f"{k:>3} | {ta_m:>9.1f} +- {ta_s:>5.1f} %    | "
              f"{ra_m:>6.1f} +- {ra_s:>4.1f} % | "
              f"{gaps.mean():>9.2f} +- {gaps.std():>5.2f} pp")
    print("=" * 74)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
