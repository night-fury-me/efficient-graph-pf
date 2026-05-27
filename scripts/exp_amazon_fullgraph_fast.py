"""C2 follow-up (fast): Full-graph vs subgraph analysis on Amazon Photo.

Instead of computing all 119K edge vulnerabilities (hours), this:
1. Uses SVD top-k to get per-edge weights from leading singular vectors
2. Samples 300 edges for column-level vulnerability + discrete ground truth
3. Computes τ between SVD-based ranking and discrete damage

Usage:
    .venv/bin/python scripts/exp_amazon_fullgraph_fast.py
"""

from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.examples.ignn_cora import IGNN
from iem.examples.ignn_amazon import _load_amazon
from iem.adversarial import (
    _compute_structural_jacobian, constrained_sensitivity_matrix,
    extract_ego_subgraph, greedy_structural_attack,
    structural_sensitivity_matrix,
)
from iem.scalable import ScalableSensitivity

SEEDS = [42, 137, 271]
N_SAMPLE = 300


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def reconverge(model, Z, ctx, max_iter=200):
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7: break
            Z = Z_new
    return Z_new


def train_ignn(data, device, seed):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        lo, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(lo[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad(); loss.backward(); optim.step()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = model(X, A_hat)
                va = float((lv.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()
    return model


def run_single(data, seed, device):
    set_seed(seed)
    model = train_ignn(data, device, seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    N = A_hat.shape[0]
    A_hat_sn = float(torch.linalg.svdvals(A_hat)[0])
    W_sn = float(torch.linalg.svdvals(model.W.weight.detach())[0])
    kappa_full = A_hat_sn * W_sn

    # ---- Full-graph: SVD-based ranking ----
    print("  SVD (matrix-free)...", end="", flush=True)
    t0 = time.time()
    op = ScalableSensitivity(
        lambda z, c: model.operator(z, c),
        Z_star, ctx, neumann_tol=1e-5,
    )
    rho_full = op.rho
    U, sigma, Vh = op.top_k_svd(k=6)
    t_svd = time.time() - t0
    print(f" {t_svd:.0f}s, rho={rho_full:.3f}", flush=True)

    # Per-edge vulnerability from leading singular vector
    # |Vh[0][k]| = edge k's contribution to the optimal attack direction
    svd_scores = Vh[0].abs().cpu().numpy()
    n_edges = op.num_edges
    edge_list = op.edge_list

    # Sample edges for discrete ground truth
    rng = random.Random(seed)
    sample_idx = rng.sample(range(n_edges), min(N_SAMPLE, n_edges))
    sample_edges = [edge_list[k] for k in sample_idx]
    sample_svd = svd_scores[sample_idx]

    # Also compute column-based vulnerability for sampled edges
    print(f"  columns for {len(sample_idx)} edges...", end="", flush=True)
    t0 = time.time()
    sample_col_vuln = []
    for idx in sample_idx:
        col = op._column(idx)
        sample_col_vuln.append(float(col.norm().item()))
    sample_col_vuln = np.array(sample_col_vuln)
    t_col = time.time() - t0
    print(f" {t_col:.0f}s", flush=True)

    # Brute-force discrete damage on sampled edges
    print(f"  brute-force discrete...", end="", flush=True)
    t0 = time.time()
    discrete_scores = []
    with torch.no_grad():
        for i, j in sample_edges:
            A_pert = A_hat.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx, "A_hat": A_pert}
            Z_pert = reconverge(model, Z_star, ctx_pert)
            discrete_scores.append(float((Z_pert - Z_star).norm()))
    discrete_scores = np.array(discrete_scores)
    t_bf = time.time() - t0
    print(f" {t_bf:.0f}s", flush=True)

    # Tau: SVD-based vs discrete
    tau_svd, _ = kendalltau(sample_svd, discrete_scores)
    # Tau: column-based vs discrete
    tau_col, _ = kendalltau(sample_col_vuln, discrete_scores)
    # Tau: SVD vs column (sanity)
    tau_svd_col, _ = kendalltau(sample_svd, sample_col_vuln)

    # ---- Subgraph (50-node) for comparison ----
    print("  subgraph (50-node)...", end="", flush=True)
    idx_sub = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx_sub][:, idx_sub]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx_sub]}
    Z_sub = reconverge(model, Z_star[idx_sub].clone(), ctx_sub)
    A_sub_sn = float(torch.linalg.svdvals(A_sub)[0])
    kappa_sub = A_sub_sn * W_sn

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, el_sub = constrained_sensitivity_matrix(S, A_sub)
    sc_scores = [float(S_c[:, k].norm()) for k in range(len(el_sub))]
    bf_sub = greedy_structural_attack(model, Z_sub, ctx_sub)
    bf_dict = {(min(i,j), max(i,j)): s for i,j,s in bf_sub}
    bf_matched = [bf_dict.get((min(i,j), max(i,j)), 0.0) for i,j in el_sub]
    tau_sub, _ = kendalltau(sc_scores, bf_matched)
    print(f" tau_sub={tau_sub:+.3f}", flush=True)

    del model, op, S, S_c, J_z, J_A
    gc.collect(); torch.cuda.empty_cache()

    return {
        "seed": seed,
        "kappa_full": kappa_full, "kappa_sub": kappa_sub,
        "rho_full": rho_full,
        "tau_full_svd": tau_svd, "tau_full_col": tau_col,
        "tau_svd_vs_col": tau_svd_col,
        "tau_sub": tau_sub,
        "n_edges": n_edges, "n_sampled": len(sample_idx),
        "t_svd": t_svd, "t_col": t_col, "t_bf": t_bf,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    data = _load_amazon(Path("datasets/amazon_photo"))
    print(f"Amazon Photo: N={data['N']}\n")

    rows = []
    for si, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({si+1}/{len(SEEDS)}) ===")
        r = run_single(data, seed, device)
        rows.append(r)
        print(f"  RESULT: tau_full(svd)={r['tau_full_svd']:+.3f}  "
              f"tau_full(col)={r['tau_full_col']:+.3f}  "
              f"tau_sub={r['tau_sub']:+.3f}")
        print(f"          kappa_full={r['kappa_full']:.3f}  kappa_sub={r['kappa_sub']:.3f}  "
              f"rho={r['rho_full']:.3f}\n")

    print("=" * 80)
    print("AMAZON PHOTO: FULL-GRAPH vs SUBGRAPH")
    print("=" * 80)
    print(f"{'Metric':<25} {'Full graph':>15} {'50-node sub':>15}")
    print("-" * 55)
    print(f"{'κ':<25} {np.mean([r['kappa_full'] for r in rows]):>15.3f} {np.mean([r['kappa_sub'] for r in rows]):>15.3f}")
    print(f"{'ρ (power-iter)':<25} {np.mean([r['rho_full'] for r in rows]):>15.3f} {'---':>15}")
    print(f"{'τ (SVD-based)':<25} {np.mean([r['tau_full_svd'] for r in rows]):>+14.3f} {'---':>15}")
    print(f"{'τ (column-based)':<25} {np.mean([r['tau_full_col'] for r in rows]):>+14.3f} {np.mean([r['tau_sub'] for r in rows]):>+14.3f}")
    print(f"{'τ (SVD vs col)':<25} {np.mean([r['tau_svd_vs_col'] for r in rows]):>+14.3f} {'---':>15}")
    print(f"{'|E|':<25} {np.mean([r['n_edges'] for r in rows]):>15.0f} {'~50':>15}")


if __name__ == "__main__":
    sys.exit(main() or 0)
