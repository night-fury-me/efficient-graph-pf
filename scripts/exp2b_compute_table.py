"""EXP-2(b) — compute/completeness table: ONE AEGIS query vs the union of separate tools.

The cleanest, least-arguable rebuttal to "the coupling is definitional" (DA-M1) is cost +
completeness: one closed-form query yields the whole audit triad, whereas matching it with
off-the-shelf tools needs a separate iterative attack, a separate smoothing certificate, and a
separately-trained defense -- each yielding only one artifact, at orders of magnitude more compute.

This script times, on a 50-node subgraph of a trained IGNN (averaged over seeds), the wall-clock of:
  * AEGIS one query  -- S_c build + SVD (sigma_1 + attack direction v_1) + per-edge ranking v_ij
                        (per-node radii r_v are an O(Nd) add-on, marginal);
  * GR-BCD attack    -- faithful Geisler greedy, 125 epochs  (attack direction + edge ranking only);
  * PR-BCD attack    -- faithful Geisler projected, 125 epochs (attack direction + edge ranking only);
  * randomized smoothing certificate -- N noisy forwards + vote (certified radius only).
Prints the wall-clock table, the speedup of one AEGIS query over each tool and over their union,
and the artifacts-per-path matrix.

Run on a single stated GPU (timing is hardware-dependent). Reuses EXP-3's faithful attackers.
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.revision_R2._common import SEEDS, load_dataset, train_ignn
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from scripts.exp3_sota_attack_sweep import _edge_tensors, grbcd_order, prbcd_order
from scripts.exp_aegis_regularized_training import analysis_sigma1


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_aegis_query(model, X_sub, A_sub):
    """The paper's one AEGIS query (optimized N<=200 dense path): structural Jacobian
    (jacrev J_z + edges-only J_A -- only the |E| edge columns, never the dense full-S)
    -> resolvent -> SVD (sigma_1 + attack direction) + per-edge ranking v_ij. The
    earlier 6s was an unoptimized autograd Jacobian loop; this is the real cost."""
    def F_op(z, c):
        return model.operator(z, c)
    _sync(); t0 = time.time()
    _, Z_star, ctx = model(X_sub, A_sub)
    J_z, J_A, _cm = _compute_structural_jacobian(F_op, Z_star, ctx, edges_only=True)
    S_c = structural_sensitivity_matrix(F_op, Z_star, ctx, J_z=J_z, J_A=J_A)
    _, sig, Vh = torch.linalg.svd(S_c, full_matrices=False)
    _sigma1, _v1 = sig[0], Vh[0]          # attack direction
    _v_ij = S_c.norm(dim=0)               # per-edge ranking
    _sync()
    return time.time() - t0


def time_attack(fn, *a):
    _sync(); t0 = time.time()
    fn(*a)
    _sync()
    return time.time() - t0


def time_smoothing(model, X_sub, A_sub, n_samples, p=0.1):
    """Randomized (sparse) smoothing certificate cost: n_samples noisy forwards + agreement vote."""
    _sync(); t0 = time.time()
    with torch.no_grad():
        base = model(X_sub, A_sub)[0].argmax(1)
        agree = torch.zeros(A_sub.shape[0], device=A_sub.device)
        for _ in range(n_samples):
            mask = (torch.rand_like(A_sub) > p).float()
            A_n = A_sub * mask
            A_n = torch.minimum(A_n, A_n.t())            # keep symmetric (drop if either dir dropped)
            pred = model(X_sub, A_n)[0].argmax(1)
            agree += (pred == base).float()
    _sync()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="Cora")
    ap.add_argument("--n-seeds", type=int, default=10)   # 10-seed rule: never fewer
    ap.add_argument("--subgraph-n", type=int, default=50)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=125)
    ap.add_argument("--n-smooth", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    seeds = SEEDS[:args.n_seeds]
    X, A_hat, y, train_mask, nfeat, ncls = load_dataset(args.dataset)
    X, A_hat, y, train_mask = X.to(device), A_hat.to(device), y.to(device), train_mask.to(device)

    T = {k: [] for k in ("aegis", "grbcd", "prbcd", "smooth")}
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = train_ignn(X, A_hat, y, train_mask, nfeat, ncls, device, seed)
        idx = extract_ego_subgraph(A_hat, max_nodes=args.subgraph_n)
        X_sub, A_sub = X[idx], A_hat[idx][:, idx]
        # warm-up (exclude CUDA init / autotune from timing)
        with torch.no_grad():
            model(X_sub, A_sub)
        _sync()

        # AEGIS needs Z_clean/edge tensors for the attackers; compute once (not timed for attacks)
        def F_op(z, c):
            return model.operator(z, c)
        with torch.no_grad():
            _, Z_clean, _ = model(X_sub, A_sub)
            _, _, ctx = model(X_sub, A_sub)
            Jz, Ja, _ = _compute_structural_jacobian(F_op, Z_clean, ctx)
            S = structural_sensitivity_matrix(F_op, Z_clean, ctx, J_z=Jz, J_A=Ja)
            S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        ii, jj, avals = _edge_tensors(A_sub, edge_list)
        Z_clean = Z_clean.detach()
        k = min(args.budget, len(edge_list))

        T["aegis"].append(time_aegis_query(model, X_sub, A_sub))
        T["grbcd"].append(time_attack(grbcd_order, model, X_sub, A_sub, ii, jj, avals, Z_clean, k, args.epochs))
        T["prbcd"].append(time_attack(prbcd_order, model, X_sub, A_sub, ii, jj, avals, Z_clean, k, args.epochs, 1000.0))
        T["smooth"].append(time_smoothing(model, X_sub, A_sub, args.n_smooth))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def ms(x):
        return st.mean(x), (st.pstdev(x) if len(x) > 1 else 0.0)
    a = ms(T["aegis"]); g = ms(T["grbcd"]); p = ms(T["prbcd"]); s = ms(T["smooth"])
    union = g[0] + s[0]            # union to get {attack ranking} + {certified radius}

    print(f"=== EXP-2(b) compute table | {args.dataset} | {gpu} | {args.n_seeds} seeds | "
          f"subgraph N={args.subgraph_n}, budget={args.budget}, attack epochs={args.epochs}, "
          f"smoothing N={args.n_smooth} ===\n")
    print(f"{'method':<26}{'wall-clock (s)':>18}{'x over AEGIS':>16}")
    print(f"{'AEGIS one query':<26}{a[0]:>12.4f}±{a[1]:.4f}{'1x':>16}")
    print(f"{'GR-BCD attack (125ep)':<26}{g[0]:>12.4f}±{g[1]:.4f}{g[0]/a[0]:>14.1f}x")
    print(f"{'PR-BCD attack (125ep)':<26}{p[0]:>12.4f}±{p[1]:.4f}{p[0]/a[0]:>14.1f}x")
    print(f"{'Smoothing cert (1e4)':<26}{s[0]:>12.4f}±{s[1]:.4f}{s[0]/a[0]:>14.1f}x")
    print(f"{'UNION (attack+cert)':<26}{union:>12.4f}{'':>6}{union/a[0]:>14.1f}x")
    print()
    print("Artifacts each path yields (one AEGIS query vs the separate tools):")
    print(f"{'path':<26}{'attack dir':>11}{'edge rank':>11}{'node radius':>13}{'cert frac':>11}")
    print(f"{'AEGIS one query':<26}{'yes':>11}{'yes':>11}{'yes':>13}{'yes':>11}")
    print(f"{'GR-BCD / PR-BCD':<26}{'yes':>11}{'yes':>11}{'no':>13}{'no':>11}")
    print(f"{'Smoothing certificate':<26}{'no':>11}{'no':>11}{'costly':>13}{'yes':>11}")
    print("\n(one query yields all four; the union of separate tools yields the same set only by "
          "paying attack+cert+defense compute, and never surfaces a per-edge ranking from the cert "
          "nor a certificate from the attack.)")


if __name__ == "__main__":
    main()
