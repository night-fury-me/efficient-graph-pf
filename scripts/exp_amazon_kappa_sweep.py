"""C2 diagnostic: κ-sweep on Amazon Photo to investigate negative τ.

Tests whether over-contraction (κ_sub = 0.14) causes the negative τ = -0.15,
and whether higher κ improves τ.

Key finding from diagnostic:
  - Full graph: ||Â||₂ = 1.0, so κ_full ≈ ||W||₂
  - 50-node subgraph: ||Â_sub||₂ = 0.20, so κ_sub ≈ 0.20 × ||W||₂
  - With spectral_norm (||W||₂ ≤ 1): κ_sub ≤ 0.20 — always over-contracted

This script:
  1. Trains IGNN_Kappa at κ_max ∈ {0.5, 0.7, 0.9, 0.99} on Amazon Photo
  2. Measures τ on 50-node subgraph (existing setup)
  3. Reports κ_sub (actual subgraph contraction) and τ

Usage:
    .venv/bin/python scripts/exp_amazon_kappa_sweep.py
"""

from __future__ import annotations

import csv
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    greedy_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_amazon import _load_amazon
from iem.certify import spectral_radius

SEEDS = [42, 137, 271, 314, 1729]  # 5 seeds for speed
KAPPA_VALUES = [0.50, 0.70, 0.90, 0.99]


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class IGNN_Kappa(nn.Module):
    def __init__(self, n_features, hidden, n_classes, kappa=0.9, A_hat_sn=1.0):
        super().__init__()
        self.hidden = hidden
        self.kappa = kappa
        self.A_hat_sn = A_hat_sn
        self.U = nn.Linear(n_features, hidden)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.xavier_normal_(self.W.weight, gain=0.5)

    def _project_W(self):
        with torch.no_grad():
            target = self.kappa / self.A_hat_sn
            current = float(torch.linalg.svdvals(self.W.weight)[0])
            if current > target and current > 1e-10:
                self.W.weight.mul_(target / current)

    def operator(self, Z, ctx):
        return F_func.relu(ctx["A_hat"] @ self.W(Z) + ctx["X_proj"])

    def forward(self, X, A_hat, max_iter=50, tol=1e-5):
        N = X.shape[0]
        X_proj = self.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}
        Z = torch.zeros(N, self.hidden, device=X.device)
        for _ in range(max_iter):
            Z_new = self.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm(), 1.0): break
            Z = Z_new
        return self.head(Z_new), Z_new, ctx


def reconverge(model, Z, ctx, max_iter=200):
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7: break
            Z = Z_new
    return Z_new


def run_single(data, seed, kappa, device):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    A_hat_sn = float(torch.linalg.svdvals(A_hat)[0])

    model = IGNN_Kappa(data["n_features"], 64, data["n_classes"],
                       kappa=kappa, A_hat_sn=A_hat_sn).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for ep in range(200):
        model.train()
        lo, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(lo[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad(); loss.backward(); optim.step()
        model._project_W()
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

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        test_acc = float((logits.argmax(1)[data["test_mask"].to(device)] == y[data["test_mask"]]).float().mean())

    # Subgraph analysis
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)

    # Actual kappa on subgraph
    A_sub_sn = float(torch.linalg.svdvals(A_sub)[0])
    W_sn = float(torch.linalg.svdvals(model.W.weight)[0])
    kappa_sub = A_sub_sn * W_sn

    # S_c + tau
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub)
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

    if not edge_list:
        return None

    # Column norm coefficient of variation (uniformity measure)
    col_norms = torch.stack([S_c[:, k].norm() for k in range(S_c.shape[1])])
    cv = float(col_norms.std() / (col_norms.mean() + 1e-10))

    # Brute-force discrete ground truth
    bf = greedy_structural_attack(model, Z_sub, ctx_sub)
    aegis_scores = [float(S_c[:, k].norm()) for k in range(len(edge_list))]
    bf_dict = {(min(i,j), max(i,j)): s for i,j,s in bf}
    bf_matched = [bf_dict.get((min(i,j), max(i,j)), 0.0) for i,j in edge_list]

    tau, _ = kendalltau(aegis_scores, bf_matched)

    del model, S, S_c, J_z, J_A
    gc.collect(); torch.cuda.empty_cache()

    return {
        "seed": seed, "kappa_max": kappa,
        "kappa_sub": kappa_sub, "A_sub_sn": A_sub_sn, "W_sn": W_sn,
        "tau": tau, "test_acc": test_acc, "col_norm_cv": cv,
        "resolvent_amp": 1.0 / (1.0 - kappa_sub),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data = _load_amazon(Path("datasets/amazon_photo"))
    print(f"Amazon Photo: N={data['N']}, features={data['n_features']}, classes={data['n_classes']}\n")

    rows = []
    for kappa in KAPPA_VALUES:
        for si, seed in enumerate(SEEDS):
            print(f"kappa={kappa:.2f} seed={seed} ({si+1}/{len(SEEDS)})", end=" ", flush=True)
            r = run_single(data, seed, kappa, device)
            if r:
                rows.append(r)
                print(f"kappa_sub={r['kappa_sub']:.3f} tau={r['tau']:+.3f} "
                      f"acc={r['test_acc']:.3f} CV={r['col_norm_cv']:.3f} "
                      f"amp={r['resolvent_amp']:.2f}x")
            else:
                print("SKIP")

    print("\n" + "=" * 95)
    print("AMAZON PHOTO κ-SWEEP SUMMARY")
    print("=" * 95)
    print(f"{'kappa_max':>10} {'kappa_sub':>10} {'||W||_2':>10} {'tau':>12} "
          f"{'Acc%':>8} {'CV':>8} {'Amp':>8}")
    print("-" * 95)
    for kappa in KAPPA_VALUES:
        subset = [r for r in rows if r["kappa_max"] == kappa]
        if not subset: continue
        print(f"{kappa:>10.2f} "
              f"{np.mean([r['kappa_sub'] for r in subset]):>10.3f} "
              f"{np.mean([r['W_sn'] for r in subset]):>10.3f} "
              f"{np.mean([r['tau'] for r in subset]):>+11.3f}±{np.std([r['tau'] for r in subset]):.3f} "
              f"{np.mean([r['test_acc'] for r in subset])*100:>7.1f} "
              f"{np.mean([r['col_norm_cv'] for r in subset]):>8.3f} "
              f"{np.mean([r['resolvent_amp'] for r in subset]):>7.2f}x")

    print(f"\n||A_sub||_2 = {rows[0]['A_sub_sn']:.3f}" if rows else "")
    print("Note: kappa_sub = ||A_sub||_2 × ||W||_2, always ≤ 0.20 on this subgraph")
    print("CV = coefficient of variation of S_c column norms (low = uniform = uninformative)")


if __name__ == "__main__":
    sys.exit(main() or 0)
