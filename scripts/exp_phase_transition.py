"""B5/P2.4 — Phase transition sweep: kappa vs resolvent divergence.

Varies kappa from 0.30 to 0.99 and measures:
  - Actual spectral radius of J_z
  - Resolvent norm ||(I - J_z)^{-1}||_2
  - Critical perturbation budget eps_crit
  - Equilibrium shift under S_c-optimal attack at eps=0.01
  - Amplification factor (should diverge as kappa -> 1)

Dataset: Cora (50-node BFS subgraph)
Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python scripts/exp_phase_transition.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
KAPPA_VALUES = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IGNN_Kappa(nn.Module):
    """IGNN with controllable spectral norm ceiling kappa.

    The contraction condition requires ||W||_2 * ||A_hat||_2 < 1.
    We set ||W||_2 = kappa / ||A_hat||_2 so that the product is exactly kappa.
    This is achieved by projecting W after each training step.
    """

    def __init__(self, n_features: int, hidden: int, n_classes: int,
                 kappa: float = 0.9, A_hat_spectral_norm: float = 1.0):
        super().__init__()
        self.hidden = hidden
        self.kappa = kappa
        self.A_hat_spectral_norm = A_hat_spectral_norm
        self.U = nn.Linear(n_features, hidden)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_classes)
        nn.init.xavier_normal_(self.W.weight, gain=0.5)

    def _project_W(self):
        """Project W so that ||W||_2 = kappa / ||A_hat||_2."""
        with torch.no_grad():
            target_norm = self.kappa / self.A_hat_spectral_norm
            current_norm = float(torch.linalg.svdvals(self.W.weight)[0])
            if current_norm > target_norm and current_norm > 1e-10:
                self.W.weight.mul_(target_norm / current_norm)

    def operator(self, Z: Tensor, ctx: dict) -> Tensor:
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return F_func.relu(A_hat @ self.W(Z) + X_proj)

    def forward(self, X: Tensor, A_hat: Tensor, max_iter: int = 50, tol: float = 1e-5):
        N = X.shape[0]
        X_proj = self.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}

        Z = torch.zeros(N, self.hidden, device=X.device)
        for k in range(max_iter):
            Z_new = self.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm(), 1.0):
                break
            Z = Z_new
        Z_star = Z_new

        logits = self.head(Z_star)
        return logits, Z_star, ctx


def train_ignn_kappa(data, device, seed, kappa, epochs=200):
    """Train IGNN with spectral norm ceiling kappa."""
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    # Compute ||A_hat||_2 for the contraction constraint
    A_hat_sn = float(torch.linalg.svdvals(A_hat)[0])

    model = IGNN_Kappa(
        data["n_features"], hidden=64, n_classes=data["n_classes"],
        kappa=kappa, A_hat_spectral_norm=A_hat_sn,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Project W to enforce kappa ceiling
        model._project_W()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float(
                    (logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]])
                    .float().mean()
                )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    return model, Z_star, ctx, A_hat_sn


def run_single(data, device, seed, kappa):
    """Run one (seed, kappa) combination and return metrics."""
    set_seed(seed)

    try:
        model, Z_star, ctx, A_hat_sn = train_ignn_kappa(data, device, seed, kappa)
    except Exception as e:
        print(f"      Training failed: {e}", flush=True)
        return None

    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    # Extract 50-node BFS subgraph
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = Z_star[idx].clone()

    # Reconverge on subgraph
    with torch.no_grad():
        for _ in range(300):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    # Compute J_z and spectral radius
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)

    D = Z_sub.numel()
    from iem.ift import compute_jacobian
    J_z = compute_jacobian(F_z, Z_sub)

    rho_actual = float(torch.linalg.eigvals(J_z).abs().max())

    # Resolvent norm ||(I - J_z)^{-1}||_2
    I_mat = torch.eye(D, device=device, dtype=J_z.dtype)
    try:
        resolvent = torch.linalg.inv(I_mat - J_z)
        resolvent_norm = float(torch.linalg.svdvals(resolvent)[0])
    except torch._C._LinAlgError:
        resolvent_norm = float("inf")

    # ||W||_2
    W_sn = float(torch.linalg.svdvals(model.W.weight.detach())[0])

    # eps_crit = (1 - kappa) / ||W||_2
    eps_crit = (1.0 - kappa) / W_sn if W_sn > 1e-10 else float("inf")

    # S_c and optimal attack at eps=0.01
    eps = 0.01
    try:
        J_z_struct, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
            J_z=J_z_struct, J_A=J_A,
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

        if S_c.shape[1] > 0:
            U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
            sigma_1 = float(sigma_c[0])

            # Optimal attack perturbation
            dA = torch.zeros_like(A_sub)
            weights = eps * Vh_c[0]
            for k, (i, j) in enumerate(edge_list):
                dA[i, j] = float(weights[k])
                dA[j, i] = float(weights[k])

            # Reconverge under perturbation
            ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
            Z = Z_sub.clone()
            with torch.no_grad():
                for _ in range(300):
                    Z_new = model.operator(Z, ctx_pert)
                    if torch.isnan(Z_new).any() or Z_new.norm() > 1e6:
                        Z_new = Z  # diverged
                        break
                    if (Z_new - Z).norm() < 1e-8:
                        break
                    Z = Z_new
            actual_shift = float((Z_new - Z_sub).norm())

            # Amplification = shift / (sigma_1 * eps)
            predicted = sigma_1 * eps
            amplification = actual_shift / (predicted) if predicted > 1e-12 else float("inf")
        else:
            sigma_1 = 0.0
            actual_shift = 0.0
            amplification = float("nan")
    except Exception as e:
        print(f"      S_c computation failed: {e}", flush=True)
        sigma_1 = float("nan")
        actual_shift = float("nan")
        amplification = float("nan")

    return {
        "kappa": kappa,
        "seed": seed,
        "rho_actual": rho_actual,
        "resolvent_norm": resolvent_norm,
        "W_spectral_norm": W_sn,
        "eps_crit": eps_crit,
        "sigma_1_Sc": sigma_1,
        "actual_shift": actual_shift,
        "amplification": amplification,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    data = _load_cora(Path("datasets/cora"))
    print(f"Cora: N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")

    all_results = []
    t_start = time.time()

    for kappa in KAPPA_VALUES:
        print(f"\n{'='*70}")
        print(f"  kappa = {kappa:.2f}")
        print(f"{'='*70}", flush=True)

        kappa_results = []
        for seed_idx, seed in enumerate(SEEDS):
            print(f"    Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ...", end=" ", flush=True)
            r = run_single(data, device, seed, kappa)
            if r:
                kappa_results.append(r)
                all_results.append(r)
                print(f"rho={r['rho_actual']:.4f}  resolvent={r['resolvent_norm']:.2f}  "
                      f"shift={r['actual_shift']:.4f}  amp={r['amplification']:.2f}", flush=True)
            else:
                print("FAILED", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if kappa_results:
            rhos = [r["rho_actual"] for r in kappa_results]
            resolvents = [r["resolvent_norm"] for r in kappa_results if r["resolvent_norm"] < 1e10]
            shifts = [r["actual_shift"] for r in kappa_results if not np.isnan(r["actual_shift"])]
            amps = [r["amplification"] for r in kappa_results
                    if not np.isnan(r["amplification"]) and r["amplification"] < 1e10]
            print(f"    => rho={np.mean(rhos):.4f}+/-{np.std(rhos):.4f}  "
                  f"resolvent={np.mean(resolvents):.2f}+/-{np.std(resolvents):.2f}  "
                  f"shift={np.mean(shifts):.4f}+/-{np.std(shifts):.4f}  "
                  f"amp={np.mean(amps):.2f}+/-{np.std(amps):.2f}" if amps else
                  f"    => rho={np.mean(rhos):.4f}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Save CSV ---
    csv_path = results_dir / "exp_phase_transition.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "kappa", "seed", "rho_actual", "resolvent_norm", "W_spectral_norm",
            "eps_crit", "sigma_1_Sc", "actual_shift", "amplification",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # --- Print summary table ---
    print("\n" + "=" * 110)
    print("PHASE TRANSITION SUMMARY (Cora, 50-node subgraph, 10 seeds)")
    print("=" * 110)
    print(f"{'kappa':>6} {'rho':>14} {'||resolvent||':>18} {'eps_crit':>14} "
          f"{'shift':>14} {'amplification':>18}")
    print("-" * 110)

    def agg(vals, fmt=".4f"):
        arr = [v for v in vals if v is not None and not np.isnan(v) and abs(v) < 1e10]
        if not arr:
            return "N/A"
        return f"{np.mean(arr):{fmt}}+/-{np.std(arr):{fmt}}"

    for kappa in KAPPA_VALUES:
        rows = [r for r in all_results if r["kappa"] == kappa]
        if not rows:
            continue
        print(f"{kappa:>6.2f} "
              f"{agg([r['rho_actual'] for r in rows]):>14} "
              f"{agg([r['resolvent_norm'] for r in rows], '.2f'):>18} "
              f"{agg([r['eps_crit'] for r in rows]):>14} "
              f"{agg([r['actual_shift'] for r in rows]):>14} "
              f"{agg([r['amplification'] for r in rows], '.2f'):>18}")

    print("\nKey result: resolvent norm and amplification should diverge as kappa -> 1")


if __name__ == "__main__":
    sys.exit(main() or 0)
