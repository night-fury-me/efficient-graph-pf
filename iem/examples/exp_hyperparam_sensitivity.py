"""P3 Experiment: Hyperparameter sensitivity (hidden dim d, spectral norm c).

Sweeps hidden dimension d={16, 32, 64, 128} and spectral norm constraint
c={0.5, 0.7, 0.9, 0.95} on Cora. Reports tightness, epsilon_crit,
certified radius, and test accuracy for each configuration.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_hyperparam_sensitivity
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    critical_perturbation_budget,
    extract_ego_subgraph,
    extract_W_spectral_norm,
    per_node_robust_radius,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
HIDDEN_DIMS = [16, 32, 64, 128]
SPECTRAL_NORMS = [0.5, 0.7, 0.9, 0.95]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IGNN_Configurable(nn.Module):
    """IGNN with configurable hidden dim and spectral norm constraint.

    Does NOT use PyTorch's spectral_norm parametrization (which forces
    ||W||_2=1 at every forward, defeating manual clamping). Instead,
    projects W after each optimizer step via SVD clamping.
    """

    def __init__(self, n_features: int, hidden: int, n_classes: int,
                 max_spectral_norm: float = 1.0):
        super().__init__()
        self.hidden = hidden
        self.U = nn.Linear(n_features, hidden)
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, n_classes)

        nn.init.xavier_normal_(self.W.weight, gain=0.5)
        self._max_sn = max_spectral_norm
        self._project_spectral_norm()

    def _project_spectral_norm(self):
        """Project W so that ||W||_2 <= max_spectral_norm."""
        with torch.no_grad():
            U, s, Vh = torch.linalg.svd(self.W.weight, full_matrices=False)
            s_clamped = s.clamp(max=self._max_sn)
            self.W.weight.copy_(U @ torch.diag(s_clamped) @ Vh)

    def operator(self, Z, ctx):
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return F_func.relu(A_hat @ self.W(Z) + X_proj)

    def forward(self, X, A_hat, max_iter=50, tol=1e-5):
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


def run_single(data, seed, device, hidden=64, max_sn=1.0):
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN_Configurable(
        data["n_features"], hidden=hidden, n_classes=data["n_classes"],
        max_spectral_norm=max_sn,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val, best_state = 0.0, None
    for ep in range(200):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        model._project_spectral_norm()

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float((logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    labels_sub = y[idx]

    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new
    logits_sub = model.head(Z_sub)

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    tight_results = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.01], n_random=3,
    )
    constr_tight = tight_results[0]["constr_tightness"]

    try:
        W_norm = float(torch.linalg.svdvals(model.W.weight.detach())[0])
    except Exception:
        W_norm = 1.0
    budget = critical_perturbation_budget(rho, W_norm)

    node_certs = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
    det_nontrivial = node_certs["radii"][node_certs["radii"] > 1e-6]

    return {
        "hidden": hidden, "max_sn": max_sn,
        "test_acc": test_acc, "rho": rho,
        "constr_tight": constr_tight,
        "eps_crit": budget["epsilon_crit"],
        "med_r": float(det_nontrivial.median()) if len(det_nontrivial) > 0 else 0.0,
        "W_norm": W_norm,
    }


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def agg_pct(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m*100:.0f}±{s*100:.0f}%"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("Loading Cora...", flush=True)
    data = _load_cora(Path("datasets/cora"))

    # Part 1: Hidden dimension sweep (fixed spectral norm = default)
    print("\n=== Hidden Dimension Sweep ===\n")
    dim_results = {d: [] for d in HIDDEN_DIMS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
        for d in HIDDEN_DIMS:
            r = run_single(data, seed, device, hidden=d, max_sn=1.0)
            dim_results[d].append(r)
            print(f"  d={d:>3}: acc={r['test_acc']:.3f} rho={r['rho']:.3f} "
                  f"tight={r['constr_tight']:.3f} eps_crit={r['eps_crit']:.3f} "
                  f"r={r['med_r']:.4f}", flush=True)

    # Part 2: Spectral norm sweep (fixed hidden=64)
    print("\n=== Spectral Norm Sweep ===\n")
    sn_results = {c: [] for c in SPECTRAL_NORMS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
        for c in SPECTRAL_NORMS:
            r = run_single(data, seed, device, hidden=64, max_sn=c)
            sn_results[c].append(r)
            print(f"  c={c:.2f}: acc={r['test_acc']:.3f} rho={r['rho']:.3f} "
                  f"tight={r['constr_tight']:.3f} eps_crit={r['eps_crit']:.3f} "
                  f"r={r['med_r']:.4f}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Tables
    print("=" * 100)
    print("HIDDEN DIMENSION SWEEP on Cora (10 seeds)")
    print("=" * 100)
    print(f"{'d':>5} {'Acc':>12} {'rho':>12} {'Tightness':>12} {'eps_crit':>12} {'Med r_v':>12}")
    print("-" * 100)
    for d in HIDDEN_DIMS:
        rs = dim_results[d]
        print(f"{d:>5} {agg([r['test_acc'] for r in rs]):>12} "
              f"{agg([r['rho'] for r in rs]):>12} "
              f"{agg([r['constr_tight'] for r in rs]):>12} "
              f"{agg([r['eps_crit'] for r in rs]):>12} "
              f"{agg([r['med_r'] for r in rs]):>12}")

    print(f"\n{'=' * 100}")
    print("SPECTRAL NORM SWEEP on Cora (10 seeds, hidden=64)")
    print("=" * 100)
    print(f"{'c':>5} {'Acc':>12} {'rho':>12} {'Tightness':>12} {'eps_crit':>12} {'Med r_v':>12} {'||W||_2':>12}")
    print("-" * 100)
    for c in SPECTRAL_NORMS:
        rs = sn_results[c]
        print(f"{c:>5.2f} {agg([r['test_acc'] for r in rs]):>12} "
              f"{agg([r['rho'] for r in rs]):>12} "
              f"{agg([r['constr_tight'] for r in rs]):>12} "
              f"{agg([r['eps_crit'] for r in rs]):>12} "
              f"{agg([r['med_r'] for r in rs]):>12} "
              f"{agg([r['W_norm'] for r in rs]):>12}")

    # Save
    results_path = Path("docs/exp_hyperparam_sensitivity_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Hyperparameter Sensitivity on Cora (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")

        f.write("## Hidden Dimension Sweep\n\n")
        f.write("| d | Acc | ρ | Tightness | ε_crit | Med r_v |\n")
        f.write("|---|---|---|---|---|---|\n")
        for d in HIDDEN_DIMS:
            rs = dim_results[d]
            f.write(f"| {d} | {agg([r['test_acc'] for r in rs])} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['eps_crit'] for r in rs])} "
                    f"| {agg([r['med_r'] for r in rs])} |\n")

        f.write("\n## Spectral Norm Sweep (hidden=64)\n\n")
        f.write("| c | Acc | ρ | Tightness | ε_crit | Med r_v | ||W||_2 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in SPECTRAL_NORMS:
            rs = sn_results[c]
            f.write(f"| {c} | {agg([r['test_acc'] for r in rs])} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['eps_crit'] for r in rs])} "
                    f"| {agg([r['med_r'] for r in rs])} "
                    f"| {agg([r['W_norm'] for r in rs])} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
