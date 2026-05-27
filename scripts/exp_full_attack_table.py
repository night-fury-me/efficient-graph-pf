"""Exp C4-4: Full attack comparison table with both metrics.

Compares ALL attack methods on BOTH equilibrium damage AND prediction flip rate:
  1. AEGIS SVD (S_c optimal)
  2. Classification-loss PGD (autograd, cross-entropy objective)
  3. Equilibrium-shift PGD (IFT gradients, same objective as AEGIS)
  4. Random perturbation

Reports both metrics to show AEGIS's SVD direction is effective even
for flipping predictions (not just maximizing equilibrium shift).

The eq-shift PGD is explicitly reframed as a "solver validation"
(SVD vs iterative optimization on the same linearized problem).

Datasets: Cora, Citeseer, WikiCS | Model: IGNN | Seeds: 10
Epsilon: 0.01, 0.05, 0.10

Output: results/full_attack_table.csv

Usage:
    .venv/bin/python scripts/exp_full_attack_table.py
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
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
EPS_VALUES = [0.01, 0.05, 0.10]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconverge(model, Z_init, ctx, max_iter=200):
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return Z_new


def load_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics
    datasets = {}
    print("Loading datasets...", flush=True)
    datasets["Cora"] = _load_cora(Path("datasets/cora"))
    datasets["Citeseer"] = _load_planetoid("citeseer", Path("datasets/citeseer"))
    datasets["WikiCS"] = _load_wikics(Path("datasets/wikics"))
    return datasets


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
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = model(X, A_hat)
                va = float((lv.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def apply_perturbation(A, edge_list, weights):
    """Apply edge-weight perturbation and return perturbed adjacency."""
    A_pert = A.clone()
    for k, (i, j) in enumerate(edge_list):
        A_pert[i, j] += weights[k]
        A_pert[j, i] += weights[k]
    return A_pert


def measure_attack(model, Z_clean, ctx_sub, A_pert, preds_clean):
    """Measure both equilibrium damage and prediction flips."""
    ctx_pert = {**ctx_sub, "A_hat": A_pert}
    Z_pert = reconverge(model, Z_clean, ctx_pert)
    damage = float((Z_pert - Z_clean).norm())
    with torch.no_grad():
        preds_pert = model.head(Z_pert).argmax(dim=1)
    n_flipped = int((preds_clean != preds_pert).sum())
    return damage, n_flipped


def pgd_attack(model, Z_clean, ctx_sub, target, edge_list, epsilon,
               objective="classification", y_sub=None, n_steps=50):
    """PGD attack with configurable objective."""
    A = ctx_sub["A_hat"]
    n_edges = len(edge_list)
    step_size = epsilon / 10.0
    # Initialize with small random noise to avoid gradient singularity
    # at ||Z_new - Z_clean|| = 0 (norm gradient is undefined at zero)
    delta_init = torch.randn(n_edges, device=A.device) * (epsilon * 0.01)
    norm_init = delta_init.norm()
    if norm_init > epsilon:
        delta_init = delta_init * (epsilon / norm_init)
    delta = delta_init.requires_grad_(True)

    for step in range(n_steps):
        A_pert = A.clone()
        for k, (i, j) in enumerate(edge_list):
            A_pert = A_pert.clone()
            A_pert[i, j] = A[i, j] + delta[k]
            A_pert[j, i] = A[j, i] + delta[k]
        ctx_pert = {**ctx_sub, "A_hat": A_pert}

        Z = Z_clean.detach().clone()
        with torch.enable_grad():
            for _ in range(50):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).detach().norm() < 1e-7:
                    break
                Z = Z_new

            if objective == "classification":
                logits = model.head(Z_new)
                loss = -F_func.cross_entropy(logits, y_sub)
            else:
                diff = Z_new - Z_clean.detach()
                loss = -diff.pow(2).sum()

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.data -= step_size * grad.sign()
            delta.data.clamp_(-epsilon / (n_edges ** 0.5), epsilon / (n_edges ** 0.5))
            norm = delta.data.norm()
            if norm > epsilon:
                delta.data *= epsilon / norm
        delta = delta.detach().requires_grad_(True)

    return delta.detach()


def run_single(ds_name, data, seed, eps, device):
    try:
        model = train_ignn(data, device, seed)

        X = data["X"].to(device)
        A_hat = data["A_hat"].to(device)
        y = data["y"].to(device)

        with torch.no_grad():
            _, Z_star, ctx = model(X, A_hat)

        idx = extract_ego_subgraph(A_hat, max_nodes=50)
        A_sub = A_hat[idx][:, idx].clone()
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
        y_sub = y[idx]

        Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)

        # Clean predictions
        with torch.no_grad():
            preds_clean = model.head(Z_sub).argmax(dim=1)

        n_nodes = len(y_sub)

        # Compute S_c for AEGIS
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        if not edge_list:
            return None
        n_edges = len(edge_list)

        # --- 1. AEGIS SVD ---
        U, sigma, Vh = torch.linalg.svd(S_c, full_matrices=False)
        svd_weights = eps * Vh[0]
        A_svd = apply_perturbation(A_sub, edge_list, svd_weights)
        dmg_svd, flips_svd = measure_attack(model, Z_sub, ctx_sub, A_svd, preds_clean)

        del S, S_c, U, sigma, Vh, J_z, J_A
        gc.collect()
        torch.cuda.empty_cache()

        # --- 2. Classification-loss PGD ---
        delta_cls = pgd_attack(
            model, Z_sub, ctx_sub, None, edge_list, eps,
            objective="classification", y_sub=y_sub
        )
        A_cls = apply_perturbation(A_sub, edge_list, delta_cls)
        dmg_cls, flips_cls = measure_attack(model, Z_sub, ctx_sub, A_cls, preds_clean)

        # --- 3. Equilibrium-shift PGD ---
        delta_shift = pgd_attack(
            model, Z_sub, ctx_sub, None, edge_list, eps,
            objective="shift"
        )
        A_shift = apply_perturbation(A_sub, edge_list, delta_shift)
        dmg_shift, flips_shift = measure_attack(model, Z_sub, ctx_sub, A_shift, preds_clean)

        # --- 4. Random perturbation ---
        rand_weights = torch.randn(n_edges, device=A_sub.device)
        rand_weights = rand_weights / rand_weights.norm() * eps
        A_rand = apply_perturbation(A_sub, edge_list, rand_weights)
        dmg_rand, flips_rand = measure_attack(model, Z_sub, ctx_sub, A_rand, preds_clean)

        return {
            "dataset": ds_name, "seed": seed, "epsilon": eps,
            "n_nodes": n_nodes, "n_edges": n_edges,
            # Equilibrium damage
            "dmg_svd": dmg_svd, "dmg_cls_pgd": dmg_cls,
            "dmg_shift_pgd": dmg_shift, "dmg_random": dmg_rand,
            # Prediction flips
            "flips_svd": flips_svd, "flips_cls_pgd": flips_cls,
            "flips_shift_pgd": flips_shift, "flips_random": flips_rand,
            # Flip rates
            "fliprate_svd": flips_svd / n_nodes,
            "fliprate_cls_pgd": flips_cls / n_nodes,
            "fliprate_shift_pgd": flips_shift / n_nodes,
            "fliprate_random": flips_rand / n_nodes,
            # Ratios
            "cls_pgd_over_svd_dmg": dmg_cls / max(dmg_svd, 1e-10),
            "shift_pgd_over_svd_dmg": dmg_shift / max(dmg_svd, 1e-10),
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return None
        raise


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t_start = time.time()

    datasets = load_datasets()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    rows = []
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        data = datasets[ds_name]
        for eps in EPS_VALUES:
            for seed_idx, seed in enumerate(SEEDS):
                print(f"[{ds_name}] eps={eps} seed={seed} ({seed_idx+1}/{len(SEEDS)})",
                      end=" ", flush=True)
                r = run_single(ds_name, data, seed, eps, device)
                if r:
                    rows.append(r)
                    print(f"SVD={r['dmg_svd']:.3f}({r['flips_svd']}f) "
                          f"ClsPGD={r['dmg_cls_pgd']:.3f}({r['flips_cls_pgd']}f) "
                          f"ShiftPGD={r['dmg_shift_pgd']:.3f}({r['flips_shift_pgd']}f) "
                          f"Rand={r['dmg_random']:.3f}({r['flips_random']}f)",
                          flush=True)
                else:
                    print("SKIP", flush=True)

    # Write CSV
    csv_path = results_dir / "full_attack_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Summary tables
    print("=" * 120)
    print("FULL ATTACK TABLE: EQUILIBRIUM DAMAGE (mean +/- std)")
    print("=" * 120)
    print(f"{'Dataset':<12} {'eps':>5} {'AEGIS SVD':>16} {'Class-PGD':>16} "
          f"{'Shift-PGD':>16} {'Random':>16} {'Cls/SVD':>8}")
    print("-" * 120)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        for eps in EPS_VALUES:
            subset = [r for r in rows if r["dataset"] == ds_name and r["epsilon"] == eps]
            if not subset:
                continue

            def fmt(key):
                vals = [r[key] for r in subset]
                return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

            ratio = np.mean([r["cls_pgd_over_svd_dmg"] for r in subset])
            print(f"{ds_name:<12} {eps:>5.2f} {fmt('dmg_svd'):>16} {fmt('dmg_cls_pgd'):>16} "
                  f"{fmt('dmg_shift_pgd'):>16} {fmt('dmg_random'):>16} {ratio:>8.2f}")

    print()
    print("=" * 120)
    print("FULL ATTACK TABLE: PREDICTION FLIP RATE % (mean +/- std)")
    print("=" * 120)
    print(f"{'Dataset':<12} {'eps':>5} {'AEGIS SVD':>16} {'Class-PGD':>16} "
          f"{'Shift-PGD':>16} {'Random':>16}")
    print("-" * 120)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        for eps in EPS_VALUES:
            subset = [r for r in rows if r["dataset"] == ds_name and r["epsilon"] == eps]
            if not subset:
                continue

            def fmtpct(key):
                vals = [r[key] * 100 for r in subset]
                return f"{np.mean(vals):.1f}+/-{np.std(vals):.1f}"

            print(f"{ds_name:<12} {eps:>5.2f} {fmtpct('fliprate_svd'):>16} "
                  f"{fmtpct('fliprate_cls_pgd'):>16} {fmtpct('fliprate_shift_pgd'):>16} "
                  f"{fmtpct('fliprate_random'):>16}")


if __name__ == "__main__":
    sys.exit(main() or 0)
