"""Classification-loss PGD attack on IGNN — non-circular baseline for P1.2.

Unlike the existing adaptive attacker (which optimizes equilibrium shift,
same objective as AEGIS SVD), this attack optimizes CLASSIFICATION LOSS
(cross-entropy) via PGD through the IGNN fixed-point iteration.

This answers: "Can an attacker targeting predictions (not equilibria)
find more damaging perturbations than AEGIS's sensitivity-optimal direction?"

Usage:
    .venv/bin/python scripts/exp_classification_attack.py
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
    structural_sensitivity_matrix,
    extract_ego_subgraph,
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


# ---------------------------------------------------------------
# Classification-Loss PGD Attack
# ---------------------------------------------------------------

def pgd_classification_attack(
    model: nn.Module,
    z_clean: torch.Tensor,
    ctx: dict,
    y_sub: torch.Tensor,
    epsilon: float,
    edge_list: list,
    n_steps: int = 50,
    A_key: str = "A_hat",
) -> dict:
    """PGD attack optimizing classification loss (cross-entropy).

    Differentiates through the IGNN fixed-point iteration via standard
    autograd to find edge perturbations that maximize misclassification.
    This uses a DIFFERENT gradient signal than AEGIS (which optimizes
    equilibrium shift), making the comparison non-circular.
    """
    A = ctx[A_key]
    n_edges = len(edge_list)
    step_size = epsilon / 10.0

    delta = torch.zeros(n_edges, device=A.device, requires_grad=True)

    for step in range(n_steps):
        A_pert = A.clone()
        for k, (i, j) in enumerate(edge_list):
            A_pert[i, j] = A_pert[i, j] + delta[k]
            A_pert[j, i] = A_pert[j, i] + delta[k]

        ctx_pert = {**ctx, A_key: A_pert}

        Z = z_clean.detach().clone()
        with torch.enable_grad():
            for _ in range(50):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).detach().norm() < 1e-7:
                    break
                Z = Z_new

            logits = model.head(Z_new)
            loss = -F_func.cross_entropy(logits, y_sub)

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

        with torch.no_grad():
            delta.data -= step_size * grad.sign()
            delta.data.clamp_(-epsilon / (n_edges ** 0.5), epsilon / (n_edges ** 0.5))
            norm = delta.data.norm()
            if norm > epsilon:
                delta.data *= epsilon / norm

        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        A_final = A.clone()
        for k, (i, j) in enumerate(edge_list):
            A_final[i, j] += delta[k]
            A_final[j, i] += delta[k]
        ctx_final = {**ctx, A_key: A_final}
        Z_final = reconverge(model, z_clean, ctx_final)
        actual_shift = float((Z_final - z_clean).norm())

        logits_final = model.head(Z_final)
        preds_clean = model.head(z_clean).argmax(dim=1)
        preds_pert = logits_final.argmax(dim=1)
        n_flipped = int((preds_clean != preds_pert).sum())

    return {
        "actual_shift": actual_shift,
        "n_flipped": n_flipped,
        "n_total": len(y_sub),
        "flip_rate": n_flipped / len(y_sub),
        "delta_norm": float(delta.detach().norm()),
    }


# ---------------------------------------------------------------
# Equilibrium-shift PGD (existing, for comparison)
# ---------------------------------------------------------------

def pgd_shift_attack(
    model, z_clean, ctx, epsilon, edge_list, n_steps=50, A_key="A_hat",
):
    """Existing adaptive attacker: optimizes equilibrium shift."""
    A = ctx[A_key]
    n_edges = len(edge_list)
    step_size = epsilon / 10.0
    delta = torch.zeros(n_edges, device=A.device, requires_grad=True)

    for step in range(n_steps):
        A_pert = A.clone()
        for k, (i, j) in enumerate(edge_list):
            A_pert[i, j] = A_pert[i, j] + delta[k]
            A_pert[j, i] = A_pert[j, i] + delta[k]
        ctx_pert = {**ctx, A_key: A_pert}
        Z = z_clean.detach().clone()
        with torch.enable_grad():
            for _ in range(50):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).detach().norm() < 1e-7:
                    break
                Z = Z_new
            shift = (Z_new - z_clean.detach()).norm()
            loss = -shift
        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
        with torch.no_grad():
            delta.data -= step_size * grad.sign()
            delta.data.clamp_(-epsilon / (n_edges ** 0.5), epsilon / (n_edges ** 0.5))
            norm = delta.data.norm()
            if norm > epsilon:
                delta.data *= epsilon / norm
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        A_final = A.clone()
        for k, (i, j) in enumerate(edge_list):
            A_final[i, j] += delta[k]
            A_final[j, i] += delta[k]
        ctx_final = {**ctx, A_key: A_final}
        Z_final = reconverge(model, z_clean, ctx_final)
        actual_shift = float((Z_final - z_clean).norm())
    return {"actual_shift": actual_shift}


# ---------------------------------------------------------------
# AEGIS SVD attack
# ---------------------------------------------------------------

def aegis_svd_attack(S_c, edge_list, A_sub, epsilon, model, z_clean, ctx):
    """AEGIS optimal attack via SVD of S_c."""
    U, sigma, Vh = torch.linalg.svd(S_c, full_matrices=False)
    weights = epsilon * Vh[0]
    with torch.no_grad():
        A_pert = A_sub.clone()
        for k, (i, j) in enumerate(edge_list):
            A_pert[i, j] += float(weights[k])
            A_pert[j, i] += float(weights[k])
        ctx_pert = {**ctx, "A_hat": A_pert}
        Z_final = reconverge(model, z_clean, ctx_pert)
        actual_shift = float((Z_final - z_clean).norm())
    return {"actual_shift": actual_shift}


# ---------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------

def load_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics
    datasets = {}
    print("Loading datasets...", flush=True)
    datasets["Cora"] = _load_cora(Path("datasets/cora"))
    datasets["Citeseer"] = _load_planetoid("citeseer", Path("datasets/citeseer"))
    datasets["WikiCS"] = _load_wikics(Path("datasets/wikics"))
    return datasets


# ---------------------------------------------------------------
# Run single (dataset, seed, epsilon)
# ---------------------------------------------------------------

def run_single(ds_name, data, seed, eps, device):
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
        optim.zero_grad(); loss.backward(); optim.step()
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
    del optim, best_state
    gc.collect()
    model.eval()

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx].clone()
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx].clone()}
    y_sub = y[idx]
    del X, A_hat, y, ctx
    gc.collect()
    torch.cuda.empty_cache()

    Z_sub = Z_star[idx].clone()
    del Z_star
    Z_sub = reconverge(model, Z_sub, ctx_sub)

    # Compute S_c for AEGIS SVD
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    del J_z, J_A, S
    if not edge_list:
        return None

    # 1. AEGIS SVD
    r_aegis = aegis_svd_attack(S_c, edge_list, A_sub, eps, model, Z_sub, ctx_sub)
    del S_c

    # 2. Classification-loss PGD
    r_cls = pgd_classification_attack(
        model, Z_sub, ctx_sub, y_sub, eps, edge_list, n_steps=50
    )

    # 3. Equilibrium-shift PGD (existing adaptive)
    r_shift = pgd_shift_attack(model, Z_sub, ctx_sub, eps, edge_list, n_steps=50)

    del model, Z_sub, ctx_sub, A_sub
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "aegis_shift": r_aegis["actual_shift"],
        "cls_pgd_shift": r_cls["actual_shift"],
        "cls_pgd_flipped": r_cls["n_flipped"],
        "cls_pgd_flip_rate": r_cls["flip_rate"],
        "shift_pgd_shift": r_shift["actual_shift"],
        "n_nodes": r_cls["n_total"],
        "n_edges": len(edge_list),
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

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
                try:
                    r = run_single(ds_name, data, seed, eps, device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("OOM", flush=True)
                        gc.collect(); torch.cuda.empty_cache()
                        continue
                    raise
                if r is None:
                    print("SKIP", flush=True)
                    continue
                ratio_cls = r["cls_pgd_shift"] / max(r["aegis_shift"], 1e-10)
                ratio_shift = r["shift_pgd_shift"] / max(r["aegis_shift"], 1e-10)
                print(f"AEGIS={r['aegis_shift']:.3f} ClsPGD={r['cls_pgd_shift']:.3f}({ratio_cls:.2f}) "
                      f"ShiftPGD={r['shift_pgd_shift']:.3f}({ratio_shift:.2f}) "
                      f"flipped={r['cls_pgd_flipped']}/{r['n_nodes']}", flush=True)
                rows.append({
                    "dataset": ds_name, "epsilon": eps, "seed": seed,
                    "aegis_shift": r["aegis_shift"],
                    "cls_pgd_shift": r["cls_pgd_shift"],
                    "shift_pgd_shift": r["shift_pgd_shift"],
                    "cls_pgd_flip_rate": r["cls_pgd_flip_rate"],
                    "cls_pgd_flipped": r["cls_pgd_flipped"],
                    "n_nodes": r["n_nodes"], "n_edges": r["n_edges"],
                })

    # Write CSV
    csv_path = results_dir / "classification_attack.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # Summary
    import statistics
    print("\n" + "=" * 100)
    print("CLASSIFICATION-LOSS PGD vs AEGIS SVD vs EQUILIBRIUM-SHIFT PGD")
    print("=" * 100)
    print(f"{'Dataset':<12} {'eps':>5} {'AEGIS shift':>14} {'ClsPGD shift':>14} {'ShiftPGD shift':>14} "
          f"{'Cls/AEGIS':>10} {'Shift/AEGIS':>12} {'Flip%':>8}")
    print("-" * 100)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        for eps in EPS_VALUES:
            subset = [r for r in rows if r["dataset"] == ds_name and r["epsilon"] == eps]
            if not subset:
                continue
            a = statistics.mean([r["aegis_shift"] for r in subset])
            c = statistics.mean([r["cls_pgd_shift"] for r in subset])
            s = statistics.mean([r["shift_pgd_shift"] for r in subset])
            f = statistics.mean([r["cls_pgd_flip_rate"] for r in subset]) * 100
            print(f"{ds_name:<12} {eps:>5.2f} {a:>14.4f} {c:>14.4f} {s:>14.4f} "
                  f"{c/max(a,1e-10):>10.2f} {s/max(a,1e-10):>12.2f} {f:>7.1f}%")


if __name__ == "__main__":
    sys.exit(main() or 0)
