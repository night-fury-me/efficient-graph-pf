"""P2 Experiment: Tightness vs. epsilon curve.

Measures how the constrained first-order tightness ratio degrades as
perturbation budget increases from eps=0.005 to eps=0.20.

Tightness = actual_shift / predicted_shift, where:
  predicted_shift = eps * sigma_1(S_c)
  actual_shift    = ||z*' - z*|| after reconverging with S_c-optimal perturbation

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_tightness_vs_epsilon
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(name, data, seed, device):
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
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float(
                    (logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]])
                    .float()
                    .mean()
                )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)
    sigma1 = float(sigma_c[0])

    results = []
    for eps in EPSILONS:
        predicted_shift = eps * sigma1

        dA = torch.zeros_like(A_sub)
        weights = eps * Vh_c[0]
        for k, (i, j) in enumerate(edge_list):
            dA[i, j] = float(weights[k])
            dA[j, i] = float(weights[k])
        ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}

        Z = Z_sub.clone()
        with torch.no_grad():
            for _ in range(200):
                Z_new = model.operator(Z, ctx_pert)
                if (Z_new - Z).norm() < 1e-8:
                    break
                Z = Z_new
        actual_shift = float((Z_new - Z_sub).norm())

        tightness = actual_shift / predicted_shift if predicted_shift > 1e-12 else float("nan")

        logits_clean = model.head(Z_sub)
        logits_pert = model.head(Z_new)
        pred_clean = logits_clean.argmax(dim=1)
        pred_pert = logits_pert.argmax(dim=1)
        n_nodes = pred_clean.shape[0]
        n_flipped = int((pred_clean != pred_pert).sum())
        flip_rate = n_flipped / n_nodes

        results.append({
            "epsilon": eps,
            "predicted_shift": predicted_shift,
            "actual_shift": actual_shift,
            "tightness": tightness,
            "flip_rate": flip_rate,
            "n_flipped": n_flipped,
        })

    return results


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None and not np.isnan(v)]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    all_results = {name: {eps: [] for eps in EPSILONS} for name, _ in datasets}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for name, data in datasets:
            r = run_single(name, data, seed, device)
            if r:
                for entry in r:
                    all_results[name][entry["epsilon"]].append(entry)
                print(f"  {name}: done", flush=True)
            else:
                print(f"  {name}: SKIP", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 110)
    print("TIGHTNESS vs EPSILON (10 seeds, constrained S_c-optimal perturbation)")
    print("=" * 110)
    print(f"{'Dataset':<12} {'eps':>6} {'Tightness':>14} {'Pred shift':>14} {'Act shift':>14} {'Flip%':>10}")
    print("-" * 110)
    for name, _ in datasets:
        for eps in EPSILONS:
            rs = all_results[name][eps]
            print(
                f"{name:<12} {eps:>6.3f} "
                f"{agg([r['tightness'] for r in rs]):>14} "
                f"{agg([r['predicted_shift'] for r in rs]):>14} "
                f"{agg([r['actual_shift'] for r in rs]):>14} "
                f"{agg([r['flip_rate'] for r in rs]):>10}"
            )
        print()

    results_path = Path("docs/exp_tightness_vs_epsilon_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Tightness vs Epsilon (10 seeds, S_c-optimal constrained perturbation)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Dataset | ε | Tightness | Pred shift | Act shift | Flip% |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, _ in datasets:
            for eps in EPSILONS:
                rs = all_results[name][eps]
                f.write(
                    f"| {name} | {eps} "
                    f"| {agg([r['tightness'] for r in rs])} "
                    f"| {agg([r['predicted_shift'] for r in rs])} "
                    f"| {agg([r['actual_shift'] for r in rs])} "
                    f"| {agg([r['flip_rate'] for r in rs])} |\n"
                )
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
