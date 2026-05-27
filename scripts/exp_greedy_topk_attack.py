"""Exp C4-1: Greedy top-k discrete edge removal attack curves.

Compares cumulative damage from removing edges in different orderings:
  1. AEGIS: top-k by S_c column-norm vulnerability ranking
  2. Degree: top-k by max(d_i, d_j)
  3. Greedy-optimal: sequential brute-force (remove most-damaging edge at each step)
  4. Random: random ordering (averaged over 5 shuffles)

This is a fully black-box, gradient-free comparison that closes the
"circular attack evaluation" critique by showing AEGIS rankings predict
discrete damage without sharing any optimization pathway.

Datasets: Cora, Citeseer, WikiCS | Model: IGNN | Seeds: 10 | k: 1..10

Output: results/greedy_topk_attack.csv

Usage:
    .venv/bin/python scripts/exp_greedy_topk_attack.py
"""

from __future__ import annotations

import csv
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
MAX_K = 10
N_RANDOM_SHUFFLES = 5


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


def get_edge_rankings(model, data, device, seed):
    """Returns (model, Z_sub, ctx_sub, edge_list, aegis_order, degree_order)."""
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

    Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)

    # Compute S_c
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    # AEGIS ranking: by S_c column norms
    col_norms = torch.stack([S_c[:, k].norm() for k in range(S_c.shape[1])])
    aegis_order = col_norms.argsort(descending=True).tolist()

    # Degree ranking: by max(d_i, d_j)
    deg = (A_sub.abs() > 1e-10).float().sum(dim=1)
    deg_scores = torch.tensor([max(float(deg[i]), float(deg[j])) for i, j in edge_list])
    degree_order = deg_scores.argsort(descending=True).tolist()

    return model, Z_sub, ctx_sub, edge_list, aegis_order, degree_order


def measure_cumulative_damage(model, Z_clean, ctx_sub, edge_list, removal_order, max_k):
    """Remove edges sequentially in given order, measure cumulative damage at each k."""
    A = ctx_sub["A_hat"]
    damages = []
    A_current = A.clone()

    for step in range(min(max_k, len(removal_order))):
        edge_idx = removal_order[step]
        i, j = edge_list[edge_idx]
        A_current = A_current.clone()
        A_current[i, j] = 0.0
        A_current[j, i] = 0.0

        ctx_pert = {**ctx_sub, "A_hat": A_current}
        Z_pert = reconverge(model, Z_clean, ctx_pert)
        damage = float((Z_pert - Z_clean).norm())
        damages.append(damage)

    return damages


def greedy_sequential_removal(model, Z_clean, ctx_sub, edge_list, max_k):
    """Brute-force greedy: at each step, remove the edge causing max damage."""
    A_current = ctx_sub["A_hat"].clone()
    remaining = list(range(len(edge_list)))
    greedy_order = []
    damages = []

    for step in range(min(max_k, len(remaining))):
        best_damage = -1
        best_idx = -1
        for edge_idx in remaining:
            i, j = edge_list[edge_idx]
            A_test = A_current.clone()
            A_test[i, j] = 0.0
            A_test[j, i] = 0.0
            ctx_test = {**ctx_sub, "A_hat": A_test}
            Z_test = reconverge(model, Z_clean, ctx_test)
            dmg = float((Z_test - Z_clean).norm())
            if dmg > best_damage:
                best_damage = dmg
                best_idx = edge_idx

        greedy_order.append(best_idx)
        remaining.remove(best_idx)
        i, j = edge_list[best_idx]
        A_current = A_current.clone()
        A_current[i, j] = 0.0
        A_current[j, i] = 0.0
        damages.append(best_damage)

    return greedy_order, damages


def run_single(ds_name, data, seed, device):
    """Run one (dataset, seed) and return rows for CSV."""
    try:
        model = train_ignn(data, device, seed)
        result = get_edge_rankings(model, data, device, seed)
        if result is None:
            return None

        model, Z_sub, ctx_sub, edge_list, aegis_order, degree_order = result
        n_edges = len(edge_list)
        k = min(MAX_K, n_edges)

        # 1. AEGIS-ranked removal
        aegis_damages = measure_cumulative_damage(
            model, Z_sub, ctx_sub, edge_list, aegis_order, k
        )

        # 2. Degree-ranked removal
        degree_damages = measure_cumulative_damage(
            model, Z_sub, ctx_sub, edge_list, degree_order, k
        )

        # 3. Greedy-optimal removal (brute-force)
        print("    greedy...", end="", flush=True)
        _, greedy_damages = greedy_sequential_removal(
            model, Z_sub, ctx_sub, edge_list, k
        )

        # 4. Random removal (averaged over N_RANDOM_SHUFFLES)
        random_damages_all = []
        for _ in range(N_RANDOM_SHUFFLES):
            rand_order = list(range(n_edges))
            random.shuffle(rand_order)
            rd = measure_cumulative_damage(
                model, Z_sub, ctx_sub, edge_list, rand_order, k
            )
            random_damages_all.append(rd)
        random_damages = [
            np.mean([random_damages_all[s][ki] for s in range(N_RANDOM_SHUFFLES)])
            for ki in range(k)
        ]

        # Build rows
        rows = []
        methods = {
            "AEGIS": aegis_damages,
            "Degree": degree_damages,
            "Greedy": greedy_damages,
            "Random": random_damages,
        }
        for method, damages in methods.items():
            for ki, dmg in enumerate(damages):
                rows.append({
                    "dataset": ds_name,
                    "seed": seed,
                    "method": method,
                    "k": ki + 1,
                    "cumulative_damage": dmg,
                })
        return rows

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(" OOM", flush=True)
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

    all_rows = []
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        data = datasets[ds_name]
        for seed_idx, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)})", end="", flush=True)
            rows = run_single(ds_name, data, seed, device)
            if rows:
                all_rows.extend(rows)
                # Print summary for this seed
                for method in ["AEGIS", "Greedy", "Degree", "Random"]:
                    subset = [r for r in rows if r["method"] == method]
                    dmg5 = [r["cumulative_damage"] for r in subset if r["k"] == 5]
                    if dmg5:
                        print(f" {method}@5={dmg5[0]:.3f}", end="")
                print(flush=True)
            else:
                print(" SKIP", flush=True)

    # Write CSV
    csv_path = results_dir / "greedy_topk_attack.csv"
    fieldnames = ["dataset", "seed", "method", "k", "cumulative_damage"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV: {csv_path}")

    # Summary table
    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 100)
    print("GREEDY TOP-K ATTACK SUMMARY (mean +/- std across seeds)")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Method':<10} {'k=1':>12} {'k=3':>12} {'k=5':>12} {'k=10':>12}")
    print("-" * 100)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        for method in ["Greedy", "AEGIS", "Degree", "Random"]:
            parts = []
            for ki in [1, 3, 5, 10]:
                vals = [r["cumulative_damage"] for r in all_rows
                        if r["dataset"] == ds_name and r["method"] == method and r["k"] == ki]
                if vals:
                    parts.append(f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}")
                else:
                    parts.append("---")
            print(f"{ds_name:<12} {method:<10} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12} {parts[3]:>12}")
        print("-" * 100)

    # AEGIS / Greedy ratio
    print("\nAEGIS / Greedy damage ratio (higher = closer to optimal):")
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        for ki in [5, 10]:
            aegis_vals = [r["cumulative_damage"] for r in all_rows
                          if r["dataset"] == ds_name and r["method"] == "AEGIS" and r["k"] == ki]
            greedy_vals = [r["cumulative_damage"] for r in all_rows
                           if r["dataset"] == ds_name and r["method"] == "Greedy" and r["k"] == ki]
            if aegis_vals and greedy_vals:
                ratios = [a / max(g, 1e-10) for a, g in zip(aegis_vals, greedy_vals)]
                print(f"  {ds_name} k={ki}: {np.mean(ratios):.3f} +/- {np.std(ratios):.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
