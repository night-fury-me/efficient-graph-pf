"""10-seed Mettack vs IFT comparison (S6 reviewer request).

Wraps the existing mettack_comparison logic in a 10-seed loop across
Cora, Citeseer, WikiCS. Reports IFT win rate and damage ratio (mean±std).

Usage:
    .venv/bin/python scripts/exp_mettack_10seed.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa

from iem.adversarial import (
    _compute_structural_jacobian,
    extract_ego_subgraph,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora
from iem.examples.mettack_comparison import (
    evaluate_attack,
    mettack_edge_scores,
)

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
MAX_K = 5


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_seed(data, seed, device):
    """Train IGNN + run IFT vs Mettack comparison for one seed."""
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(100):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    S_size = len(idx)
    A_sub = A_hat[idx][:, idx]
    X_sub = X[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    with torch.no_grad():
        pseudo_labels = model.head(Z_sub).argmax(dim=1)

    n_edges = int((A_sub.abs() > 1e-10).sum() - S_size) // 2
    if n_edges < 3:
        return None

    # IFT ranking
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    ift_edges = [(i, j) for i, j, _ in attack["all_edge_vulnerabilities"]]

    # Mettack ranking
    mettack_ranked = mettack_edge_scores(
        X_sub, A_sub, pseudo_labels,
        n_features=data["n_features"], n_classes=data["n_classes"],
    )
    mettack_edges = [(i, j) for i, j, _ in mettack_ranked]

    # Damage comparison at k=1..MAX_K
    max_k = min(MAX_K, n_edges)
    damage_ratios = []
    ift_wins = 0

    all_edges_list = [
        (i, j) for i in range(S_size) for j in range(i + 1, S_size)
        if A_sub[i, j].abs() > 1e-10
    ]

    for k in range(1, max_k + 1):
        dmg_ift = evaluate_attack(model, Z_sub, ctx_sub, ift_edges[:k])
        dmg_met = evaluate_attack(model, Z_sub, ctx_sub, mettack_edges[:k])

        dmg_rands = []
        for _ in range(10):
            rand_remove = random.sample(all_edges_list, min(k, len(all_edges_list)))
            dmg_rands.append(evaluate_attack(model, Z_sub, ctx_sub, rand_remove))
        dmg_rand = np.mean(dmg_rands)

        if dmg_ift >= dmg_met:
            ift_wins += 1
        ratio = dmg_ift / dmg_met if dmg_met > 1e-10 else float("inf")
        ratio_vs_rand = dmg_ift / dmg_rand if dmg_rand > 1e-10 else float("inf")
        damage_ratios.append({
            "k": k, "ift": dmg_ift, "met": dmg_met, "rand": dmg_rand,
            "ratio_vs_met": ratio, "ratio_vs_rand": ratio_vs_rand,
        })

    return {
        "ift_wins": ift_wins,
        "total_k": max_k,
        "damage_ratios": damage_ratios,
    }


def agg(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.2f}±{s:.2f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = [
        ("Cora", _load_cora(Path("datasets/cora"))),
        ("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))),
        ("WikiCS", _load_wikics(Path("datasets/wikics"))),
    ]

    grand_ift_wins = 0
    grand_total = 0

    for ds_name, data in datasets:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        all_ratios_vs_met = {k: [] for k in range(1, MAX_K + 1)}
        all_ratios_vs_rand = {k: [] for k in range(1, MAX_K + 1)}
        ds_ift_wins = 0
        ds_total = 0

        for si, seed in enumerate(SEEDS):
            print(f"  seed {seed} ({si+1}/{len(SEEDS)})...", end=" ", flush=True)
            r = run_single_seed(data, seed, device)
            if r is None:
                print("SKIP (too few edges)", flush=True)
                continue

            ds_ift_wins += r["ift_wins"]
            ds_total += r["total_k"]

            for d in r["damage_ratios"]:
                all_ratios_vs_met[d["k"]].append(d["ratio_vs_met"])
                all_ratios_vs_rand[d["k"]].append(d["ratio_vs_rand"])

            wins = r["ift_wins"]
            total = r["total_k"]
            print(f"IFT wins {wins}/{total}, "
                  f"mean ratio vs Met: {np.mean([d['ratio_vs_met'] for d in r['damage_ratios']]):.2f}x",
                  flush=True)

        grand_ift_wins += ds_ift_wins
        grand_total += ds_total

        print(f"\n  {ds_name} summary (10 seeds):")
        print(f"  IFT win rate: {ds_ift_wins}/{ds_total} ({100*ds_ift_wins/max(ds_total,1):.0f}%)")
        print(f"  {'k':>4} {'IFT/Met ratio':>16} {'IFT/Rand ratio':>16}")
        for k in range(1, MAX_K + 1):
            if all_ratios_vs_met[k]:
                print(f"  {k:>4} {agg(all_ratios_vs_met[k]):>16} {agg(all_ratios_vs_rand[k]):>16}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"GRAND TOTAL (10 seeds × 3 datasets)")
    print(f"{'='*60}")
    print(f"IFT wins: {grand_ift_wins}/{grand_total} ({100*grand_ift_wins/max(grand_total,1):.0f}%)")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
