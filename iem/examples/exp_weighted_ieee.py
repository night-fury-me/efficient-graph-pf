"""P2-8 Experiment: Binary vs weighted adjacency for IEEE power flow.

Compares AEGIS N-1 contingency ranking quality when the ContractiveGCN-PF
uses binary adjacency (current) vs admittance-weighted adjacency.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_weighted_ieee
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
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.adversarial import (
    _compute_structural_jacobian,
    optimal_structural_attack,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.contractive_pf import ContractiveGCN_PF

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

IEEE_CASES = [
    ("case14", "datasets/IEEE_case14_2000.parquet", 14),
    ("case30", "datasets/IEEE_case30_2000.parquet", 30),
    ("case57", "datasets/IEEE_case57_2000.parquet", 57),
    ("case118", "datasets/IEEE_case118_2000.parquet", 118),
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedContractiveGCN_PF(ContractiveGCN_PF):
    """ContractiveGCN_PF with admittance-weighted adjacency."""

    def _build_adjacency(self, Y):
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        A = Y.abs()
        A.fill_diagonal_(0.0)
        max_val = A.max()
        if max_val > 1e-10:
            A = A / max_val
        A.fill_diagonal_(1.0)
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D


def brute_force_n1(model, Z_star, ctx, edges, n_iter=50):
    A_orig = ctx["A_hat"]
    scores = []
    with torch.no_grad():
        for i, j in edges:
            A_pert = A_orig.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx, "A_hat": A_pert}
            Z = Z_star.clone()
            for _ in range(n_iter):
                Z = model.operator(Z, ctx_pert)
            scores.append(float((Z - Z_star).norm()))
    return scores


def precision_at_k(pred_ranking, true_ranking, k):
    if len(pred_ranking) < k or len(true_ranking) < k:
        k = min(len(pred_ranking), len(true_ranking))
    if k == 0:
        return 0.0
    true_topk = set()
    for i, j, _ in true_ranking[:k]:
        true_topk.add((min(i, j), max(i, j)))
    hits = 0
    for i, j, _ in pred_ranking[:k]:
        if (min(i, j), max(i, j)) in true_topk:
            hits += 1
    return hits / k


def run_single(case_name, ds_path, N_expected, seed, device, model_cls):
    set_seed(seed)

    if not Path(ds_path).exists():
        return None

    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = model_cls(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(30):
        model.train()
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()

    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)))
    with torch.no_grad():
        V_pred, ctx_pf = model(
            batch["bus_type"].to(device), batch["Lines_connected"].to(device),
            None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device), batch["V_start"].to(device),
            batch["sizes"].to(device),
        )

    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    N = int(batch["sizes"][0].item())

    A_sub = A_hat[:N, :N]
    X_proj_sub = ctx_pf["X_proj"][:N]
    Z_sub = Z_star[:N]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    # Prediction error
    V_true = batch["V_newton"].to(device)
    mae = float((V_pred - V_true).abs().mean())

    # Edges
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_sub[i, j].abs() > 1e-10:
                edges.append((i, j))
    if len(edges) < 3:
        return None

    # Tightness
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

    # AEGIS ranking
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    aegis_ranking = [(i, j, v) for i, j, v in attack["all_edge_vulnerabilities"]]

    # Brute-force N-1
    bf_scores = brute_force_n1(model, Z_sub, ctx_sub, edges)
    bf_ranking = sorted(
        [(edges[k][0], edges[k][1], bf_scores[k]) for k in range(len(edges))],
        key=lambda x: x[2], reverse=True,
    )

    # Kendall tau
    bf_dict = {(min(i, j), max(i, j)): rank for rank, (i, j, _) in enumerate(bf_ranking)}
    common = []
    for rank, (i, j, _) in enumerate(aegis_ranking):
        key = (min(i, j), max(i, j))
        if key in bf_dict:
            common.append((rank, bf_dict[key]))
    tau = kendalltau(*zip(*common))[0] if len(common) >= 3 else None

    p5 = precision_at_k(aegis_ranking, bf_ranking, 5)
    p10 = precision_at_k(aegis_ranking, bf_ranking, 10)

    return {
        "case": case_name, "N": N, "edges": len(edges),
        "tight": constr_tight, "tau": tau, "p5": p5, "p10": p10, "mae": mae,
    }


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    variants = [
        ("Binary", ContractiveGCN_PF),
        ("Weighted", WeightedContractiveGCN_PF),
    ]

    all_results = {}
    for var_name, model_cls in variants:
        all_results[var_name] = {name: [] for name, _, _ in IEEE_CASES}
        for seed_idx, seed in enumerate(SEEDS):
            print(f"=== {var_name} | Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
            for case_name, ds_path, N_exp in IEEE_CASES:
                r = run_single(case_name, ds_path, N_exp, seed, device, model_cls)
                if r:
                    all_results[var_name][case_name].append(r)
                    tau_s = f"{r['tau']:+.3f}" if r['tau'] is not None else "N/A"
                    print(f"  {case_name}: tau={tau_s} P@10={r['p10']:.2f} MAE={r['mae']:.4f}", flush=True)
                else:
                    print(f"  {case_name}: SKIP", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 130)
    print("BINARY vs WEIGHTED ADJACENCY — IEEE N-1 COMPARISON (10 seeds)")
    print("=" * 130)
    print(f"{'Variant':<10} {'Case':<10} {'Tight':>12} {'τ':>12} {'P@5':>10} {'P@10':>10} {'MAE':>12}")
    print("-" * 130)
    for var_name, _ in variants:
        for case_name, _, _ in IEEE_CASES:
            rs = all_results[var_name][case_name]
            if not rs:
                continue
            print(f"{var_name:<10} {case_name:<10} "
                  f"{agg([r['tight'] for r in rs]):>12} "
                  f"{agg([r['tau'] for r in rs]):>12} "
                  f"{agg([r['p5'] for r in rs], '.2f'):>10} "
                  f"{agg([r['p10'] for r in rs], '.2f'):>10} "
                  f"{agg([r['mae'] for r in rs], '.4f'):>12}")
        print()

    results_path = Path("docs/exp_weighted_ieee_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Binary vs Weighted Adjacency — IEEE N-1 (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Variant | Case | Tight | τ | P@5 | P@10 | MAE |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for var_name, _ in variants:
            for case_name, _, _ in IEEE_CASES:
                rs = all_results[var_name][case_name]
                if not rs:
                    continue
                f.write(f"| {var_name} | {case_name} "
                        f"| {agg([r['tight'] for r in rs])} "
                        f"| {agg([r['tau'] for r in rs])} "
                        f"| {agg([r['p5'] for r in rs], '.2f')} "
                        f"| {agg([r['p10'] for r in rs], '.2f')} "
                        f"| {agg([r['mae'] for r in rs], '.4f')} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
