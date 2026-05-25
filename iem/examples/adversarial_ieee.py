"""Adversarial Equilibrium Theory on IEEE power flow benchmarks.

Runs constrained tightness, N-1 ranking (IFT vs brute-force), and phase
transition analysis on IEEE case14/30/57/118. No subgraph extraction needed
for case14/30/57 (small enough for full Jacobian). Case118 uses full grid.

Multi-seed (10 seeds) for publication-ready mean±std.

Usage:
    .venv/bin/python -m iem.examples.adversarial_ieee
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import models  # noqa: register model builders

from iem.adversarial import (
    _compute_structural_jacobian,
    certified_shift_bound,
    constrained_sensitivity_matrix,
    critical_perturbation_budget,
    extract_W_spectral_norm,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.contractive_pf import ContractiveGCN_PF

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader

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


def run_ieee_single(case_name, ds_path, N_expected, seed, device):
    """Train ContractiveGCN-PF + adversarial analysis on one IEEE case."""
    set_seed(seed)

    if not Path(ds_path).exists():
        return None

    from torch.utils.data import Subset
    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(30):
        model.train()
        total_loss = 0
        n_batch = 0
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
            total_loss += loss.item()
            n_batch += 1

    model.eval()
    train_rmse = (total_loss / max(n_batch, 1)) ** 0.5

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
    X_proj = ctx_pf["X_proj"]
    N = int(batch["sizes"][0].item())

    # Extract single grid (first in batch)
    A_sub = A_hat[:N, :N]
    X_proj_sub = X_proj[:N]
    Z_sub = Z_star[:N]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    # Reconverge
    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    n_edges = int((A_sub.abs() > 1e-10).sum() - N) // 2
    D = Z_sub.numel()

    # Spectral radius
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    # Compute S
    t0 = time.time()
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )
    t_total = time.time() - t0

    # Constrained tightness
    tight_results = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.01], n_random=3,
    )
    constr_tight = tight_results[0]["constr_tightness"]
    atk_adv = tight_results[0]["attack_advantage"]

    # Constrained sigma_1
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    sigma_1_c = float(torch.linalg.svdvals(S_c)[0]) if S_c.shape[1] > 0 else 0

    # Critical budget
    try:
        W_norm = extract_W_spectral_norm(model)
    except ValueError:
        W_norm = 1.0
    budget = critical_perturbation_budget(rho, W_norm)

    # N-1 ranking: IFT vs brute-force
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    ift_vuln = {(i, j): v for i, j, v in attack["all_edge_vulnerabilities"]}

    bf = greedy_structural_attack(model, Z_sub, ctx_sub)
    bf_rank = {(i, j): r for r, (i, j, _) in enumerate(bf)}

    # Kendall tau
    common = []
    for i, j, bf_shift in bf:
        key = (i, j)
        if key in ift_vuln:
            common.append((bf_shift, ift_vuln[key]))
    tau_n1 = None
    top5_agree = None
    if len(common) >= 3:
        a, b = zip(*common)
        tau_n1, _ = kendalltau(a, b)
        k = min(5, len(common))
        bf_top = set(range(k))
        ift_ranked = sorted(range(len(common)), key=lambda i: common[i][1], reverse=True)
        ift_top = set(ift_ranked[:k])
        top5_agree = len(bf_top & ift_top) / k

    return {
        "case": case_name, "N": N, "edges": n_edges, "D": D,
        "rho": rho, "sigma_1_c": sigma_1_c,
        "constr_tight": constr_tight, "atk_adv": atk_adv,
        "eps_crit": budget["epsilon_crit"],
        "tau_n1": tau_n1, "top5": top5_agree,
        "train_rmse": train_rmse, "t_total": t_total,
    }


def agg(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.3f}±{s:.3f}"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    all_results = {name: [] for name, _, _ in IEEE_CASES}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for case_name, ds_path, N_exp in IEEE_CASES:
            r = run_ieee_single(case_name, ds_path, N_exp, seed, device)
            if r:
                all_results[case_name].append(r)
                print(f"  {case_name}: N={r['N']} rho={r['rho']:.3f} tight={r['constr_tight']:.3f} "
                      f"tau={r['tau_n1']:+.3f} t={r['t_total']:.1f}s" if r['tau_n1'] else
                      f"  {case_name}: N={r['N']} rho={r['rho']:.3f} tight={r['constr_tight']:.3f} "
                      f"tau=N/A t={r['t_total']:.1f}s", flush=True)
            else:
                print(f"  {case_name}: SKIP (dataset not found)", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Summary table
    print("=" * 100)
    print("IEEE POWER FLOW — ADVERSARIAL ANALYSIS (10 seeds)")
    print("=" * 100)
    print(f"{'Case':<10} {'N':>4} {'|E|':>5} {'rho':>12} {'Tightness':>12} "
          f"{'AtkAdv':>12} {'eps_crit':>12} {'N-1 tau':>12} {'Top-5':>12} {'Time':>8}")
    print("-" * 100)
    for case_name, _, _ in IEEE_CASES:
        rs = all_results[case_name]
        if not rs:
            print(f"{case_name:<10} — dataset not found")
            continue
        N = rs[0]["N"]
        E = rs[0]["edges"]
        print(f"{case_name:<10} {N:>4} {E:>5} "
              f"{agg([r['rho'] for r in rs]):>12} "
              f"{agg([r['constr_tight'] for r in rs]):>12} "
              f"{agg([r['atk_adv'] for r in rs]):>12} "
              f"{agg([r['eps_crit'] for r in rs]):>12} "
              f"{agg([r['tau_n1'] for r in rs]):>12} "
              f"{agg([r['top5'] for r in rs]):>12} "
              f"{np.mean([r['t_total'] for r in rs]):>7.1f}s")

    # Save
    results_path = Path("docs/ieee_adversarial_results.md")
    with open(results_path, "w") as f:
        f.write("# IEEE Power Flow — Adversarial Analysis (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Case | N | |E| | ρ | Tightness | AtkAdv | ε_crit | N-1 τ | Top-5 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for case_name, _, _ in IEEE_CASES:
            rs = all_results[case_name]
            if not rs:
                continue
            f.write(f"| {case_name} | {rs[0]['N']} | {rs[0]['edges']} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['atk_adv'] for r in rs])} "
                    f"| {agg([r['eps_crit'] for r in rs])} "
                    f"| {agg([r['tau_n1'] for r in rs])} "
                    f"| {agg([r['top5'] for r in rs])} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
