"""P2 Experiment: Top-k precision for N-1 contingency + LODF comparison.

Reports precision@5, precision@10 of AEGIS vulnerability ranking vs
brute-force N-1 ground truth on IEEE case14/30/57/118. Also computes
LODF-based linear screening ranking for comparison.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_topk_precision_ieee
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
    constrained_sensitivity_matrix,
    greedy_structural_attack,
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


def compute_lodf_ranking(A_sub: torch.Tensor) -> list:
    """Approximate LODF-based ranking using DC power flow linearization.

    LODF (Line Outage Distribution Factor) measures the change in power
    flow on remaining lines when one line is tripped. We approximate this
    via the graph Laplacian pseudoinverse (DC approximation: B-matrix).

    Returns list of (i, j, lodf_severity) sorted by descending severity.
    """
    N = A_sub.shape[0]
    A_bin = (A_sub.abs() > 1e-10).float()
    A_bin.fill_diagonal_(0)

    # Build susceptance matrix (B = A for unit-impedance DC approximation)
    B = A_bin.clone()
    L = torch.diag(B.sum(dim=1)) - B

    # Pseudoinverse of Laplacian (remove rank-1 null space)
    try:
        eigvals, eigvecs = torch.linalg.eigh(L.cpu())
        eigvals = eigvals.to(A_sub.device)
        eigvecs = eigvecs.to(A_sub.device)
        mask = eigvals.abs() > 1e-8
        L_pinv = (eigvecs[:, mask] @ torch.diag(1.0 / eigvals[mask]) @ eigvecs[:, mask].T)
    except Exception:
        return []

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_bin[i, j] > 0.5:
                # PTDF for line (i,j): X_ij = L_pinv[i,i] + L_pinv[j,j] - 2*L_pinv[i,j]
                x_ij = float(L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j])
                # LODF severity ~ 1/x_ij (lines with small reactance are more critical)
                severity = 1.0 / max(abs(x_ij), 1e-10)
                edges.append((i, j, severity))

    edges.sort(key=lambda x: x[2], reverse=True)
    return edges


def precision_at_k(pred_ranking: list, true_ranking: list, k: int) -> float:
    """Compute precision@k: fraction of predicted top-k in true top-k."""
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


def run_single(case_name, ds_path, N_expected, seed, device):
    set_seed(seed)

    if not Path(ds_path).exists():
        return None

    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
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
    X_proj = ctx_pf["X_proj"]
    N = int(batch["sizes"][0].item())

    A_sub = A_hat[:N, :N]
    X_proj_sub = X_proj[:N]
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

    n_edges = int((A_sub.abs() > 1e-10).sum() - N) // 2

    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    # Constrained tightness
    tight_results = validate_bound_tightness(
        lambda z, c: model.operator(z, c), model, Z_sub, ctx_sub, S,
        epsilons=[0.01], n_random=3,
    )
    constr_tight = tight_results[0]["constr_tightness"]

    # AEGIS vulnerability ranking
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    aegis_ranking = [(i, j, v) for i, j, v in attack["all_edge_vulnerabilities"]]

    # Brute-force N-1 ground truth
    bf = greedy_structural_attack(model, Z_sub, ctx_sub)

    # LODF ranking
    lodf_ranking = compute_lodf_ranking(A_sub)

    # Kendall tau: AEGIS vs brute-force
    bf_dict = {(min(i, j), max(i, j)): rank for rank, (i, j, _) in enumerate(bf)}
    aegis_common = []
    for rank, (i, j, _) in enumerate(aegis_ranking):
        key = (min(i, j), max(i, j))
        if key in bf_dict:
            aegis_common.append((rank, bf_dict[key]))
    tau_aegis = None
    if len(aegis_common) >= 3:
        a, b = zip(*aegis_common)
        tau_aegis, _ = kendalltau(a, b)

    # Kendall tau: LODF vs brute-force
    lodf_common = []
    for rank, (i, j, _) in enumerate(lodf_ranking):
        key = (min(i, j), max(i, j))
        if key in bf_dict:
            lodf_common.append((rank, bf_dict[key]))
    tau_lodf = None
    if len(lodf_common) >= 3:
        a, b = zip(*lodf_common)
        tau_lodf, _ = kendalltau(a, b)

    # Precision@k
    p_at_5_aegis = precision_at_k(aegis_ranking, bf, 5)
    p_at_10_aegis = precision_at_k(aegis_ranking, bf, 10)
    p_at_5_lodf = precision_at_k(lodf_ranking, bf, 5)
    p_at_10_lodf = precision_at_k(lodf_ranking, bf, 10)

    return {
        "case": case_name, "N": N, "edges": n_edges,
        "rho": rho, "constr_tight": constr_tight,
        "tau_aegis": tau_aegis, "tau_lodf": tau_lodf,
        "p5_aegis": p_at_5_aegis, "p10_aegis": p_at_10_aegis,
        "p5_lodf": p_at_5_lodf, "p10_lodf": p_at_10_lodf,
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

    all_results = {name: [] for name, _, _ in IEEE_CASES}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        for case_name, ds_path, N_exp in IEEE_CASES:
            r = run_single(case_name, ds_path, N_exp, seed, device)
            if r:
                all_results[case_name].append(r)
                tau_s = f"{r['tau_aegis']:+.3f}" if r['tau_aegis'] is not None else "N/A"
                print(f"  {case_name}: tau={tau_s} P@5={r['p5_aegis']:.2f} P@10={r['p10_aegis']:.2f}",
                      flush=True)
            else:
                print(f"  {case_name}: SKIP", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Table
    print("=" * 120)
    print("IEEE N-1 TOP-K PRECISION + LODF COMPARISON (10 seeds)")
    print("=" * 120)
    print(f"{'Case':<10} {'N':>4} {'|E|':>5} {'Tight':>12} {'AEGIS τ':>12} "
          f"{'P@5':>10} {'P@10':>10} {'LODF τ':>12} {'LODF P@5':>10} {'LODF P@10':>10}")
    print("-" * 120)
    for case_name, _, _ in IEEE_CASES:
        rs = all_results[case_name]
        if not rs:
            print(f"{case_name:<10} — dataset not found")
            continue
        N = rs[0]["N"]
        E = rs[0]["edges"]
        print(f"{case_name:<10} {N:>4} {E:>5} "
              f"{agg([r['constr_tight'] for r in rs]):>12} "
              f"{agg([r['tau_aegis'] for r in rs]):>12} "
              f"{agg([r['p5_aegis'] for r in rs], '.2f'):>10} "
              f"{agg([r['p10_aegis'] for r in rs], '.2f'):>10} "
              f"{agg([r['tau_lodf'] for r in rs]):>12} "
              f"{agg([r['p5_lodf'] for r in rs], '.2f'):>10} "
              f"{agg([r['p10_lodf'] for r in rs], '.2f'):>10}")

    # Save
    results_path = Path("docs/exp_topk_precision_ieee_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# IEEE N-1 Top-K Precision + LODF Comparison (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| Case | N | |E| | Tight | AEGIS τ | P@5 | P@10 | LODF τ | LODF P@5 | LODF P@10 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for case_name, _, _ in IEEE_CASES:
            rs = all_results[case_name]
            if not rs:
                continue
            f.write(f"| {case_name} | {rs[0]['N']} | {rs[0]['edges']} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['tau_aegis'] for r in rs])} "
                    f"| {agg([r['p5_aegis'] for r in rs], '.2f')} "
                    f"| {agg([r['p10_aegis'] for r in rs], '.2f')} "
                    f"| {agg([r['tau_lodf'] for r in rs])} "
                    f"| {agg([r['p5_lodf'] for r in rs], '.2f')} "
                    f"| {agg([r['p10_lodf'] for r in rs], '.2f')} |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
