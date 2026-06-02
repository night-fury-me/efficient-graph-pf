"""Subgraph vs full-graph vulnerability ranking validation (R2 reviewer request).

For case14 and case30 (small enough for exact full-graph S_c), compare per-edge
vulnerability rankings from BFS subgraph extraction against the full-graph ground
truth. Reports Kendall τ and P@5 across 10 seeds.

Usage:
    .venv/bin/python scripts/exp_subgraph_vs_fullgraph.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: register model builders

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.contractive_pf import ContractiveGCN_PF
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass

CASES = [
    ("case14", "datasets/IEEE_case14_2000.parquet", 14, [8, 10, 12]),
    ("case30", "datasets/IEEE_case30_2000.parquet", 30, [10, 15, 20, 25]),
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(ds_path, device, seed):
    set_seed(seed)
    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(30):
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
        _, ctx_pf = model(
            batch["bus_type"].to(device), batch["Lines_connected"].to(device),
            None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device), batch["V_start"].to(device),
            batch["sizes"].to(device),
        )
    return model, ctx_pf, int(batch["sizes"][0].item())


def reconverge(model, Z_init, ctx, max_iter=200, tol=1e-7):
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < tol:
                return Z_new
            Z = Z_new
    return Z_new


def compute_edge_vulnerability(model, Z_star, ctx):
    """Return dict mapping (i_orig, j_orig) -> vulnerability score."""
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_star, ctx,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_star, ctx, J_z=J_z, J_A=J_A,
    )
    A = ctx["A_hat"]
    S_c, edge_list = constrained_sensitivity_matrix(S, A)
    vuln = {}
    for k, (i, j) in enumerate(edge_list):
        vuln[(i, j)] = float(S_c[:, k].norm())
    return vuln


def run_single(case_name, ds_path, N_full, sub_sizes, seed, device):
    model, ctx_pf, N = train_model(ds_path, device, seed)
    assert N == N_full, f"Expected N={N_full}, got {N}"

    A_full = ctx_pf["A_hat"][:N, :N]
    X_proj_full = ctx_pf["X_proj"][:N]
    Z_full = ctx_pf["Z_star"][:N]
    ctx_full = {"A_hat": A_full, "X_proj": X_proj_full}

    Z_full = reconverge(model, Z_full, ctx_full)

    # Full-graph vulnerability (ground truth)
    vuln_full = compute_edge_vulnerability(model, Z_full, ctx_full)

    results = []
    for sub_N in sub_sizes:
        if sub_N >= N:
            results.append({"sub_N": sub_N, "tau": 1.0, "p5": 1.0, "n_common": len(vuln_full)})
            continue

        sub_nodes = extract_ego_subgraph(A_full, max_nodes=sub_N)
        node_list = sub_nodes.tolist()

        A_sub = A_full[sub_nodes][:, sub_nodes]
        X_proj_sub = X_proj_full[sub_nodes]
        Z_sub = Z_full[sub_nodes]
        ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

        Z_sub = reconverge(model, Z_sub, ctx_sub)

        vuln_sub_local = compute_edge_vulnerability(model, Z_sub, ctx_sub)

        # Map subgraph indices back to original
        vuln_sub = {}
        for (i_local, j_local), v in vuln_sub_local.items():
            i_orig, j_orig = node_list[i_local], node_list[j_local]
            key = (min(i_orig, j_orig), max(i_orig, j_orig))
            vuln_sub[key] = v

        # Normalize full-graph keys too
        vuln_full_norm = {}
        for (i, j), v in vuln_full.items():
            key = (min(i, j), max(i, j))
            vuln_full_norm[key] = v

        # Common edges
        common = sorted(set(vuln_sub.keys()) & set(vuln_full_norm.keys()))
        if len(common) < 3:
            results.append({"sub_N": sub_N, "tau": None, "p5": None, "n_common": len(common)})
            continue

        full_scores = [vuln_full_norm[e] for e in common]
        sub_scores = [vuln_sub[e] for e in common]
        tau, _ = kendalltau(full_scores, sub_scores)

        # P@5: do the top-5 edges by subgraph ranking overlap with top-5 by full-graph?
        k = min(5, len(common))
        full_top = set(sorted(range(len(common)), key=lambda i: full_scores[i], reverse=True)[:k])
        sub_top = set(sorted(range(len(common)), key=lambda i: sub_scores[i], reverse=True)[:k])
        p_at_k = len(full_top & sub_top) / k

        results.append({"sub_N": sub_N, "tau": tau, "p5": p_at_k, "n_common": len(common)})

    return results


def agg(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A", None, None
    m, s = np.mean(arr), np.std(arr)
    return f"{m:.2f}±{s:.2f}", m, s


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    all_results = {}
    for case_name, ds_path, N, sub_sizes in CASES:
        if not Path(ds_path).exists():
            print(f"SKIP {case_name}: dataset not found at {ds_path}")
            continue

        key = case_name
        all_results[key] = {s: {"tau": [], "p5": [], "n_common": []} for s in sub_sizes}

        for si, seed in enumerate(SEEDS):
            print(f"  {case_name} seed {seed} ({si+1}/{len(SEEDS)})...", end=" ", flush=True)
            res = run_single(case_name, ds_path, N, sub_sizes, seed, device)
            for r in res:
                s = r["sub_N"]
                all_results[key][s]["tau"].append(r["tau"])
                all_results[key][s]["p5"].append(r["p5"])
                all_results[key][s]["n_common"].append(r["n_common"])
            taus = [r["tau"] for r in res if r["tau"] is not None]
            print(f"τ={[f'{t:.2f}' for t in taus]}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s\n")

    # Summary
    print("=" * 75)
    print("SUBGRAPH vs FULL-GRAPH VULNERABILITY RANKING (10 seeds)")
    print("=" * 75)
    print(f"{'Case':<10} {'Sub N':>6} {'|Common|':>10} {'Kendall τ':>14} {'P@5':>14}")
    print("-" * 75)
    for case_name, _, N, sub_sizes in CASES:
        if case_name not in all_results:
            continue
        for s in sub_sizes:
            d = all_results[case_name][s]
            tau_str, tau_m, _ = agg(d["tau"])
            p5_str, _, _ = agg(d["p5"])
            nc = int(np.mean([c for c in d["n_common"] if c]))
            print(f"{case_name:<10} {s:>6} {nc:>10} {tau_str:>14} {p5_str:>14}")
        print(f"{case_name:<10} {N:>6} {'(full)':>10} {'1.00±0.00':>14} {'1.00±0.00':>14}")
        print("-" * 75)


if __name__ == "__main__":
    main()
