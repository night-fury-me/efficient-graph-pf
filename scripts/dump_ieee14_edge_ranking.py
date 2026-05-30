"""Dump per-edge AEGIS vulnerability and brute-force N-1 severity for IEEE
case14, used by paper/figures/fig_ieee14_case.tex.

Reuses the same training + AEGIS pipeline as
`iem.examples.adversarial_ieee` but exposes the per-edge ranking arrays
that the summary script discards. Multi-seed mean rankings are saved so
the figure's edge highlights are not seed-sensitive.

Output:
    paper/figures/data/ieee14_edges.csv
        columns: u, v, aegis_score_mean, aegis_score_std,
                 n1_severity_mean, n1_severity_std,
                 aegis_rank_mean, n1_rank_mean, seeds_used

Usage:
    .venv/bin/python scripts/dump_ieee14_edge_ranking.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401 -- register model builders
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.adversarial_ieee import set_seed
from iem.examples.contractive_pf import ContractiveGCN_PF

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
DEFAULT_CASE = "case14"
DEFAULT_PARQUET = "datasets/IEEE_case14_2000.parquet"


def run_one(case_name: str, ds_path: str, seed: int, device: torch.device) -> dict | None:
    set_seed(seed)
    if not Path(ds_path).exists():
        return None

    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    train_ds = Subset(ds, range(min(200, len(ds))))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ep in range(30):
        model.train()
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device),
                batch["Lines_connected"].to(device),
                None,
                batch["Y_Lines"].to(device),
                batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device),
                batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)))
    with torch.no_grad():
        _V, ctx_pf = model(
            batch["bus_type"].to(device),
            batch["Lines_connected"].to(device),
            None,
            batch["Y_Lines"].to(device),
            batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device),
            batch["V_start"].to(device),
            batch["sizes"].to(device),
        )

    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]
    X_proj = ctx_pf["X_proj"]
    N = int(batch["sizes"][0].item())

    A_sub = A_hat[:N, :N]
    Z_sub = Z_star[:N]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj[:N]}

    Z = Z_sub.clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    Z_sub = Z_new

    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A,
    )

    # AEGIS per-edge ranking
    attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
    aegis = {(int(i), int(j)): float(v) for i, j, v in attack["all_edge_vulnerabilities"]}

    # Brute-force N-1 ranking
    bf = greedy_structural_attack(model, Z_sub, ctx_sub)
    n1 = {(int(i), int(j)): float(s) for i, j, s in bf}

    return {"aegis": aegis, "n1": n1, "N": N}


def aggregate(results: list[dict]) -> dict[tuple[int, int], dict]:
    """Aggregate per-edge across seeds."""
    edges: set[tuple[int, int]] = set()
    for r in results:
        edges.update(r["aegis"].keys())
        edges.update(r["n1"].keys())

    out: dict[tuple[int, int], dict] = {}
    for key in edges:
        a_vals = [r["aegis"].get(key, 0.0) for r in results]
        n_vals = [r["n1"].get(key, 0.0) for r in results]
        out[key] = {
            "aegis_score_mean": float(np.mean(a_vals)),
            "aegis_score_std": float(np.std(a_vals)),
            "n1_severity_mean": float(np.mean(n_vals)),
            "n1_severity_std": float(np.std(n_vals)),
        }

    # Compute mean ranks (lower index = more critical)
    for r in results:
        a_sorted = sorted(r["aegis"].items(), key=lambda kv: -kv[1])
        n_sorted = sorted(r["n1"].items(), key=lambda kv: -kv[1])
        a_rank = {k: i for i, (k, _) in enumerate(a_sorted)}
        n_rank = {k: i for i, (k, _) in enumerate(n_sorted)}
        for k in out:
            out[k].setdefault("_aegis_ranks", []).append(a_rank.get(k, len(a_sorted)))
            out[k].setdefault("_n1_ranks", []).append(n_rank.get(k, len(n_sorted)))

    for k, v in out.items():
        v["aegis_rank_mean"] = float(np.mean(v.pop("_aegis_ranks")))
        v["n1_rank_mean"] = float(np.mean(v.pop("_n1_ranks")))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3, help="Number of seeds to average (default: 3)")
    p.add_argument("--case", default=DEFAULT_CASE)
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--out", default="paper/figures/data/ieee14_edges.csv")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, seeds={args.seeds}, case={args.case}")

    t0 = time.time()
    results = []
    for i, seed in enumerate(SEEDS[: args.seeds]):
        print(f"  seed {seed} ({i+1}/{args.seeds})...", flush=True)
        r = run_one(args.case, args.parquet, seed, device)
        if r is None:
            print(f"    skipped (dataset {args.parquet} not found)")
            sys.exit(1)
        results.append(r)
        print(f"    edges={len(r['aegis'])}, t={time.time()-t0:.1f}s")

    agg = aggregate(results)
    N = results[0]["N"]

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("u,v,aegis_score_mean,aegis_score_std,n1_severity_mean,n1_severity_std,"
                "aegis_rank_mean,n1_rank_mean,seeds_used\n")
        for (u, v), d in sorted(agg.items(), key=lambda kv: kv[1]["aegis_rank_mean"]):
            f.write(f"{u+1},{v+1},{d['aegis_score_mean']:.6f},{d['aegis_score_std']:.6f},"
                    f"{d['n1_severity_mean']:.6f},{d['n1_severity_std']:.6f},"
                    f"{d['aegis_rank_mean']:.2f},{d['n1_rank_mean']:.2f},{args.seeds}\n")

    print(f"\nWrote {len(agg)} edges to {out_path} (N={N}, total t={time.time()-t0:.1f}s)")


if __name__ == "__main__":
    sys.exit(main() or 0)
