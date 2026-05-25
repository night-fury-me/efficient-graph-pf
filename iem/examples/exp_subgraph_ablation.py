"""P1 Experiment: Subgraph size ablation for certificate stability.

Varies BFS ego-subgraph size N={30, 50, 100, 200} on Cora and measures
constrained tightness, median certified radius, coverage, spectral radius,
and wall-clock time. Validates that N=50 is sufficient.

Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python -m iem.examples.exp_subgraph_ablation
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
    extract_ego_subgraph,
    per_node_robust_radius,
    structural_sensitivity_matrix,
    validate_bound_tightness,
)
from iem.certify import spectral_radius
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
SUBGRAPH_SIZES = [30, 50, 100, 200]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(data, seed, device):
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
                val_acc = float((logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]]).float().mean())
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    # Use same center node for all subgraph sizes (highest degree)
    center = int(A_hat.sum(dim=1).argmax().item())

    results_by_N = []
    for max_nodes in SUBGRAPH_SIZES:
        t0 = time.time()

        idx = extract_ego_subgraph(A_hat, max_nodes=max_nodes, center=center)
        actual_N = len(idx)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
        labels_sub = y[idx]

        Z_sub = Z_star[idx].clone()
        with torch.no_grad():
            for _ in range(200):
                Z_new = model.operator(Z_sub, ctx_sub)
                if (Z_new - Z_sub).norm() < 1e-7:
                    break
                Z_sub = Z_new
        Z_sub = Z_new
        logits_sub = model.head(Z_sub)

        n_edges = int((A_sub.abs() > 1e-10).sum() - actual_N) // 2

        def F_z(z, _ctx=ctx_sub):
            return model.operator(z.reshape(Z_sub.shape), _ctx).reshape(-1)
        rho = spectral_radius(F_z, Z_sub)

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

        node_certs = per_node_robust_radius(S, Z_sub, logits_sub, labels_sub, rho, model.head)
        det_nontrivial = node_certs["radii"][node_certs["radii"] > 1e-6]

        t_total = time.time() - t0

        results_by_N.append({
            "max_nodes": max_nodes,
            "actual_N": actual_N,
            "n_edges": n_edges,
            "rho": rho,
            "constr_tight": constr_tight,
            "med_r": float(det_nontrivial.median()) if len(det_nontrivial) > 0 else 0.0,
            "cert_coverage": node_certs["frac_correct_and_certified"],
            "t_total": t_total,
        })

    return results_by_N


def agg(vals, fmt=".3f"):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}±{s:{fmt}}"


def agg_pct(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m*100:.0f}±{s*100:.0f}%"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("Loading Cora...", flush=True)
    data = _load_cora(Path("datasets/cora"))

    # Organize results by subgraph size
    all_results = {N: [] for N in SUBGRAPH_SIZES}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ===", flush=True)
        results_by_N = run_single(data, seed, device)
        for entry in results_by_N:
            all_results[entry["max_nodes"]].append(entry)
            print(f"  N={entry['max_nodes']:>3}: tight={entry['constr_tight']:.3f} "
                  f"r={entry['med_r']:.4f} rho={entry['rho']:.3f} t={entry['t_total']:.1f}s",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    # Table
    print("=" * 100)
    print("SUBGRAPH SIZE ABLATION on Cora (10 seeds)")
    print("=" * 100)
    print(f"{'N':>5} {'Tightness':>14} {'Med r_v':>14} {'Cert%':>12} {'rho':>14} {'Time':>10}")
    print("-" * 100)
    for N in SUBGRAPH_SIZES:
        rs = all_results[N]
        print(f"{N:>5} "
              f"{agg([r['constr_tight'] for r in rs]):>14} "
              f"{agg([r['med_r'] for r in rs]):>14} "
              f"{agg_pct([r['cert_coverage'] for r in rs]):>12} "
              f"{agg([r['rho'] for r in rs]):>14} "
              f"{np.mean([r['t_total'] for r in rs]):>9.1f}s")

    # Save
    results_path = Path("docs/exp_subgraph_ablation_results.md")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Subgraph Size Ablation on Cora (10 seeds)\n\n")
        f.write(f"Seeds: {SEEDS}\n\n")
        f.write("| N | Tightness | Med r_v | Cert% | ρ | Time |\n")
        f.write("|---|---|---|---|---|---|\n")
        for N in SUBGRAPH_SIZES:
            rs = all_results[N]
            f.write(f"| {N} "
                    f"| {agg([r['constr_tight'] for r in rs])} "
                    f"| {agg([r['med_r'] for r in rs])} "
                    f"| {agg_pct([r['cert_coverage'] for r in rs])} "
                    f"| {agg([r['rho'] for r in rs])} "
                    f"| {np.mean([r['t_total'] for r in rs]):.1f}s |\n")
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
