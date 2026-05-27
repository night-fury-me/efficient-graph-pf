"""Proposition 3(b) — Ranking transfer monotonicity coverage.

Checks the sufficient condition from Proposition 3(b):
    w_{k1} >= w_{k2}  whenever  v_{k1} > v_{k2}

where w_k = A_hat[i,j] (normalized edge weight) and v_k = ||S_c[:, k]||
(column norm of the constrained sensitivity matrix).

For each (dataset, seed): train IGNN, extract 50-node BFS subgraph,
compute S_c, then check all edge pairs (k1, k2) where v_{k1} > v_{k2}
and report the fraction satisfying w_{k1} >= w_{k2}.

Datasets:  Cora, Citeseer, WikiCS
Seeds:     10

Output: results/prop3b_coverage.txt

Usage:
    .venv/bin/python scripts/exp_prop3b_coverage.py
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

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def agg(vals, fmt=".4f"):
    arr = [v for v in vals if v is not None and not np.isnan(v)]
    if not arr:
        return "N/A"
    m, s = np.mean(arr), np.std(arr)
    return f"{m:{fmt}}+/-{s:{fmt}}"


# ---------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------

def load_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics

    datasets = {}
    print("Loading datasets...", flush=True)

    print("  Cora...", flush=True)
    datasets["Cora"] = _load_cora(Path("datasets/cora"))

    print("  Citeseer...", flush=True)
    datasets["Citeseer"] = _load_planetoid("citeseer", Path("datasets/citeseer"))

    print("  WikiCS...", flush=True)
    datasets["WikiCS"] = _load_wikics(Path("datasets/wikics"))

    for name, d in datasets.items():
        print(f"    {name}: N={d['N']}, feat={d['n_features']}, classes={d['n_classes']}",
              flush=True)
    return datasets


# ---------------------------------------------------------------
# Reconverge IGNN on subgraph
# ---------------------------------------------------------------

def reconverge(model, Z_init, ctx_sub, max_iter=200):
    """Reconverge IGNN fixed point under given context."""
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_sub)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return Z_new


# ---------------------------------------------------------------
# Run single (dataset, seed) combination
# ---------------------------------------------------------------

def run_single(dataset_name, data, seed, device):
    """Returns (n_satisfy, n_pairs, coverage) or None on failure."""
    set_seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    try:
        # Train IGNN
        model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val, best_state = 0.0, None
        for ep in range(200):
            model.train()
            logits, _, _ = model(X, A_hat)
            loss = F_func.cross_entropy(logits[data["train_mask"].to(device)],
                                        y[data["train_mask"]])
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (ep + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    logits_v, _, _ = model(X, A_hat)
                    val_acc = float((logits_v.argmax(1)[data["val_mask"].to(device)]
                                    == y[data["val_mask"]]).float().mean())
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            _, Z_star, ctx = model(X, A_hat)

        # Extract 50-node BFS subgraph
        idx = extract_ego_subgraph(A_hat, max_nodes=50)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

        # Reconverge on subgraph
        Z_sub = Z_star[idx].clone()
        Z_sub = reconverge(model, Z_sub, ctx_sub)

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

        # v_k = column norms of S_c
        v = torch.linalg.norm(S_c, dim=0)  # shape: (n_edges,)

        # w_k = normalized edge weights A_hat[i,j] for each edge
        w = torch.tensor(
            [float(A_sub[i, j]) for i, j in edge_list],
            device=S_c.device,
        )

        n_edges = len(edge_list)
        n_satisfy = 0
        n_pairs = 0

        # Check all pairs (k1, k2) where v_{k1} > v_{k2}
        for k1 in range(n_edges):
            for k2 in range(k1 + 1, n_edges):
                if v[k1] > v[k2]:
                    n_pairs += 1
                    if w[k1] >= w[k2]:
                        n_satisfy += 1
                elif v[k2] > v[k1]:
                    n_pairs += 1
                    if w[k2] >= w[k1]:
                        n_satisfy += 1
                # v[k1] == v[k2]: skip (no strict ordering required)

        coverage = n_satisfy / n_pairs if n_pairs > 0 else float("nan")
        return (n_satisfy, n_pairs, coverage)

    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return None


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t_start = time.time()

    datasets = load_datasets()
    DATASET_NAMES = ["Cora", "Citeseer", "WikiCS"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "prop3b_coverage.txt"

    all_results = {}  # dataset -> list of coverage values

    lines = []  # lines for output file

    lines.append("Proposition 3(b) Ranking Condition Coverage")
    lines.append("=" * 70)
    lines.append(f"Condition: w_{{k1}} >= w_{{k2}} whenever v_{{k1}} > v_{{k2}}")
    lines.append(f"  w_k = A_hat[i,j] (normalized edge weight)")
    lines.append(f"  v_k = ||S_c[:, k]|| (column norm of constrained sensitivity)")
    lines.append(f"Seeds: {SEEDS}")
    lines.append(f"Subgraph size: 50 nodes (BFS)")
    lines.append("")

    for ds_name in DATASET_NAMES:
        data = datasets[ds_name]
        coverages = []
        lines.append(f"--- {ds_name} ---")

        for seed_idx, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
            result = run_single(ds_name, data, seed, device)

            if result is not None:
                n_satisfy, n_pairs, coverage = result
                coverages.append(coverage)
                line = (f"  seed={seed:>5d}  "
                        f"satisfy={n_satisfy:>6d}/{n_pairs:<6d}  "
                        f"coverage={coverage:.4f}")
                lines.append(line)
                print(f"    {n_satisfy}/{n_pairs} pairs satisfy => {coverage:.4f}", flush=True)
            else:
                lines.append(f"  seed={seed:>5d}  SKIP")
                print(f"    SKIP", flush=True)

        all_results[ds_name] = coverages
        lines.append("")

    # Summary
    elapsed = time.time() - t_start
    lines.append("=" * 70)
    lines.append("SUMMARY (mean +/- std across seeds)")
    lines.append("=" * 70)
    lines.append(f"{'Dataset':<15} {'Coverage':>20} {'N seeds':>10}")
    lines.append("-" * 50)

    for ds_name in DATASET_NAMES:
        cvs = all_results.get(ds_name, [])
        if cvs:
            m, s = np.mean(cvs), np.std(cvs)
            lines.append(f"{ds_name:<15} {m:.4f}+/-{s:.4f}       {len(cvs):>5d}")
        else:
            lines.append(f"{ds_name:<15} {'N/A':>20} {'0':>10}")
    lines.append("-" * 50)

    # Grand aggregate
    all_cvs = []
    for cvs in all_results.values():
        all_cvs.extend(cvs)
    if all_cvs:
        m, s = np.mean(all_cvs), np.std(all_cvs)
        lines.append(f"{'Overall':<15} {m:.4f}+/-{s:.4f}       {len(all_cvs):>5d}")

    lines.append("")
    lines.append(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Write output
    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nResults saved to {out_path}", flush=True)

    # Print summary
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    sys.exit(main() or 0)
