"""Exp C4-3: Random direction search in S_c perturbation space.

Validates SVD optimality by sampling M=1000 random unit directions
in R^|E| and evaluating ||S_c * d|| for each. Compares:
  - Best-of-M damage / SVD-optimal damage (sigma_1)
  - How many random samples needed to reach 90% of SVD damage
  - Distribution stats (mean, median, max, std)

If the SVD direction is genuinely special, best-of-1000 should be
substantially below sigma_1. If many directions achieve similar damage,
the S_c landscape is flat and SVD optimality is less meaningful.

Datasets: Cora, Citeseer, WikiCS | Model: IGNN | Seeds: 10

Output: results/random_direction_search.csv

Usage:
    .venv/bin/python scripts/exp_random_direction_search.py
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
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
M_SAMPLES = 1000


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


def run_single(ds_name, data, seed, device):
    try:
        model = train_ignn(data, device, seed)

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

        n_edges = len(edge_list)

        # SVD optimal: sigma_1
        U, sigma, Vh = torch.linalg.svd(S_c, full_matrices=False)
        sigma_1 = float(sigma[0])
        sigma_2 = float(sigma[1]) if len(sigma) > 1 else 0.0

        # Random direction search
        damages = []
        for _ in range(M_SAMPLES):
            d = torch.randn(n_edges, device=S_c.device)
            d = d / d.norm()
            shift = (S_c @ d).norm()
            damages.append(float(shift))

        damages = np.array(damages)

        # Stats
        best_of_m = damages.max()
        ratio_best = best_of_m / max(sigma_1, 1e-10)

        # How many samples to reach 90% of SVD
        threshold_90 = 0.9 * sigma_1
        running_max = np.maximum.accumulate(damages)
        samples_to_90 = int(np.argmax(running_max >= threshold_90)) + 1 if running_max[-1] >= threshold_90 else M_SAMPLES

        return {
            "dataset": ds_name,
            "seed": seed,
            "sigma_1": sigma_1,
            "sigma_2": sigma_2,
            "sigma_gap": sigma_1 - sigma_2,
            "sigma_gap_ratio": (sigma_1 - sigma_2) / max(sigma_1, 1e-10),
            "best_of_M": best_of_m,
            "ratio_best_to_svd": ratio_best,
            "mean_random": float(damages.mean()),
            "median_random": float(np.median(damages)),
            "std_random": float(damages.std()),
            "p90_random": float(np.percentile(damages, 90)),
            "p99_random": float(np.percentile(damages, 99)),
            "samples_to_90pct": samples_to_90,
            "n_edges": n_edges,
            "M": M_SAMPLES,
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
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

    rows = []
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        data = datasets[ds_name]
        for seed_idx, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)})", end=" ", flush=True)
            r = run_single(ds_name, data, seed, device)
            if r:
                rows.append(r)
                print(f"sigma1={r['sigma_1']:.3f} best/svd={r['ratio_best_to_svd']:.4f} "
                      f"gap%={r['sigma_gap_ratio']:.3f} "
                      f"samples_to_90%={r['samples_to_90pct']}",
                      flush=True)
            else:
                print("SKIP", flush=True)

    # Write CSV
    csv_path = results_dir / "random_direction_search.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 95)
    print("RANDOM DIRECTION SEARCH SUMMARY (mean +/- std across seeds)")
    print("=" * 95)
    print(f"{'Dataset':<12} {'sigma_1':>10} {'Best/SVD':>10} {'Mean/SVD':>10} "
          f"{'Gap%':>8} {'Samp90':>8}")
    print("-" * 95)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        subset = [r for r in rows if r["dataset"] == ds_name]
        if not subset:
            continue
        s1 = np.mean([r["sigma_1"] for r in subset])
        br = np.mean([r["ratio_best_to_svd"] for r in subset])
        mr = np.mean([r["mean_random"] / max(r["sigma_1"], 1e-10) for r in subset])
        gap = np.mean([r["sigma_gap_ratio"] for r in subset])
        s90 = np.mean([r["samples_to_90pct"] for r in subset])
        print(f"{ds_name:<12} {s1:>10.3f} {br:>10.4f} {mr:>10.4f} {gap:>8.3f} {s90:>8.0f}")

    print("\nInterpretation:")
    print("  Best/SVD < 1.0: SVD direction is genuinely special (not achievable by random search)")
    print("  Gap% > 0.1: sigma_1 is well-separated from sigma_2 (stable direction)")
    print("  Samp90 = M: random search never reaches 90% of SVD (strong SVD advantage)")


if __name__ == "__main__":
    sys.exit(main() or 0)
