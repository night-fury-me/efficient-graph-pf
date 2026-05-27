"""Exp C4-5: Finite-difference per-edge sensitivity baseline.

For each edge (i,j): perturb edge weight by delta=0.001, run forward pass,
measure ||Delta_z|| / delta. This gives a per-edge sensitivity ranking
WITHOUT any IFT, Neumann series, or SVD — just |E| forward passes.

Compares:
  1. FD ranking vs S_c ranking (Kendall tau) — should be ~1.0 if IFT is correct
  2. FD ranking vs discrete ground truth (tau) — matches S_c's transfer?
  3. S_c ranking vs discrete ground truth (tau) — reference

If tau_FD ~= tau_Sc, the IFT machinery doesn't help for ranking.
If tau_Sc > tau_FD, the resolvent amplification captures something
that forward-pass probing misses.

Also reports timing: FD requires |E| forward passes, S_c requires IFT + SVD.

Datasets: Cora, Citeseer, WikiCS | Model: IGNN | Seeds: 10

Output: results/finite_difference_baseline.csv

Usage:
    .venv/bin/python scripts/exp_finite_difference_baseline.py
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
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _load_cora

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
FD_DELTA = 0.001


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


def finite_difference_sensitivity(model, Z_clean, ctx_sub, edge_list, delta=FD_DELTA):
    """Per-edge finite-difference sensitivity: perturb each edge, measure shift."""
    A = ctx_sub["A_hat"]
    sensitivities = []

    with torch.no_grad():
        for i, j in edge_list:
            A_pert = A.clone()
            A_pert[i, j] += delta
            A_pert[j, i] += delta
            ctx_pert = {**ctx_sub, "A_hat": A_pert}
            Z_pert = reconverge(model, Z_clean, ctx_pert)
            shift = float((Z_pert - Z_clean).norm())
            sensitivities.append(shift / delta)

    return torch.tensor(sensitivities)


def brute_force_discrete_ranking(model, Z_clean, ctx_sub, edge_list):
    """Brute-force single-edge removal: rank by discrete damage."""
    A = ctx_sub["A_hat"]
    damages = []
    with torch.no_grad():
        for i, j in edge_list:
            A_pert = A.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx_sub, "A_hat": A_pert}
            Z_pert = reconverge(model, Z_clean, ctx_pert)
            damages.append(float((Z_pert - Z_clean).norm()))
    return torch.tensor(damages)


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

        n_edges_before = len([(i, j) for i in range(A_sub.shape[0])
                              for j in range(i+1, A_sub.shape[0])
                              if A_sub[i, j].abs() > 1e-10])

        # 1. S_c ranking (via IFT)
        t0 = time.time()
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        if not edge_list:
            return None
        sc_scores = torch.stack([S_c[:, k].norm() for k in range(S_c.shape[1])])
        t_sc = time.time() - t0

        # 2. Finite-difference ranking
        t0 = time.time()
        fd_scores = finite_difference_sensitivity(model, Z_sub, ctx_sub, edge_list)
        t_fd = time.time() - t0

        # 3. Brute-force discrete ground truth
        t0 = time.time()
        discrete_damages = brute_force_discrete_ranking(model, Z_sub, ctx_sub, edge_list)
        t_discrete = time.time() - t0

        # Kendall tau correlations
        sc_np = sc_scores.cpu().numpy()
        fd_np = fd_scores.cpu().numpy()
        disc_np = discrete_damages.numpy()

        tau_sc_vs_disc, _ = kendalltau(sc_np, disc_np)
        tau_fd_vs_disc, _ = kendalltau(fd_np, disc_np)
        tau_sc_vs_fd, _ = kendalltau(sc_np, fd_np)

        # P@10
        k10 = min(10, len(edge_list))
        gt_top = set(discrete_damages.argsort(descending=True)[:k10].tolist())
        sc_top = set(sc_scores.argsort(descending=True)[:k10].tolist())
        fd_top = set(fd_scores.argsort(descending=True)[:k10].tolist())
        p10_sc = len(sc_top & gt_top) / k10
        p10_fd = len(fd_top & gt_top) / k10

        return {
            "dataset": ds_name,
            "seed": seed,
            "n_edges": len(edge_list),
            "tau_sc_vs_discrete": tau_sc_vs_disc,
            "tau_fd_vs_discrete": tau_fd_vs_disc,
            "tau_sc_vs_fd": tau_sc_vs_fd,
            "p10_sc": p10_sc,
            "p10_fd": p10_fd,
            "time_sc": t_sc,
            "time_fd": t_fd,
            "time_discrete": t_discrete,
            "speedup_fd_over_sc": t_sc / max(t_fd, 1e-6),
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
                print(f"tau(Sc,disc)={r['tau_sc_vs_discrete']:+.3f} "
                      f"tau(FD,disc)={r['tau_fd_vs_discrete']:+.3f} "
                      f"tau(Sc,FD)={r['tau_sc_vs_fd']:+.3f} "
                      f"t_sc={r['time_sc']:.2f}s t_fd={r['time_fd']:.2f}s",
                      flush=True)
            else:
                print("SKIP", flush=True)

    # Write CSV
    csv_path = results_dir / "finite_difference_baseline.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 110)
    print("FINITE-DIFFERENCE BASELINE SUMMARY (mean +/- std across seeds)")
    print("=" * 110)
    print(f"{'Dataset':<12} {'tau(Sc,disc)':>14} {'tau(FD,disc)':>14} {'tau(Sc,FD)':>14} "
          f"{'P@10_Sc':>10} {'P@10_FD':>10} {'t_Sc(s)':>10} {'t_FD(s)':>10}")
    print("-" * 110)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        subset = [r for r in rows if r["dataset"] == ds_name]
        if not subset:
            continue

        def fmt(key):
            vals = [r[key] for r in subset]
            return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

        def fmtt(key):
            vals = [r[key] for r in subset]
            return f"{np.mean(vals):.2f}"

        print(f"{ds_name:<12} {fmt('tau_sc_vs_discrete'):>14} {fmt('tau_fd_vs_discrete'):>14} "
              f"{fmt('tau_sc_vs_fd'):>14} {fmt('p10_sc'):>10} {fmt('p10_fd'):>10} "
              f"{fmtt('time_sc'):>10} {fmtt('time_fd'):>10}")

    print()
    print("Interpretation:")
    print("  tau(Sc,FD) ~= 1.0: S_c and FD produce the same ranking (IFT is consistent)")
    print("  tau(Sc,disc) > tau(FD,disc): IFT resolvent adds value over naive forward probing")
    print("  tau(Sc,disc) ~= tau(FD,disc): S_c ranking = FD ranking (IFT adds no ranking value)")
    print()
    print("Key question: Does the IFT resolvent (I-J_z)^{-1} amplification")
    print("change the ranking, or does it just scale all sensitivities uniformly?")


if __name__ == "__main__":
    sys.exit(main() or 0)
