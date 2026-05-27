"""Exp C4-2: Gradient-based targeted attack (Nettack-style, independent baseline).

For each edge (i,j), computes the classification-loss gradient
d L_CE / d A_ij via standard autograd through the IGNN fixed-point iteration.
Edges are ranked by |gradient| magnitude.

This is a genuinely independent attack baseline:
  - Different objective: classification loss, not equilibrium shift
  - Different gradient: autograd through unrolled FP iteration, not IFT resolvent
  - Different ranking: per-edge gradient magnitude, not S_c column norms

Compares:
  1. AEGIS ranking (S_c column norms) vs discrete ground truth (tau_aegis)
  2. Gradient ranking (|dL/dA_ij|) vs discrete ground truth (tau_grad)
  3. Cross-correlation: AEGIS ranking vs gradient ranking (tau_cross)

Also: targeted attack on most-vulnerable node (smallest r_v) — does the
gradient-targeted removal flip the prediction? Does AEGIS predict which
nodes are successfully attacked?

Datasets: Cora, Citeseer, WikiCS | Model: IGNN | Seeds: 10

Output: results/gradient_targeted_attack.csv

Usage:
    .venv/bin/python scripts/exp_gradient_targeted_attack.py
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


def compute_classification_gradients(model, Z_clean, ctx_sub, y_sub, edge_list):
    """Compute per-edge classification-loss gradient via autograd through FP iteration.

    Returns gradient magnitudes |dL_CE / dA_ij| for each edge.
    Uses standard autograd (not IFT resolvent) — a genuinely different
    differentiation pathway.
    """
    A = ctx_sub["A_hat"]
    n_edges = len(edge_list)

    delta = torch.zeros(n_edges, device=A.device, requires_grad=True)

    A_pert = A.clone()
    for k, (i, j) in enumerate(edge_list):
        A_pert = A_pert.clone()
        A_pert[i, j] = A[i, j] + delta[k]
        A_pert[j, i] = A[j, i] + delta[k]

    ctx_pert = {**ctx_sub, "A_hat": A_pert}

    Z = Z_clean.detach().clone()
    with torch.enable_grad():
        for _ in range(50):
            Z_new = model.operator(Z, ctx_pert)
            if (Z_new - Z).detach().norm() < 1e-7:
                break
            Z = Z_new

        logits = model.head(Z_new)
        loss = F_func.cross_entropy(logits, y_sub)

    grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]
    return grad.detach().abs()


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
        y = data["y"].to(device)

        with torch.no_grad():
            _, Z_star, ctx = model(X, A_hat)

        idx = extract_ego_subgraph(A_hat, max_nodes=50)
        A_sub = A_hat[idx][:, idx]
        ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
        y_sub = y[idx]

        Z_sub = reconverge(model, Z_star[idx].clone(), ctx_sub)

        # 1. AEGIS S_c ranking
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
        if not edge_list:
            return None

        aegis_scores = torch.stack([S_c[:, k].norm() for k in range(S_c.shape[1])])

        # 2. Classification-gradient ranking
        grad_scores = compute_classification_gradients(
            model, Z_sub, ctx_sub, y_sub, edge_list
        )

        # 3. Brute-force discrete ground truth
        discrete_damages = brute_force_discrete_ranking(
            model, Z_sub, ctx_sub, edge_list
        )

        # Compute Kendall tau correlations
        tau_aegis, _ = kendalltau(aegis_scores.cpu().numpy(), discrete_damages.numpy())
        tau_grad, _ = kendalltau(grad_scores.cpu().numpy(), discrete_damages.numpy())
        tau_cross, _ = kendalltau(aegis_scores.cpu().numpy(), grad_scores.cpu().numpy())

        # P@10 for both methods
        k10 = min(10, len(edge_list))
        gt_top10 = set(discrete_damages.argsort(descending=True)[:k10].tolist())
        aegis_top10 = set(aegis_scores.argsort(descending=True)[:k10].tolist())
        grad_top10 = set(grad_scores.argsort(descending=True)[:k10].tolist())
        p10_aegis = len(aegis_top10 & gt_top10) / k10
        p10_grad = len(grad_top10 & gt_top10) / k10

        # Targeted attack: remove top-1 edge by gradient ranking, check flips
        with torch.no_grad():
            preds_clean = model.head(Z_sub).argmax(dim=1)

        top_grad_edge = grad_scores.argmax().item()
        i, j = edge_list[top_grad_edge]
        A_pert = A_sub.clone()
        A_pert[i, j] = 0.0
        A_pert[j, i] = 0.0
        ctx_pert = {**ctx_sub, "A_hat": A_pert}
        with torch.no_grad():
            Z_pert = reconverge(model, Z_sub, ctx_pert)
            preds_pert = model.head(Z_pert).argmax(dim=1)
        grad_flips = int((preds_clean != preds_pert).sum())

        top_aegis_edge = aegis_scores.argmax().item()
        i, j = edge_list[top_aegis_edge]
        A_pert2 = A_sub.clone()
        A_pert2[i, j] = 0.0
        A_pert2[j, i] = 0.0
        ctx_pert2 = {**ctx_sub, "A_hat": A_pert2}
        with torch.no_grad():
            Z_pert2 = reconverge(model, Z_sub, ctx_pert2)
            preds_pert2 = model.head(Z_pert2).argmax(dim=1)
        aegis_flips = int((preds_clean != preds_pert2).sum())

        return {
            "dataset": ds_name,
            "seed": seed,
            "tau_aegis_vs_discrete": tau_aegis,
            "tau_grad_vs_discrete": tau_grad,
            "tau_aegis_vs_grad": tau_cross,
            "p10_aegis": p10_aegis,
            "p10_grad": p10_grad,
            "flips_grad_top1": grad_flips,
            "flips_aegis_top1": aegis_flips,
            "n_nodes": len(y_sub),
            "n_edges": len(edge_list),
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(" OOM", flush=True)
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
                print(f"tau_aegis={r['tau_aegis_vs_discrete']:+.3f} "
                      f"tau_grad={r['tau_grad_vs_discrete']:+.3f} "
                      f"tau_cross={r['tau_aegis_vs_grad']:+.3f} "
                      f"flips(grad/aegis)={r['flips_grad_top1']}/{r['flips_aegis_top1']}",
                      flush=True)
            else:
                print("SKIP", flush=True)

    # Write CSV
    csv_path = results_dir / "gradient_targeted_attack.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # Summary
    elapsed = time.time() - t_start
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 100)
    print("GRADIENT-TARGETED ATTACK SUMMARY (mean +/- std across seeds)")
    print("=" * 100)
    print(f"{'Dataset':<12} {'tau_AEGIS':>12} {'tau_Grad':>12} {'tau_Cross':>12} "
          f"{'P@10_AEGIS':>12} {'P@10_Grad':>12} {'Flips_G':>8} {'Flips_A':>8}")
    print("-" * 100)
    for ds_name in ["Cora", "Citeseer", "WikiCS"]:
        subset = [r for r in rows if r["dataset"] == ds_name]
        if not subset:
            continue

        def fmt(key):
            vals = [r[key] for r in subset]
            return f"{np.mean(vals):.3f}+/-{np.std(vals):.3f}"

        print(f"{ds_name:<12} {fmt('tau_aegis_vs_discrete'):>12} {fmt('tau_grad_vs_discrete'):>12} "
              f"{fmt('tau_aegis_vs_grad'):>12} {fmt('p10_aegis'):>12} {fmt('p10_grad'):>12} "
              f"{np.mean([r['flips_grad_top1'] for r in subset]):>7.1f} "
              f"{np.mean([r['flips_aegis_top1'] for r in subset]):>7.1f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
