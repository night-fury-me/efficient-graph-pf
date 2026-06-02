"""B6/P2.2 — Comprehensive breach rates at multiple epsilon values.

For each (dataset, epsilon) pair:
  1. Train IGNN, extract 50-node BFS subgraph
  2. Compute S_c and per-node first-order sensitivity radii r_v
  3. Apply S_c-optimal perturbation at magnitude epsilon
  4. Re-converge IGNN to new equilibrium
  5. Check each node: does predicted class change? (breach)
  6. Report breach rate and whether breaches respect the radius

Datasets: Cora, Citeseer, Pubmed (subgraph), WikiCS, Amazon Photo
Epsilons: [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
Seeds: [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

Usage:
    .venv/bin/python scripts/exp_breach_rates.py
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
    per_node_robust_radius,
    structural_sensitivity_matrix,
)
from iem.certify import spectral_radius
from iem.examples.ignn_amazon import _load_amazon
from iem.examples.ignn_citeseer_pubmed import _load_planetoid
from iem.examples.ignn_cora import IGNN, _load_cora
from iem.examples.ignn_wikics import _load_wikics

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
EPSILONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets():
    """Load all five datasets."""
    datasets = []
    datasets.append(("Cora", _load_cora(Path("datasets/cora"))))
    datasets.append(("Citeseer", _load_planetoid("citeseer", Path("datasets/citeseer"))))
    datasets.append(("Pubmed", _load_planetoid("pubmed", Path("datasets/pubmed"))))
    datasets.append(("WikiCS", _load_wikics(Path("datasets/wikics"))))
    datasets.append(("AmazonPhoto", _load_amazon(Path("datasets/amazon_photo"))))
    return datasets


def train_ignn(data, device, seed, epochs=200):
    """Train IGNN with early stopping."""
    set_seed(seed)
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None

    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits_v, _, _ = model(X, A_hat)
                val_acc = float(
                    (logits_v.argmax(1)[data["val_mask"]] == y[data["val_mask"]])
                    .float().mean()
                )
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)

    return model, Z_star, ctx


def run_breach_analysis(model, Z_star, ctx, data, device, seed):
    """For one trained model, compute breach rates across all epsilons.

    Returns a list of dicts, one per epsilon.
    """
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    # Extract 50-node BFS subgraph
    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}
    Z_sub = Z_star[idx].clone()

    # Reconverge on subgraph
    with torch.no_grad():
        for _ in range(300):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    labels_sub = y[idx]
    N_sub = len(idx)

    # Clean predictions
    with torch.no_grad():
        logits_clean = model.head(Z_sub)
    preds_clean = logits_clean.argmax(dim=1)

    # Spectral radius
    def F_z(z):
        return model.operator(z.reshape(Z_sub.shape), ctx_sub).reshape(-1)
    rho = spectral_radius(F_z, Z_sub)

    # Compute S and S_c
    try:
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub,
            J_z=J_z, J_A=J_A,
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    except Exception as e:
        print(f"      S_c computation failed: {e}", flush=True)
        return []

    if S_c.shape[1] == 0 or not edge_list:
        print("      No edges in subgraph", flush=True)
        return []

    # Per-node robust radii
    radius_info = per_node_robust_radius(
        S, Z_sub, logits_clean, labels_sub, rho, model.head,
    )
    radii = radius_info["radii"].to(device)  # (N_sub,)

    # SVD of S_c for optimal attack direction
    U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)

    results = []

    for eps in EPSILONS:
        # Build S_c-optimal perturbation
        dA = torch.zeros_like(A_sub)
        weights = eps * Vh_c[0]
        for k, (i, j) in enumerate(edge_list):
            dA[i, j] = float(weights[k])
            dA[j, i] = float(weights[k])

        # Re-converge under perturbation
        ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
        Z = Z_sub.clone()
        with torch.no_grad():
            diverged = False
            for _ in range(300):
                Z_new = model.operator(Z, ctx_pert)
                if torch.isnan(Z_new).any() or Z_new.norm() > 1e6:
                    diverged = True
                    break
                if (Z_new - Z).norm() < 1e-8:
                    break
                Z = Z_new

        if diverged:
            results.append({
                "epsilon": eps,
                "breach_rate": float("nan"),
                "n_breached": -1,
                "n_vulnerable": -1,
                "radius_respected": float("nan"),
                "status": "diverged",
            })
            continue

        # Check predictions after perturbation
        with torch.no_grad():
            logits_pert = model.head(Z_new)
        preds_pert = logits_pert.argmax(dim=1)

        # Breach = prediction changed
        breached = (preds_pert != preds_clean)
        n_breached = int(breached.sum().item())

        # Only count nodes with nontrivial sensitivity (r_v > 0)
        has_radius = (radii > 1e-8)
        n_vulnerable = int(has_radius.sum().item())

        if n_vulnerable > 0:
            breach_rate = n_breached / n_vulnerable
        else:
            breach_rate = 0.0

        # Check: breaches should only happen for nodes where eps > r_v
        # "radius_respected" = fraction of breached nodes where eps >= r_v
        if n_breached > 0:
            breached_radii = radii[breached]
            respected = (breached_radii <= eps + 1e-8)
            radius_respected = float(respected.float().mean())
        else:
            radius_respected = 1.0  # no breaches => trivially respected

        results.append({
            "epsilon": eps,
            "breach_rate": breach_rate,
            "n_breached": n_breached,
            "n_vulnerable": n_vulnerable,
            "n_total": N_sub,
            "radius_respected": radius_respected,
            "status": "OK",
        })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("Loading datasets ...", flush=True)
    datasets = load_datasets()

    all_results = []
    t_start = time.time()

    for ds_name, data in datasets:
        print(f"\n{'='*70}")
        print(f"  {ds_name}: N={data['N']}, features={data['n_features']}, "
              f"classes={data['n_classes']}")
        print(f"{'='*70}", flush=True)

        for seed_idx, seed in enumerate(SEEDS):
            print(f"\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ...", flush=True)

            try:
                model, Z_star, ctx = train_ignn(data, device, seed)

                breach_results = run_breach_analysis(
                    model, Z_star, ctx, data, device, seed,
                )

                for r in breach_results:
                    r["dataset"] = ds_name
                    r["seed"] = seed
                    all_results.append(r)

                    if r["status"] == "OK":
                        print(f"    eps={r['epsilon']:.2f}: breach_rate={r['breach_rate']:.3f} "
                              f"({r['n_breached']}/{r['n_vulnerable']} vulnerable) "
                              f"radius_respected={r['radius_respected']:.2f}", flush=True)
                    else:
                        print(f"    eps={r['epsilon']:.2f}: {r['status']}", flush=True)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                err_msg = str(e)[:80]
                print(f"    ERROR: {err_msg}", flush=True)
                for eps in EPSILONS:
                    all_results.append({
                        "dataset": ds_name,
                        "seed": seed,
                        "epsilon": eps,
                        "breach_rate": float("nan"),
                        "n_breached": -1,
                        "n_vulnerable": -1,
                        "n_total": -1,
                        "radius_respected": float("nan"),
                        "status": f"ERROR: {err_msg}",
                    })

            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Save CSV ---
    csv_path = results_dir / "exp_breach_rates.csv"
    fieldnames = [
        "dataset", "seed", "epsilon", "breach_rate", "n_breached",
        "n_vulnerable", "n_total", "radius_respected", "status",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            # Ensure all fields present
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")

    # --- Print summary table ---
    print("\n" + "=" * 100)
    print("BREACH RATE SUMMARY (mean +/- std over 10 seeds)")
    print("=" * 100)

    def agg(vals):
        arr = [v for v in vals if v is not None and not np.isnan(v)]
        if not arr:
            return "N/A"
        return f"{np.mean(arr):.3f}+/-{np.std(arr):.3f}"

    # Header
    eps_header = "".join(f"{'eps='+str(e):>18}" for e in EPSILONS)
    print(f"{'Dataset':<15}{eps_header}")
    print("-" * (15 + 18 * len(EPSILONS)))

    ds_names = ["Cora", "Citeseer", "Pubmed", "WikiCS", "AmazonPhoto"]
    for ds_name in ds_names:
        row_str = f"{ds_name:<15}"
        for eps in EPSILONS:
            rows = [r for r in all_results
                    if r["dataset"] == ds_name
                    and abs(r["epsilon"] - eps) < 1e-6
                    and r["status"] == "OK"]
            rates = [r["breach_rate"] for r in rows]
            row_str += f"{agg(rates):>18}"
        print(row_str)

    # Radius-respected summary
    print(f"\n{'Radius Respected (fraction of breaches where eps >= r_v)':}")
    print(f"{'Dataset':<15}{eps_header}")
    print("-" * (15 + 18 * len(EPSILONS)))

    for ds_name in ds_names:
        row_str = f"{ds_name:<15}"
        for eps in EPSILONS:
            rows = [r for r in all_results
                    if r["dataset"] == ds_name
                    and abs(r["epsilon"] - eps) < 1e-6
                    and r["status"] == "OK"]
            respected = [r["radius_respected"] for r in rows]
            row_str += f"{agg(respected):>18}"
        print(row_str)

    print("\nKey result: breach rate should increase with epsilon; "
          "breaches should respect per-node radii (radius_respected ~ 1.0)")


if __name__ == "__main__":
    sys.exit(main() or 0)
