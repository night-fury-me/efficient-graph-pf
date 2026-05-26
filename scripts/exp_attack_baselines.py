"""B2 / P1.2 — Structured attack baselines comparison.

Compares AEGIS (SVD-optimal) attack against 3 heuristic baselines:
  1. Degree-proportional: weight ~ max(d_i, d_j), normalized to ||dA||_F = eps
  2. Spectral heuristic: perturbation along top eigenvector of A, edge-restricted
  3. Betweenness centrality: weight ~ edge betweenness centrality

For each method: apply perturbation at eps=0.01, measure ||Delta z*||.
Report AtkAdv = AEGIS_damage / baseline_damage.

Datasets:  Cora, Citeseer, WikiCS (IGNN only, 50-node subgraph)
Seeds: 10

Output: results/attack_baselines.csv

Usage:
    .venv/bin/python scripts/exp_attack_baselines.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import networkx as nx
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
EPS = 0.01


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def agg(vals, fmt=".3f"):
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
# Reconverge IGNN on perturbed adjacency
# ---------------------------------------------------------------

def reconverge(model, Z_init, ctx_pert, max_iter=200):
    """Reconverge IGNN fixed point under perturbed context."""
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx_pert)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return Z_new


# ---------------------------------------------------------------
# Attack methods
# ---------------------------------------------------------------

def aegis_svd_attack(S_c, Vh_c, edge_list, A_sub, eps):
    """AEGIS optimal attack: dA along leading right singular vector of S_c."""
    weights = eps * Vh_c[0]
    dA = torch.zeros_like(A_sub)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    return dA


def degree_proportional_attack(A_sub, edge_list, eps):
    """Degree-proportional: perturbation weight ~ max(d_i, d_j)."""
    # Compute degrees from the (un-normalized) adjacency pattern
    # A_sub is normalized, so use nonzero pattern to get degree
    deg = (A_sub.abs() > 1e-10).float().sum(dim=1)

    raw_weights = torch.zeros(len(edge_list), device=A_sub.device)
    for k, (i, j) in enumerate(edge_list):
        raw_weights[k] = max(float(deg[i]), float(deg[j]))

    # Normalize to ||dA||_F = eps
    norm = raw_weights.norm()
    if norm < 1e-10:
        return torch.zeros_like(A_sub)
    weights = raw_weights / norm * eps

    dA = torch.zeros_like(A_sub)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    return dA


def spectral_heuristic_attack(A_sub, edge_list, eps):
    """Spectral heuristic: perturbation along top eigenvector of A, edge-restricted."""
    # Top eigenvector of A_sub
    eigenvalues, eigenvectors = torch.linalg.eigh(A_sub)
    v1 = eigenvectors[:, -1]  # leading eigenvector (largest eigenvalue)

    # Perturbation: outer product restricted to existing edges
    raw_weights = torch.zeros(len(edge_list), device=A_sub.device)
    for k, (i, j) in enumerate(edge_list):
        raw_weights[k] = v1[i] * v1[j]

    # Normalize to ||dA||_F = eps
    norm = raw_weights.norm()
    if norm < 1e-10:
        return torch.zeros_like(A_sub)
    weights = raw_weights / norm * eps

    dA = torch.zeros_like(A_sub)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    return dA


def betweenness_centrality_attack(A_sub, edge_list, eps):
    """Betweenness centrality: perturbation weight ~ edge betweenness."""
    N = A_sub.shape[0]

    # Build networkx graph from adjacency
    G = nx.Graph()
    for i, j in edge_list:
        G.add_edge(i, j)

    # Compute edge betweenness centrality
    ebc = nx.edge_betweenness_centrality(G, normalized=True)

    raw_weights = torch.zeros(len(edge_list), device=A_sub.device)
    for k, (i, j) in enumerate(edge_list):
        # networkx stores edges with min(i,j) first
        key = (i, j) if (i, j) in ebc else (j, i)
        raw_weights[k] = ebc.get(key, 0.0)

    # Normalize to ||dA||_F = eps
    norm = raw_weights.norm()
    if norm < 1e-10:
        return torch.zeros_like(A_sub)
    weights = raw_weights / norm * eps

    dA = torch.zeros_like(A_sub)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    return dA


def random_attack(edge_list, A_sub, eps):
    """Random baseline: uniform random perturbation weights, normalized."""
    raw_weights = torch.randn(len(edge_list), device=A_sub.device)
    norm = raw_weights.norm()
    if norm < 1e-10:
        return torch.zeros_like(A_sub)
    weights = raw_weights / norm * eps

    dA = torch.zeros_like(A_sub)
    for k, (i, j) in enumerate(edge_list):
        dA[i, j] = float(weights[k])
        dA[j, i] = float(weights[k])
    return dA


# ---------------------------------------------------------------
# Run single (dataset, seed) combination
# ---------------------------------------------------------------

def run_single(dataset_name, data, seed, device):
    """Returns list of dicts with method, damage, etc., or None on failure."""
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

        # Extract subgraph
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

        U_c, sigma_c, Vh_c = torch.linalg.svd(S_c, full_matrices=False)

        # Generate attack perturbations
        attacks = {
            "AEGIS (SVD)": aegis_svd_attack(S_c, Vh_c, edge_list, A_sub, EPS),
            "Degree": degree_proportional_attack(A_sub, edge_list, EPS),
            "Spectral": spectral_heuristic_attack(A_sub, edge_list, EPS),
            "Betweenness": betweenness_centrality_attack(A_sub, edge_list, EPS),
            "Random": random_attack(edge_list, A_sub, EPS),
        }

        # Measure damage for each method
        results = []
        for method_name, dA in attacks.items():
            ctx_pert = {**ctx_sub, "A_hat": A_sub + dA}
            Z_pert = reconverge(model, Z_sub, ctx_pert)
            damage = float((Z_pert - Z_sub).norm())
            results.append({
                "dataset": dataset_name,
                "seed": seed,
                "method": method_name,
                "damage": damage,
            })

        # Compute AtkAdv vs Random for each method
        random_damage = [r["damage"] for r in results if r["method"] == "Random"][0]
        for r in results:
            if random_damage > 1e-12:
                r["atk_adv_vs_random"] = r["damage"] / random_damage
            else:
                r["atk_adv_vs_random"] = float("nan")

        return results

    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"      OOM error, trying CPU...", flush=True)
            torch.cuda.empty_cache()
            try:
                return run_single(dataset_name, data, seed, torch.device("cpu"))
            except Exception as e2:
                print(f"      CPU fallback failed: {e2}", flush=True)
                return None
        print(f"      Error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"      Error: {e}", flush=True)
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
    csv_path = results_dir / "attack_baselines.csv"

    all_rows = []

    for ds_name in DATASET_NAMES:
        data = datasets[ds_name]
        for seed_idx, seed in enumerate(SEEDS):
            print(f"[{ds_name}] seed={seed} ({seed_idx+1}/{len(SEEDS)})", flush=True)
            results = run_single(ds_name, data, seed, device)
            if results is not None:
                for r in results:
                    all_rows.append(r)
                    print(f"    {r['method']:<20} damage={r['damage']:.6f}  "
                          f"AtkAdv={r['atk_adv_vs_random']:.2f}x", flush=True)
            else:
                print(f"    SKIP", flush=True)

    # Write CSV
    fieldnames = ["dataset", "seed", "method", "damage", "atk_adv_vs_random"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved to {csv_path}", flush=True)

    # Summary table
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 95)
    print("ATTACK BASELINES SUMMARY (mean +/- std across seeds)")
    print("=" * 95)
    print(f"{'Dataset':<15} {'Method':<20} {'Damage':>16} {'AtkAdv vs Random':>20}")
    print("-" * 95)

    methods = ["AEGIS (SVD)", "Degree", "Spectral", "Betweenness", "Random"]
    for ds_name in DATASET_NAMES:
        for method in methods:
            subset = [r for r in all_rows
                      if r["dataset"] == ds_name and r["method"] == method]
            if not subset:
                continue
            damages = [r["damage"] for r in subset]
            advs = [r["atk_adv_vs_random"] for r in subset]
            print(f"{ds_name:<15} {method:<20} {agg(damages, '.6f'):>16} "
                  f"{agg(advs):>20}")
        print("-" * 95)

    # Cross-dataset AtkAdv for AEGIS
    print("\nAEGIS AtkAdv vs Random (cross-dataset):")
    aegis_all = [r["atk_adv_vs_random"] for r in all_rows if r["method"] == "AEGIS (SVD)"]
    print(f"  Overall: {agg(aegis_all)}")
    for ds_name in DATASET_NAMES:
        aegis_ds = [r["atk_adv_vs_random"] for r in all_rows
                    if r["method"] == "AEGIS (SVD)" and r["dataset"] == ds_name]
        print(f"  {ds_name}: {agg(aegis_ds)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
