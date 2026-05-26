"""B4 / P2.7 — Degree-vulnerability correlation analysis.

For each (dataset, architecture, seed):
  1. Train model, extract subgraph, compute S_c
  2. Compute per-edge vulnerability v_ij = ||[S_c]_{:,k}||_2
  3. Compute max(d_i, d_j) for each edge (subgraph degrees)
  4. Compute Kendall tau between vulnerability and max-degree
  5. If tau > 0.3: compute degree-normalized scores v_ij / max(d_i, d_j)
     and check whether the residual still correlates with discrete damage

Datasets:  Cora, Citeseer, Pubmed, WikiCS, Amazon Photo
Architectures: IGNN, GCN-2, GCN-4, APPNP (representative subset)
Seeds: 10

Output: results/degree_correlation.csv

Usage:
    .venv/bin/python scripts/exp_degree_correlation.py
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
# Explicit GNN architectures (representative subset)
# ---------------------------------------------------------------

class ExplicitGCN(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, hidden))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, n_classes)

    def forward_hidden(self, X, A_hat):
        Z = F_func.dropout(X, p=self.dropout, training=self.training)
        for layer in self.layers:
            Z = F_func.relu(A_hat @ layer(Z))
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


class ExplicitAPPNP(nn.Module):
    def __init__(self, n_features, hidden, n_classes, n_prop=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.mlp = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, n_classes)
        self.n_prop = n_prop
        self.alpha = alpha

    def forward_hidden(self, X, A_hat):
        X_drop = F_func.dropout(X, p=self.dropout, training=self.training)
        H_0 = self.mlp(X_drop)
        Z = H_0
        for _ in range(self.n_prop):
            Z = F_func.dropout(Z, p=self.dropout, training=self.training)
            Z = (1 - self.alpha) * (A_hat @ Z) + self.alpha * H_0
        return Z

    def forward(self, X, A_hat):
        Z = self.forward_hidden(X, A_hat)
        return self.head(Z), Z


# Model-specific hyperparameters
_HP = {
    "GCN-2":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "GCN-4":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
    "APPNP":  {"lr": 0.01,  "wd": 5e-4, "epochs": 400},
}


# ---------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------

def load_all_datasets():
    from iem.examples.ignn_citeseer_pubmed import _load_planetoid
    from iem.examples.ignn_wikics import _load_wikics
    from iem.examples.ignn_amazon import _load_amazon

    datasets = {}
    print("Loading datasets...", flush=True)

    print("  Cora...", flush=True)
    datasets["Cora"] = _load_cora(Path("datasets/cora"))

    print("  Citeseer...", flush=True)
    datasets["Citeseer"] = _load_planetoid("citeseer", Path("datasets/citeseer"))

    print("  Pubmed...", flush=True)
    datasets["Pubmed"] = _load_planetoid("pubmed", Path("datasets/pubmed"))

    print("  WikiCS...", flush=True)
    datasets["WikiCS"] = _load_wikics(Path("datasets/wikics"))

    print("  Amazon Photo...", flush=True)
    datasets["Amazon Photo"] = _load_amazon(Path("datasets/amazon_photo"))

    for name, d in datasets.items():
        print(f"    {name}: N={d['N']}, feat={d['n_features']}, classes={d['n_classes']}",
              flush=True)
    return datasets


# ---------------------------------------------------------------
# Sensitivity computation (explicit GNNs)
# ---------------------------------------------------------------

def compute_explicit_sensitivity(model, X_sub, A_sub, eps_fd=1e-4):
    N = A_sub.shape[0]
    with torch.no_grad():
        Z_base = model.forward_hidden(X_sub, A_sub).reshape(-1)
    D = Z_base.shape[0]
    S = torch.zeros(D, N * N, device=A_sub.device)
    with torch.no_grad():
        for idx in range(N * N):
            i, j = idx // N, idx % N
            A_pert = A_sub.clone()
            A_pert[i, j] += eps_fd
            Z_pert = model.forward_hidden(X_sub, A_pert).reshape(-1)
            S[:, idx] = (Z_pert - Z_base) / eps_fd
    return S, Z_base


# ---------------------------------------------------------------
# Brute-force edge-removal (ground truth for residual correlation)
# ---------------------------------------------------------------

def brute_force_edge_removal(model, X_sub, A_sub, edge_list, is_ignn=False,
                             ctx_sub=None, Z_sub=None):
    shifts = []
    with torch.no_grad():
        if is_ignn:
            for i, j in edge_list:
                A_pert = A_sub.clone()
                A_pert[i, j] = 0.0
                A_pert[j, i] = 0.0
                ctx_pert = {**ctx_sub, "A_hat": A_pert}
                Z = Z_sub.clone()
                for _ in range(100):
                    Z_new = model.operator(Z, ctx_pert)
                    if (Z_new - Z).norm() < 1e-7:
                        break
                    Z = Z_new
                shifts.append(float((Z_new - Z_sub).norm()))
        else:
            Z_base = model.forward_hidden(X_sub, A_sub)
            for i, j in edge_list:
                A_pert = A_sub.clone()
                A_pert[i, j] = 0.0
                A_pert[j, i] = 0.0
                Z_pert = model.forward_hidden(X_sub, A_pert)
                shifts.append(float((Z_pert - Z_base).norm()))
    return shifts


# ---------------------------------------------------------------
# Train explicit GNN
# ---------------------------------------------------------------

def train_explicit(model_name, model, X, A_hat, y, train_mask, val_mask, seed, device):
    set_seed(seed)
    hp = _HP.get(model_name, {"lr": 0.01, "wd": 5e-4, "epochs": 200})
    optim = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    best_val, best_state = 0.0, None
    patience, patience_counter = 50, 0
    for ep in range(hp["epochs"]):
        model.train()
        logits, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            logits_v, _ = model(X, A_hat)
            val_acc = float((logits_v.argmax(1)[val_mask] == y[val_mask]).float().mean())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()


# ---------------------------------------------------------------
# Run single (dataset, architecture, seed) combination
# ---------------------------------------------------------------

def run_single(dataset_name, arch_name, data, seed, device):
    """Returns dict with tau_vuln_deg, tau_residual_disc, etc., or None."""
    set_seed(seed)

    try:
        if arch_name == "IGNN":
            return _run_ignn(data, seed, device)
        else:
            return _run_explicit(arch_name, data, seed, device)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"      OOM, trying CPU...", flush=True)
            torch.cuda.empty_cache()
            try:
                cpu = torch.device("cpu")
                if arch_name == "IGNN":
                    return _run_ignn(data, seed, cpu)
                else:
                    return _run_explicit(arch_name, data, seed, cpu)
            except Exception as e2:
                print(f"      CPU fallback failed: {e2}", flush=True)
                return None
        print(f"      Error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"      Error: {e}", flush=True)
        return None


def _compute_degree_correlation(A_sub, edge_list, cont_scores, disc_scores):
    """Core analysis: degree vs vulnerability correlation + residual check."""
    # Compute subgraph degrees from the adjacency pattern
    deg = (A_sub.abs() > 1e-10).float().sum(dim=1)

    # max(d_i, d_j) for each edge
    max_deg = np.array([max(float(deg[i]), float(deg[j])) for i, j in edge_list])

    # Kendall tau between vulnerability and max-degree
    tau_vuln_deg, _ = kendalltau(cont_scores, max_deg)

    # If high correlation: compute degree-normalized residual
    tau_residual_disc = float("nan")
    if tau_vuln_deg > 0.3:
        # Degree-normalized vulnerability
        max_deg_safe = np.maximum(max_deg, 1.0)
        residual_scores = cont_scores / max_deg_safe

        # Check if residual still correlates with discrete damage
        if disc_scores is not None and len(disc_scores) >= 3:
            tau_residual_disc, _ = kendalltau(residual_scores, disc_scores)

    return {
        "tau_vuln_deg": tau_vuln_deg,
        "tau_residual_disc": tau_residual_disc,
        "mean_max_deg": float(np.mean(max_deg)),
        "n_edges": len(edge_list),
    }


def _run_ignn(data, seed, device):
    """IGNN-specific analysis."""
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

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": ctx["X_proj"][idx]}

    # Reconverge on subgraph
    Z_sub = Z_star[idx].clone()
    with torch.no_grad():
        for _ in range(200):
            Z_new = model.operator(Z_sub, ctx_sub)
            if (Z_new - Z_sub).norm() < 1e-7:
                break
            Z_sub = Z_new
    Z_sub = Z_new

    # Compute S via IFT
    J_z, J_A, _ = _compute_structural_jacobian(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub
    )
    S = structural_sensitivity_matrix(
        lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
    )
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    # Continuous vulnerability
    cont_scores = np.array([float(S_c[:, k].norm()) for k in range(len(edge_list))])

    # Discrete ground truth (for residual correlation check)
    disc_scores = np.array(brute_force_edge_removal(
        model, None, A_sub, edge_list, is_ignn=True, ctx_sub=ctx_sub, Z_sub=Z_sub
    ))

    return _compute_degree_correlation(A_sub, edge_list, cont_scores, disc_scores)


def _run_explicit(arch_name, data, seed, device):
    """Explicit GNN analysis."""
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    nf, nc = data["n_features"], data["n_classes"]

    model_map = {
        "GCN-2":  lambda: ExplicitGCN(nf, 64, nc, n_layers=2, dropout=0.5),
        "GCN-4":  lambda: ExplicitGCN(nf, 64, nc, n_layers=4, dropout=0.5),
        "APPNP":  lambda: ExplicitAPPNP(nf, 64, nc, n_prop=10, alpha=0.1, dropout=0.5),
    }
    model = model_map[arch_name]().to(device)
    train_explicit(arch_name, model, X, A_hat, y,
                   data["train_mask"].to(device), data["val_mask"].to(device),
                   seed, device)

    idx = extract_ego_subgraph(A_hat, max_nodes=50)
    A_sub = A_hat[idx][:, idx]
    X_sub = X[idx]

    # Compute S via finite differences
    S, Z_base = compute_explicit_sensitivity(model, X_sub, A_sub)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)
    if not edge_list:
        return None

    # Continuous vulnerability
    cont_scores = np.array([float(S_c[:, k].norm()) for k in range(len(edge_list))])

    # Discrete ground truth
    disc_scores = np.array(brute_force_edge_removal(
        model, X_sub, A_sub, edge_list, is_ignn=False
    ))

    return _compute_degree_correlation(A_sub, edge_list, cont_scores, disc_scores)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    t_start = time.time()

    datasets = load_all_datasets()

    ARCHITECTURES = ["IGNN", "GCN-2", "GCN-4", "APPNP"]
    DATASET_NAMES = ["Cora", "Citeseer", "Pubmed", "WikiCS", "Amazon Photo"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "degree_correlation.csv"

    rows = []

    for ds_name in DATASET_NAMES:
        data = datasets[ds_name]
        for arch in ARCHITECTURES:
            for seed_idx, seed in enumerate(SEEDS):
                print(f"[{ds_name}] [{arch}] seed={seed} ({seed_idx+1}/{len(SEEDS)})",
                      flush=True)
                r = run_single(ds_name, arch, data, seed, device)
                if r is not None:
                    row = {
                        "dataset": ds_name,
                        "architecture": arch,
                        "seed": seed,
                        "tau_vuln_deg": r["tau_vuln_deg"],
                        "tau_residual_disc": r["tau_residual_disc"],
                        "mean_max_deg": r["mean_max_deg"],
                        "n_edges": r["n_edges"],
                    }
                    rows.append(row)
                    residual_s = (f"{r['tau_residual_disc']:+.3f}"
                                  if not np.isnan(r["tau_residual_disc"])
                                  else "N/A (tau<0.3)")
                    print(f"    tau(vuln,deg)={r['tau_vuln_deg']:+.3f}  "
                          f"tau(residual,disc)={residual_s}  "
                          f"mean_max_deg={r['mean_max_deg']:.1f}  "
                          f"edges={r['n_edges']}", flush=True)
                else:
                    print(f"    SKIP", flush=True)

    # Write CSV
    fieldnames = ["dataset", "architecture", "seed", "tau_vuln_deg",
                  "tau_residual_disc", "mean_max_deg", "n_edges"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to {csv_path}", flush=True)

    # Summary table
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n")

    print("=" * 110)
    print("DEGREE-VULNERABILITY CORRELATION SUMMARY (mean +/- std across seeds)")
    print("=" * 110)
    print(f"{'Dataset':<15} {'Arch':<8} {'tau(vuln,deg)':>16} {'tau(resid,disc)':>18} "
          f"{'mean_max_deg':>14} {'#edges':>8}")
    print("-" * 110)

    for ds_name in DATASET_NAMES:
        for arch in ARCHITECTURES:
            subset = [r for r in rows
                      if r["dataset"] == ds_name and r["architecture"] == arch]
            if not subset:
                continue
            tau_vd = [r["tau_vuln_deg"] for r in subset]
            tau_rd = [r["tau_residual_disc"] for r in subset]
            mdeg = [r["mean_max_deg"] for r in subset]
            edges = [r["n_edges"] for r in subset]
            print(f"{ds_name:<15} {arch:<8} {agg(tau_vd):>16} {agg(tau_rd):>18} "
                  f"{agg(mdeg, '.1f'):>14} {np.mean(edges):>8.0f}")

    # High-correlation cases summary
    print("\n--- High degree-correlation cases (tau(vuln,deg) > 0.3) ---")
    high_corr = [r for r in rows if r["tau_vuln_deg"] > 0.3]
    n_high = len(high_corr)
    n_total = len(rows)
    print(f"  {n_high}/{n_total} runs have tau(vuln,deg) > 0.3 ({100*n_high/n_total:.0f}%)"
          if n_total > 0 else "  No data")

    if high_corr:
        residual_taus = [r["tau_residual_disc"] for r in high_corr
                         if not np.isnan(r["tau_residual_disc"])]
        if residual_taus:
            print(f"  For those cases, tau(residual,disc) = {agg(residual_taus)}")
            print(f"  Residual still predictive (tau > 0.3): "
                  f"{sum(1 for t in residual_taus if t > 0.3)}/{len(residual_taus)}")

    print(f"\n{len(rows)} successful runs out of "
          f"{len(DATASET_NAMES) * len(ARCHITECTURES) * len(SEEDS)} attempted")


if __name__ == "__main__":
    sys.exit(main() or 0)
