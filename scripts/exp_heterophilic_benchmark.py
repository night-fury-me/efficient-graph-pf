"""M5: Heterophilic benchmark — Texas, Cornell, Wisconsin (WebKB).

Runs AEGIS tau evaluation on heterophilic graphs to test whether
continuous-to-discrete transfer holds beyond homophilic datasets.

Datasets: Texas (183 nodes), Cornell (183), Wisconsin (251)
  - Heterophily ratio: ~0.1 (vs >0.7 for Cora/Citeseer)
  - 5 classes each
  - 10 random train/val/test splits from Pei et al. 2020

Architectures: IGNN, GCN-2, GCN-4, APPNP
Seeds: 10

Output: results/heterophilic_benchmark.csv

Usage:
    .venv/bin/python scripts/exp_heterophilic_benchmark.py
"""

from __future__ import annotations

import csv
import io
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    greedy_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
try:
    import os as _aegis_os
    _aegis_s = _aegis_os.environ.get('AEGIS_SEEDS')
    if _aegis_s: SEEDS = [int(_x) for _x in _aegis_s.split(',') if _x.strip()]
except Exception:
    pass
GEOM_GCN_BASE = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master"


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def reconverge(model, Z, ctx, max_iter=200):
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7: break
            Z = Z_new
    return Z_new


def _download(url, dst):
    if not dst.exists():
        print(f"    downloading {url}...", flush=True)
        urllib.request.urlretrieve(url, str(dst))


def _load_heterophilic(name: str, data_dir: Path) -> dict:
    """Load Texas/Cornell/Wisconsin from Geom-GCN format."""
    data_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = data_dir / name

    # Download files
    ds_dir.mkdir(exist_ok=True)
    _download(f"{GEOM_GCN_BASE}/new_data/{name}/out1_node_feature_label.txt",
              ds_dir / "out1_node_feature_label.txt")
    _download(f"{GEOM_GCN_BASE}/new_data/{name}/out1_graph_edges.txt",
              ds_dir / "out1_graph_edges.txt")
    for i in range(10):
        for split in ["train", "val", "test"]:
            _download(
                f"{GEOM_GCN_BASE}/splits/{name}_split_0.6_0.2_{i}.npz",
                ds_dir / f"split_{i}.npz",
            )

    # Parse node features and labels
    with open(ds_dir / "out1_node_feature_label.txt") as f:
        lines = f.readlines()[1:]  # skip header
    node_data = {}
    for line in lines:
        parts = line.strip().split('\t')
        node_id = int(parts[0])
        features = list(map(float, parts[1].split(',')))
        label = int(parts[2])
        node_data[node_id] = (features, label)

    N = len(node_data)
    n_features = len(node_data[0][0])
    X = torch.zeros(N, n_features, dtype=torch.float32)
    y = torch.zeros(N, dtype=torch.long)
    for nid in range(N):
        X[nid] = torch.tensor(node_data[nid][0])
        y[nid] = node_data[nid][1]

    n_classes = int(y.max().item()) + 1

    # Parse edges
    with open(ds_dir / "out1_graph_edges.txt") as f:
        lines = f.readlines()[1:]  # skip header
    edges = []
    for line in lines:
        parts = line.strip().split('\t')
        edges.append((int(parts[0]), int(parts[1])))

    # Build adjacency
    A = torch.zeros(N, N, dtype=torch.float32)
    for i, j in edges:
        if i < N and j < N:
            A[i, j] = 1.0
            A[j, i] = 1.0

    # Symmetric normalization: D^{-1/2}(A+I)D^{-1/2}
    A_self = A + torch.eye(N)
    deg = A_self.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A_self @ D_inv_sqrt

    # Load split 0 for default train/val/test masks
    split_file = ds_dir / "split_0.npz"
    if split_file.exists():
        splits = np.load(split_file, allow_pickle=True)
        train_mask = torch.tensor(splits["train_mask"].astype(bool), dtype=torch.bool)[:N]
        val_mask = torch.tensor(splits["val_mask"].astype(bool), dtype=torch.bool)[:N]
        test_mask = torch.tensor(splits["test_mask"].astype(bool), dtype=torch.bool)[:N]
    else:
        # Fallback: random 60/20/20 split
        perm = torch.randperm(N)
        n_train = int(0.6 * N)
        n_val = int(0.2 * N)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        val_mask[perm[n_train:n_train+n_val]] = True
        test_mask[perm[n_train+n_val:]] = True

    # Compute homophily ratio
    n_same = 0
    n_total = 0
    for i, j in edges:
        if i < N and j < N:
            n_total += 1
            if y[i] == y[j]:
                n_same += 1
    homophily = n_same / max(n_total, 1)

    return {
        "X": X, "A_hat": A_hat, "y": y,
        "N": N, "n_features": n_features, "n_classes": n_classes,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
        "homophily": homophily,
    }


# ---------------------------------------------------------------
# Explicit GNN architectures (matching exp_tau_all_datasets.py)
# ---------------------------------------------------------------

class GCN_K(nn.Module):
    def __init__(self, n_in, hidden, n_out, K=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, hidden))
        for _ in range(K - 2):
            self.layers.append(nn.Linear(hidden, hidden))
        self.layers.append(nn.Linear(hidden, n_out))

    def forward(self, X, A_hat):
        Z = X
        for i, lin in enumerate(self.layers):
            Z = A_hat @ lin(Z)
            if i < len(self.layers) - 1:
                Z = F_func.relu(Z)
        return Z, Z, {"A_hat": A_hat, "X_proj": X}

    def operator(self, Z, ctx):
        return F_func.relu(ctx["A_hat"] @ self.layers[0](Z) + ctx.get("X_proj", 0))

    def head(self, Z):
        return self.layers[-1](Z)


class APPNP_Model(nn.Module):
    def __init__(self, n_in, hidden, n_out, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = nn.Linear(n_in, hidden)
        self.lin2 = nn.Linear(hidden, n_out)
        self.K = K
        self.alpha = alpha

    def forward(self, X, A_hat):
        H = F_func.relu(self.lin1(X))
        H = self.lin2(H)
        Z = H
        for _ in range(self.K):
            Z = (1 - self.alpha) * A_hat @ Z + self.alpha * H
        return Z, Z, {"A_hat": A_hat, "X_proj": H}

    def head(self, Z):
        return Z

    def operator(self, Z, ctx):
        return (1 - self.alpha) * ctx["A_hat"] @ Z + self.alpha * ctx["X_proj"]


def build_model(arch, n_features, n_classes, device):
    if arch == "IGNN":
        return IGNN(n_features, hidden=64, n_classes=n_classes).to(device)
    elif arch == "GCN-2":
        return GCN_K(n_features, 64, n_classes, K=2).to(device)
    elif arch == "GCN-4":
        return GCN_K(n_features, 64, n_classes, K=4).to(device)
    elif arch == "APPNP":
        return APPNP_Model(n_features, 64, n_classes).to(device)
    raise ValueError(f"Unknown arch: {arch}")


def train_model(model, data, device, seed, epochs=200):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"].to(device)], y[data["train_mask"]])
        optim.zero_grad(); loss.backward(); optim.step()
        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                lv, _, _ = model(X, A_hat)
                va = float((lv.argmax(1)[data["val_mask"].to(device)] == y[data["val_mask"]]).float().mean())
            if va > best_val:
                best_val = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits, _, _ = model(X, A_hat)
        test_acc = float((logits.argmax(1)[data["test_mask"].to(device)] == y[data["test_mask"]]).float().mean())
    return test_acc


def run_single(ds_name, data, arch, seed, device):
    set_seed(seed)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)

    model = build_model(arch, data["n_features"], data["n_classes"], device)
    test_acc = train_model(model, data, device, seed)

    with torch.no_grad():
        _, Z_star, ctx = model(X, A_hat)

    # Full-graph dense analysis (these are tiny graphs, N < 300)
    N = data["N"]
    ctx_full = {"A_hat": A_hat, "X_proj": ctx["X_proj"]}

    Z_eq = reconverge(model, Z_star, ctx_full) if arch == "IGNN" else Z_star

    # Build edge list
    edge_list = []
    for i in range(N):
        for j in range(i + 1, N):
            if A_hat[i, j].abs() > 1e-10:
                edge_list.append((i, j))
    if len(edge_list) < 3:
        return None

    try:
        use_ift = (arch == "IGNN" or arch == "APPNP")
        if use_ift:
            J_z, J_A, _ = _compute_structural_jacobian(
                lambda z, c: model.operator(z, c), Z_eq, ctx_full)
            S = structural_sensitivity_matrix(
                lambda z, c: model.operator(z, c), Z_eq, ctx_full, J_z=J_z, J_A=J_A)
            S_c, edge_list = constrained_sensitivity_matrix(S, A_hat)
            aegis_scores = [float(S_c[:, k].norm()) for k in range(len(edge_list))]
        else:
            # Finite-difference for explicit GNNs (GCN-2, GCN-4)
            delta = 0.001
            aegis_scores = []
            with torch.no_grad():
                for i, j in edge_list:
                    A_pert = A_hat.clone()
                    A_pert[i, j] += delta
                    A_pert[j, i] += delta
                    Z_pert, _, _ = model(X, A_pert)
                    shift = float((Z_pert - Z_star).norm())
                    aegis_scores.append(shift / delta)

        # Brute-force discrete ground truth
        discrete_scores = []
        with torch.no_grad():
            for i, j in edge_list:
                A_pert = A_hat.clone()
                A_pert[i, j] = 0.0
                A_pert[j, i] = 0.0
                if arch == "IGNN":
                    ctx_pert = {**ctx_full, "A_hat": A_pert}
                    Z_pert = reconverge(model, Z_eq, ctx_pert)
                else:
                    Z_pert, _, _ = model(X, A_pert)
                discrete_scores.append(float((Z_pert - Z_eq).norm()))

        tau, _ = kendalltau(aegis_scores, discrete_scores)

        k10 = min(10, len(edge_list))
        gt_top = set(np.argsort(discrete_scores)[-k10:])
        ae_top = set(np.argsort(aegis_scores)[-k10:])
        p10 = len(gt_top & ae_top) / k10

        return {
            "dataset": ds_name, "architecture": arch, "seed": seed,
            "tau": tau, "p10": p10, "test_acc": test_acc,
            "n_nodes": N, "n_edges": len(edge_list),
            "homophily": data["homophily"],
        }

    except Exception as e:
        print(f" Error: {e}", flush=True)
        return None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_dir = Path("datasets/heterophilic")
    datasets = {}
    for name in ["texas", "cornell", "wisconsin"]:
        print(f"Loading {name}...", flush=True)
        try:
            datasets[name] = _load_heterophilic(name, data_dir)
            d = datasets[name]
            print(f"  N={d['N']}, feat={d['n_features']}, classes={d['n_classes']}, "
                  f"homophily={d['homophily']:.3f}")
        except Exception as e:
            print(f"  Failed: {e}")

    ARCHS = ["IGNN", "GCN-2", "GCN-4", "APPNP"]
    rows = []

    for ds_name, data in datasets.items():
        for arch in ARCHS:
            for si, seed in enumerate(SEEDS):
                print(f"[{ds_name}/{arch}] seed={seed} ({si+1}/{len(SEEDS)})",
                      end=" ", flush=True)
                r = run_single(ds_name, data, arch, seed, device)
                if r:
                    rows.append(r)
                    print(f"tau={r['tau']:+.3f} P@10={r['p10']:.2f} acc={r['test_acc']:.3f}")
                else:
                    print("SKIP")

    # Write CSV
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "heterophilic_benchmark.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV: {csv_path}")

    # Summary
    print("\n" + "=" * 90)
    print("HETEROPHILIC BENCHMARK SUMMARY (mean +/- std across seeds)")
    print("=" * 90)
    print(f"{'Dataset':<12} {'Homoph':>7} {'Arch':<8} {'tau':>14} {'P@10':>10} {'Acc%':>8}")
    print("-" * 90)
    for ds_name in datasets:
        for arch in ARCHS:
            subset = [r for r in rows if r["dataset"] == ds_name and r["architecture"] == arch]
            if not subset: continue
            h = subset[0]["homophily"]
            tau_vals = [r["tau"] for r in subset]
            p10_vals = [r["p10"] for r in subset]
            acc_vals = [r["test_acc"] for r in subset]
            print(f"{ds_name:<12} {h:>7.3f} {arch:<8} "
                  f"{np.mean(tau_vals):>+6.3f}±{np.std(tau_vals):.3f} "
                  f"{np.mean(p10_vals):>6.3f}±{np.std(p10_vals):.3f} "
                  f"{np.mean(acc_vals)*100:>7.1f}")
        print("-" * 90)


if __name__ == "__main__":
    sys.exit(main() or 0)
