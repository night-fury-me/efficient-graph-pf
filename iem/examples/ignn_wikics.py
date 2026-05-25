"""IEM validation on 4th domain: IGNN on WikiCS (Wikipedia Computer Science).

Wikipedia articles as nodes, hyperlinks as edges. Node classification into
10 CS sub-fields. Different from citation (directed academic references)
and e-commerce (co-purchase) — encyclopedic knowledge graph structure.

Dataset: WikiCS (Mernyei & Cangea, 2020, ICML Workshop)
  - 11,701 nodes (articles), 216,123 edges (hyperlinks)
  - 300-dim node features (GloVe embeddings of article text)
  - 10 classes (CS sub-fields)

Usage:
    .venv/bin/python -m iem.examples.ignn_wikics
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem.examples.ignn_cora import IGNN

WIKICS_URL = "https://github.com/pmernyei/wiki-cs-dataset/raw/master/dataset"
WIKICS_FILES = ["data.json"]


def _download_wikics(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    dst = data_dir / "data.json"
    if not dst.exists():
        print(f"  downloading WikiCS...", flush=True)
        urllib.request.urlretrieve(WIKICS_URL + "/data.json", str(dst))


def _load_wikics(data_dir: Path) -> dict:
    """Load WikiCS into dense tensors."""
    _download_wikics(data_dir)
    with open(data_dir / "data.json") as f:
        raw = json.load(f)

    X = torch.tensor(raw["features"], dtype=torch.float32)
    y = torch.tensor(raw["labels"], dtype=torch.long)
    N = X.shape[0]

    # Build adjacency from edge list
    edges = raw["links"]
    row, col = [], []
    for src, dsts in enumerate(edges):
        for dst in dsts:
            row.append(src)
            col.append(dst)
    adj = sp.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)), shape=(N, N)
    )
    adj = adj + adj.T  # symmetrize
    adj = adj + sp.eye(N)  # self-loops
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    A_hat_dense = torch.tensor(A_hat.toarray(), dtype=torch.float32)

    # Use first provided train/val/test split
    train_masks = raw["train_masks"]
    val_masks = raw["val_masks"]
    test_mask_raw = raw["test_mask"]
    train_mask = torch.tensor(train_masks[0], dtype=torch.bool)
    val_mask = torch.tensor(val_masks[0], dtype=torch.bool)
    test_mask = torch.tensor(test_mask_raw, dtype=torch.bool)

    return {
        "X": X, "A_hat": A_hat_dense, "y": y, "N": N,
        "n_features": X.shape[1], "n_classes": int(y.max()) + 1,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/wikics")

    print("=== Loading WikiCS ===", flush=True)
    data = _load_wikics(data_dir)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    print(f"  N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")
    print(f"  train={data['train_mask'].sum()}, val={data['val_mask'].sum()}, test={data['test_mask'].sum()}")

    # Train IGNN
    print("\n=== Training IGNN (50 epochs) ===", flush=True)
    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for ep in range(1, 51):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[data["train_mask"]], y[data["train_mask"]])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if ep % 10 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                logits, _, _ = model(X, A_hat)
                pred = logits.argmax(dim=1)
                val_acc = float((pred[data["val_mask"]] == y[data["val_mask"]]).float().mean())
                test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
            print(f"  ep {ep:3d} | loss {loss.item():.4f} | val {val_acc:.3f} | test {test_acc:.3f}", flush=True)

    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    print(f"\n  Final test accuracy: {test_acc:.3f}")
    residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
    print(f"  Fixed-point residual: {residual:.2e}")

    # --- IEM on 50-node subgraph ---
    print("\n=== IEM on WikiCS ===", flush=True)
    from iem import IEMiner

    deg = A_hat.sum(dim=1)
    center = int(deg.argmax().item())
    neighbors = (A_hat[center] > 0).nonzero(as_tuple=True)[0]
    subgraph_idx = neighbors[:50]
    S = len(subgraph_idx)
    print(f"  Subgraph: {S} nodes around center {center} (deg={int(deg[center].item())})")

    A_sub = A_hat[subgraph_idx][:, subgraph_idx]
    X_proj_sub = ctx["X_proj"][subgraph_idx]
    Z_star_sub = Z_star[subgraph_idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    miner = IEMiner(lambda z, c=ctx_sub: model.operator(z, c), Z_star_sub, ctx_sub, method="direct")

    print("\n  --- Contractivity ---", flush=True)
    report = miner.certify()
    print(f"  rho = {report['rho']:.4f}, contractive = {report['is_contractive']}")

    print("\n  --- Node Attribution (top 5) ---", flush=True)
    phi = miner.node_attribution("X_proj")
    top5 = phi.argsort(descending=True)[:5]
    for rank, idx in enumerate(top5.tolist()):
        real_idx = int(subgraph_idx[idx].item())
        label = int(y[real_idx].item())
        print(f"    #{rank+1}: article {real_idx} (field {label}), phi={phi[idx]:.4e}")
    n_nz = int((phi > 1e-6).sum().item())
    print(f"  Nonzero: {n_nz}/{S}")

    print("\n  --- Certified bounds (eps=0.1) ---", flush=True)
    cert = miner.certified_bound(phi, epsilon=0.1)
    print(f"  rho={cert['rho']:.4f}, max_bound={cert['max_bound']:.4e}")
    if cert.get("max_rank_perturbation") and cert["max_rank_perturbation"] != float("inf"):
        print(f"  Rank stability: {cert['max_rank_perturbation']:.4e}")

    print("\n=== WIKICS + IEM: COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
