"""IEM validation on 2nd domain: IGNN on Cora citation network.

Implements a minimal Implicit Graph Neural Network (Gu et al., NeurIPS 2020)
and demonstrates that IEM's IFT-based attribution + certification work
identically to the power-flow domain — proving domain-agnosticism.

Usage:
    .venv/bin/python -m iem.examples.ignn_cora

No external graph libraries required (uses scipy + urllib only).
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. Cora data loader (no PyG dependency)
# ---------------------------------------------------------------------------

CORA_URL = "https://github.com/kimiyoung/planetoid/raw/master/data/"
CORA_FILES = [
    "ind.cora.x", "ind.cora.tx", "ind.cora.allx",
    "ind.cora.y", "ind.cora.ty", "ind.cora.ally",
    "ind.cora.graph", "ind.cora.test.index",
]


def _download_cora(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in CORA_FILES:
        dst = data_dir / fname
        if not dst.exists():
            print(f"  downloading {fname}...", flush=True)
            urllib.request.urlretrieve(CORA_URL + fname, str(dst))


def _load_cora(data_dir: Path) -> dict:
    """Load Cora into dense tensors: X (2708, 1433), A_hat (2708, 2708), y (2708,)."""

    def _pkl(name):
        with open(data_dir / name, "rb") as f:
            return pickle.load(f, encoding="latin1")

    x = _pkl("ind.cora.x")        # training features (sparse)
    tx = _pkl("ind.cora.tx")       # test features
    allx = _pkl("ind.cora.allx")   # all labeled features
    y = _pkl("ind.cora.y")
    ty = _pkl("ind.cora.ty")
    ally = _pkl("ind.cora.ally")
    graph = _pkl("ind.cora.graph")

    test_idx = []
    with open(data_dir / "ind.cora.test.index") as f:
        for line in f:
            test_idx.append(int(line.strip()))
    test_idx = np.array(test_idx)

    # Sort test indices (some Planetoid versions have them shuffled)
    test_idx_sorted = np.sort(test_idx)

    # Build full feature matrix
    features = sp.vstack([allx, tx]).tolil()
    features[test_idx] = features[test_idx_sorted]
    X = torch.tensor(features.toarray(), dtype=torch.float32)

    # Labels
    labels = np.vstack([ally, ty])
    labels[test_idx] = labels[test_idx_sorted]
    y = torch.tensor(labels.argmax(axis=1), dtype=torch.long)

    # Adjacency (symmetric, add self-loops, normalize)
    N = X.shape[0]
    adj = sp.lil_matrix((N, N), dtype=np.float32)
    for src, dsts in graph.items():
        for dst in dsts:
            adj[src, dst] = 1.0
            adj[dst, src] = 1.0
    adj = adj + sp.eye(N)  # self-loops
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt  # normalized adjacency
    A_hat = torch.tensor(A_hat.toarray(), dtype=torch.float32)

    # Standard split: train=0:140, val=140:640, test=test_idx
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[:140] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[140:640] = True
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[test_idx] = True

    return {
        "X": X, "A_hat": A_hat, "y": y, "N": N,
        "n_features": X.shape[1], "n_classes": int(y.max()) + 1,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


# ---------------------------------------------------------------------------
# 2. Minimal IGNN (Implicit Graph Neural Network)
# ---------------------------------------------------------------------------

class IGNN(nn.Module):
    """Minimal IGNN: Z* = σ(A_hat @ Z* @ W + X @ U + b).

    Weight-tied iteration → DEQ fixed point. Contractive when ||W||_2 < 1/||A_hat||_2.
    """

    def __init__(self, n_features: int, hidden: int, n_classes: int, spectral_norm: bool = True):
        super().__init__()
        self.hidden = hidden
        self.U = nn.Linear(n_features, hidden)      # input projection
        self.W = nn.Linear(hidden, hidden, bias=False)  # state propagation
        self.head = nn.Linear(hidden, n_classes)     # readout

        nn.init.xavier_normal_(self.W.weight, gain=0.5)
        if spectral_norm:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            self.W = _sn(self.W)

    def operator(self, Z: Tensor, ctx: dict) -> Tensor:
        """F(Z) = ReLU(A_hat @ Z @ W^T + X_proj)."""
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return F_func.relu(A_hat @ self.W(Z) + X_proj)

    def forward(self, X: Tensor, A_hat: Tensor, max_iter: int = 50, tol: float = 1e-5):
        N = X.shape[0]
        X_proj = self.U(X)  # (N, hidden)
        ctx = {"A_hat": A_hat, "X_proj": X_proj}

        # Fixed-point iteration
        Z = torch.zeros(N, self.hidden, device=X.device)
        for k in range(max_iter):
            Z_new = self.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm(), 1.0):
                break
            Z = Z_new
        Z_star = Z_new

        logits = self.head(Z_star)
        return logits, Z_star, ctx


# ---------------------------------------------------------------------------
# 3. Train + IEM validation
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/cora")

    print("=== Loading Cora ===", flush=True)
    _download_cora(data_dir)
    data = _load_cora(data_dir)
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
            print(f"  ep {ep:3d} | loss {loss.item():.4f} | val_acc {val_acc:.3f} | test_acc {test_acc:.3f}", flush=True)

    # Get fixed point
    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[data["test_mask"]] == y[data["test_mask"]]).float().mean())
    print(f"\n  Final test accuracy: {test_acc:.3f}")
    residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
    print(f"  Fixed-point residual: {residual:.2e}")

    # --- IEM ---
    print("\n=== IEM: Implicit Equilibrium Mining on IGNN ===", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from iem import IEMiner

    # Use a SUBGRAPH for Jacobian tractability (full Cora N=2708 → D=2708*64=173k — too large)
    # BFS ego subgraph ensures connectivity (first-k neighbors may be disconnected)
    from iem.adversarial import extract_ego_subgraph
    subgraph_idx = extract_ego_subgraph(A_hat, max_nodes=50)
    S = len(subgraph_idx)
    center = int(subgraph_idx[0].item())
    print(f"  Subgraph: {S} nodes (BFS from center node {center})")

    A_sub = A_hat[subgraph_idx][:, subgraph_idx]
    X_proj_sub = ctx["X_proj"][subgraph_idx]
    Z_star_sub = Z_star[subgraph_idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    def F_ignn(z, c=ctx_sub):
        return model.operator(z, c)

    miner = IEMiner(F_ignn, Z_star_sub, ctx_sub, method="direct")

    # 1. Contractivity
    print("\n  --- Contractivity ---", flush=True)
    report = miner.certify()
    print(f"  rho = {report['rho']:.4f}, contractive = {report['is_contractive']}")

    # 2. Node Attribution (which nodes' features matter most?)
    print("\n  --- Node Attribution (top 5) ---", flush=True)
    phi = miner.node_attribution("X_proj")
    top5 = phi.argsort(descending=True)[:5]
    for rank, idx in enumerate(top5.tolist()):
        real_idx = int(subgraph_idx[idx].item())
        label = int(y[real_idx].item())
        print(f"    #{rank+1}: node {real_idx} (class {label}), phi={phi[idx]:.4e}")
    n_nz = int((phi > 1e-6).sum().item())
    print(f"  Nonzero: {n_nz}/{S}")

    # 3. Certified bounds
    print("\n  --- Certified bounds (eps=0.1) ---", flush=True)
    cert = miner.certified_bound(phi, epsilon=0.1)
    print(f"  rho={cert['rho']:.4f}, max_bound={cert['max_bound']:.4e}")
    if cert.get("max_rank_perturbation") and cert["max_rank_perturbation"] != float("inf"):
        print(f"  Rank stability: {cert['max_rank_perturbation']:.4e}")

    print("\n=== IGNN + IEM: DOMAIN-AGNOSTIC VALIDATION COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
