"""IEM validation on a fraud-detection domain: IGNN on Amazon Fraud (CARE-GNN).

Users as nodes, behavioural similarity links as edges. Binary node
classification (benign vs. fraudulent reviewer). A 6th, security-flavoured
domain for the continuous-to-discrete transfer experiment, distinct from
citation, co-purchase, and power-flow graphs.

Dataset: Amazon Fraud (Dou et al., CARE-GNN, 2020; McAuley & Leskovec base)
  - 11,944 users (nodes), homogeneous ``homo`` adjacency (already symmetric)
  - 25-dim handcrafted node features
  - 2 classes (0 = benign, 1 = fraud)

The raw .mat is expected at ``datasets/amazon_fraud/Amazon.mat`` with keys
``homo`` [N x N sparse], ``features`` [N x 25 sparse], ``label`` [1 x N].

Usage:
    .venv/bin/python -m iem.examples.ignn_amazon_fraud
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse IGNN from Cora example
from iem.examples.ignn_cora import IGNN


def _load_amazon_fraud(data_dir: Path) -> dict:
    """Load Amazon Fraud (CARE-GNN) into dense tensors.

    Returns the same dict contract as ``ignn_amazon._load_amazon``:
    X [N,F] float32, A_hat [N,N] float32 dense (D^-1/2 (A+A^T+I) D^-1/2),
    y [N] int64, N, n_features, n_classes, and bool train/val/test masks.
    """
    mat_path = data_dir / "Amazon.mat"
    data = sio.loadmat(str(mat_path))

    # Features (sparse in .mat) and labels. ``homo`` is already symmetric.
    feats = sp.csr_matrix(data["features"])
    X = torch.tensor(feats.toarray(), dtype=torch.float32)
    y = torch.tensor(data["label"].ravel().astype(np.int64))

    adj = sp.csr_matrix(data["homo"])

    # Alignment guard: node i of A, features, and label must stay aligned.
    assert adj.shape[0] == X.shape[0] == y.shape[0], (
        f"misaligned: A={adj.shape[0]}, X={X.shape[0]}, y={y.shape[0]}"
    )

    # Adjacency (sparse -> dense normalized) — identical recipe to _load_amazon.
    adj = adj + adj.T  # symmetrize (no-op for already-symmetric homo)
    adj = adj + sp.eye(adj.shape[0])  # self-loops
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt

    N = X.shape[0]
    # Random 60/20/20 split (same convention + seed as the other loaders).
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    n_train = int(0.6 * N)
    n_val = int(0.2 * N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:n_train]] = True
    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[perm[n_train:n_train + n_val]] = True
    test_mask = torch.zeros(N, dtype=torch.bool)
    test_mask[perm[n_train + n_val:]] = True

    # Full graph (11944x11944 < Pubmed's 19717) -> dense float32.
    A_hat_dense = torch.tensor(A_hat.toarray(), dtype=torch.float32)

    return {
        "X": X, "A_hat": A_hat_dense, "y": y, "N": N,
        "n_features": X.shape[1], "n_classes": int(y.max()) + 1,
        "train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/amazon_fraud")

    print("=== Loading Amazon Fraud ===", flush=True)
    data = _load_amazon_fraud(data_dir)
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    print(f"  N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")
    print(f"  train={data['train_mask'].sum()}, val={data['val_mask'].sum()}, test={data['test_mask'].sum()}")

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

    print("\n=== AMAZON FRAUD: COMPLETE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
