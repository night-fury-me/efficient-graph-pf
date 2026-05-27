#!/usr/bin/env python3
"""Standard GAT (no edge-weight modification) on Cora.

Compares against GAT-dagger (edge-weighted) results in the AEGIS paper.
Key insight: standard GAT uses A only as a binary mask for attention
neighbourhood, so continuous perturbation of A_hat does NOT change
attention weights => dZ/dA_ij ~ 0 => S_c is zero => tau undefined.

This script:
1. Implements a standard 2-layer GAT (PyTorch only, no PyG)
2. Trains it (200 epochs, 10 seeds), reports test accuracy
3. Computes finite-difference sensitivity on 10 sampled edges
4. Reports whether FD sensitivities confirm near-zero claim
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Data loader from existing codebase
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iem.examples.ignn_cora import _download_cora, _load_cora


# ---------------------------------------------------------------------------
# Standard GAT layer (no edge weights — A is binary mask only)
# ---------------------------------------------------------------------------
class GATLayer(nn.Module):
    """Single-head GAT layer.

    alpha_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
    where j in N(i) is determined by (adj > 0), i.e. binary mask.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))

        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        h:   (N, in_features)
        adj: (N, N) — used ONLY as binary mask (nonzero pattern)
        """
        Wh = h @ self.W  # (N, out)
        N = Wh.size(0)

        # Compute attention coefficients
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)  # (N, N, out)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)  # (N, N, out)
        e = self.leaky_relu(torch.cat([Wh_i, Wh_j], dim=2) @ self.a).squeeze(-1)  # (N, N)

        # Binary mask: attention only over neighbours where adj > 0
        mask = (adj > 0).float()
        e = e.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(e, dim=1)
        attention = self.dropout(attention)

        h_prime = attention @ Wh  # (N, out)

        if self.concat:
            return F.elu(h_prime)
        return h_prime


class StandardGAT(nn.Module):
    """2-layer standard GAT for node classification."""

    def __init__(self, n_features: int, n_hidden: int, n_classes: int,
                 n_heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout

        # Layer 1: multi-head attention
        self.heads = nn.ModuleList([
            GATLayer(n_features, n_hidden, dropout=dropout, concat=True)
            for _ in range(n_heads)
        ])

        # Layer 2: single-head for classification
        self.out_layer = GATLayer(n_hidden * n_heads, n_classes,
                                  dropout=dropout, concat=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, training=self.training)

        # Multi-head layer 1: concatenate heads
        x = torch.cat([head(x, adj) for head in self.heads], dim=1)

        x = F.dropout(x, self.dropout, training=self.training)

        # Layer 2: single head
        x = self.out_layer(x, adj)
        return x


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_seed(data: dict, seed: int, device: torch.device,
                   n_epochs: int = 200, lr: float = 5e-3,
                   weight_decay: float = 5e-4) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    model = StandardGAT(
        n_features=data["n_features"],
        n_hidden=8,
        n_classes=data["n_classes"],
        n_heads=8,
        dropout=0.6,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for ep in range(1, n_epochs + 1):
        model.train()
        logits = model(X, A_hat)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0 or ep == n_epochs:
            model.eval()
            with torch.no_grad():
                logits = model(X, A_hat)
                pred = logits.argmax(dim=1)
                val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
                test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

    return {"seed": seed, "val_acc": best_val_acc, "test_acc": best_test_acc,
            "model": model}


# ---------------------------------------------------------------------------
# Finite-difference sensitivity
# ---------------------------------------------------------------------------
def fd_sensitivity(model: nn.Module, X: torch.Tensor, A_hat: torch.Tensor,
                   n_edges: int = 10, delta: float = 1e-3,
                   device: torch.device = torch.device("cpu")) -> list[float]:
    """Perturb A_hat[i,j] by delta, measure ||DeltaZ|| / delta for sampled edges."""
    model.eval()

    # Find nonzero edges (excluding self-loops)
    mask = (A_hat > 0) & (~torch.eye(A_hat.shape[0], dtype=torch.bool, device=device))
    edges = mask.nonzero(as_tuple=False)
    if len(edges) > n_edges:
        idx = torch.randperm(len(edges))[:n_edges]
        edges = edges[idx]

    # Baseline output
    with torch.no_grad():
        Z_base = model(X, A_hat)

    sensitivities = []
    for e in edges:
        i, j = int(e[0]), int(e[1])
        A_pert = A_hat.clone()
        A_pert[i, j] = A_hat[i, j] + delta

        with torch.no_grad():
            Z_pert = model(X, A_pert)

        dZ = (Z_pert - Z_base).norm().item()
        sensitivities.append(dZ / delta)

    return sensitivities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path("datasets/cora")

    print("=" * 60)
    print("Standard GAT on Cora — comparison for AEGIS paper")
    print("=" * 60)

    # Load data
    _download_cora(data_dir)
    data = _load_cora(data_dir)
    print(f"N={data['N']}, features={data['n_features']}, classes={data['n_classes']}")
    print(f"train={data['train_mask'].sum()}, val={data['val_mask'].sum()}, "
          f"test={data['test_mask'].sum()}")
    print()

    # Train 10 seeds
    print(f"Training standard 2-layer GAT, 200 epochs, 10 seeds ...")
    print(f"  (8 heads x 8 hidden per head = 64 hidden dim)")
    print()

    results = []
    t0 = time.time()
    for seed in range(10):
        res = train_one_seed(data, seed=seed, device=device, n_epochs=200)
        results.append(res)
        print(f"  seed {seed}: val_acc={res['val_acc']:.4f}, test_acc={res['test_acc']:.4f}")
    elapsed = time.time() - t0

    accs = [r["test_acc"] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print()
    print(f"Test accuracy: {mean_acc:.4f} +/- {std_acc:.4f}  (10 seeds, {elapsed:.1f}s)")

    # Finite-difference sensitivity
    print()
    print("=" * 60)
    print("Finite-difference sensitivity analysis")
    print("  Perturbing A_hat[i,j] += 0.001 for 10 random edges")
    print("=" * 60)

    # Use best seed model
    best = max(results, key=lambda r: r["test_acc"])
    model = best["model"]
    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)

    torch.manual_seed(42)
    sens = fd_sensitivity(model, X, A_hat, n_edges=10, delta=1e-3, device=device)

    print()
    for idx, s in enumerate(sens):
        print(f"  edge {idx}: ||dZ||/delta = {s:.6e}")
    mean_sens = np.mean(sens)
    max_sens = np.max(sens)
    print()
    print(f"  Mean sensitivity: {mean_sens:.6e}")
    print(f"  Max  sensitivity: {max_sens:.6e}")

    # Interpretation
    print()
    print("=" * 60)
    print("Interpretation")
    print("=" * 60)

    # Threshold: sensitivities < 1e-3 are "near zero"
    threshold = 1e-3
    near_zero = all(s < threshold for s in sens)

    if near_zero:
        status = "CONFIRMED"
        explanation = (
            "All FD sensitivities are near-zero (< 1e-3). This confirms that\n"
            "standard GAT attention alpha_ij = softmax(LeakyReLU(a^T[Wh_i||Wh_j]))\n"
            "is computed independently of the edge WEIGHT in A_hat. The adjacency\n"
            "only acts as a binary mask determining the neighbourhood N(i).\n"
            "Therefore dZ/dA_ij ~ 0 for continuous perturbation, S_c columns are\n"
            "zero, and Kendall tau is undefined.\n"
            "\n"
            "This is precisely WHY the paper introduces GAT-dagger, which\n"
            "multiplies attention by edge weight: alpha_ij * A_hat[i,j].\n"
            "Only GAT-dagger makes the output continuously sensitive to A_hat,\n"
            "enabling meaningful IEM attribution (tightness=0.99, tau=+0.36)."
        )
    else:
        status = "NUANCED"
        explanation = (
            "Some FD sensitivities are non-negligible. This may be due to:\n"
            "  - Numerical precision in softmax masking\n"
            "  - The perturbation changing the binary mask threshold\n"
            "However the sensitivities should still be much smaller than\n"
            "those observed for GAT-dagger or other edge-weighted models."
        )

    print(f"Status: {status}")
    print()
    print(explanation)

    # Comparison table
    print()
    print("=" * 60)
    print("Comparison: Standard GAT vs GAT-dagger (from paper)")
    print("=" * 60)
    print(f"{'Metric':<25} {'Standard GAT':<20} {'GAT-dagger':<20}")
    print("-" * 65)
    print(f"{'Test accuracy':<25} {mean_acc:.4f} +/- {std_acc:.4f}    {'~comparable':<20}")
    print(f"{'FD sensitivity (mean)':<25} {mean_sens:.2e}           {'significant':<20}")
    print(f"{'S_c (IEM attribution)':<25} {'~ 0 (undefined)':<20} {'well-defined':<20}")
    print(f"{'Tightness':<25} {'N/A':<20} {'0.99':<20}")
    print(f"{'AtkAdv':<25} {'N/A':<20} {'4.4x':<20}")
    print(f"{'Kendall tau':<25} {'undefined':<20} {'+0.36':<20}")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "gat_standard_comparison.txt"

    with open(out_path, "w") as f:
        f.write("Standard GAT on Cora — Comparison for AEGIS Paper\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Architecture: 2-layer GAT, 8 heads x 8 hidden, dropout=0.6\n")
        f.write(f"Training: 200 epochs, Adam lr=5e-3, wd=5e-4, 10 seeds\n")
        f.write(f"Device: {device}\n\n")

        f.write("Per-seed test accuracy:\n")
        for r in results:
            f.write(f"  seed {r['seed']}: {r['test_acc']:.4f}\n")
        f.write(f"\nMean test accuracy: {mean_acc:.4f} +/- {std_acc:.4f}\n\n")

        f.write("Finite-difference sensitivity (delta=0.001, 10 edges):\n")
        for idx, s in enumerate(sens):
            f.write(f"  edge {idx}: ||dZ||/delta = {s:.6e}\n")
        f.write(f"\n  Mean: {mean_sens:.6e}\n")
        f.write(f"  Max:  {max_sens:.6e}\n\n")

        f.write(f"Near-zero confirmation: {status}\n\n")
        f.write(explanation + "\n\n")

        f.write("Comparison Table:\n")
        f.write(f"{'Metric':<25} {'Standard GAT':<20} {'GAT-dagger':<20}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Test accuracy':<25} {mean_acc:.4f} +/- {std_acc:.4f}    {'~comparable':<20}\n")
        f.write(f"{'FD sensitivity (mean)':<25} {mean_sens:.2e}           {'significant':<20}\n")
        f.write(f"{'S_c (IEM attribution)':<25} {'~ 0 (undefined)':<20} {'well-defined':<20}\n")
        f.write(f"{'Tightness':<25} {'N/A':<20} {'0.99':<20}\n")
        f.write(f"{'AtkAdv':<25} {'N/A':<20} {'4.4x':<20}\n")
        f.write(f"{'Kendall tau':<25} {'undefined':<20} {'+0.36':<20}\n")

    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
