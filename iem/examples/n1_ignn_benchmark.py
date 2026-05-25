"""N-1 edge criticality benchmark for IGNN across all three domains.

For each domain (Cora, Amazon Photo, WikiCS):
  1. Train IGNN, get fixed point Z*
  2. IEM: compute ∂Z*/∂A_hat_ij for each edge via FD + (I-J)⁻¹
  3. Brute-force: remove each edge from A_hat, re-iterate, measure ||ΔZ||
  4. Compare rankings via Kendall τ

Usage:
    .venv/bin/python -m iem.examples.n1_ignn_benchmark
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from iem import IEMiner
from iem.ift import compute_jacobian
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora
from iem.examples.ignn_amazon import _load_amazon
from iem.examples.ignn_wikics import _load_wikics


def _train_ignn(model, X, A_hat, y, train_mask, val_mask, test_mask, epochs=50, device="cuda"):
    """Train IGNN, return final test acc + Z_star + ctx."""
    import torch.nn.functional as F_func
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(1, epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F_func.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
    model.eval()
    with torch.no_grad():
        logits, Z_star, ctx = model(X, A_hat)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[test_mask] == y[test_mask]).float().mean())
    return test_acc, Z_star, ctx


def _get_subgraph(A_hat, Z_star, ctx, y, max_nodes=50):
    """Extract a subgraph around the highest-degree node."""
    deg = A_hat.sum(dim=1)
    center = int(deg.argmax().item())
    neighbors = (A_hat[center] > 0).nonzero(as_tuple=True)[0]
    idx = neighbors[:max_nodes]
    S = len(idx)
    A_sub = A_hat[idx][:, idx]
    X_proj_sub = ctx["X_proj"][idx]
    Z_sub = Z_star[idx]
    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}

    # Active edges in subgraph
    edges = []
    for i in range(S):
        for j in range(i + 1, S):
            if A_sub[i, j].abs() > 1e-6:
                edges.append((i, j))
    return idx, Z_sub, ctx_sub, edges, S


def _brute_force_edges(model, Z_star, ctx, edges, n_iter=50):
    """Remove each edge, re-iterate, measure ΔZ. No re-normalization."""
    A_orig = ctx["A_hat"]
    scores = torch.zeros(len(edges), device=A_orig.device)
    with torch.no_grad():
        for idx, (i, j) in enumerate(edges):
            A_pert = A_orig.clone()
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0
            ctx_pert = {**ctx, "A_hat": A_pert}
            Z = Z_star.clone()
            for _ in range(n_iter):
                Z = model.operator(Z, ctx_pert)
            scores[idx] = (Z - Z_star).norm().item()
    return scores


def _iem_edges(model, Z_star, ctx, edges):
    """IEM edge sensitivity via FD on operator + (I-J)⁻¹."""
    D = Z_star.numel()
    device = Z_star.device
    t0 = time.time()

    def F_z(z):
        return model.operator(z.reshape(Z_star.shape), ctx).reshape(-1)

    J = compute_jacobian(F_z, Z_star)
    I_mat = torch.eye(D, device=device, dtype=J.dtype)
    A_sys = I_mat - J
    try:
        A_inv = torch.linalg.inv(A_sys)
    except torch._C._LinAlgError:
        rho = torch.linalg.eigvals(J).abs().max().item()
        lam = max(rho - 0.99, 0.01)
        A_inv = torch.linalg.inv((1 + lam) * I_mat - J)

    A_hat = ctx["A_hat"]
    eps = 1e-4
    scores = torch.zeros(len(edges), device=device)
    with torch.no_grad():
        f_base = model.operator(Z_star, ctx).reshape(-1)
        for idx, (i, j) in enumerate(edges):
            A_pert = A_hat.clone()
            A_pert[i, j] += eps
            A_pert[j, i] += eps
            ctx_pert = {**ctx, "A_hat": A_pert}
            f_pert = model.operator(Z_star, ctx_pert).reshape(-1)
            dF = (f_pert - f_base) / eps
            dz = A_inv @ dF
            scores[idx] = dz.norm().item()

    return scores, time.time() - t0


def run_domain(name, data, device):
    """Train IGNN + run N-1 benchmark on one domain."""
    print(f"\n{'='*60}", flush=True)
    print(f"=== {name}: N={data['N']}, features={data['n_features']}, classes={data['n_classes']} ===", flush=True)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)

    model = IGNN(data["n_features"], hidden=64, n_classes=data["n_classes"]).to(device)
    test_acc, Z_star, ctx = _train_ignn(
        model, X, A_hat, y,
        data["train_mask"].to(device), data["val_mask"].to(device), data["test_mask"].to(device),
        epochs=50, device=device,
    )
    residual = (model.operator(Z_star, ctx) - Z_star).norm().item()
    print(f"  test_acc={test_acc:.3f}, residual={residual:.2e}", flush=True)

    # Subgraph
    idx, Z_sub, ctx_sub, edges, S = _get_subgraph(A_hat, Z_star, ctx, y)
    n_edges = len(edges)
    print(f"  Subgraph: {S} nodes, {n_edges} edges", flush=True)

    if n_edges < 3:
        print(f"  Too few edges — skipping N-1", flush=True)
        return None

    # ρ
    miner = IEMiner(lambda z, c=ctx_sub: model.operator(z, c), Z_sub, ctx_sub, method="direct")
    rho = miner.rho
    print(f"  rho={rho:.4f}, contractive={rho < 1}", flush=True)

    # Brute-force
    t0 = time.time()
    bf_scores = _brute_force_edges(model, Z_sub, ctx_sub, edges)
    bf_time = time.time() - t0

    # IEM
    iem_scores, iem_time = _iem_edges(model, Z_sub, ctx_sub, edges)

    # Compare
    tau, p = kendalltau(bf_scores.cpu().numpy(), iem_scores.cpu().numpy())
    k = min(5, n_edges)
    bf_top = set(bf_scores.argsort(descending=True)[:k].tolist())
    iem_top = set(iem_scores.argsort(descending=True)[:k].tolist())
    agree = len(bf_top & iem_top) / k
    speedup = bf_time / max(iem_time, 1e-6)

    print(f"  τ={tau:+.3f} (p={p:.2e}), top-{k}={agree:.0%}, "
          f"BF={bf_time:.2f}s, IEM={iem_time:.2f}s, speedup={speedup:.1f}×", flush=True)

    return {"name": name, "acc": test_acc, "rho": rho, "tau": tau,
            "top5": agree, "speedup": speedup, "n_edges": n_edges}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = [
        ("Cora (Citations)", _load_cora(Path("datasets/cora"))),
        ("Amazon Photo (E-commerce)", _load_amazon(Path("datasets/amazon_photo"))),
        ("WikiCS (Encyclopedia)", _load_wikics(Path("datasets/wikics"))),
    ]

    results = []
    for name, data in datasets:
        r = run_domain(name, data, device)
        if r:
            results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("=== CROSS-DOMAIN N-1 SUMMARY ===", flush=True)
    print(f"{'Domain':<30} {'Acc':>6} {'ρ':>6} {'τ':>7} {'Top-5':>6} {'Speedup':>8}", flush=True)
    print("-" * 70, flush=True)
    for r in results:
        print(f"{r['name']:<30} {r['acc']:>5.1%} {r['rho']:>6.3f} {r['tau']:>+6.3f} {r['top5']:>5.0%} {r['speedup']:>7.1f}×", flush=True)
    if results:
        mean_tau = np.mean([r["tau"] for r in results])
        mean_top5 = np.mean([r["top5"] for r in results])
        print("-" * 70, flush=True)
        print(f"{'MEAN':<30} {'':>6} {'':>6} {mean_tau:>+6.3f} {mean_top5:>5.0%}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
