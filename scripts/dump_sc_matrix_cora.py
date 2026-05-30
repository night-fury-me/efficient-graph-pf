"""Dump the structural sensitivity matrix S_c on a small Cora ego-graph,
used by paper/figures/fig_sc_heatmap.tex.

Trains IGNN on Cora (one seed), extracts a small BFS ego-subgraph, then
materialises S_c (rows = node-channels, cols = edge perturbations) and
its leading right singular vector v_1.

Output:
    paper/figures/data/sc_matrix.npy   -- (D, |E|) numpy array of |S_c|
    paper/figures/data/sc_v1.npy       -- (|E|,)  numpy array of |v_1|
    paper/figures/data/sc_meta.json    -- N, d, |E|, sigma_1, sigma_2

Usage:
    .venv/bin/python scripts/dump_sc_matrix_cora.py [--N 10 --hidden 4 --seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    structural_sensitivity_matrix,
)
from iem.examples.ignn_cora import IGNN, _download_cora, _load_cora


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=10, help="Subgraph size (default 10)")
    p.add_argument("--hidden", type=int, default=4, help="IGNN hidden dim (default 4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out-dir", default="paper/figures/data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load Cora ----
    data_dir = ROOT / "iem" / "examples" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _download_cora(data_dir)
    data = _load_cora(data_dir)

    X = data["X"].to(device)
    A_hat = data["A_hat"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)

    print(f"Cora: N={X.shape[0]}, F={X.shape[1]}, classes={data['n_classes']}")
    print(f"device={device}, hidden={args.hidden}, target subgraph N={args.N}")

    # ---- Train IGNN ----
    model = IGNN(data["n_features"], hidden=args.hidden, n_classes=data["n_classes"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        logits, _, _ = model(X, A_hat)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if ep % 10 == 0:
            print(f"  ep {ep}: loss={loss.item():.4f}, t={time.time()-t0:.1f}s")

    # ---- Extract small ego subgraph ----
    model.eval()
    sub_idx = extract_ego_subgraph(A_hat, max_nodes=args.N)
    N_sub = len(sub_idx)
    print(f"Subgraph: N_sub={N_sub} (target {args.N})")
    A_sub = A_hat[sub_idx][:, sub_idx]
    X_sub = X[sub_idx]

    # ---- Forward pass to fixed point (use the model's own operator) ----
    X_proj_sub = model.U(X_sub)
    ctx = {"A_hat": A_sub, "X_proj": X_proj_sub}

    with torch.no_grad():
        Z = torch.zeros(N_sub, args.hidden, device=device)
        for _ in range(200):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-8:
                break
            Z = Z_new
        Z_star = Z

    def operator(z, c):
        return model.operator(z, c)

    # ---- Compute S, S_c ----
    print("Computing S, S_c...", flush=True)
    J_z, J_A, _ = _compute_structural_jacobian(operator, Z_star, ctx)
    S = structural_sensitivity_matrix(operator, Z_star, ctx, J_z=J_z, J_A=J_A)
    S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

    print(f"  S shape: {tuple(S.shape)}, S_c shape: {tuple(S_c.shape)}, |E|={len(edge_list)}")

    # ---- SVD ----
    U_svd, sigma, Vh = torch.linalg.svd(S_c.cpu(), full_matrices=False)
    v_1 = Vh[0].abs().cpu().numpy()
    sigma_np = sigma.cpu().numpy()

    # ---- Save ----
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sc_np = S_c.abs().cpu().numpy()
    np.save(out_dir / "sc_matrix.npy", sc_np)
    np.save(out_dir / "sc_v1.npy", v_1)

    meta = {
        "N_sub": int(N_sub),
        "hidden": int(args.hidden),
        "n_edges": int(len(edge_list)),
        "seed": int(args.seed),
        "sigma_1": float(sigma_np[0]),
        "sigma_2": float(sigma_np[1]) if len(sigma_np) > 1 else 0.0,
        "gap_ratio": float((sigma_np[0] - sigma_np[1]) / sigma_np[0]) if len(sigma_np) > 1 else 1.0,
        "sc_shape": list(sc_np.shape),
        "max_abs_sc": float(sc_np.max()),
    }
    with (out_dir / "sc_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:")
    print(f"  {out_dir / 'sc_matrix.npy'}  shape {sc_np.shape}")
    print(f"  {out_dir / 'sc_v1.npy'}      shape {v_1.shape}")
    print(f"  {out_dir / 'sc_meta.json'}   sigma_1={meta['sigma_1']:.3f}, "
          f"gap={meta['gap_ratio']:.3f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
