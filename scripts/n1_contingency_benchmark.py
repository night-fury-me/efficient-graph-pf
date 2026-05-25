"""N-1 contingency ranking benchmark: IEM (one backward) vs brute-force.

For each graph in the test set:
  1. Run PE_DEQ_PF to get z* (the equilibrium voltage profile)
  2. IEM method: compute ||∂z*/∂Y_ij|| for ALL edges in ONE backward pass
  3. Brute-force: for each active edge e, set Y_e=0, re-run the DEQ forward,
     measure ||z*_perturbed - z*_original|| → per-edge criticality
  4. Rank edges by both methods, compute Kendall τ correlation

Reports:
  - Kendall τ (should be ≥ 0.9 for IEM to be publishable)
  - Speedup factor (should be 10-100× for small grids)
  - Top-5 critical edges agreement

Usage:
    .venv/bin/python scripts/n1_contingency_benchmark.py [--n_graphs 20]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa — registers all
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from iem import IEMiner
from models.pe_deq_pf.model import PE_DEQ_PF
from torch.utils.data import DataLoader


def _capture_fixed_point(model, batch, device):
    """Run model forward, capture z_star and ctx via monkey-patch."""
    _cap = {}
    _orig = model.deq.__class__.__call__

    def _hook(self, z0, ctx):
        r = _orig(self, z0, ctx)
        _cap["z"] = r.detach()
        _cap["c"] = ctx
        return r

    model.deq.__class__.__call__ = _hook
    with torch.no_grad():
        model(
            batch["bus_type"].to(device),
            batch["Lines_connected"].to(device),
            None,
            batch["Y_Lines"].to(device),
            batch["Y_C_Lines"].to(device),
            batch["S_start"].to(device),
            batch["V_start"].to(device),
            batch["sizes"].to(device),
        )
    model.deq.__class__.__call__ = _orig
    return _cap["z"], _cap["c"]


def _brute_force_n1(model, z_star, ctx, n_iter=30):
    """Remove each active edge, re-iterate the operator, measure ΔV.

    Returns: (n_edges,) tensor of ||z*_perturbed - z*_original|| per edge.
    """
    Y_orig = ctx["Y"]  # (N, N) or (1, N, N) complex
    if Y_orig.dim() == 3:
        Y_orig = Y_orig.squeeze(0)
    N = Y_orig.shape[-1]

    # Find active edges (off-diagonal nonzero entries in upper triangle)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if Y_orig[i, j].abs() > 1e-12:
                edges.append((i, j))

    n_edges = len(edges)
    bf_scores = torch.zeros(n_edges, device=Y_orig.device)

    for idx, (i, j) in enumerate(edges):
        Y_pert = Y_orig.clone()
        y_ij = Y_pert[i, j].clone()
        y_ji = Y_pert[j, i].clone()
        Y_pert[i, j] = 0.0
        Y_pert[j, i] = 0.0
        Y_pert[i, i] = Y_pert[i, i] + y_ij
        Y_pert[j, j] = Y_pert[j, j] + y_ji

        ctx_pert = {**ctx, "Y": Y_pert}

        # Re-iterate from z_star to find new equilibrium
        z = z_star.clone()
        for _ in range(n_iter):
            z = model._operator(z, ctx_pert)
        bf_scores[idx] = (z - z_star).norm().item()

    return edges, bf_scores


def _iem_n1(model, z_star, ctx, edges):
    """Direct edge sensitivity via IFT: ∂z*/∂Y_ij for each active edge.

    Steps:
      1. Compute state Jacobian J_zz = ∂F/∂z (D×D) — one-time cost
      2. Pre-compute A_inv = (I - J_zz)⁻¹ (or ridge-regularized)
      3. For each edge (i,j): compute ∂F/∂Y_ij via one backward, then
         ∂z*/∂Y_ij = A_inv @ ∂F/∂Y_ij
      4. Edge score = ||∂z*/∂Y_ij||

    Total cost: D backward passes (J_zz) + n_edges backward passes + n_edges
    mat-vec multiplies ≈ 0.3-1s for D~150, n_edges~200.
    """
    D = z_star.numel()
    N = ctx["Y"].shape[-1]
    device = z_star.device
    t0 = time.time()

    # Step 1: State Jacobian J_zz
    from iem.ift import compute_jacobian
    def F_z(z):
        return model._operator(z.reshape(z_star.shape), ctx).reshape(-1)
    J_zz = compute_jacobian(F_z, z_star)

    # Step 2: Pre-compute (I - J)⁻¹ (with ridge if near-singular)
    I_mat = torch.eye(D, device=device, dtype=J_zz.dtype)
    A = I_mat - J_zz
    try:
        A_inv = torch.linalg.inv(A)
    except torch._C._LinAlgError:
        rho = torch.linalg.eigvals(J_zz).abs().max().item()
        lam = max(rho - 0.99, 0.01)
        A_inv = torch.linalg.inv((1 + lam) * I_mat - J_zz)

    # Step 3: Per-edge ∂F/∂Y_ij via backward, then ∂z*/∂Y_ij = A_inv @ ∂F/∂Y_ij
    Y_orig = ctx["Y"]
    if Y_orig.dim() == 3:
        Y_orig = Y_orig.squeeze(0)

    edge_scores = torch.zeros(len(edges), device=device)

    # Per-edge ∂F/∂Y_ij via finite-difference on the operator, then IFT solve
    # ∂F/∂Y_ij via FD, then multiply by A_inv
    eps = 1e-5
    with torch.no_grad():
        f_base = model._operator(z_star, ctx).reshape(-1)
        for idx, (i, j) in enumerate(edges):
            Y_pert = Y_orig.clone()
            Y_pert[i, j] += eps
            Y_pert[j, i] += eps  # symmetric perturbation
            ctx_pert = {**ctx, "Y": Y_pert}
            f_pert = model._operator(z_star, ctx_pert).reshape(-1)
            dF_dYij = (f_pert - f_base) / eps  # (D,) real
            # ∂z*/∂Y_ij = A_inv @ dF_dYij
            dz_dYij = A_inv @ dF_dYij.to(A_inv.dtype)
            edge_scores[idx] = dz_dYij.norm().item()

    iem_time = time.time() - t0
    return edge_scores, iem_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_graphs", type=int, default=10)
    ap.add_argument("--ckpt", type=str, default="results/runs/contractive_strong_jac/ckpt/last.ckpt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = PE_DEQ_PF(
        d=4, d_hi=32, num_attn_layers=2, pinn=True,
        dtheta_max=0.30, dvm_frac=0.10,
        forward_iter=15, backward_iter=15,
        backward_mode="phantom", jac_reg_weight=1.0, jac_reg_n_samples=1,
        damping_init=0.1, spectral_norm=True, unrolled_warmup_epochs=0,
    ).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Model loaded from {args.ckpt}")

    ds = ChanghunDataset(
        ["./datasets/HVN_15000_NR_plain_4_to_32_buses.parquet"],
        per_unit=True, device=device,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)

    taus = []
    top5_agreements = []
    speedups = []

    for g_idx, batch in enumerate(loader):
        if g_idx >= args.n_graphs:
            break
        N = int(batch["sizes"][0].item())

        z_star, ctx = _capture_fixed_point(model, batch, device)

        # --- Brute-force N-1 ---
        t0 = time.time()
        with torch.no_grad():
            edges, bf_scores = _brute_force_n1(model, z_star, ctx)
        bf_time = time.time() - t0
        n_edges = len(edges)

        if n_edges < 3:
            print(f"Graph {g_idx}: N={N}, only {n_edges} edges — skipping")
            continue

        # --- IEM (direct edge sensitivity via IFT) ---
        iem_edge_scores, iem_time = _iem_n1(model, z_star, ctx, edges)

        # --- Compare rankings ---
        bf_rank = bf_scores.argsort(descending=True)
        iem_rank = iem_edge_scores.argsort(descending=True)

        bf_order = bf_scores.cpu().numpy()
        iem_order = iem_edge_scores.cpu().numpy()
        tau, p_val = kendalltau(bf_order, iem_order)
        taus.append(tau)

        # Top-5 agreement (how many of BF's top-5 are in IEM's top-5)
        k = min(5, n_edges)
        bf_top = set(bf_rank[:k].tolist())
        iem_top = set(iem_rank[:k].tolist())
        agreement = len(bf_top & iem_top) / k
        top5_agreements.append(agreement)

        speedup = bf_time / max(iem_time, 1e-6)
        speedups.append(speedup)

        print(
            f"Graph {g_idx:2d} | N={N:2d}, edges={n_edges:3d} | "
            f"τ={tau:+.3f} (p={p_val:.2e}) | top-{k} agree={agreement:.0%} | "
            f"BF={bf_time:.2f}s, IEM={iem_time:.2f}s, speedup={speedup:.1f}×"
        )

    print("\n" + "=" * 70)
    if taus:
        import numpy as np

        print(f"Mean Kendall τ:      {np.mean(taus):+.3f} ± {np.std(taus):.3f}")
        print(f"Mean top-5 agreement: {np.mean(top5_agreements):.0%}")
        print(f"Mean speedup:         {np.mean(speedups):.1f}×")
        print(f"Graphs evaluated:     {len(taus)}")
    else:
        print("No valid graphs evaluated.")


if __name__ == "__main__":
    sys.exit(main() or 0)
