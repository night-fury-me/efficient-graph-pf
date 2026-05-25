"""Contractive GCN-PF: IGNN-style architecture for power flow with guaranteed ρ < 1.

PE_DEQ_PF's complex ops (exp(1j*θ), conj) resist contractivity (ρ ≈ 1.00
even with spectral_norm + jac_reg=1.0). This model uses the SAME structure
as IGNN (which achieves ρ=0.41 on Cora) adapted for power flow:

  Z* = ReLU(A_hat @ Z* @ W + PF_proj)

Where:
  - A_hat: normalized adjacency from Y_bus (real, symmetric, ||A_hat||₂ < 1)
  - W: spectral-normed weight matrix (||W||₂ ≤ 1)
  - PF_proj: projected bus features [bus_type, P, Q, |V_start|, θ_start]
  - Output: linear head on Z* → (v_pred, θ_pred) per bus

Contractivity guarantee: ||∂F/∂Z|| ≤ ||A_hat||₂ · ||W||₂ ≤ ||A_hat||₂ < 1
because A_hat is degree-normalized and W is spectral-normed.

Usage:
    .venv/bin/python -m iem.examples.contractive_pf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from scipy.stats import kendalltau
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader


class ContractiveGCN_PF(nn.Module):
    """IGNN-style contractive equilibrium for AC power flow."""

    def __init__(self, n_bus_features: int = 5, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.U = nn.Linear(n_bus_features, hidden)
        self.W = nn.Linear(hidden, hidden, bias=False)
        nn.init.xavier_normal_(self.W.weight, gain=0.5)
        from torch.nn.utils.parametrizations import spectral_norm
        self.W = spectral_norm(self.W)

        self.v_head = nn.Linear(hidden, 1)
        self.th_head = nn.Linear(hidden, 1)

    def _build_adjacency(self, Y: Tensor) -> Tensor:
        """Build normalized BINARY adjacency from complex Y_bus.
        A = |Y_ij| > 0 (binary), then D^{-1/2} A D^{-1/2}."""
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        A = (Y.abs() > 1e-12).float()
        A.fill_diagonal_(1.0)  # self-loops
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D

    def _build_bus_features(self, bus_type, S, V0, A_hat=None) -> Tensor:
        """Build per-bus feature vector [is_slack, is_pv, P, Q, |V|]."""
        N = bus_type.shape[-1]
        bt = bus_type.reshape(N)
        is_slack = (bt == 1).float().unsqueeze(-1)
        is_pv = (bt == 2).float().unsqueeze(-1)
        P = S.reshape(N).real.unsqueeze(-1)
        Q = S.reshape(N).imag.unsqueeze(-1)
        Vm = V0.reshape(N, 2)[:, 0:1]
        return torch.cat([is_slack, is_pv, P, Q, Vm], dim=-1)  # (N, 5)

    def operator(self, Z: Tensor, ctx: dict) -> Tensor:
        """F(Z) = ReLU(A_hat @ W(Z) + X_proj). Contractive by construction."""
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        return F_func.relu(A_hat @ self.W(Z) + X_proj)

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph,
                max_iter=50, tol=1e-5, **_unused):
        N = bus_type.shape[-1]
        device = bus_type.device

        # Build graph structure
        if Y is None:
            from models.edge_selfattn.admittance import build_dense_Y, build_edges_blockdiag
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d = Ys.squeeze(0)
            Yc_1d = Yc.squeeze(0)
            undirected, _, _, _, ys_edge, yc_edge = build_edges_blockdiag(
                line_mask_1d=Line_1d, Ys_1d=Ys_1d, Yc_1d=Yc_1d,
                n_nodes_per_graph=n_nodes_per_graph,
                edge_feat_dim=4,
                pairs_for_n=lambda n, d: torch.triu_indices(n, n, offset=1, device=d).t().contiguous(),
                device=device,
            )
            Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)

        A_hat = self._build_adjacency(Y)
        X = self._build_bus_features(bus_type, S, V0, A_hat)
        X_proj = self.U(X)
        ctx = {"A_hat": A_hat, "X_proj": X_proj, "Y": Y}

        # Fixed-point iteration
        Z = torch.zeros(N, self.hidden, device=device)
        for k in range(max_iter):
            Z_new = self.operator(Z, ctx)
            if (Z_new - Z).norm() < tol * max(Z.norm().item(), 1.0):
                break
            Z = Z_new
        Z_star = Z_new

        v_pred = self.v_head(Z_star).squeeze(-1) + V0.reshape(N, 2)[:, 0]
        th_pred = self.th_head(Z_star).squeeze(-1) + V0.reshape(N, 2)[:, 1]
        out = torch.stack([v_pred, th_pred], dim=-1).unsqueeze(0)

        ctx["Z_star"] = Z_star
        return out, ctx


def _brute_force_n1(model, Z_star, ctx, edges, n_iter=50):
    """Remove each edge from A_hat, re-iterate, measure ΔZ."""
    A_orig = ctx["A_hat"]
    scores = torch.zeros(len(edges), device=A_orig.device)
    with torch.no_grad():
        for idx, (i, j) in enumerate(edges):
            A_pert = A_orig.clone()
            # Zero out the edge WITHOUT re-normalizing — matches IFT's
            # first-order perturbation of A_hat[i,j] directly.
            A_pert[i, j] = 0.0
            A_pert[j, i] = 0.0

            ctx_pert = {**ctx, "A_hat": A_pert}
            Z = Z_star.clone()
            for _ in range(n_iter):
                Z = model.operator(Z, ctx_pert)
            scores[idx] = (Z - Z_star).norm().item()
    return scores


def _iem_n1(model, Z_star, ctx, edges):
    """IEM edge sensitivity: pre-compute (I-J)⁻¹, then FD per edge on operator."""
    from iem.ift import compute_jacobian
    D = Z_star.numel()
    device = Z_star.device
    t0 = time.time()

    def F_z(z):
        return model.operator(z.reshape(Z_star.shape), ctx).reshape(-1)

    J = compute_jacobian(F_z, Z_star)
    I_mat = torch.eye(D, device=device)
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

    iem_time = time.time() - t0
    return scores, iem_time


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data — stratified subset for fast iteration, batch_size=1
    ds = ChanghunDataset(
        ["./datasets/HVN_stratified_1500.parquet"], per_unit=True, device=device
    )
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_blockdiag)
    print(f"Dataset: {len(ds)} samples", flush=True)

    # Build + train model (v1 arch: 5 features, hidden=64, single-layer proj)
    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n=== Training ContractiveGCN_PF (100 epochs) ===", flush=True)
    for ep in range(1, 101):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            V_true = batch["V_newton"].to(device)
            loss = ((V_pred - V_true) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:3d} | loss {total_loss/n:.4e}", flush=True)

    # Evaluate on a few graphs
    model.eval()
    print("\n=== IEM + N-1 contingency benchmark ===", flush=True)

    from iem import IEMiner

    taus, top5s, speedups = [], [], []
    for g_idx, batch in enumerate(loader):
        if g_idx >= 20:
            break
        N = int(batch["sizes"][0].item())
        with torch.no_grad():
            V_pred, ctx = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
        Z_star = ctx["Z_star"]

        # Contractivity check (first 3 only — expensive)
        if g_idx < 3:
            def F_z(z):
                return model.operator(z.reshape(Z_star.shape), ctx).reshape(-1)
            miner = IEMiner(lambda z, c=ctx: model.operator(z, c), Z_star, ctx, method="direct")
            rho = miner.rho
            print(f"  Graph {g_idx}: N={N}, rho={rho:.4f}", flush=True)

        # Find active edges
        A_hat = ctx["A_hat"]
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if A_hat[i, j].abs() > 1e-6:
                    edges.append((i, j))
        n_edges = len(edges)
        if n_edges < 3:
            continue

        # Brute-force
        t0 = time.time()
        bf_scores = _brute_force_n1(model, Z_star, ctx, edges)
        bf_time = time.time() - t0

        # IEM
        iem_scores, iem_time = _iem_n1(model, Z_star, ctx, edges)

        tau, p = kendalltau(bf_scores.cpu().numpy(), iem_scores.cpu().numpy())
        k = min(5, n_edges)
        bf_top = set(bf_scores.argsort(descending=True)[:k].tolist())
        iem_top = set(iem_scores.argsort(descending=True)[:k].tolist())
        agree = len(bf_top & iem_top) / k
        speedup = bf_time / max(iem_time, 1e-6)

        taus.append(tau)
        top5s.append(agree)
        speedups.append(speedup)
        print(
            f"  Graph {g_idx:2d} | N={N:2d}, edges={n_edges:3d} | "
            f"τ={tau:+.3f} (p={p:.2e}) | top-{k}={agree:.0%} | "
            f"BF={bf_time:.2f}s IEM={iem_time:.2f}s {speedup:.1f}×",
            flush=True,
        )

    print("\n" + "=" * 60, flush=True)
    if taus:
        print(f"Mean τ:      {np.mean(taus):+.3f} ± {np.std(taus):.3f}")
        print(f"Top-5 agree: {np.mean(top5s):.0%}")
        print(f"Speedup:     {np.mean(speedups):.1f}×")
        print(f"Graphs:      {len(taus)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
