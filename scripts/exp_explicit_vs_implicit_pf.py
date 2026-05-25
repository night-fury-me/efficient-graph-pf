"""Explicit GCN vs Implicit GCN vs PIGNN-Attn-LS power flow residual comparison.

Trains ContractiveGCN_PF (implicit/equilibrium), a standard 4-layer GCN, and
PIGNN-Attn-LS (physics-informed with line search) on IEEE case14 and case30.
Compares voltage RMSE and power-balance residual ΔS (10 seeds).

Usage:
    .venv/bin/python scripts/exp_explicit_vs_implicit_pf.py
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa: register model builders

from iem.examples.contractive_pf import ContractiveGCN_PF
from models.edge_selfattn.model import GNSMsg_EdgeSelfAttn
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]

CASES = [
    ("case14", "datasets/IEEE_case14_2000.parquet", 14),
    ("case30", "datasets/IEEE_case30_2000.parquet", 30),
]


class ExplicitGCN_PF(nn.Module):
    """Standard K-layer GCN for power flow (no equilibrium, no contractivity)."""

    def __init__(self, n_bus_features: int = 5, hidden: int = 64, n_layers: int = 4):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(n_bus_features, hidden)
        self.layers = nn.ModuleList([
            nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)
        ])
        self.v_head = nn.Linear(hidden, 1)
        self.th_head = nn.Linear(hidden, 1)

    def _build_adjacency(self, Y: Tensor) -> Tensor:
        if Y.dim() == 3:
            Y = Y.squeeze(0)
        A = (Y.abs() > 1e-12).float()
        A.fill_diagonal_(1.0)
        deg = A.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D = torch.diag(deg_inv_sqrt)
        return D @ A @ D

    def _build_bus_features(self, bus_type, S, V0) -> Tensor:
        N = bus_type.shape[-1]
        bt = bus_type.reshape(N)
        is_slack = (bt == 1).float().unsqueeze(-1)
        is_pv = (bt == 2).float().unsqueeze(-1)
        P = S.reshape(N).real.unsqueeze(-1)
        Q = S.reshape(N).imag.unsqueeze(-1)
        Vm = V0.reshape(N, 2)[:, 0:1]
        return torch.cat([is_slack, is_pv, P, Q, Vm], dim=-1)

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph, **_unused):
        N = bus_type.shape[-1]
        device = bus_type.device

        if Y is None:
            from models.edge_selfattn.admittance import build_dense_Y, build_edges_blockdiag
            Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys_1d = Ys.squeeze(0)
            Yc_1d = Yc.squeeze(0)
            undirected, _, _, _, ys_edge, yc_edge = build_edges_blockdiag(
                line_mask_1d=Line_1d, Ys_1d=Ys_1d, Yc_1d=Yc_1d,
                n_nodes_per_graph=n_nodes_per_graph, edge_feat_dim=4,
                pairs_for_n=lambda n, d: torch.triu_indices(n, n, offset=1, device=d).t().contiguous(),
                device=device,
            )
            Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)

        A_hat = self._build_adjacency(Y)
        X = self._build_bus_features(bus_type, S, V0)
        Z = self.input_proj(X)

        for layer in self.layers:
            Z = F_func.relu(A_hat @ layer(Z))

        v_pred = self.v_head(Z).squeeze(-1) + V0.reshape(N, 2)[:, 0]
        th_pred = self.th_head(Z).squeeze(-1) + V0.reshape(N, 2)[:, 1]
        out = torch.stack([v_pred, th_pred], dim=-1).unsqueeze(0)
        return out, {"Y": Y}


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_Y_bus(batch, device):
    """Build dense Y_bus from batch data for ΔS computation."""
    N = int(batch["sizes"][0].item())
    from models.edge_selfattn.admittance import build_dense_Y, build_edges_blockdiag
    Line = batch["Lines_connected"].to(device)
    Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
    Ys_1d = batch["Y_Lines"].to(device).squeeze(0)
    Yc_1d = batch["Y_C_Lines"].to(device).squeeze(0)
    undirected, _, _, _, ys_edge, yc_edge = build_edges_blockdiag(
        line_mask_1d=Line_1d, Ys_1d=Ys_1d, Yc_1d=Yc_1d,
        n_nodes_per_graph=batch["sizes"].to(device), edge_feat_dim=4,
        pairs_for_n=lambda n, d: torch.triu_indices(n, n, offset=1, device=d).t().contiguous(),
        device=device,
    )
    return build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)[:N, :N]


def compute_power_balance_residual(V_2d, Y_bus):
    """ΔS = |S(V_pred) - S(V_true)| per bus."""
    Vm = V_2d[:, 0]
    Va = V_2d[:, 1]
    V_complex = Vm * torch.exp(1j * Va)
    I_computed = Y_bus @ V_complex
    return V_complex * I_computed.conj()


def forward_model(model, batch, device, model_type):
    """Unified forward pass for all three model types."""
    args = (
        batch["bus_type"].to(device), batch["Lines_connected"].to(device),
        None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
        batch["S_start"].to(device), batch["V_start"].to(device),
        batch["sizes"].to(device),
    )
    result = model(*args)

    if model_type == "pignn":
        if isinstance(result, tuple) and len(result) == 2:
            V_pred, phys_loss = result
            return V_pred, phys_loss
        return result, None
    else:
        V_pred, ctx = result
        return V_pred, ctx


def train_and_evaluate(model, ds, device, model_type, n_epochs=30):
    train_ds = Subset(ds, range(min(200, len(ds))))
    test_ds = Subset(ds, range(200, min(400, len(ds))))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(n_epochs):
        model.train()
        for batch in train_loader:
            V_pred, extra = forward_model(model, batch, device, model_type)
            V_true = batch["V_newton"].to(device)
            loss = ((V_pred - V_true) ** 2).mean()
            if model_type == "pignn" and extra is not None:
                loss = loss + 0.1 * extra
            optim.zero_grad()
            loss.backward()
            optim.step()

    model.eval()
    v_errors, th_errors, ds_residuals = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            N = int(batch["sizes"][0].item())
            V_pred, _ = forward_model(model, batch, device, model_type)
            V_true = batch["V_newton"].to(device).reshape(N, 2)
            V_pred_2d = V_pred.reshape(N, 2)

            v_errors.append(float(((V_pred_2d[:, 0] - V_true[:, 0]) ** 2).mean().sqrt()))
            th_errors.append(float(((V_pred_2d[:, 1] - V_true[:, 1]) ** 2).mean().sqrt()))

            Y_bus = build_Y_bus(batch, device)
            S_pred = compute_power_balance_residual(V_pred_2d, Y_bus)
            S_true = compute_power_balance_residual(V_true, Y_bus)
            ds_residuals.append(float((S_pred - S_true).abs().mean()))

    return {
        "v_rmse": np.mean(v_errors),
        "th_rmse": np.mean(th_errors),
        "delta_s": np.mean(ds_residuals),
    }


def run_single(case_name, ds_path, N_expected, seed, device):
    ds = ChanghunDataset([ds_path], per_unit=True, device=device)
    results = {}

    # IGNN (implicit equilibrium)
    set_seed(seed)
    ignn = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    results["IGNN"] = train_and_evaluate(ignn, ds, device, "ignn", n_epochs=30)

    # GCN-4 (explicit, no physics)
    set_seed(seed)
    gcn = ExplicitGCN_PF(n_bus_features=5, hidden=64, n_layers=4).to(device)
    results["GCN-4"] = train_and_evaluate(gcn, ds, device, "gcn", n_epochs=30)

    # PIGNN-Attn-LS (explicit, physics-informed with line search)
    set_seed(seed)
    pignn = GNSMsg_EdgeSelfAttn(
        d=10, d_hi=32, K=30, pinn=True, use_armijo=True,
        n_heads=4, num_attn_layers=1,
    ).to(device)
    results["PIGNN"] = train_and_evaluate(pignn, ds, device, "pignn", n_epochs=30)

    return results


def agg(vals):
    m, s = np.mean(vals), np.std(vals)
    return f"{m:.4f}±{s:.4f}", m, s


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t0 = time.time()

    model_names = ["IGNN", "GCN-4", "PIGNN"]

    for case_name, ds_path, N in CASES:
        if not Path(ds_path).exists():
            print(f"SKIP {case_name}: dataset not found")
            continue

        all_res = {m: {"v_rmse": [], "th_rmse": [], "delta_s": []} for m in model_names}

        for si, seed in enumerate(SEEDS):
            print(f"  {case_name} seed {seed} ({si+1}/{len(SEEDS)})...", end=" ", flush=True)
            res = run_single(case_name, ds_path, N, seed, device)
            ds_strs = []
            for m in model_names:
                for k in all_res[m]:
                    all_res[m][k].append(res[m][k])
                ds_strs.append(f"{m}={res[m]['delta_s']:.3f}")
            print(f"ΔS: {', '.join(ds_strs)}", flush=True)

        print(f"\n{'='*80}")
        print(f"  {case_name} (N={N}) — 10 seeds")
        print(f"{'='*80}")
        header = f"  {'Metric':<20}"
        for m in model_names:
            header += f" {m:>20}"
        print(header)
        print(f"  {'-'*76}")
        for k in ["v_rmse", "th_rmse", "delta_s"]:
            label = {
                "v_rmse": "|V| RMSE (p.u.)",
                "th_rmse": "θ RMSE (p.u.)",
                "delta_s": "ΔS residual (p.u.)",
            }[k]
            row = f"  {label:<20}"
            for m in model_names:
                s, _, _ = agg(all_res[m][k])
                row += f" {s:>20}"
            print(row)
        print()

    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
