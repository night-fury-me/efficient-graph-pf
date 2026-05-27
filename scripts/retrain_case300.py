"""Retrain case300 with proper settings to improve model quality.

Current: 200 training samples, 30 epochs → ΔS = 10.9 p.u., θ = 0.394
Target:  1600 training samples, 200 epochs → measure improvement

Reports: |V| RMSE, θ RMSE, ΔS (power-balance residual), τ, P@10

Usage:
    .venv/bin/python scripts/retrain_case300.py
"""

from __future__ import annotations

import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_func
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import models  # noqa
from iem.adversarial import (
    _compute_structural_jacobian,
    constrained_sensitivity_matrix,
    extract_ego_subgraph,
    greedy_structural_attack,
    optimal_structural_attack,
    structural_sensitivity_matrix,
)
from iem.examples.contractive_pf import ContractiveGCN_PF
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from torch.utils.data import DataLoader, Subset

SEEDS = [42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999]
DS_PATH = "datasets/IEEE_case300_2000.parquet"
N_TRAIN = 1600
N_EPOCHS = 200
HIDDEN = 64
LR = 1e-3
SUBGRAPH_SIZE = 200


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconverge(model, Z_init, ctx, max_iter=200):
    Z = Z_init.clone()
    with torch.no_grad():
        for _ in range(max_iter):
            Z_new = model.operator(Z, ctx)
            if (Z_new - Z).norm() < 1e-7:
                break
            Z = Z_new
    return Z_new


def compute_physics_metrics(V_pred, V_true, Y_bus, S_start):
    """Compute |V| RMSE, θ RMSE, and ΔS (power-balance residual).

    V_pred, V_true: [batch, N, 2] or [N, 2] — columns are (|V|, θ)
    Y_bus: [N, N] complex — bus admittance matrix
    S_start: [batch, N] or [N] — injected apparent power per bus
    """
    V_pred_c = V_pred.detach().cpu()
    V_true_c = V_true.detach().cpu()

    # Squeeze batch dimensions
    while V_pred_c.dim() > 2:
        V_pred_c = V_pred_c.squeeze(0)
    while V_true_c.dim() > 2:
        V_true_c = V_true_c.squeeze(0)

    N = V_pred_c.shape[0]

    # |V| and θ (columns 0 and 1)
    V_mag_pred = V_pred_c[:, 0]
    V_ang_pred = V_pred_c[:, 1]
    V_mag_true = V_true_c[:, 0]
    V_ang_true = V_true_c[:, 1]

    rmse_vmag = float(((V_mag_pred - V_mag_true) ** 2).mean().sqrt())
    rmse_vang = float(((V_ang_pred - V_ang_true) ** 2).mean().sqrt())

    # ΔS: power-balance residual  S_calc = V * conj(Y * V)
    if Y_bus is not None:
        try:
            Y = Y_bus.detach().cpu()
            if Y.dim() == 3:
                Y = Y.squeeze(0)
            Y = Y[:N, :N]

            V_complex = V_mag_pred * torch.exp(1j * V_ang_pred)
            I_complex = Y.to(torch.complex64) @ V_complex.to(torch.complex64)
            S_calc = V_complex * I_complex.conj()

            S_inj = S_start.detach().cpu()
            while S_inj.dim() > 1:
                S_inj = S_inj.squeeze(0)
            S_inj = S_inj[:N].to(torch.complex64)

            delta_s = float((S_calc - S_inj).abs().mean())
        except Exception as e:
            print(f"    ΔS computation error: {e}")
            delta_s = float("nan")
    else:
        delta_s = float("nan")

    return rmse_vmag, rmse_vang, delta_s


def run_single(seed, device):
    set_seed(seed)

    ds = ChanghunDataset([DS_PATH], per_unit=True, device=device)
    n_total = len(ds)
    n_train = min(N_TRAIN, n_total)

    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if val_idx else None
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_blockdiag)

    model = ContractiveGCN_PF(n_bus_features=5, hidden=HIDDEN).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=N_EPOCHS)

    best_val_loss = float("inf")
    best_state = None

    t_start = time.time()
    for ep in range(N_EPOCHS):
        model.train()
        for batch in train_loader:
            V_pred, _ = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
            loss = ((V_pred - batch["V_newton"].to(device)) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()

        if val_ds and (ep + 1) % 20 == 0:
            model.eval()
            val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_blockdiag)
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    V_pred, _ = model(
                        batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                        None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                        batch["S_start"].to(device), batch["V_start"].to(device),
                        batch["sizes"].to(device),
                    )
                    val_losses.append(float(((V_pred - batch["V_newton"].to(device)) ** 2).mean()))
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    t_train = time.time() - t_start

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Evaluate on first sample
    eval_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)
    eval_batch = next(iter(eval_loader))

    with torch.no_grad():
        V_pred, ctx_pf = model(
            eval_batch["bus_type"].to(device), eval_batch["Lines_connected"].to(device),
            None, eval_batch["Y_Lines"].to(device), eval_batch["Y_C_Lines"].to(device),
            eval_batch["S_start"].to(device), eval_batch["V_start"].to(device),
            eval_batch["sizes"].to(device),
        )

    N = int(eval_batch["sizes"][0].item())
    # Squeeze batch dim: [1, N, 2] → [N, 2]
    V_pred_flat = V_pred.squeeze(0)[:N]
    V_true_flat = eval_batch["V_newton"].to(device).squeeze(0)[:N]

    Y_bus = ctx_pf.get("Y")
    S_start = eval_batch["S_start"].to(device).squeeze(0)[:N]

    rmse_v, rmse_theta, delta_s = compute_physics_metrics(V_pred_flat, V_true_flat, Y_bus, S_start)

    # AEGIS analysis on subgraph
    Z_star = ctx_pf["Z_star"]
    A_hat = ctx_pf["A_hat"]

    if N > SUBGRAPH_SIZE:
        idx = extract_ego_subgraph(A_hat[:N, :N], max_nodes=SUBGRAPH_SIZE)
        A_sub = A_hat[:N, :N][idx][:, idx]
        X_proj_sub = ctx_pf["X_proj"][:N][idx]
        Z_sub = Z_star[:N][idx]
    else:
        idx = list(range(N))
        A_sub = A_hat[:N, :N]
        X_proj_sub = ctx_pf["X_proj"][:N]
        Z_sub = Z_star[:N]

    ctx_sub = {"A_hat": A_sub, "X_proj": X_proj_sub}
    Z_sub = reconverge(model, Z_sub, ctx_sub)

    # S_c + tau
    try:
        J_z, J_A, _ = _compute_structural_jacobian(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub
        )
        S = structural_sensitivity_matrix(
            lambda z, c: model.operator(z, c), Z_sub, ctx_sub, J_z=J_z, J_A=J_A
        )
        S_c, edge_list = constrained_sensitivity_matrix(S, A_sub)

        attack = optimal_structural_attack(S, A_sub, epsilon=0.01)
        aegis_ranking = attack["all_edge_vulnerabilities"]

        bf_ranking = greedy_structural_attack(model, Z_sub, ctx_sub)

        aegis_scores = [v for _, _, v in aegis_ranking]
        bf_scores_dict = {(min(i, j), max(i, j)): s for i, j, s in bf_ranking}
        bf_matched = [bf_scores_dict.get((min(i, j), max(i, j)), 0.0) for i, j, _ in aegis_ranking]

        tau, _ = kendalltau(aegis_scores, bf_matched)

        k10 = min(10, len(aegis_ranking))
        gt_top = set((min(i, j), max(i, j)) for i, j, _ in bf_ranking[:k10])
        ae_top = set((min(i, j), max(i, j)) for i, j, _ in aegis_ranking[:k10])
        p10 = len(gt_top & ae_top) / k10
    except Exception as e:
        print(f"    AEGIS error: {e}")
        tau, p10 = float("nan"), float("nan")

    return {
        "seed": seed,
        "rmse_vmag": rmse_v,
        "rmse_theta": rmse_theta,
        "delta_s": delta_s,
        "tau": tau,
        "p10": p10,
        "t_train": t_train,
        "n_train": n_train,
        "n_sub": len(idx),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training case300: {N_TRAIN} samples, {N_EPOCHS} epochs, hidden={HIDDEN}")
    print()

    results = []
    for seed_idx, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})", end=" ... ", flush=True)
        r = run_single(seed, device)
        results.append(r)
        print(f"|V|={r['rmse_vmag']:.4f} θ={r['rmse_theta']:.4f} ΔS={r['delta_s']:.3f} "
              f"τ={r['tau']:+.3f} P@10={r['p10']:.2f} ({r['t_train']:.0f}s)")
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("CASE300 RETRAINED — SUMMARY (10 seeds)")
    print("=" * 80)

    def fmt(key):
        vals = [r[key] for r in results if not np.isnan(r[key])]
        if not vals:
            return "N/A"
        return f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"

    print(f"  |V| RMSE:  {fmt('rmse_vmag')}   (old: 0.031)")
    print(f"  θ RMSE:    {fmt('rmse_theta')}   (old: 0.394)")
    print(f"  ΔS:        {fmt('delta_s')}   (old: 10.9)")
    print(f"  τ:         {fmt('tau')}   (old: +0.72)")
    print(f"  P@10:      {fmt('p10')}   (old: 0.87)")
    print(f"  Training:  {N_TRAIN} samples, {N_EPOCHS} epochs")


if __name__ == "__main__":
    sys.exit(main() or 0)
