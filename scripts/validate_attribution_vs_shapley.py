"""Validate IFT attribution against exact Shapley on small graphs.

Computes both ift_attribution (O(n) gradient-based) and exact_shapley
(O(n·2^n) coalition enumeration) on small HVN graphs (n ≤ 16 buses),
then reports Spearman/Kendall rank correlation between them.

If correlation is high (ρ > 0.8), IFT attribution is a valid fast proxy.

Usage:
    .venv/bin/python scripts/validate_attribution_vs_shapley.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import models  # noqa
from data_loading.collate import collate_blockdiag
from data_loading.dataset import ChanghunDataset
from iem import IEMiner
from iem.ift import compute_jacobian, param_sensitivity
from iem.shapley import exact_shapley, ift_attribution
from models.pe_deq_pf.model import PE_DEQ_PF
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the contractive model (ρ < 1)
    print("=== Loading ContractiveGCN-PF + HVN ===", flush=True)
    from iem.examples.contractive_pf import ContractiveGCN_PF

    ds = ChanghunDataset(
        ["./datasets/HVN_stratified_1500.parquet"], per_unit=True, device=device
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_blockdiag)

    # Train quickly
    model = ContractiveGCN_PF(n_bus_features=5, hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    import torch.nn.functional as F_func

    print("  Training 50 epochs...", flush=True)
    for ep in range(50):
        model.train()
        for batch in loader:
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
    model.eval()
    print("  Done.", flush=True)

    # Validate on small graphs (n ≤ 16 for exact Shapley tractability)
    print("\n=== Validating IFT attribution vs exact Shapley ===", flush=True)
    print(f"{'Graph':>6} {'N':>3} {'Spearman':>10} {'Kendall':>10} {'IFT_top1':>10} {'Shap_top1':>10}", flush=True)

    spearman_vals = []
    kendall_vals = []
    top1_agree = 0
    n_tested = 0

    for g_idx, batch in enumerate(loader):
        N = int(batch["sizes"][0].item())
        if N > 16:
            continue
        if n_tested >= 20:
            break

        with torch.no_grad():
            V_pred, ctx = model(
                batch["bus_type"].to(device), batch["Lines_connected"].to(device),
                None, batch["Y_Lines"].to(device), batch["Y_C_Lines"].to(device),
                batch["S_start"].to(device), batch["V_start"].to(device),
                batch["sizes"].to(device),
            )
        Z_star = ctx["Z_star"]
        A_hat = ctx["A_hat"]
        X_proj = ctx["X_proj"]
        ctx_sub = {"A_hat": A_hat, "X_proj": X_proj}

        # IFT attribution
        ift_attr = param_sensitivity(
            lambda z, c: model.operator(z, c),
            Z_star, ctx_sub, "X_proj", method="direct",
        ).squeeze().detach().cpu()

        # For exact Shapley: define value function as ||V_pred - V_newton||²
        V_newton = batch["V_newton"].to(device)

        def value_fn(Z):
            v = model.v_head(Z).squeeze(-1) + batch["V_start"].to(device).reshape(N, 2)[:, 0]
            th = model.th_head(Z).squeeze(-1) + batch["V_start"].to(device).reshape(N, 2)[:, 1]
            V = torch.stack([v, th], dim=-1).unsqueeze(0)
            return -((V - V_newton) ** 2).mean()

        # Baseline: zero X_proj
        baseline = torch.zeros_like(Z_star)

        shap = exact_shapley(
            value_fn=value_fn,
            z_star=Z_star,
            baseline=baseline,
            player_dim=0,
            n_players=N,
        ).detach().cpu()

        # Rank correlation
        ift_np = ift_attr.numpy() if ift_attr.dim() <= 1 else ift_attr.norm(dim=-1).numpy()
        shap_np = shap.abs().numpy()

        if len(ift_np) >= 3:
            sp, _ = spearmanr(ift_np, shap_np)
            kt, _ = kendalltau(ift_np, shap_np)
            spearman_vals.append(sp)
            kendall_vals.append(kt)

            ift_top = int(ift_np.argmax())
            shap_top = int(shap_np.argmax())
            if ift_top == shap_top:
                top1_agree += 1

            print(f"{g_idx:>6} {N:>3} {sp:>+10.3f} {kt:>+10.3f} {ift_top:>10} {shap_top:>10}", flush=True)
            n_tested += 1

    print(f"\n{'='*60}", flush=True)
    if spearman_vals:
        print(f"Mean Spearman ρ:  {np.mean(spearman_vals):+.3f} ± {np.std(spearman_vals):.3f}")
        print(f"Mean Kendall τ:   {np.mean(kendall_vals):+.3f} ± {np.std(kendall_vals):.3f}")
        print(f"Top-1 agreement:  {top1_agree}/{n_tested} = {top1_agree/n_tested:.0%}")
        print(f"Graphs tested:    {n_tested}")
        print()
        if np.mean(spearman_vals) > 0.8:
            print("VERDICT: IFT attribution is a VALID fast proxy for Shapley (ρ > 0.8)")
        elif np.mean(spearman_vals) > 0.5:
            print("VERDICT: IFT attribution MODERATELY correlates with Shapley (0.5 < ρ < 0.8)")
        else:
            print("VERDICT: IFT attribution WEAKLY correlates with Shapley (ρ < 0.5)")


if __name__ == "__main__":
    sys.exit(main() or 0)
