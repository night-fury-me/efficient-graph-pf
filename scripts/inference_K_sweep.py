"""Inference-time K sweep on a trained PE_DEQ_PF checkpoint.

Verifies the architectural claim: K is decoupled from training, the solver
exits early on residual tolerance, and increasing K at inference is free
(except for runtime) -- no retraining needed.
"""
from __future__ import annotations

import math
import sys

import torch

import models  # noqa: F401
from models.registry import build_model
from train.data import build_dataloaders


@torch.no_grad()
def eval_with_K(ckpt: str, K: int, val_loader, device, model_name: str = 'PE_DEQ_PF') -> dict:
    m = build_model(model_name, device=device, d=4, d_hi=32, K=K,
                    num_attn_layers=2, dtheta_max=0.30, dvm_frac=0.10,
                    gamma=0.9, v_limit=True, use_armijo=True, pinn=True)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False), strict=True)
    m.pinn = True
    m.eval()
    m.deq.solver_kwargs['max_iter'] = max(K, 2)

    sp = ss = sm = sa = n = 0
    fwd_iters = []
    for b in val_loader:
        out = m(b['bus_type'].to(device), b['Lines_connected'].to(device), None,
                b['Y_Lines'].to(device), b['Y_C_Lines'].to(device),
                b['S_start'].to(device), b['V_start'].to(device), b['sizes'].to(device))
        Vp, phys = out if isinstance(out, tuple) else (out, None)
        Vt = b['V_newton'].to(device); B = b['bus_type'].size(0)
        sp += float(phys.item())*B
        diff = Vp - Vt
        ss += float((diff*diff).mean().item())*B
        sm += float((diff[..., 0]**2).mean().item())*B
        sa += float((diff[..., 1]**2).mean().item())*B
        n += B
        fwd_iters.append(len(m.deq.forward_res))

    return dict(K=K,
                avg_iters_used=sum(fwd_iters)/len(fwd_iters),
                val_phys=sp/n,
                val_rmse=(ss/n)**0.5,
                val_rmse_mag=(sm/n)**0.5,
                val_rmse_ang_deg=(sa/n)**0.5 * 180 / math.pi)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = build_dataloaders(
        parquet_paths=['./datasets/HVN_15000_NR_plain_4_to_32_buses.parquet'],
        per_unit=True, device=device, batch_size=64, block_diag=True,
        seed=42, split_mode='ratio', train_ratio=0.8, valid_ratio=0.1,
    )
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        './results/runs/260522-203831_9d04/ckpt/best.ckpt'
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'PE_DEQ_PF'

    print(f"checkpoint: {ckpt}")
    print(f"model: {model_name}")
    print(f"{'K':>4} | {'iters used':>10} | {'val_phys':>12} | {'val_rmse':>12} | {'val_mag':>12} | {'val_ang°':>10}")
    print('-' * 84)
    for K in [5, 10, 15, 30, 50, 100]:
        r = eval_with_K(ckpt, K, splits.val_loader, device, model_name)
        print(f"{K:>4} | {r['avg_iters_used']:>10.1f} | {r['val_phys']:>12.4e} | "
              f"{r['val_rmse']:>12.4e} | {r['val_rmse_mag']:>12.4e} | {r['val_rmse_ang_deg']:>10.4f}")


if __name__ == '__main__':
    main()
