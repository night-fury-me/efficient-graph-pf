"""Run the trained model on a real LVN batch, instrument the K-iteration to
capture the actual (dv, dth) values produced per iteration, and verify
whether they're being clamped, near-zero, or just unhelpfully distributed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from data_loading.dataset import ChanghunDataset
from data_loading.collate import collate_blockdiag
from torch.utils.data import DataLoader
from models.edge_selfattn.model import GNSMsg_EdgeSelfAttn

CKPT = Path('results/runs/260523-073214_0e54/ckpt/best.ckpt')


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Build model matching diag A config: d=4, d_hi=16, K=10, VnFeat
    model = GNSMsg_EdgeSelfAttn(
        d=4, d_hi=16, K=10, num_attn_layers=1,
        pinn=True, gamma=0.9, v_limit=True, use_armijo=True,
        dtheta_max=0.30, dvm_frac=0.10,
        bus_feat_extra_dim=1,  # VnFeat
    )
    model.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=False))
    model.to(device).eval()
    print(f'Model loaded, {sum(p.numel() for p in model.parameters()):,} params')

    # Load one batch
    ds = ChanghunDataset(['./datasets/LVN_converted_n36000.parquet'], per_unit=True, device=device)
    print(f'Dataset: {len(ds)} rows')
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_blockdiag)
    batch = next(iter(loader))

    bus_type = batch['bus_type'].to(device)
    Line = batch['Lines_connected'].to(device)
    Ys = batch['Y_Lines'].to(device)
    Yc = batch['Y_C_Lines'].to(device)
    Sstart = batch['S_start'].to(device)
    Vstart = batch['V_start'].to(device)
    Vnewton = batch['V_newton'].to(device)
    sizes = batch['sizes'].to(device)
    vn_log = batch['vn_log'].to(device)

    print(f'Batch shapes: bus_type {bus_type.shape}, V_start {Vstart.shape}, V_newton {Vnewton.shape}')

    # --- Monkey-patch the model to record per-iteration deltas ---
    captured = {'dv': [], 'dth': [], 'dv_clamped': [], 'dth_clamped': []}
    orig_constraints = model._apply_constraints

    def spy_constraints(*, v, dth, dv, slack_mask, pv_mask):
        captured['dv'].append(dv.detach().clone())
        captured['dth'].append(dth.detach().clone())
        dth_c, dv_c = orig_constraints(v=v, dth=dth, dv=dv, slack_mask=slack_mask, pv_mask=pv_mask)
        captured['dv_clamped'].append(dv_c.detach().clone())
        captured['dth_clamped'].append(dth_c.detach().clone())
        return dth_c, dv_c
    model._apply_constraints = spy_constraints

    with torch.no_grad():
        Vpred, _ = model(bus_type, Line, None, Ys, Yc, Sstart, Vstart, sizes, vn_log=vn_log)

    print(f'\nVpred shape: {Vpred.shape}  Vnewton shape: {Vnewton.shape}')
    print()
    print('=== Per-iteration dv stats (BEFORE clamp) ===')
    print(f'{"k":>3} {"dv_mean":>12} {"dv_std":>12} {"dv_abs_max":>12} {"dv_clamped_pct":>15}')
    for k in range(10):
        dv = captured['dv'][k]
        dv_c = captured['dv_clamped'][k]
        clamped_pct = float((dv != dv_c).float().mean().item()) * 100
        print(f'{k:>3} {dv.mean().item():+12.4e} {dv.std().item():12.4e} '
              f'{dv.abs().max().item():12.4e} {clamped_pct:>14.2f}%')

    print()
    print('=== Per-iteration dth stats (BEFORE clamp) ===')
    print(f'{"k":>3} {"dth_mean":>12} {"dth_std":>12} {"dth_abs_max":>12} {"dth_clamped_pct":>15}')
    for k in range(10):
        dth = captured['dth'][k]
        dth_c = captured['dth_clamped'][k]
        clamped_pct = float((dth != dth_c).float().mean().item()) * 100
        print(f'{k:>3} {dth.mean().item():+12.4e} {dth.std().item():12.4e} '
              f'{dth.abs().max().item():12.4e} {clamped_pct:>14.2f}%')

    print()
    print('=== Predicted V_pred (after K iterations) vs V_start vs V_newton ===')
    # Vpred shape: (B, N, 2) where last dim is (v, th)
    # Vstart shape: (B, N, 2)
    v_pred = Vpred[..., 0]
    th_pred = Vpred[..., 1]
    v_start = Vstart[..., 0]
    th_start = Vstart[..., 1]
    v_newton = Vnewton[..., 0]
    th_newton = Vnewton[..., 1]

    print(f'v_pred  : mean={v_pred.mean().item():.4f}  std={v_pred.std().item():.4f}')
    print(f'v_start : mean={v_start.mean().item():.4f}  std={v_start.std().item():.4f}')
    print(f'v_newton: mean={v_newton.mean().item():.4f}  std={v_newton.std().item():.4f}')
    print()
    print(f'(v_pred - v_start): mean={(v_pred - v_start).mean().item():+.4e}  '
          f'std={(v_pred - v_start).std().item():.4e}  '
          f'abs_max={(v_pred - v_start).abs().max().item():.4e}')
    print(f'(v_pred - v_newton): mean={(v_pred - v_newton).mean().item():+.4e}  '
          f'std={(v_pred - v_newton).std().item():.4e}  '
          f'abs_max={(v_pred - v_newton).abs().max().item():.4e}')
    print(f'(v_start - v_newton): mean={(v_start - v_newton).mean().item():+.4e}  '
          f'std={(v_start - v_newton).std().item():.4e}  '
          f'abs_max={(v_start - v_newton).abs().max().item():.4e}')

    print()
    print('=== INTERPRETATION ===')
    delta_pred = (v_pred - v_start).abs().mean().item()
    delta_needed = (v_newton - v_start).abs().mean().item()
    ratio = delta_pred / delta_needed if delta_needed > 0 else 0.0
    print(f'  Mean |v_pred - v_start|  : {delta_pred:.4e}  (what model moves)')
    print(f'  Mean |v_newton - v_start|: {delta_needed:.4e}  (what model needs to move)')
    print(f'  Ratio = {ratio:.3f}  (1.0 = model moves enough; 0.0 = model frozen at V_start)')

    return 0


if __name__ == '__main__':
    sys.exit(main())
