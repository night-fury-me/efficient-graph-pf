"""Diagnose: did training move the output heads off their zero initialization?

If the per-head weight L2 norms are still essentially zero (< 1e-3) after
3 epochs of MSE training, then the gradient signal isn't reaching the heads
(or is being killed by the chained K-step backprop). If they moved
substantially (> 1e-2) but the model still outputs V_start, then there's
something downstream (clamps, constraints) suppressing the head deltas.

Run: .venv/bin/python scripts/inspect_trained_heads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


CKPT = Path('results/runs/260523-073214_0e54/ckpt/best.ckpt')  # Diag A, 3 epochs MSE


def main() -> int:
    print(f'Loading {CKPT}...', flush=True)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'  {len(sd)} tensors\n', flush=True)

    # --- Output head weight norms per K iteration ---
    print('=== Output head weight L2 norms (per K iteration) ===')
    print(f'{"k":>3} {"theta_w":>12} {"theta_b":>12} {"v_w":>12} {"v_b":>12} {"m_w":>12} {"m_b":>12}')
    for k in range(10):
        tw = sd[f'theta_head.{k}.weight'].norm().item()
        tb = sd[f'theta_head.{k}.bias'].norm().item()
        vw = sd[f'v_head.{k}.weight'].norm().item()
        vb = sd[f'v_head.{k}.bias'].norm().item()
        mw = sd[f'm_head.{k}.weight'].norm().item()
        mb = sd[f'm_head.{k}.bias'].norm().item()
        print(f'{k:>3} {tw:>12.4e} {tb:>12.4e} {vw:>12.4e} {vb:>12.4e} {mw:>12.4e} {mb:>12.4e}')

    print()
    print('=== Reference: attention block & in_proj norms (should be ~O(1)) ===')
    print(f'  in_proj.weight       : {sd["in_proj.weight"].norm().item():.4e}')
    print(f'  blocks.0.q.weight    : {sd["blocks.0.q.weight"].norm().item():.4e}')
    print(f'  blocks.0.k.weight    : {sd["blocks.0.k.weight"].norm().item():.4e}')
    print(f'  blocks.0.v.weight    : {sd["blocks.0.v.weight"].norm().item():.4e}')
    print(f'  blocks.0.out.weight  : {sd["blocks.0.out.weight"].norm().item():.4e}')
    print(f'  blocks.0.ffn.0.weight: {sd["blocks.0.ffn.0.weight"].norm().item():.4e}')

    # --- Compare with fresh-init model ---
    print()
    print('=== Compare against FRESH zero-init reference ===')
    # All theta/v/m heads init'd to zero, so their norm should be 0 at init.
    # Print delta = (trained norm) for each head — that IS the delta from init.
    th_total = sum(sd[f'theta_head.{k}.weight'].norm().item() for k in range(10))
    v_total = sum(sd[f'v_head.{k}.weight'].norm().item() for k in range(10))
    m_total = sum(sd[f'm_head.{k}.weight'].norm().item() for k in range(10))
    print(f'  Sum of theta_head weight norms across K=10: {th_total:.4e}')
    print(f'  Sum of v_head   weight norms across K=10  : {v_total:.4e}')
    print(f'  Sum of m_head   weight norms across K=10  : {m_total:.4e}')

    print()
    print('=== INTERPRETATION ===')
    if th_total < 1e-3 and v_total < 1e-3:
        print('  ==> Heads are still essentially ZERO. Gradient is not reaching them.')
        print('      Root cause CONFIRMED: zero-init traps the model at V_start.')
        print('      Fix: replace zero init with small Xavier (gain=0.01).')
    elif th_total > 1e-2 or v_total > 1e-2:
        print('  ==> Heads moved off zero significantly. Init is NOT the killer.')
        print('      The downstream constraints/clamps must be suppressing the deltas.')
        print('      Inspect _apply_constraints and v_min/v_max behavior.')
    else:
        print(f'  ==> Heads moved slightly (theta_total={th_total:.4e}, v_total={v_total:.4e}).')
        print('      Gradient flows but is weak. Either:')
        print('      (a) Xavier init would unlock faster convergence, OR')
        print('      (b) The optimum the heads find still has dv≈0 on average.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
