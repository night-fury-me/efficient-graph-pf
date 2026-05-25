"""Sanity-check the proposed fix BEFORE editing model.py:

  dth = dtheta_max * tanh(theta_head(x))
  dv  = dvm_frac   * tanh(v_head(x))

Monkey-patch a fresh model's forward to use this bounded-head formulation,
do one forward+backward pass on a real batch, and verify:
  1. dv and dth remain inside their bounds (no clamp ever triggers)
  2. Gradient flows to all heads (non-zero head weight gradients)
  3. The final V_pred actually differs from V_start non-trivially
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from data_loading.dataset import ChanghunDataset
from data_loading.collate import collate_blockdiag
from torch.utils.data import DataLoader
from models.edge_selfattn.model import GNSMsg_EdgeSelfAttn


def patched_forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph, *, vn_log=None, **_unused):
    """Same as original forward but with bounded heads + no _apply_constraints clamp.
    Slack/PV masks are still applied."""
    from models.edge_selfattn.admittance import (
        build_dense_Y, build_edges_blockdiag, build_edges_plain,
    )
    import math

    device = bus_type.device
    B, N = bus_type.shape

    P_set, Q_set = S.real, S.imag
    v = V0[..., 0].clone()
    th = V0[..., 1].clone()
    m = v.new_zeros(B, N, self.d)

    edge_index_dir = None
    edge_feat_dir = None
    edge_index_dir_list = None
    edge_feat_dir_list = None

    if n_nodes_per_graph is not None:
        Line_1d = Line.squeeze(0) if Line.dim() == 2 else Line
        Ys_1d = Ys.squeeze(0)
        Yc_1d = Yc.squeeze(0)
        undirected, _, edge_index_dir, edge_feat_dir, ys_edge, yc_edge = build_edges_blockdiag(
            line_mask_1d=Line_1d, Ys_1d=Ys_1d, Yc_1d=Yc_1d,
            n_nodes_per_graph=n_nodes_per_graph,
            edge_feat_dim=self.edge_feat_dim,
            pairs_for_n=self._pairs_for_n,
            device=device,
        )
        if Y is None:
            Y = build_dense_Y(N, undirected, ys_edge, yc_edge, device=device)
    else:
        pairs = self._pairs_for_n(N, device)
        edge_index_dir_list, edge_feat_dir_list, undirected_list, mask_list = build_edges_plain(
            Line=Line, Ys=Ys, Yc=Yc, N=N,
            edge_feat_dim=self.edge_feat_dim, pairs=pairs, device=device,
        )

    slack_mask = bus_type == 1
    pv_mask = bus_type == 2

    captured = {'dv_raw': [], 'dth_raw': []}

    for k in range(self.K):
        Vc = v * torch.exp(1j * th)
        Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)
        Sc = Vc * Ic.conj()
        DP = (P_set - Sc.real)
        DQ = (Q_set - Sc.imag)
        DP = DP.masked_fill(slack_mask, 0.0)
        DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

        bus_feat = torch.stack([v, th, DP, DQ], dim=-1)
        if self.bus_feat_extra_dim > 0:
            if vn_log is None:
                extra = bus_feat.new_zeros(bus_feat.shape[:-1] + (self.bus_feat_extra_dim,))
            else:
                extra = vn_log.unsqueeze(-1)
            bus_feat = torch.cat([bus_feat, extra], dim=-1)
        x = self.in_proj(torch.cat([bus_feat, m], dim=-1))

        if n_nodes_per_graph is not None:
            x = self._apply_blocks(x, edge_index_dir, edge_feat_dir)

        head_idx = 0 if self.tied_heads else k
        # === BOUNDED HEADS (the proposed fix) ===
        dth = self.dtheta_max * torch.tanh(self.theta_head[head_idx](x).squeeze(-1))
        dv = self.dvm_frac * torch.tanh(self.v_head[head_idx](x).squeeze(-1))
        dm = torch.tanh(self.m_head[head_idx](x))
        dm = F.layer_norm(dm, dm.shape[-1:])

        # Slack/PV masks (still needed: those buses must NOT move)
        dth = dth.masked_fill(slack_mask, 0.0)
        dv = dv.masked_fill(slack_mask | pv_mask, 0.0)

        captured['dv_raw'].append(dv.detach().clone())
        captured['dth_raw'].append(dth.detach().clone())

        # Direct update — no clamp!
        th = (th + dth + math.pi) % (2 * math.pi) - math.pi
        v = v + dv  # NO clamp — tanh keeps it bounded per step
        m = m + dm

    out = torch.stack([v, th], dim=-1)
    return out, captured


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Fresh model — same config as diag A
    model = GNSMsg_EdgeSelfAttn(
        d=4, d_hi=16, K=10, num_attn_layers=1,
        pinn=False, gamma=0.9, v_limit=True, use_armijo=False,
        dtheta_max=0.30, dvm_frac=0.10,
        bus_feat_extra_dim=1,
    )
    model.to(device).eval()
    print(f'Fresh model, {sum(p.numel() for p in model.parameters()):,} params')

    ds = ChanghunDataset(['./datasets/LVN_converted_n36000.parquet'], per_unit=True, device=device)
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

    print(f'\n=== Sanity check 1: forward pass with FRESH ZERO-INIT heads ===')
    print('(Output heads are zero-init, so all dv/dth should be exactly 0)')
    with torch.no_grad():
        Vpred, captured = patched_forward(model, bus_type, Line, None, Ys, Yc, Sstart, Vstart, sizes, vn_log=vn_log)
    for k in range(10):
        dv = captured['dv_raw'][k]
        dth = captured['dth_raw'][k]
        print(f'  k={k} dv:[{dv.abs().max().item():.4e}, in_bound={dv.abs().max().item() <= 0.10}]  '
              f'dth:[{dth.abs().max().item():.4e}, in_bound={dth.abs().max().item() <= 0.30}]')

    print()
    print(f'=== Sanity check 2: forward with RANDOM heads (worst case) ===')
    print('(Re-init heads with large random values; tanh must still bound the outputs)')
    with torch.no_grad():
        for j in range(10):
            torch.nn.init.normal_(model.theta_head[j].weight, std=10.0)
            torch.nn.init.normal_(model.theta_head[j].bias, std=10.0)
            torch.nn.init.normal_(model.v_head[j].weight, std=10.0)
            torch.nn.init.normal_(model.v_head[j].bias, std=10.0)
    with torch.no_grad():
        Vpred, captured = patched_forward(model, bus_type, Line, None, Ys, Yc, Sstart, Vstart, sizes, vn_log=vn_log)
    for k in range(10):
        dv = captured['dv_raw'][k]
        dth = captured['dth_raw'][k]
        all_bounded_v = dv.abs().max().item() <= 0.10 + 1e-6
        all_bounded_th = dth.abs().max().item() <= 0.30 + 1e-6
        print(f'  k={k} dv_max={dv.abs().max().item():.4f} (≤0.10? {all_bounded_v})  '
              f'dth_max={dth.abs().max().item():.4f} (≤0.30? {all_bounded_th})  '
              f'dv_mean={dv.mean().item():+.4f}  dth_mean={dth.mean().item():+.4f}')

    # Sanity check 3: gradient flow
    print()
    print(f'=== Sanity check 3: gradient flow on MSE loss ===')
    # Re-init to fresh state
    model = GNSMsg_EdgeSelfAttn(
        d=4, d_hi=16, K=10, num_attn_layers=1,
        pinn=False, gamma=0.9, v_limit=True, use_armijo=False,
        dtheta_max=0.30, dvm_frac=0.10,
        bus_feat_extra_dim=1,
    )
    model.to(device).train()
    Vpred, _ = patched_forward(model, bus_type, Line, None, Ys, Yc, Sstart, Vstart, sizes, vn_log=vn_log)
    loss = ((Vpred - Vnewton) ** 2).mean()
    print(f'  Loss = {loss.item():.4e}')
    loss.backward()
    head_grads_v = [model.v_head[k].weight.grad.norm().item() for k in range(10)]
    head_grads_th = [model.theta_head[k].weight.grad.norm().item() for k in range(10)]
    head_grad_v_bias = [model.v_head[k].bias.grad.norm().item() for k in range(10)]
    print(f'  v_head.weight gradient norms per K iter: {[f"{g:.3e}" for g in head_grads_v]}')
    print(f'  v_head.bias   gradient norms per K iter: {[f"{g:.3e}" for g in head_grad_v_bias]}')
    print(f'  th_head.weight gradient norms per K iter: {[f"{g:.3e}" for g in head_grads_th]}')

    all_v_nonzero = all(g > 1e-8 for g in head_grads_v)
    all_th_nonzero = all(g > 1e-8 for g in head_grads_th)
    all_bias_nonzero = all(g > 1e-8 for g in head_grad_v_bias)
    print()
    print('=== VERDICT ===')
    print(f'  All v_head WEIGHT gradients non-zero: {all_v_nonzero}')
    print(f'  All v_head BIAS gradients non-zero  : {all_bias_nonzero}  <-- KEY (biases were stuck at 0 in old run)')
    print(f'  All th_head WEIGHT gradients non-zero: {all_th_nonzero}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
