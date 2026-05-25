"""Per-graph descriptor extractor for HyperDEQ-PF pilot.

Computes a 12-dim feature vector per graph from the batch tensors that the
training loop already provides (no schema change needed).

Features (all log10 of |x| + 1e-12 where applicable, except ratios):
  [0]  log10(N_buses)
  [1]  log10(num_active_edges)
  [2]  log10(mean degree)
  [3]  log10(mean |S_start| over PQ buses)
  [4]  log10(std  |S_start| over PQ buses)
  [5]  log10(max  |S_start| over PQ buses)
  [6]  log10(mean |V_start|)             -- per-unit, should be ~0
  [7]  log10(std  |V_start|)
  [8]  log10(mean |Y_Lines| over active edges)
  [9]  log10(std  |Y_Lines|)
  [10] num_slack / N
  [11] num_PV / N

The voltage-class signal (HVN vs MVN) shows up most strongly in features
[3-5] and [8-9] because the per-unit S/Y normalisations applied during
dataset creation differ across voltage classes.
"""

from __future__ import annotations

import torch

DESCRIPTOR_DIM = 12
_EPS = 1e-12


def _safe_log10(x: torch.Tensor) -> torch.Tensor:
    return torch.log10(x.abs() + _EPS)


@torch.no_grad()
def per_graph_descriptor(
    *,
    bus_type: torch.Tensor,
    Line: torch.Tensor,
    Ys: torch.Tensor,
    S_start: torch.Tensor,
    V_start: torch.Tensor,
    n_nodes_per_graph: torch.Tensor | None,
) -> torch.Tensor:
    """Returns descriptors of shape (G, DESCRIPTOR_DIM) where G = #graphs in batch.

    Block-diagonal layout: bus_type has shape (1, M=sum(N_i)) and
    n_nodes_per_graph = [N_1, N_2, ...]. Per-graph slicing extracts each
    sub-graph's features.

    Plain layout: bus_type (B, N), all graphs have same N.
    """
    block_diag = n_nodes_per_graph is not None
    if block_diag:
        # bus_type (1, M); Line / Ys are (1, P_total) over all canonical pairs
        sizes = n_nodes_per_graph
        G = int(sizes.numel())
        descriptors = []
        node_offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
        # Pair-offset for canonical pair indices per graph: P_i = N_i*(N_i-1)/2
        pair_sizes = (sizes * (sizes - 1) // 2)
        pair_offsets = torch.cat([pair_sizes.new_zeros(1), pair_sizes.cumsum(0)])
        bt = bus_type.squeeze(0)
        S_flat = S_start.squeeze(0)
        V_flat = V_start.squeeze(0)
        Line_flat = Line.squeeze(0) if Line.dim() == 2 else Line
        Ys_flat = Ys.squeeze(0)
        for g in range(G):
            n0, n1 = int(node_offsets[g]), int(node_offsets[g + 1])
            p0, p1 = int(pair_offsets[g]), int(pair_offsets[g + 1])
            descriptors.append(
                _descriptor_one(
                    bt[n0:n1],
                    Line_flat[p0:p1].to(torch.bool),
                    Ys_flat[p0:p1],
                    S_flat[n0:n1],
                    V_flat[n0:n1],
                )
            )
        return torch.stack(descriptors, dim=0)
    else:
        B, N = bus_type.shape
        descriptors = []
        for b in range(B):
            descriptors.append(
                _descriptor_one(
                    bus_type[b],
                    Line[b].to(torch.bool),
                    Ys[b],
                    S_start[b],
                    V_start[b],
                )
            )
        return torch.stack(descriptors, dim=0)


def _descriptor_one(
    bt: torch.Tensor,   # (N,)
    line_mask: torch.Tensor,  # (P,) bool over canonical pairs
    Ys: torch.Tensor,   # (P,) complex
    S: torch.Tensor,    # (N,) complex
    V: torch.Tensor,    # (N, 2) [v, theta]
) -> torch.Tensor:
    """Build the 12-dim descriptor for one graph."""
    N = bt.numel()
    n_edges = int(line_mask.sum())
    mean_deg = (2 * n_edges) / max(N, 1)

    # |S| over PQ buses (bus_type == 3 in HVN/MVN convention)
    pq_mask = bt == 3
    if pq_mask.any():
        S_pq = S[pq_mask].abs()
        s_mean = S_pq.mean()
        s_std = S_pq.std() if S_pq.numel() > 1 else S_pq.new_tensor(_EPS)
        s_max = S_pq.max()
    else:
        s_mean = s_std = s_max = S.new_tensor(_EPS, dtype=torch.float32).abs()

    # |V_start| — PU should be near 1.0
    v_abs = V[..., 0].abs()
    v_mean = v_abs.mean()
    v_std = v_abs.std() if v_abs.numel() > 1 else v_abs.new_tensor(_EPS)

    # |Y_Lines| over active edges
    if n_edges > 0:
        Ys_active = Ys[line_mask].abs()
        y_mean = Ys_active.mean()
        y_std = Ys_active.std() if Ys_active.numel() > 1 else Ys_active.new_tensor(_EPS)
    else:
        y_mean = y_std = Ys.new_tensor(_EPS).abs()

    n_slack = float((bt == 1).sum().item())
    n_pv = float((bt == 2).sum().item())

    feats = torch.tensor(
        [
            float(torch.log10(torch.tensor(float(N)))),
            float(torch.log10(torch.tensor(float(max(n_edges, 1))))),
            float(torch.log10(torch.tensor(float(max(mean_deg, _EPS))))),
            float(_safe_log10(s_mean)),
            float(_safe_log10(s_std)),
            float(_safe_log10(s_max)),
            float(_safe_log10(v_mean)),
            float(_safe_log10(v_std)),
            float(_safe_log10(y_mean)),
            float(_safe_log10(y_std)),
            n_slack / max(N, 1),
            n_pv / max(N, 1),
        ],
        dtype=torch.float32,
        device=bt.device,
    )
    return feats
