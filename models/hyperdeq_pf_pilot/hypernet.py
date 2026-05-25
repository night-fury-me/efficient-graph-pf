"""Multi-point FiLM hypernet for HyperDEQ-PF.

v2 (this file): produces FiLM (gamma, beta) at (num_attn_layers + 1)
modulation points: once after in_proj, and once after each
EdgeSelfAttnBlock output. This keeps the per-graph conditioning alive
through the attention blocks so that the theta/v/m heads see a
voltage-class-aware representation -- v1 only modulated in_proj, which
caused angle collapse on cross-voltage transfer (HVN-trained -> MVN
zero-shot rmse 10x worse).

Total output dim: (num_attn_layers + 1) * 2 * d_model.
For pilot defaults (d_model=32, num_attn_layers=2): 3 * 2 * 32 = 192.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .descriptor import DESCRIPTOR_DIM


class FiLMHypernet(nn.Module):
    """Maps graph descriptor g -> per-layer FiLM (gamma, beta) pairs.

    Output is a list of (gamma_i, beta_i) tuples of length n_points =
    num_attn_layers + 1. Each gamma_i, beta_i has shape (G, d_model).
    gamma_i is initialised to ~1 + delta (residual around identity) and
    beta_i to ~0 -- so the model starts identical to plain PE_DEQ_PF.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_points: int,
        hidden: int = 128,
        descriptor_dim: int = DESCRIPTOR_DIM,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_points = int(n_points)
        self.descriptor_dim = int(descriptor_dim)

        self.trunk = nn.Sequential(
            nn.Linear(self.descriptor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # One (gamma, beta) head per modulation point. Zero init so model
        # starts as plain PE_DEQ_PF (gamma=1+0, beta=0).
        self.gamma_heads = nn.ModuleList(
            [nn.Linear(hidden, self.d_model) for _ in range(self.n_points)]
        )
        self.beta_heads = nn.ModuleList(
            [nn.Linear(hidden, self.d_model) for _ in range(self.n_points)]
        )
        for h in list(self.gamma_heads) + list(self.beta_heads):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, g: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """g: (G, descriptor_dim) -> list of n_points (gamma, beta) tuples."""
        h = self.trunk(g)
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for gh, bh in zip(self.gamma_heads, self.beta_heads):
            out.append((1.0 + gh(h), bh(h)))
        return out
