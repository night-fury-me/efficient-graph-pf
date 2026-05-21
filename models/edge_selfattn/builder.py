"""Builder for GNSMsg_EdgeSelfAttn.

Importing this module decorates `_build` with `@register_model`, which
self-registers the model in `models.registry.MODEL_REGISTRY`. Because
`models/edge_selfattn/__init__.py` imports this file, simply importing the
`edge_selfattn` package is enough to make the model buildable by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import register_model

from .model import GNSMsg_EdgeSelfAttn


@register_model("GNSMsg_EdgeSelfAttn")
def _build(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    gamma: float,
    v_limit: bool,
    use_armijo: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    **_unused,
) -> nn.Module:
    return GNSMsg_EdgeSelfAttn(
        d=d,
        d_hi=d_hi,
        K=K,
        pinn=pinn,
        gamma=gamma,
        v_limit=v_limit,
        use_armijo=use_armijo,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        num_attn_layers=num_attn_layers,
    ).to(device)
