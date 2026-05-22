"""Builder for PE_DEQ_PF.

Importing this module decorates `_build` with `@register_model`, which
self-registers the model in `models.registry.MODEL_REGISTRY`. Because
`models/pe_deq_pf/__init__.py` imports this file, simply importing the
`pe_deq_pf` package is enough to make the model buildable by name.

Maps the shared config surface (d, d_hi, K, ...) used by the existing
GNSMsg_EdgeSelfAttn builder so the same YAML configs can be reused. The
options `gamma`, `v_limit`, and `use_armijo` are accepted but ignored:
DEQ replaces both the per-iteration loss discount and the line-search.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import register_model

from .model import PE_DEQ_PF


@register_model("PE_DEQ_PF")
def _build(
    *,
    d: int,
    d_hi: int,
    K: int,
    pinn: bool,
    dtheta_max: float,
    dvm_frac: float,
    num_attn_layers: int,
    device: torch.device,
    **_unused,
) -> nn.Module:
    return PE_DEQ_PF(
        d=d,
        d_hi=d_hi,
        num_attn_layers=num_attn_layers,
        pinn=pinn,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        forward_iter=K,
        backward_iter=K,
    ).to(device)
