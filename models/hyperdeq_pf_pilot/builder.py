"""Builder registration for HyperDEQ_PF_Pilot."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.registry import register_model

from .model import HyperDEQ_PF_Pilot


@register_model("HyperDEQ_PF_Pilot")
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
    """HyperDEQ-PF pilot: PE_DEQ_PF + FiLM-conditioning on in_proj.

    Same defaults as PE_DEQ_PF (plain). Adds a FiLMHypernet (~few hundred
    extra params) that conditions on a 12-dim graph descriptor.

    Recommended: train with pure MSE (no --PINN) so the cross-voltage
    signal isn't muddled by the LVN-style adversarial PINN landscape.
    """
    return HyperDEQ_PF_Pilot(
        d=d,
        d_hi=d_hi,
        num_attn_layers=num_attn_layers,
        pinn=pinn,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        forward_iter=K,
        backward_iter=K,
        backward_mode="phantom",
    ).to(device)
