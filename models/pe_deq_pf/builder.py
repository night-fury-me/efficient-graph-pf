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
    """Default: phantom-gradient backward, no Jacobian regularization."""
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


@register_model("PE_DEQ_PF_JacReg")
def _build_jacreg(
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
    """Exact IFT backward + Jacobian regularization (Bai+Koltun+Kolter 2021).

    Jacobian penalty drives F toward contractivity, which is what makes the
    IFT inverse well-conditioned -- giving exact (not phantom) gradients
    once training has stabilized the operator. In practice, the chaotic
    pre-contractive phase can be long and stochastic; prefer
    PE_DEQ_PF_Phantom_JacReg for robust training.
    """
    return PE_DEQ_PF(
        d=d,
        d_hi=d_hi,
        num_attn_layers=num_attn_layers,
        pinn=pinn,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        forward_iter=K,
        backward_iter=K,
        backward_mode="ift",
        jac_reg_weight=0.1,
        jac_reg_n_samples=1,
    ).to(device)


@register_model("PE_DEQ_PF_Stable")
def _build_stable(
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
    """All-three-suggestions recipe:

      1. Curriculum: unrolled K-step BPTT for first 20 epochs, then DEQ.
      2. Strong Jacobian regularization (lambda = 1.0) — 10x what we used in
         PE_DEQ_PF_JacReg, applied during both warmup and DEQ phases.
      3. Architecture: K=5 (shallower), damping_init=0.05 (gentler step),
         spectral_norm on attention-block weights (||W||_2 <= 1 prior).

    The CLI's `K` is overridden to 5 to enforce the recipe.
    """
    return PE_DEQ_PF(
        d=d,
        d_hi=d_hi,
        num_attn_layers=num_attn_layers,
        pinn=pinn,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        forward_iter=5,           # K=5 -- shallower DEQ
        backward_iter=5,
        backward_mode="ift",
        jac_reg_weight=1.0,       # 10x stronger than PE_DEQ_PF_JacReg
        jac_reg_n_samples=1,
        damping_init=0.05,        # gentler step => F closer to identity
        spectral_norm=True,       # ||W||_2 <= 1 on attention weights
        unrolled_warmup_epochs=20,
    ).to(device)


@register_model("PE_DEQ_PF_Phantom_JacReg")
def _build_phantom_jacreg(
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
    """Phantom backward + Jacobian regularization -- the robust DEQ recipe.

    Phantom gradient gives a 1-step backward through the trailing f(z*) call;
    it is always well-defined (does not require contractivity). The Jacobian
    penalty still encourages spectral radius < 1, so the forward solve
    converges to a true fixed point even though the backward never depends
    on it.
    """
    return PE_DEQ_PF(
        d=d,
        d_hi=d_hi,
        num_attn_layers=num_attn_layers,
        pinn=pinn,
        dtheta_max=dtheta_max,
        dvm_frac=dvm_frac,
        forward_iter=K,
        backward_iter=K,
        backward_mode="phantom",
        jac_reg_weight=0.1,
        jac_reg_n_samples=1,
    ).to(device)
